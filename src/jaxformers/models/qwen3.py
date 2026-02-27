import copy
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention
from safetensors import safe_open
from transformers import (
    AddedToken,
    AutoConfig,
    PreTrainedConfig,
    PreTrainedTokenizerFast,
)

from jaxformers.distributed.parallel import ParallelDims
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
)
from jaxformers.modeling_utils import (
    AdditionalConfig,
    DEFAULT_ADDITIONAL_CONFIG,
    logical_to_physical,
    Model,
)

from ..attention_utils import ATTENTION_INTERFACE
from ..dispatch.einsum import einsum
from ..distributed import (
    BATCH,
    CONTEXT,
    FSDP,
    MODEL,
    mutate_sharding_rule_parallel_dims,
    SEQ,
)


LayerWeights = TypeVar("LayerWeights")
ModelWeights = TypeVar("ModelWeights")
AxisName = str | tuple[str, ...] | None


SHARDING_RULES = {
    "none": None,
    "batch": BATCH,
    "fsdp": FSDP,
    "model": MODEL,
    "sequence": SEQ,
    "context": CONTEXT,
}


Config: TypeAlias = PreTrainedConfig


def get_rope_theta(cfg: Config) -> float:
    rope_parameters = cfg.rope_parameters
    return float(rope_parameters["rope_theta"])


def apply_rope(x: jax.Array, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])  # (B, T)
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))  # (H/2, )
    inp = einsum(
        "bt,h-> bth", positions, freq, precision=jax.lax.Precision.HIGHEST
    )  # (B, T, H/2)
    x1, x2 = x[:, :, :, : H // 2], x[:, :, :, H // 2 :]  # (B, T, N, H/2)
    sin, cos = (
        jnp.sin(inp).astype(x.dtype)[:, :, None, :],
        jnp.cos(inp).astype(x.dtype)[:, :, None, :],
    )  # (B, T, 1, H/2)

    return jnp.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
    )  # (B, T, N, H)


def rms_norm(x: jax.Array, gamma, eps):
    rms = jnp.sqrt(jnp.pow(x.astype(jnp.float32), 2).mean(-1, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def forward_layer(
    config: Config,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    layer_idx: int,
    kv=None,
    pos=0,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    B, T, D = x.shape
    rules = config.sharding_rules

    x_norm = rms_norm(x, w["input_layernorm"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    q = einsum(
        "btd,md->btm",
        x_norm,
        w["q_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )
    k = einsum(
        "btd,md->btm",
        x_norm,
        w["k_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )
    v = einsum(
        "btd,md->btm",
        x_norm,
        w["v_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )

    q = rearrange(
        q,
        "b t (n h) -> b t n h",
        n=config.num_attention_heads,
        h=config.head_dim,
    )

    k = rearrange(
        k,
        "b t (k h) -> b t k h",
        k=config.num_key_value_heads,
        h=config.head_dim,
    )
    v = rearrange(
        v,
        "b t (k h) -> b t k h",
        k=config.num_key_value_heads,
        h=config.head_dim,
    )

    q = rms_norm(q, w["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, w["k_norm"], config.rms_norm_eps)

    rope_theta = get_rope_theta(config)
    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attn_impl = config.additional_config["attn_implementation"]
    attention_interface = ATTENTION_INTERFACE[attn_impl]
    q_sharding = jax.NamedSharding(
        jax.sharding.get_abstract_mesh(),
        logical_to_physical(
            ("batch", "context", "model", "none"), config.sharding_rules
        ),
    )
    attn_output = attention_interface(
        q, k, v, mask=inputs["attention_mask"], q_sharding=q_sharding
    )  # (B, T, N, H)

    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
    o = einsum(
        "btd,ed->bte",
        attn_output,
        w["o_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    x += o
    x_norm = rms_norm(x, w["post_attention_layernorm"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    # FFN
    act_fn = jax.nn.silu
    gate = act_fn(
        einsum(
            "btd,fd->btf",
            x_norm,
            w["gate_proj"],
            preferred_element_type=jnp.float32,
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
    )

    up = einsum(
        "btd,fd->btf",
        x_norm,
        w["up_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )

    x += einsum(
        "btf,df->btd",
        gate * up,
        w["down_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    return x, kv


def make_mask(config, input_ids, attention_mask=None, segment_ids=None, **kwargs):
    attn_implementation = config.additional_config["attn_implementation"]
    if attn_implementation not in ATTENTION_MASK_INTERFACE:
        return None

    if attention_mask is not None:
        attention_mask = attention_mask.astype(jnp.bool_)

    attention_mask = make_causal_mask(
        attn_implementation, input_ids, attention_mask, segment_ids
    )
    return attention_mask


def forward(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    input_ids: Int[Array, "B T"],
    kv: None = None,
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    rules = config.sharding_rules
    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    input_ids = (
        weights["model.embed_tokens.weight"]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )

    return_kv = kv is not None
    if kv is None:
        kv = defaultdict(lambda: None)
    inputs["attention_mask"] = make_mask(config, input_ids, **inputs)

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}."
        layer_weights = {
            "input_layernorm": weights[f"{prefix}input_layernorm.weight"],
            "post_attention_layernorm": weights[
                f"{prefix}post_attention_layernorm.weight"
            ],
            "q_proj": weights[f"{prefix}self_attn.q_proj.weight"],
            "k_proj": weights[f"{prefix}self_attn.k_proj.weight"],
            "v_proj": weights[f"{prefix}self_attn.v_proj.weight"],
            "o_proj": weights[f"{prefix}self_attn.o_proj.weight"],
            "q_norm": weights[f"{prefix}self_attn.q_norm.weight"],
            "k_norm": weights[f"{prefix}self_attn.k_norm.weight"],
            "gate_proj": weights[f"{prefix}mlp.gate_proj.weight"],
            "up_proj": weights[f"{prefix}mlp.up_proj.weight"],
            "down_proj": weights[f"{prefix}mlp.down_proj.weight"],
        }

        if config.additional_config["remat_layer"]:
            fwd = jax.remat(partial(forward_layer, config))
        else:
            fwd = partial(forward_layer, config)

        input_ids, kv[layer_idx] = fwd(
            input_ids, layer_weights, layer_idx, kv[layer_idx], pos, **inputs
        )  # sharding: (batch, seq, None)


    input_ids = rms_norm(
        input_ids, weights["model.norm.weight"], config.rms_norm_eps
    )  # (batch, "seq", "None")

    return (input_ids, kv) if return_kv else input_ids


def embed(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    input_ids: Int[Array, "B T"],
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    rules = config.sharding_rules
    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    input_ids = (
        weights["model.embed_tokens.weight"]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )
    return input_ids

def unembed(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    input_ids: Int[Array, "B T H"],
    dtype: jnp.dtype = jnp.float32,
    compute_dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    rules = config.sharding_rules

    # loss_sharding = (
    #     ("batch", "context", "model")
    #     if config.additional_config.get("loss_parallel")
    #     else ("batch", "context", "none")
    # )

    out_embed = (
        weights["model.embed_tokens.weight"]
        if config.tie_word_embeddings
        else weights["lm_head.weight"]
    )

    logits = einsum(
        "btd,vd-> btv",
        input_ids,
        out_embed,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        preferred_element_type=compute_dtype,
    )

    return logits.astype(dtype)


def save_safetensors(weights: PyTree[ModelWeights], path: str | Path):
    pass


def load(
    model_id: str,
    parallel_dims: ParallelDims,
    devices: list | None = None,
    local_dir: str | None = None,
    multihost: bool = False,
    additional_config: AdditionalConfig | None = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> Model:

    additional_config = {
        **DEFAULT_ADDITIONAL_CONFIG,
        **(additional_config or {}),
    }

    sharding_rules = mutate_sharding_rule_parallel_dims(
        dict(SHARDING_RULES),
        parallel_dims,
        sequence_parallelism=additional_config["sequence_parallelism"],
    )

    model_ckpt_dir = Path(snapshot_download(repo_id=model_id, local_dir=local_dir))

    tokenizer_config_path = model_ckpt_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())

    tokenizer_file = str(model_ckpt_dir / "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        added_tokens_decoder={
            int(k): AddedToken(**v)
            for k, v in tokenizer_config["added_tokens_decoder"].items()
        },
    )
    if "chat_template" in tokenizer_config:
        tokenizer.chat_template = tokenizer_config["chat_template"]

    cfg = AutoConfig.from_pretrained(model_ckpt_dir)
    if not isinstance(cfg, PreTrainedConfig):
        raise TypeError(f"Expected HF config, got {type(cfg)!r}")
    cfg.additional_config = additional_config
    cfg.parallel_dims = parallel_dims
    cfg.sharding_rules = sharding_rules

    if multihost:
        jax.distributed.initialize()

    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(AxisType.Explicit for _ in axis_names)
    mesh = jax.make_mesh(
        axis_shapes, axis_names, axis_types=axis_types, devices=devices
    )
    jax.set_mesh(mesh)

    def get_sharding(key):
        if "self_attn.q_proj" in key:
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        if "self_attn.k_proj" in key:
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        if "self_attn.v_proj" in key:
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        if "mlp.gate_proj" in key:
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        if "mlp.up_proj" in key:
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        if "self_attn.o_proj" in key:
            return logical_to_physical(("fsdp", "model"), sharding_rules)
        if "mlp.down_proj" in key:
            return logical_to_physical(("fsdp", "model"), sharding_rules)
        if "embed_tokens" in key:
            return logical_to_physical(("model", "none"), sharding_rules)
        if "lm_head" in key:
            return logical_to_physical(("model", "none"), sharding_rules)
        return P()

    weights = {}
    t0 = time.monotonic()
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = jax.device_put(
                    f.get_tensor(key).astype(param_dtype), get_sharding(key)
                )
    diff = time.monotonic() - t0
    print(f"model loaded at {diff}s ")

    return Model(
        config=cfg,
        weights=weights,
        forward=partial(forward, cfg),
        tokenizer=tokenizer,
        embed = partial(embed, cfg),
        unembed = partial(unembed, cfg),
        lm_head_key = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm.head.weight"
    )


def apply_rope_ragged(x: jax.Array, theta: float, positions: jax.Array) -> jax.Array:
    # x: [T, N, H]
    T, N, H = x.shape
    if H % 2 != 0:
        raise ValueError("RoPE head_dim must be even")

    pos = positions.astype(jnp.float32)
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))  # (H/2,)
    inp = einsum("t,h->th", pos, freq, precision=jax.lax.Precision.HIGHEST)  # (T,H/2)

    x1, x2 = x[:, :, : H // 2], x[:, :, H // 2 :]
    sin = jnp.sin(inp).astype(x.dtype)[:, None, :]
    cos = jnp.cos(inp).astype(x.dtype)[:, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_rope_cache(
    *,
    max_model_len: int,
    head_dim: int,
    theta: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    if max_model_len < 1:
        raise ValueError("max_model_len must be >= 1")

    pos = jnp.arange(max_model_len, dtype=jnp.float32)[:, None]
    freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    inp = pos * freq[None, :]
    sin = jnp.sin(inp).astype(dtype)
    cos = jnp.cos(inp).astype(dtype)
    return sin, cos


def apply_rope_ragged_cached(
    x: jax.Array,
    rope_sin: jax.Array,
    rope_cos: jax.Array,
    positions: jax.Array,
) -> jax.Array:
    T, N, H = x.shape
    if H % 2 != 0:
        raise ValueError("RoPE head_dim must be even")

    sin = rope_sin[positions][:, None, :].astype(x.dtype)
    cos = rope_cos[positions][:, None, :].astype(x.dtype)

    x1, x2 = x[:, :, : H // 2], x[:, :, H // 2 :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def forward_layer_inference_ragged(
    config: Config,
    x: Float[Array, "T D"],
    w: dict[str, Array],
    ragged_attention,
    kv_pages: jax.Array,
    positions: jax.Array,
    page_ids: jax.Array,
    page_offsets: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    rope_sin: jax.Array,
    rope_cos: jax.Array,
    *,
    dtype: jnp.dtype,
):
    rules = config.sharding_rules

    x_norm = rms_norm(x, w["input_layernorm"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "none"), rules))

    q = einsum(
        "td,md->tm",
        x_norm,
        w["q_proj"],
        preferred_element_type=dtype,
        out_sharding=logical_to_physical(("batch", "model"), rules),
    )
    k = einsum(
        "td,md->tm",
        x_norm,
        w["k_proj"],
        preferred_element_type=dtype,
        out_sharding=logical_to_physical(("batch", "model"), rules),
    )
    v = einsum(
        "td,md->tm",
        x_norm,
        w["v_proj"],
        preferred_element_type=dtype,
        out_sharding=logical_to_physical(("batch", "model"), rules),
    )

    q = rearrange(
        q,
        "t (n h) -> t n h",
        n=config.num_attention_heads,
        h=config.head_dim,
    )
    k = rearrange(
        k,
        "t (k h) -> t k h",
        k=config.num_key_value_heads,
        h=config.head_dim,
    )
    v = rearrange(
        v,
        "t (k h) -> t k h",
        k=config.num_key_value_heads,
        h=config.head_dim,
    )

    q = rms_norm(q, w["q_norm"], config.rms_norm_eps)
    k = rms_norm(k, w["k_norm"], config.rms_norm_eps)

    q = apply_rope_ragged_cached(q, rope_sin, rope_cos, positions)
    k = apply_rope_ragged_cached(k, rope_sin, rope_cos, positions)

    kv = jnp.stack([k, v], axis=2).reshape(
        k.shape[0], config.num_key_value_heads * 2, config.head_dim
    )
    kv_pages = kv_pages.at[page_ids, page_offsets].set(
        kv,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        wrap_negative_indices=False,
    )

    attn_out = ragged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)

    attn_out = rearrange(attn_out, "t n h -> t (n h)")
    o = einsum(
        "td,ed->te",
        attn_out,
        w["o_proj"],
        preferred_element_type=dtype,
        out_sharding=logical_to_physical(("batch", "none"), rules),
    )
    x = x + o

    x_norm = rms_norm(x, w["post_attention_layernorm"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "none"), rules))

    gate = jax.nn.silu(
        einsum(
            "td,fd->tf",
            x_norm,
            w["gate_proj"],
            preferred_element_type=dtype,
            out_sharding=logical_to_physical(("batch", "model"), rules),
        )
    )
    up = einsum(
        "td,fd->tf",
        x_norm,
        w["up_proj"],
        preferred_element_type=dtype,
        out_sharding=logical_to_physical(("batch", "model"), rules),
    )
    x = x + einsum(
        "tf,df->td",
        gate * up,
        w["down_proj"],
        preferred_element_type=dtype,
        out_sharding=logical_to_physical(("batch", "none"), rules),
    )

    return x, kv_pages


def forward_inference_ragged_paged(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    kv_cache: tuple[jax.Array, ...],
    token_ids: jax.Array,
    positions: jax.Array,
    page_ids: jax.Array,
    page_offsets: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    rope_sin: jax.Array,
    rope_cos: jax.Array,
    *,
    dtype: jnp.dtype,
    ragged_attention,
):
    rules = config.sharding_rules
    x = (
        weights["model.embed_tokens.weight"]
        .at[token_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "none"), rules))
        .astype(dtype)
    )

    new_kv_cache: list[jax.Array] = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}."
        layer_weights = {
            "input_layernorm": weights[f"{prefix}input_layernorm.weight"],
            "post_attention_layernorm": weights[
                f"{prefix}post_attention_layernorm.weight"
            ],
            "q_proj": weights[f"{prefix}self_attn.q_proj.weight"],
            "k_proj": weights[f"{prefix}self_attn.k_proj.weight"],
            "v_proj": weights[f"{prefix}self_attn.v_proj.weight"],
            "o_proj": weights[f"{prefix}self_attn.o_proj.weight"],
            "q_norm": weights[f"{prefix}self_attn.q_norm.weight"],
            "k_norm": weights[f"{prefix}self_attn.k_norm.weight"],
            "gate_proj": weights[f"{prefix}mlp.gate_proj.weight"],
            "up_proj": weights[f"{prefix}mlp.up_proj.weight"],
            "down_proj": weights[f"{prefix}mlp.down_proj.weight"],
        }

        x, kv_pages = forward_layer_inference_ragged(
            config,
            x,
            layer_weights,
            ragged_attention,
            kv_cache[layer_idx],
            positions,
            page_ids,
            page_offsets,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            rope_sin,
            rope_cos,
            dtype=dtype,
        )
        new_kv_cache.append(kv_pages)

    x = rms_norm(x, weights["model.norm.weight"], config.rms_norm_eps).astype(dtype)
    return x, tuple(new_kv_cache)
