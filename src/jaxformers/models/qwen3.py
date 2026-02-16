import copy
import json
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
from jaxtyping import Array, Float, Int, PyTree
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
    "qkv_embed": FSDP,
    "q_heads": MODEL,
    "kv_heads": MODEL,
    "o_heads": MODEL,
    "mlp_up_embed": FSDP,
    "mlp_up_ffw": MODEL,
    "mlp_down_ffw": MODEL,
    "mlp_down_embed": FSDP,
    "vocab_in": MODEL,
    "vocab_out": None,
}


Config: TypeAlias = PreTrainedConfig


def get_rope_theta(cfg: Config) -> float:
    rope_parameters = getattr(cfg, "rope_parameters", None)
    if (
        not isinstance(rope_parameters, dict)
        or rope_parameters.get("rope_theta") is None
    ):
        raise TypeError(
            "Qwen-3 config must define `rope_parameters` as a dict with a `rope_theta` key "
            "(e.g. {'rope_theta': 1000000, ...})."
        )
    return float(rope_parameters["rope_theta"])


def apply_rope(x: jax.Array, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])  # (B, T)
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))  # (H/2, )
    inp = jnp.einsum(
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
    cfg: Config,
    layer_idx: int,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    kv=None,
    pos=0,
    **inputs,
):
    B, T, D = x.shape
    rules = cfg.sharding_rules

    x_norm = rms_norm(x, w["input_layernorm"], cfg.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    q = jnp.einsum(
        "btd,md->btm",
        x_norm,
        w["q_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "q_heads"), rules),
    )
    k = jnp.einsum(
        "btd,md->btm",
        x_norm,
        w["k_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "kv_heads"), rules),
    )
    v = jnp.einsum(
        "btd,md->btm",
        x_norm,
        w["v_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "kv_heads"), rules),
    )

    q = rearrange(
        q,
        "b t (n h) -> b t n h",
        n=cfg.num_attention_heads,
        h=cfg.head_dim,
    )

    k = rearrange(
        k,
        "b t (k h) -> b t k h",
        k=cfg.num_key_value_heads,
        h=cfg.head_dim,
    )
    v = rearrange(
        v,
        "b t (k h) -> b t k h",
        k=cfg.num_key_value_heads,
        h=cfg.head_dim,
    )

    q = rms_norm(q, w["q_norm"], cfg.rms_norm_eps)
    k = rms_norm(k, w["k_norm"], cfg.rms_norm_eps)

    rope_theta = get_rope_theta(cfg)
    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attn_impl = cfg.additional_config["attn_implementation"]
    attention_interface = ATTENTION_INTERFACE[attn_impl]

    attn_output = attention_interface(
        q, k, v, mask=inputs["attention_mask"]
    )  # (B, T, N, H)

    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
    o = jnp.einsum(
        "btd,ed->bte",
        attn_output,
        w["o_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    x += o
    x_norm = rms_norm(x, w["post_attention_layernorm"], cfg.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    # FFN
    act_fn = jax.nn.silu
    gate = act_fn(
        jnp.einsum(
            "btd,fd->btf",
            x_norm,
            w["gate_proj"],
            preferred_element_type=jnp.float32,
            out_sharding=logical_to_physical(("batch", "context", "mlp_up_ffw"), rules),
        )
    )

    up = jnp.einsum(
        "btd,fd->btf",
        x_norm,
        w["up_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "mlp_up_ffw"), rules),
    )

    x += jnp.einsum(
        "btf,df->btd",
        gate * up,
        w["down_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    return x, kv


def make_mask(config, input_ids, attention_mask=None, segment_ids=None):
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
    **inputs,
):
    B, T = input_ids.shape
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

        if config.additional_config["gradient_checkpointing"]:
            fwd = jax.remat(partial(forward_layer, config, layer_idx))
        else:
            fwd = partial(forward_layer, config, layer_idx)

        input_ids, kv[layer_idx] = fwd(
            input_ids, layer_weights, kv[layer_idx], pos, **inputs
        )  # sharding: (batch, seq, None)

    out_embed = (
        weights["model.embed_tokens.weight"]
        if config.tie_word_embeddings
        else weights["lm_head.weight"]
    )
    input_ids = rms_norm(
        input_ids, weights["model.norm.weight"], config.rms_norm_eps
    )  # (batch, "seq", "None")

    # btd (batch, seq, none), vd (model, none) -> btv (batch, seq, none)
    # to achieve this
    logits = jnp.einsum(
        "btd,vd-> btv",
        input_ids,
        out_embed,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
        preferred_element_type=input_ids.dtype,
    )

    return (logits, kv) if return_kv else logits


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
            return logical_to_physical(("q_heads", "qkv_embed"), sharding_rules)
        if "self_attn.k_proj" in key:
            return logical_to_physical(("q_heads", "qkv_embed"), sharding_rules)
        if "self_attn.v_proj" in key:
            return logical_to_physical(("q_heads", "qkv_embed"), sharding_rules)
        if "mlp.gate_proj" in key:
            return logical_to_physical(("mlp_up_ffw", "mlp_up_embed"), sharding_rules)
        if "mlp.up_proj" in key:
            return logical_to_physical(("mlp_up_ffw", "mlp_up_embed"), sharding_rules)
        if "self_attn.o_proj" in key:
            return logical_to_physical(("qkv_embed", "o_heads"), sharding_rules)
        if "mlp.down_proj" in key:
            return logical_to_physical(
                ("mlp_down_embed", "mlp_down_ffw"), sharding_rules
            )
        if "embed_tokens" in key:
            return logical_to_physical(("vocab_in", "vocab_out"), sharding_rules)
        if "lm_head" in key:
            return logical_to_physical(("vocab_in", "vocab_out"), sharding_rules)
        return P()

    weights = {}
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = jax.device_put(
                    f.get_tensor(key).astype(param_dtype), get_sharding(key)
                )

    return Model(
        config=cfg,
        weights=weights,
        forward=partial(forward, cfg),
        tokenizer=tokenizer,
    )
