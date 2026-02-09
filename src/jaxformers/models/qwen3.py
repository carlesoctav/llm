import copy
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PyTree
from requests.utils import DEFAULT_ACCEPT_ENCODING
from safetensors import safe_open
from transformers import AddedToken, PreTrainedTokenizerFast

from jaxformers.attention_utils import eager_dot_product_attention
from jaxformers.distributed.parallel import check_mesh_axis_for_inference, ParallelDims
from jaxformers.masking_utils import ATTENTION_MASK_INTERFACE

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


def logical_to_physical(logical, rules):
    spec = [rules[lo] for lo in logical]
    flat_leaves = jtu.tree_leaves(spec)
    if len(flat_leaves) != len(set(flat_leaves)):
        raise ValueError(
            f"Colliding physical axes from translating logical spec {logical} -> {spec}"
        )

    return P(*spec)


class AdditionalConfig(TypedDict):
    # training
    gradient_checkpointing: bool = True

    # training and inference
    attn_implementation: str = "sdpa"

    # inference
    max_num_batched_token: int = 2048
    max_model_len: int = 2048
    max_num_request: int = 256
    page_size: int = 128
    num_pages: int  # (max_model_len * max_num_request // page_size)


DEFAULT_ADDITIONAL_CONFIG = {
    "gradient_checkpointing": True,
    "attn_implementation": "sdpa",
    "max_num_batched_token": 2048,
    "max_model_len": 2048,
    "max_num_request": 256,
    "page_size": 256,
}


def mutable_check_additional_config_for_inference(config: Config):
    additional_config = config["additional_config"]
    additional_config["attn_implementation"] = "ragged_paged_dot_product_attention"
    additional_config["gradient_checkpointing"] = False
    page_size = additional_config["page_size"]

    additional_config["num_pages"] = (
        additional_config["max_model_len"] * additional_config["max_num_request"]
        + page_size
        - 1
    ) // page_size


class Config(TypedDict):
    model_type: str = "qwen3"
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: dict[str, Any]
    rope_theta: int
    sliding_window: int | None
    tie_word_embeddings: bool
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    additional_config: AdditionalConfig

    # come from args
    sharding_rules: dict[str, AxisName]
    parallel_dims: ParallelDims


@dataclass
class Model:
    config: Config
    weights: PyTree[Array, "ModelWeights"]
    forward: Callable
    tokenizer: PreTrainedTokenizerFast


@dataclass
class InferenceModel:
    config: Config
    weights: PyTree[Array, "ModelWeights"]
    forward: Callable
    compute_logits: Callable
    forward_embedding: Callable
    init_kv: Callable
    tokenizer: PreTrainedTokenizerFast


def init_kv(L, K, H, num_pages, page_size, dtype = jnp.bfloat16):
    kv = [jnp.zeros((num_pages, page_size, 2*K, H), dtype = dtype) for _ in range(L)]
    return kv


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
    rules = cfg.get("sharding_rules", SHARDING_RULES)

    x_norm = rms_norm(x, w["input_layernorm"], cfg["rms_norm_eps"])
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
        n=cfg["num_attention_heads"],
        h=cfg["head_dim"],
    )

    k = rearrange(
        k,
        "b t (k h) -> b t k h",
        k=cfg["num_key_value_heads"],
        h=cfg["head_dim"],
    )
    v = rearrange(
        v,
        "b t (k h) -> b t k h",
        k=cfg["num_key_value_heads"],
        h=cfg["head_dim"],
    )

    q = rms_norm(q, w["q_norm"], cfg["rms_norm_eps"])
    k = rms_norm(k, w["k_norm"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg["rope_theta"], pos)
    k = apply_rope(k, cfg["rope_theta"], pos)

    if kv is not None:
        raise NotImplementedError
        # kv = jax.lax.dynamic_update_slice(kv, jnp.stack([k, v]), (0, 0, pos, 0, 0))
        # k, v = kv

    attn_impl = cfg["additional_config"]["attn_implementation"]
    attention_fn = ATTENTION_INTERFACE[attn_impl]

    # think again about this
    S = k.shape[1]
    q_pos = pos + jnp.arange(T, dtype=jnp.int32)
    kv_pos = jnp.arange(S, dtype=jnp.int32)
    causal = q_pos[:, None] >= kv_pos[None, :]  # (T, S)
    padding = inputs.get("attention_mask", None)
    if padding is not None:
        # padding is typically (B, S) with 1 for tokens, 0 for padding
        pad = padding.astype(bool)
        attn_mask = causal[None, None, :, :] & pad[:, None, None, :]
        attn_mask = jnp.broadcast_to(attn_mask, (B, cfg["num_attention_heads"], T, S))
    else:
        attn_mask = causal  # (T, S) broadcasted by the attention implementation

    attn_output = attention_fn(q, k, v, mask=attn_mask)  # (B, T, N, H)

    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
    o = jnp.einsum(
        "btd,ed->bte",
        attn_output,
        w["o_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    x += o
    x_norm = rms_norm(x, w["post_attention_layernorm"], cfg["rms_norm_eps"])
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


def compute_logits(cfg, input_ids, weights):
    pass

def embed_tokens(config, input_ids, weights):
    pass


def forward(
    cfg: Config,
    input_ids: Int[Array, "B T"],
    weights: PyTree[Array, "ModelWeights"],
    kv: None = None,
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    **inputs,
):
    B, T = input_ids.shape
    rules = cfg.get("sharding_rules", SHARDING_RULES)
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

    for layer_idx in range(cfg["num_hidden_layers"]):
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

        if cfg.get("additional_config", {}).get("gradient_checkpointing", None):
            fwd = jax.remat(partial(forward_layer, cfg, layer_idx))
        else:
            fwd = partial(forward_layer, cfg, layer_idx)

        input_ids, kv[layer_idx] = fwd(
            input_ids, layer_weights, kv[layer_idx], pos, **inputs
        )  # sharding: (batch, seq, None)

    out_embed = (
        weights["model.embed_tokens.weight"]
        if cfg["tie_word_embeddings"]
        else weights["lm_head.weight"]
    )
    input_ids = rms_norm(
        input_ids, weights["model.norm.weight"], cfg["rms_norm_eps"]
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
    hf_ckpt_dir: str = "~/weights/huggingface",
    multihost: bool = False,
    config_kwargs: AdditionalConfig | None = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> Model:

    if not config_kwargs:
        config_kwargs = dict(DEFAULT_ADDITIONAL_CONFIG)

    sharding_rules = mutate_sharding_rule_parallel_dims(
        dict(SHARDING_RULES), parallel_dims
    )
    model_ckpt_dir = Path(hf_ckpt_dir).expanduser() / model_id

    if not model_ckpt_dir.exists():
        snapshot_download(repo_id=model_id, local_dir=model_ckpt_dir)

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

    cfg_path = model_ckpt_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg = Config(
        **cfg,
        **config_kwargs,
        parallel_dims=parallel_dims,
        sharding_rules=sharding_rules,
    )

    L, N, K, H, D = (
        cfg["num_hidden_layers"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
        cfg["hidden_size"],
    )

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
        tokenizer = tokenizer,
    )


def load_inference(
    model_or_model_id: Model | str,
    parallel_dims: ParallelDims,
    devices: list | None = None,
    hf_ckpt_dir: str = "~/weights/huggingface",
    multihost: bool = False,
    config_kwargs: AdditionalConfig | None = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    kv_dtype: jnp.dtype = jnp.bfloat16,
):
    if isinstance(model_or_model_id, Model):
        model = model_or_model_id
        print("using cololcated mode by passing Model")
    elif isinstance(model_or_model_id, str):
        model_id = model_or_model_id
        model = load(
            model_id=model_id,
            parallel_dims=parallel_dims,
            devices=devices,
            hf_ckpt_dir=hf_ckpt_dir,
            multihost=multihost,
            config_kwargs=config_kwargs,
            param_dtype=param_dtype,
        )
    else:
        raise TypeError("model_or_model_id must be a Model instance or a model id string")

    config = copy.deepcopy(model.config)
    check_mesh_axis_for_inference(config["parallel_dims"])
    mutable_check_additional_config_for_inference(config)

    L, K, H, num_pages, page_size = (
        config["num_hidden_layers"],
        config["num_key_value_heads"],
        config["head_dim"],
        config["additional_config"]["num_pages"],
        config["additional_config"]["page_size"],
    )

    return InferenceModel(
        config=config,
        weights=model.weights,
        forward=partial(forward, config),
        init_kv=partial(init_kv, L, K, H, num_pages, page_size, kv_dtype),
        compute_logits=partial(compute_logits, config),
        embed_tokens=partial(embed_tokens, config),
        tokenizer = model.tokenizer
    )
