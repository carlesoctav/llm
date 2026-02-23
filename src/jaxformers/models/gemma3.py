from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PyTree
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedConfig,
)

from jaxformers.distributed.parallel import ParallelDims
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
    make_sliding_window_causal_mask,
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


def apply_rope(x: jax.Array, theta: float, pos=0):
    bsz, seqlen, _nheads, head_dim = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(seqlen)[None, :], [bsz, seqlen])
    freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    inp = einsum(
        "bt,h->bth",
        positions,
        freq,
        precision=jax.lax.Precision.HIGHEST,
    )
    x1, x2 = x[:, :, :, : head_dim // 2], x[:, :, :, head_dim // 2 :]
    sin, cos = (
        jnp.sin(inp).astype(x.dtype)[:, :, None, :],
        jnp.cos(inp).astype(x.dtype)[:, :, None, :],
    )
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def gemma_rms_norm(x: jax.Array, weight: jax.Array, eps: float):
    x_fp32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.square(x_fp32).mean(-1, keepdims=True) + eps)
    out = (x_fp32 / rms) * (1.0 + weight.astype(jnp.float32))
    return out.astype(x.dtype)


def get_activation_fn(hidden_activation: str) -> Callable[[jax.Array], jax.Array]:
    if hidden_activation in ("gelu_pytorch_tanh", "gelu_new"):
        return lambda x: jax.nn.gelu(x, approximate=True)
    if hidden_activation == "gelu":
        return lambda x: jax.nn.gelu(x, approximate=False)
    if hidden_activation == "silu":
        return jax.nn.silu
    if hidden_activation == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported hidden activation {hidden_activation!r}")


def get_rope_theta(cfg: Config, attention_type: str) -> float:
    rope_parameters = getattr(cfg, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        raise TypeError(
            "Gemma-3 config must define `rope_parameters` as a dict with per-attention-type "
            "settings (e.g. {'full_attention': {'rope_theta': ...}, ...})."
        )

    attn_cfg = rope_parameters.get(attention_type)
    if not isinstance(attn_cfg, dict) or attn_cfg.get("rope_theta") is None:
        raise KeyError(
            f"Missing `rope_theta` for attention_type={attention_type!r} in `rope_parameters`. "
            f"Available keys: {sorted(rope_parameters.keys())!r}"
        )
    return float(attn_cfg["rope_theta"])


def make_mask_mapping(config, input_embeds, attention_mask=None, segment_ids=None, **kwargs):
    attn_impl = config.additional_config["attn_implementation"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return {
            "full_attention": None,
            "sliding_attention": None,
        }

    if attention_mask is not None:
        attention_mask = attention_mask.astype(jnp.bool_)

    full_mask = make_causal_mask(attn_impl, input_embeds, attention_mask, segment_ids)
    window_size = getattr(config, "sliding_window", None)
    if window_size is None:
        sliding_mask = full_mask
    else:
        sliding_mask = make_sliding_window_causal_mask(
            attn_impl,
            input_embeds,
            int(window_size),
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    if attn_impl == "sdpa":
        if full_mask is not None and full_mask.ndim == 3:
            full_mask = full_mask[:, None, :, :]
        if sliding_mask is not None and sliding_mask.ndim == 3:
            sliding_mask = sliding_mask[:, None, :, :]

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


def linear_3d(x, w, b=None, *, out_sharding=None):
    y = einsum(
        "btd,md->btm",
        x,
        w,
        preferred_element_type=x.dtype,
        out_sharding=out_sharding,
    )
    if b is not None:
        y = y + b[None, None, :]
    return y


def forward_layer(
    cfg: Config,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    layer_idx: int,
    rope_theta: float,
    kv=None,
    pos=0,
    **inputs,
):
    rules = cfg.sharding_rules
    act_fn = get_activation_fn(cfg.hidden_activation)
    head_dim = cfg.head_dim

    residual = x
    x_norm = gemma_rms_norm(x, w["input_layernorm"], cfg.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    q = linear_3d(
        x_norm,
        w["q_proj"],
        w.get("q_proj_bias"),
        out_sharding=logical_to_physical(("batch", "context", "none"), rules),
    )
    k = linear_3d(
        x_norm,
        w["k_proj"],
        w.get("k_proj_bias"),
        out_sharding=logical_to_physical(("batch", "context", "none"), rules),
    )
    v = linear_3d(
        x_norm,
        w["v_proj"],
        w.get("v_proj_bias"),
        out_sharding=logical_to_physical(("batch", "context", "none"), rules),
    )

    q = rearrange(q, "b t (n h) -> b t n h", n=cfg.num_attention_heads, h=head_dim)
    k = rearrange(k, "b t (k h) -> b t k h", k=cfg.num_key_value_heads, h=head_dim)
    v = rearrange(v, "b t (k h) -> b t k h", k=cfg.num_key_value_heads, h=head_dim)

    q = gemma_rms_norm(q, w["q_norm"], cfg.rms_norm_eps)
    k = gemma_rms_norm(k, w["k_norm"], cfg.rms_norm_eps)

    query_pre_attn_scalar = float(getattr(cfg, "query_pre_attn_scalar", cfg.head_dim))
    q = q * jnp.sqrt(jnp.array(cfg.head_dim / query_pre_attn_scalar, dtype=q.dtype))

    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attn_impl = cfg.additional_config["attn_implementation"]
    attention_interface = ATTENTION_INTERFACE[attn_impl]
    attn_output = attention_interface(q, k, v, mask=inputs["attention_mask"])
    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
    attn_output = linear_3d(
        attn_output,
        w["o_proj"],
        w.get("o_proj_bias"),
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    attn_output = gemma_rms_norm(
        attn_output,
        w["post_attention_layernorm"],
        cfg.rms_norm_eps,
    )
    x = residual + attn_output

    residual = x
    x_norm = gemma_rms_norm(x, w["pre_feedforward_layernorm"], cfg.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    gate = act_fn(
        linear_3d(
            x_norm,
            w["gate_proj"],
            w.get("gate_proj_bias"),
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
    )
    up = linear_3d(
        x_norm,
        w["up_proj"],
        w.get("up_proj_bias"),
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )
    ffw = einsum(
        "btf,df->btd",
        gate * up,
        w["down_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )
    if w.get("down_proj_bias") is not None:
        ffw = ffw + w["down_proj_bias"][None, None, :]

    ffw = gemma_rms_norm(ffw, w["post_feedforward_layernorm"], cfg.rms_norm_eps)
    x = residual + ffw
    return x, kv


def forward(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    input_ids: Int[Array, "B T"],
    kv: None = None,
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    attention_mask: Int[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    logits_to_keep: int = 0,
    **inputs,
):
    rules = config.sharding_rules
    model_prefix = "model"
    embed_key = f"{model_prefix}.embed_tokens.weight"
    final_norm_key = f"{model_prefix}.norm.weight"
    lm_head_key = "lm_head.weight"

    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    x = (
        weights[embed_key]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )
    x *= jnp.sqrt(jnp.array(config.hidden_size, dtype=dtype))

    return_kv = kv is not None
    if kv is None:
        kv = defaultdict(lambda: None)

    mask_mapping = make_mask_mapping(
        config,
        x,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    layer_types = config.layer_types
    for layer_idx in range(int(config.num_hidden_layers)):
        attention_type = layer_types[layer_idx % len(layer_types)]
        rope_theta = get_rope_theta(config, attention_type)

        prefix = f"{model_prefix}.layers.{layer_idx}."
        layer_weights = {
            "input_layernorm": weights[f"{prefix}input_layernorm.weight"],
            "post_attention_layernorm": weights[
                f"{prefix}post_attention_layernorm.weight"
            ],
            "pre_feedforward_layernorm": weights[
                f"{prefix}pre_feedforward_layernorm.weight"
            ],
            "post_feedforward_layernorm": weights[
                f"{prefix}post_feedforward_layernorm.weight"
            ],
            "q_proj": weights[f"{prefix}self_attn.q_proj.weight"],
            "q_proj_bias": weights.get(f"{prefix}self_attn.q_proj.bias"),
            "k_proj": weights[f"{prefix}self_attn.k_proj.weight"],
            "k_proj_bias": weights.get(f"{prefix}self_attn.k_proj.bias"),
            "v_proj": weights[f"{prefix}self_attn.v_proj.weight"],
            "v_proj_bias": weights.get(f"{prefix}self_attn.v_proj.bias"),
            "o_proj": weights[f"{prefix}self_attn.o_proj.weight"],
            "o_proj_bias": weights.get(f"{prefix}self_attn.o_proj.bias"),
            "q_norm": weights[f"{prefix}self_attn.q_norm.weight"],
            "k_norm": weights[f"{prefix}self_attn.k_norm.weight"],
            "gate_proj": weights[f"{prefix}mlp.gate_proj.weight"],
            "gate_proj_bias": weights.get(f"{prefix}mlp.gate_proj.bias"),
            "up_proj": weights[f"{prefix}mlp.up_proj.weight"],
            "up_proj_bias": weights.get(f"{prefix}mlp.up_proj.bias"),
            "down_proj": weights[f"{prefix}mlp.down_proj.weight"],
            "down_proj_bias": weights.get(f"{prefix}mlp.down_proj.bias"),
        }

        if config.additional_config["remat_layer"]:
            fwd = jax.remat(partial(forward_layer, config))
        else:
            fwd = partial(forward_layer, config)

        layer_mask = mask_mapping.get(attention_type)
        x, kv[layer_idx] = fwd(
            x,
            layer_weights,
            layer_idx,
            rope_theta,
            kv[layer_idx],
            pos,
            attention_mask=layer_mask,
            **inputs,
        )

    x = gemma_rms_norm(x, weights[final_norm_key], config.rms_norm_eps)
    if getattr(config, "tie_word_embeddings", True) or lm_head_key not in weights:
        out_embed = weights[embed_key]
    else:
        out_embed = weights[lm_head_key]

    if logits_to_keep:
        x = x[:, -int(logits_to_keep) :, :]

    loss_sharding = (
        ("batch", "context", "model")
        if config.additional_config.get("loss_parallel")
        else ("batch", "context", "none")
    )
    logits = einsum(
        "btd,vd->btv",
        x,
        out_embed,
        out_sharding=logical_to_physical(loss_sharding, rules),
        preferred_element_type=x.dtype,
    )

    softcap = getattr(config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = jnp.tanh(logits / softcap) * softcap

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
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_dir, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_ckpt_dir)
    if not isinstance(cfg, PreTrainedConfig):
        raise TypeError(f"Expected HF config, got {type(cfg)!r}")

    if multihost:
        jax.distributed.initialize()

    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(AxisType.Explicit for _ in axis_names)
    mesh = jax.make_mesh(
        axis_shapes,
        axis_names,
        axis_types=axis_types,
        devices=devices,
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
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = jax.device_put(
                    f.get_tensor(key).astype(param_dtype),
                    get_sharding(key),
                )

    if "model.embed_tokens.weight" not in weights:
        raise KeyError(
            "Could not locate Gemma-3 text embedding weights. "
            "Expected 'model.embed_tokens.weight'. "
            "If this is a multimodal checkpoint, load it via your gemma3_mm module."
        )

    cfg.additional_config = additional_config
    cfg.parallel_dims = parallel_dims
    cfg.sharding_rules = sharding_rules

    return Model(
        config=cfg,
        weights=weights,
        forward=partial(forward, cfg),
        tokenizer=tokenizer,
    )
