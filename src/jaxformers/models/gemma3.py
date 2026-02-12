import json
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypedDict, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PyTree
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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


class Config(TypedDict):
    model_type: str
    attention_bias: bool
    attention_dropout: float
    attn_logit_softcapping: float | None
    bos_token_id: int
    eos_token_id: int
    final_logit_softcapping: float | None
    head_dim: int
    hidden_activation: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_types: list[str]
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int | None
    query_pre_attn_scalar: float
    rms_norm_eps: float
    rope_parameters: dict[str, Any] | None
    sliding_window: int | None
    tie_word_embeddings: bool
    use_bidirectional_attention: bool
    use_cache: bool
    vocab_size: int

    additional_config: AdditionalConfig

    # filled in load()
    sharding_rules: dict[str, AxisName]
    parallel_dims: ParallelDims
    model_prefix: str
    lm_head_key: str


@dataclass
class Model:
    config: Config
    weights: PyTree[Array, "ModelWeights"]
    forward: Callable
    tokenizer: PreTrainedTokenizerBase
    opt_state: PyTree["ModelWeights"] | None = None


def apply_rope(x: jax.Array, theta: float, pos=0):
    bsz, seqlen, _nheads, head_dim = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(seqlen)[None, :], [bsz, seqlen])
    freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    inp = jnp.einsum(
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


def get_rope_theta(cfg: Config, layer_idx: int) -> float:
    rope_parameters = cfg.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        layer_types = cfg.get("layer_types")
        if isinstance(layer_types, list) and layer_idx < len(layer_types):
            layer_rope = rope_parameters.get(layer_types[layer_idx])
            if (
                isinstance(layer_rope, dict)
                and layer_rope.get("rope_theta") is not None
            ):
                return float(layer_rope["rope_theta"])
        for rope_cfg in rope_parameters.values():
            if isinstance(rope_cfg, dict) and rope_cfg.get("rope_theta") is not None:
                return float(rope_cfg["rope_theta"])

    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_idx < len(layer_types):
        layer_type = layer_types[layer_idx]
        if (
            layer_type == "sliding_attention"
            and cfg.get("rope_local_base_freq") is not None
        ):
            return float(cfg["rope_local_base_freq"])
        if layer_type == "full_attention" and cfg.get("rope_theta") is not None:
            return float(cfg["rope_theta"])
            
    if cfg.get("rope_theta") is not None:
        return float(cfg["rope_theta"])
    return 10_000.0


def make_mask_mapping(config, input_embeds, attention_mask=None, segment_ids=None):
    attn_impl = config["additional_config"]["attn_implementation"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return {
            "full_attention": None,
            "sliding_attention": None,
        }

    if attention_mask is not None:
        attention_mask = attention_mask.astype(jnp.bool_)

    full_mask = make_causal_mask(attn_impl, input_embeds, attention_mask, segment_ids)
    window_size = config.get("sliding_window")
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
    y = jnp.einsum(
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
    layer_idx: int,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    kv=None,
    pos=0,
    **inputs,
):
    rules = cfg["sharding_rules"]
    act_fn = get_activation_fn(cfg["hidden_activation"])
    head_dim = cfg["head_dim"]

    residual = x
    x_norm = gemma_rms_norm(x, w["input_layernorm"], cfg["rms_norm_eps"])
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

    q = rearrange(q, "b t (n h) -> b t n h", n=cfg["num_attention_heads"], h=head_dim)
    k = rearrange(k, "b t (k h) -> b t k h", k=cfg["num_key_value_heads"], h=head_dim)
    v = rearrange(v, "b t (k h) -> b t k h", k=cfg["num_key_value_heads"], h=head_dim)

    q = gemma_rms_norm(q, w["q_norm"], cfg["rms_norm_eps"])
    k = gemma_rms_norm(k, w["k_norm"], cfg["rms_norm_eps"])

    query_pre_attn_scalar = float(cfg.get("query_pre_attn_scalar", cfg["head_dim"]))
    q = q * jnp.sqrt(jnp.array(cfg["head_dim"] / query_pre_attn_scalar, dtype=q.dtype))

    rope_theta = get_rope_theta(cfg, layer_idx)
    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attn_impl = cfg["additional_config"]["attn_implementation"]
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
        cfg["rms_norm_eps"],
    )
    x = residual + attn_output

    residual = x
    x_norm = gemma_rms_norm(x, w["pre_feedforward_layernorm"], cfg["rms_norm_eps"])
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    gate = act_fn(
        linear_3d(
            x_norm,
            w["gate_proj"],
            w.get("gate_proj_bias"),
            out_sharding=logical_to_physical(("batch", "context", "mlp_up_ffw"), rules),
        )
    )
    up = linear_3d(
        x_norm,
        w["up_proj"],
        w.get("up_proj_bias"),
        out_sharding=logical_to_physical(("batch", "context", "mlp_up_ffw"), rules),
    )
    ffw = jnp.einsum(
        "btf,df->btd",
        gate * up,
        w["down_proj"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )
    if w.get("down_proj_bias") is not None:
        ffw = ffw + w["down_proj_bias"][None, None, :]

    ffw = gemma_rms_norm(ffw, w["post_feedforward_layernorm"], cfg["rms_norm_eps"])
    x = residual + ffw
    return x, kv


def forward(
    config: Config,
    input_ids: Int[Array, "B T"],
    weights: PyTree[Array, "ModelWeights"],
    kv: None = None,
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    attention_mask: Int[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
    logits_to_keep: int = 0,
    **inputs,
):
    rules = config["sharding_rules"]
    model_prefix = config["model_prefix"]
    embed_key = f"{model_prefix}.embed_tokens.weight"
    final_norm_key = f"{model_prefix}.norm.weight"

    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    x = (
        weights[embed_key]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )
    x *= jnp.sqrt(jnp.array(config["hidden_size"], dtype=dtype))

    return_kv = kv is not None
    if kv is None:
        kv = defaultdict(lambda: None)

    mask_mapping = make_mask_mapping(
        config,
        x,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )

    for layer_idx in range(config["num_hidden_layers"]):
        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and layer_idx < len(layer_types):
            attention_type = layer_types[layer_idx]
        else:
            pattern = int(config.get("sliding_window_pattern", 6))
            attention_type = (
                "sliding_attention"
                if bool((layer_idx + 1) % pattern)
                else "full_attention"
            )

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

        if config["additional_config"]["gradient_checkpointing"]:
            fwd = jax.remat(partial(forward_layer, config, layer_idx))
        else:
            fwd = partial(forward_layer, config, layer_idx)

        layer_mask = mask_mapping.get(attention_type)
        x, kv[layer_idx] = fwd(
            x,
            layer_weights,
            kv[layer_idx],
            pos,
            attention_mask=layer_mask,
            **inputs,
        )

    x = gemma_rms_norm(x, weights[final_norm_key], config["rms_norm_eps"])
    if config.get("tie_word_embeddings", True) or config["lm_head_key"] not in weights:
        out_embed = weights[embed_key]
    else:
        out_embed = weights[config["lm_head_key"]]

    if logits_to_keep:
        x = x[:, -int(logits_to_keep) :, :]

    logits = jnp.einsum(
        "btd,vd->btv",
        x,
        out_embed,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
        preferred_element_type=x.dtype,
    )

    softcap = config.get("final_logit_softcapping")
    if softcap is not None:
        logits = jnp.tanh(logits / softcap) * softcap

    return (logits, kv) if return_kv else logits


def save_safetensors(weights: PyTree[ModelWeights], path: str | Path):
    pass


def detect_weight_prefixes(weights: dict[str, jax.Array]):
    if "model.embed_tokens.weight" in weights:
        return "model", "lm_head.weight"
    if "language_model.model.embed_tokens.weight" in weights:
        return "language_model.model", "language_model.lm_head.weight"

    embed_key = next(
        (key for key in weights if key.endswith("model.embed_tokens.weight")),
        None,
    )
    if embed_key is None:
        raise KeyError("Could not locate Gemma-3 embedding weights in checkpoint.")

    model_prefix = embed_key[: -len(".embed_tokens.weight")]
    lm_head_key = (
        "language_model.lm_head.weight"
        if model_prefix.startswith("language_model.")
        else "lm_head.weight"
    )
    return model_prefix, lm_head_key


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

    cfg_path = model_ckpt_dir / "config.json"
    raw_cfg = json.loads(cfg_path.read_text())
    if isinstance(raw_cfg.get("text_config"), dict):
        base_cfg = dict(raw_cfg["text_config"])
    else:
        base_cfg = dict(raw_cfg)

    if base_cfg.get("layer_types") is None:
        pattern = int(
            base_cfg.get(
                "sliding_window_pattern", raw_cfg.get("sliding_window_pattern", 6)
            )
        )
        num_layers = int(base_cfg["num_hidden_layers"])
        base_cfg["layer_types"] = [
            "sliding_attention" if bool((i + 1) % pattern) else "full_attention"
            for i in range(num_layers)
        ]

    if base_cfg.get("rope_parameters") is None and (
        base_cfg.get("rope_theta") is not None
        or base_cfg.get("rope_local_base_freq") is not None
    ):
        full_theta = float(base_cfg.get("rope_theta") or 10_000.0)
        sliding_theta = float(base_cfg.get("rope_local_base_freq") or full_theta)
        base_cfg["rope_parameters"] = {
            "full_attention": {"rope_theta": full_theta},
            "sliding_attention": {"rope_theta": sliding_theta},
        }

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
                    f.get_tensor(key).astype(param_dtype),
                    get_sharding(key),
                )

    model_prefix, lm_head_key = detect_weight_prefixes(weights)

    cfg = Config(
        **base_cfg,
        additional_config=additional_config,
        parallel_dims=parallel_dims,
        sharding_rules=sharding_rules,
        model_prefix=model_prefix,
        lm_head_key=lm_head_key,
    )

    return Model(
        config=cfg,
        weights=weights,
        forward=partial(forward, cfg),
        tokenizer=tokenizer,
    )
