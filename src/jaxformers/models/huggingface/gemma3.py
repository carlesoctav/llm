import fnmatch
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig

from jaxformers.distributed.parallel import ParallelDims
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
    make_sliding_window_causal_mask,
)
from jaxformers.modeling_utils import (
    AdditionalConfig,
    DEFAULT_ADDITIONAL_CONFIG,
    load_weights,
    logical_to_physical,
    Model,
)

from ...attention_utils import ATTENTION_INTERFACE
from ...dispatch.einsum import einsum
from ...distributed import (
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


def get_sharding(key, sharding_rules):
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
        return logical_to_physical(("model", "fsdp"), sharding_rules)
    if "lm_head" in key:
        return logical_to_physical(("model", "fsdp"), sharding_rules)
    return P()


Config: TypeAlias = PreTrainedConfig
Initializer: TypeAlias = Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], jax.Array]


def _match_first_initializer(
    key: str,
    initializers: dict[str, Initializer],
) -> Initializer | None:
    for pattern, init_fn in initializers.items():
        if fnmatch.fnmatchcase(key, pattern):
            return init_fn
    return None


def _default_initializer_range(config: Config) -> float:
    std = getattr(config, "initializer_range", None)
    if std is None:
        return 0.02
    std_float = float(std)
    return std_float if std_float != 0.0 else 0.02


def _default_initializer_for_key(config: Config, key: str) -> Initializer:
    if key.endswith(".bias"):
        return jax.nn.initializers.zeros

    if (
        key.endswith(".norm.weight")
        or ".layernorm.weight" in key
        or "norm.weight" in key
    ):
        return jax.nn.initializers.zeros

    return jax.nn.initializers.truncated_normal(
        stddev=_default_initializer_range(config)
    )


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


def get_rope_theta(config: Config, attention_type: str) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        attn_config = rope_parameters.get(attention_type)
        if isinstance(attn_config, dict) and attn_config.get("rope_theta") is not None:
            return float(attn_config["rope_theta"])

    if attention_type == "sliding_attention":
        rope_local_base_freq = getattr(config, "rope_local_base_freq", None)
        if rope_local_base_freq is not None:
            return float(rope_local_base_freq)

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)

    raise KeyError(
        "Gemma-3 config is missing RoPE settings. Expected either per-attention "
        "`rope_parameters`, `rope_theta`, or `rope_local_base_freq`."
    )


def make_mask(config, input_embeds, attention_mask=None, segment_ids=None, **kwargs):
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

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


def linear_3d(x, w, b=None, *, out_sharding=None):
    y = einsum(
        "btf,df->btd",
        x,
        w,
        preferred_element_type=x.dtype,
        out_sharding=out_sharding,
    )
    if b is not None:
        y = y + b[None, None, :]
    return y


def forward_layer(
    config: Config,
    x: Float[Array, "B T D"],
    w: PyTree[Array, "LayerWeights"],
    layer_idx: int,
    rope_theta: float,
    pos=0,
    **inputs,
):
    rules = config.sharding_rules
    act_fn = get_activation_fn(config.hidden_activation)
    head_dim = config.head_dim

    residual = x
    x_norm = gemma_rms_norm(x, w["input_layernorm"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))
    q_sharding = jax.NamedSharding(
        jax.sharding.get_abstract_mesh(),
        logical_to_physical(
            ("batch", "context", "model", "none"), config.sharding_rules
        ),
    )

    attn_impl = config.additional_config["attn_implementation"]
    attention_interface = ATTENTION_INTERFACE[attn_impl]

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

    q = rearrange(q, "b t (n h) -> b t n h", n=config.num_attention_heads, h=head_dim)
    k = rearrange(k, "b t (k h) -> b t k h", k=config.num_key_value_heads, h=head_dim)
    v = rearrange(v, "b t (k h) -> b t k h", k=config.num_key_value_heads, h=head_dim)

    q = gemma_rms_norm(q, w["q_norm"], config.rms_norm_eps)
    k = gemma_rms_norm(k, w["k_norm"], config.rms_norm_eps)

    query_pre_attn_scalar = float(
        getattr(config, "query_pre_attn_scalar", config.head_dim)
    )
    q = q * jnp.sqrt(jnp.array(config.head_dim / query_pre_attn_scalar, dtype=q.dtype))

    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attn_output = attention_interface(
        q, k, v, mask=inputs["attention_mask"], q_sharding=q_sharding
    )

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
        config.rms_norm_eps,
    )
    x = residual + attn_output

    residual = x
    x_norm = gemma_rms_norm(x, w["pre_feedforward_layernorm"], config.rms_norm_eps)
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

    ffw = gemma_rms_norm(ffw, w["post_feedforward_layernorm"], config.rms_norm_eps)
    x = residual + ffw
    return x


def forward(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    input_ids: Int[Array, "B T"],
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    rules = config.sharding_rules
    model_prefix = "model"
    embed_key = f"{model_prefix}.embed_tokens.weight"
    final_norm_key = f"{model_prefix}.norm.weight"

    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    x = (
        weights[embed_key]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )
    x *= jnp.sqrt(jnp.array(config.hidden_size, dtype=dtype))


    mask_mapping = make_mask(
        config,
        x,
        **inputs,
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

        layer_inputs = {**inputs, "attention_mask": mask_mapping.get(attention_type)}
        x = fwd(
            x,
            layer_weights,
            layer_idx,
            rope_theta,
            pos,
            **layer_inputs,
        )

    x = gemma_rms_norm(x, weights[final_norm_key], config.rms_norm_eps)
    # Note: x[1] are sharded across the tensor-parallel (TP) axis.
    # To compute the loss we must all_gather those shards into a full sequence.
    # The sharded-token approach should enable loss parallelism, but here we rely on
    # xla_chunked cross-entropy instead. Ideally the cross-entropy op would
    # infer loss parallelism from sharding annotations, but that is not yet
    # supported in this implementation.
    x = reshard(
        x, logical_to_physical(("batch", "context", "none"), config.sharding_rules)
    )

    return x


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
    input_embeds = (
        weights["model.embed_tokens.weight"]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )
    input_embeds *= jnp.sqrt(jnp.array(config.hidden_size, dtype=dtype))
    return input_embeds


def unembed(
    config: Config,
    weights: PyTree[Array, "ModelWeights"],
    hidden_states: Float[Array, "B T D"],
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    rules = config.sharding_rules
    out_embed = (
        weights["model.embed_tokens.weight"]
        if getattr(config, "tie_word_embeddings", True)
        else weights["lm_head.weight"]
    )
    logits = einsum(
        "btd,vd->btv",
        hidden_states,
        out_embed,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        preferred_element_type=jnp.float32,
    )
    return logits


def save_safetensors(weights: PyTree[ModelWeights], path: str | Path):
    pass


def init(
    config: Config,
    parallel_dims: ParallelDims,
    devices: list | None = None,
    multihost: bool = False,
    additional_config: AdditionalConfig | None = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    *,
    rngs: PRNGKeyArray,
    initializers: dict[str, Initializer] | None = None,
    tokenizer=None,
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

    def get_sharding(key: str):
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
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        if "lm_head" in key:
            return logical_to_physical(("model", "fsdp"), sharding_rules)
        return P()

    counter = 0

    def init_param(name: str, shape: tuple[int, ...]) -> Array:
        nonlocal counter
        key = jax.random.fold_in(rngs, counter)
        counter += 1

        if initializers is None:
            init_fn = _default_initializer_for_key(config, name)
        else:
            init_fn = _match_first_initializer(name, initializers)
            if init_fn is None:
                raise KeyError(
                    f"No initializer matched {name!r}. "
                    "Provide a matching pattern (e.g. '*' as a fallback)."
                )

        arr = init_fn(key, shape, dtype=param_dtype)
        # if initializers is None and name == "model.embed_tokens.weight":
        #     pad_token_id = getattr(config, "pad_token_id", None)
        #     if pad_token_id is not None:
        #         arr = arr.at[int(pad_token_id), :].set(jnp.zeros(shape[1], dtype=arr.dtype))
        return jax.device_put(arr, get_sharding(name))

    hidden_size = int(getattr(config, "hidden_size"))
    intermediate_size = int(getattr(config, "intermediate_size"))
    num_hidden_layers = int(getattr(config, "num_hidden_layers"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    num_key_value_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
    vocab_size = int(getattr(config, "vocab_size"))

    weights: dict[str, Array] = {}

    weights["model.embed_tokens.weight"] = init_param(
        "model.embed_tokens.weight", (vocab_size, hidden_size)
    )
    if not getattr(config, "tie_word_embeddings", True):
        weights["lm_head.weight"] = init_param(
            "lm_head.weight", (vocab_size, hidden_size)
        )

    attention_bias = bool(getattr(config, "attention_bias", False))
    for layer_idx in range(num_hidden_layers):
        prefix = f"model.layers.{layer_idx}."
        weights[f"{prefix}input_layernorm.weight"] = init_param(
            f"{prefix}input_layernorm.weight", (hidden_size,)
        )
        weights[f"{prefix}post_attention_layernorm.weight"] = init_param(
            f"{prefix}post_attention_layernorm.weight", (hidden_size,)
        )
        weights[f"{prefix}pre_feedforward_layernorm.weight"] = init_param(
            f"{prefix}pre_feedforward_layernorm.weight", (hidden_size,)
        )
        weights[f"{prefix}post_feedforward_layernorm.weight"] = init_param(
            f"{prefix}post_feedforward_layernorm.weight", (hidden_size,)
        )

        q_out = num_attention_heads * head_dim
        kv_out = num_key_value_heads * head_dim

        weights[f"{prefix}self_attn.q_proj.weight"] = init_param(
            f"{prefix}self_attn.q_proj.weight", (q_out, hidden_size)
        )
        weights[f"{prefix}self_attn.k_proj.weight"] = init_param(
            f"{prefix}self_attn.k_proj.weight", (kv_out, hidden_size)
        )
        weights[f"{prefix}self_attn.v_proj.weight"] = init_param(
            f"{prefix}self_attn.v_proj.weight", (kv_out, hidden_size)
        )
        weights[f"{prefix}self_attn.o_proj.weight"] = init_param(
            f"{prefix}self_attn.o_proj.weight", (hidden_size, q_out)
        )

        if attention_bias:
            weights[f"{prefix}self_attn.q_proj.bias"] = init_param(
                f"{prefix}self_attn.q_proj.bias", (q_out,)
            )
            weights[f"{prefix}self_attn.k_proj.bias"] = init_param(
                f"{prefix}self_attn.k_proj.bias", (kv_out,)
            )
            weights[f"{prefix}self_attn.v_proj.bias"] = init_param(
                f"{prefix}self_attn.v_proj.bias", (kv_out,)
            )
            weights[f"{prefix}self_attn.o_proj.bias"] = init_param(
                f"{prefix}self_attn.o_proj.bias", (hidden_size,)
            )

        weights[f"{prefix}self_attn.q_norm.weight"] = init_param(
            f"{prefix}self_attn.q_norm.weight", (head_dim,)
        )
        weights[f"{prefix}self_attn.k_norm.weight"] = init_param(
            f"{prefix}self_attn.k_norm.weight", (head_dim,)
        )

        weights[f"{prefix}mlp.gate_proj.weight"] = init_param(
            f"{prefix}mlp.gate_proj.weight", (intermediate_size, hidden_size)
        )
        weights[f"{prefix}mlp.up_proj.weight"] = init_param(
            f"{prefix}mlp.up_proj.weight", (intermediate_size, hidden_size)
        )
        weights[f"{prefix}mlp.down_proj.weight"] = init_param(
            f"{prefix}mlp.down_proj.weight", (hidden_size, intermediate_size)
        )

    weights["model.norm.weight"] = init_param("model.norm.weight", (hidden_size,))

    config.additional_config = additional_config
    config.parallel_dims = parallel_dims
    config.sharding_rules = sharding_rules

    return Model(
        name=__name__,
        config=config,
        weights=weights,
        forward=partial(forward, config),
        tokenizer=tokenizer,
        embed=partial(embed, config),
        unembed=partial(unembed, config),
        lm_head_key=(
            "model.embed_tokens.weight"
            if getattr(config, "tie_word_embeddings", True)
            else "lm_head.weight"
        ),
    )


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
    config = AutoConfig.from_pretrained(model_ckpt_dir)
    if not isinstance(config, PreTrainedConfig):
        raise TypeError(f"Expected HF config, got {type(config)!r}")

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

    weights = load_weights(model_ckpt_dir, param_dtype, sharding_rules, get_sharding)

    if "model.embed_tokens.weight" not in weights:
        raise KeyError(
            "Could not locate Gemma-3 text embedding weights. "
            "Expected 'model.embed_tokens.weight'. "
            "If this is a multimodal checkpoint, load it via your gemma3_mm module."
        )

    config.additional_config = additional_config
    config.parallel_dims = parallel_dims
    config.sharding_rules = sharding_rules

    return Model(
        name=__name__,
        config=config,
        weights=weights,
        forward=partial(forward, config),
        tokenizer=tokenizer,
        embed=partial(embed, config),
        unembed=partial(unembed, config),
        lm_head_key=(
            "model.embed_tokens.weight"
            if getattr(config, "tie_word_embeddings", True)
            else "lm_head.weight"
        ),
    )
