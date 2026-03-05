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
from transformers import (
    AutoTokenizer,
    Gemma3TextConfig,
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
    load_weights,
    load_weights_vectorize,
    logical_to_physical,
    Model,
)
from jaxformers.print_utils import tree_pprint
from jaxformers.scan_utils import make_scan_fwd

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


SHARDING_RULES = {
    "none": None,
    "batch": BATCH,
    "fsdp": FSDP,
    "model": MODEL,
    "sequence": SEQ,
    "context": CONTEXT,
}


Config: TypeAlias = Gemma3TextConfig
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
        stddev=_default_initializer_range(config), lower=-3, upper=3
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
    if not isinstance(rope_parameters, dict):
        raise TypeError(
            "Gemma-3 config must define `rope_parameters` as a dict with per-attention-type "
            "settings (e.g. {'full_attention': {'rope_theta': ...}, ...})."
        )

    attn_config = rope_parameters.get(attention_type)
    if not isinstance(attn_config, dict) or attn_config.get("rope_theta") is None:
        raise KeyError(
            f"Missing `rope_theta` for attention_type={attention_type!r} in `rope_parameters`. "
            f"Available keys: {sorted(rope_parameters.keys())!r}"
        )
    return float(attn_config["rope_theta"])

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
    layer_idx: Int[Array, ""],
    rope_theta: Float[Array, ""],
    pos=0,
    **inputs,
):
    rules = config.sharding_rules
    act_fn = get_activation_fn(config.hidden_activation)
    head_dim = config.head_dim
    attention_mask = jax.lax.select(inputs["is_sliding"], inputs["attention_mask"]["sliding_attention"], inputs["attention_mask"]["full_attention"])

    residual = x
    x_norm = gemma_rms_norm(x, w["input_layernorm.weight"], config.rms_norm_eps)
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
        w["self_attn.q_proj.weight"],
        w.get("self_attn.q_proj.bias"),
        out_sharding=logical_to_physical(("batch", "context", "none"), rules),
    )
    k = linear_3d(
        x_norm,
        w["self_attn.k_proj.weight"],
        w.get("self_attn.k_proj.bias"),
        out_sharding=logical_to_physical(("batch", "context", "none"), rules),
    )
    v = linear_3d(
        x_norm,
        w["self_attn.v_proj.weight"],
        w.get("self_attn.v_proj.bias"),
        out_sharding=logical_to_physical(("batch", "context", "none"), rules),
    )

    q = rearrange(q, "b t (n h) -> b t n h", n=config.num_attention_heads, h=head_dim)
    k = rearrange(k, "b t (k h) -> b t k h", k=config.num_key_value_heads, h=head_dim)
    v = rearrange(v, "b t (k h) -> b t k h", k=config.num_key_value_heads, h=head_dim)

    q = gemma_rms_norm(q, w["self_attn.q_norm.weight"], config.rms_norm_eps)
    k = gemma_rms_norm(k, w["self_attn.k_norm.weight"], config.rms_norm_eps)

    query_pre_attn_scalar = float(
        getattr(config, "query_pre_attn_scalar", config.head_dim)
    )
    q = q * jnp.sqrt(jnp.array(config.head_dim / query_pre_attn_scalar, dtype=q.dtype))

    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attn_output = attention_interface(
        q, k, v, mask=attention_mask, q_sharding=q_sharding
    )

    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
    attn_output = linear_3d(
        attn_output,
        w["self_attn.o_proj.weight"],
        w.get("self_attn.o_proj.bias"),
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )

    attn_output = gemma_rms_norm(
        attn_output,
        w["post_attention_layernorm.weight"],
        config.rms_norm_eps,
    )
    x = residual + attn_output

    residual = x
    x_norm = gemma_rms_norm(
        x, w["pre_feedforward_layernorm.weight"], config.rms_norm_eps
    )
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    gate = act_fn(
        linear_3d(
            x_norm,
            w["mlp.gate_proj.weight"],
            w.get("mlp.gate_proj.bias"),
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
    )
    up = linear_3d(
        x_norm,
        w["mlp.up_proj.weight"],
        w.get("mlp.up_proj.bias"),
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )

    ffw = einsum(
        "btf,df->btd",
        gate * up,
        w["mlp.down_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "sequence", "none"), rules),
    )
    if w.get("mlp.down_proj.bias") is not None:
        ffw = ffw + w["mlp.down_proj.bias"][None, None, :]

    ffw = gemma_rms_norm(
        ffw, w["post_feedforward_layernorm.weight"], config.rms_norm_eps
    )
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

    if config.additional_config["remat_layer"]:
        fwd = jax.remat(partial(forward_layer, config))
    else:
        fwd = partial(forward_layer, config)

    mask_mapping = make_mask(
        config,
        x,
        **inputs,
    )
    input_kwargs = {
        "attention_mask": mask_mapping,
        "rope_theta": jnp.asarray([get_rope_theta(config, attention_type) for attention_type in config.layer_types]),
        "is_sliding": jnp.array([t == "sliding_attention" for t in config.layer_types]),
        "layer_idx" : jnp.asarray(list(range(0, config.num_hidden_layers))),
        "pos": jnp.asarray(0),
        "rngs": jax.random.split(rngs, config.num_hidden_layers) if rngs is not None else rngs,
    }
    x = make_scan_fwd(fwd, config.num_hidden_layers)(x, weights, input_kwargs)
    x = gemma_rms_norm(x, weights[final_norm_key], config.rms_norm_eps)

    # Note: x[1] are sharded across the tensor-parallel (TP) mesh.
    # To compute the loss we must all_gather those shards into a full sequence.
    # why shard token? The sharded-token approach should enable loss parallelism, but here we rely on
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
    config = Gemma3TextConfig.from_pretrained(model_ckpt_dir)
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

    weights = partial(
        load_weights_vectorize, "model.layers.", config.num_hidden_layers
    )(model_ckpt_dir, param_dtype, sharding_rules, get_sharding)

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
