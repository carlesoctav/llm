from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from enum import auto, StrEnum
from functools import partial
from pathlib import Path
from typing import Callable, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from transformers import AutoConfig, AutoTokenizer, PreTrainedConfig

from jaxformers.distributed.parallel import ParallelDims
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
    make_decode_mask,
)
from jaxformers.modeling_utils import (
    AdditionalConfig,
    DEFAULT_ADDITIONAL_CONFIG,
    load_weights,
    logical_to_physical,
    Model,
)
from jaxformers.scan_utils import make_scan_fwd

from ...attention_utils import ATTENTION_INTERFACE, update_kv_cache
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
Config: TypeAlias = PreTrainedConfig
Initializer: TypeAlias = Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], jax.Array]
LAYER_PREFIX = "model.layers."
LAYER_PATTERN = re.compile(rf"{re.escape(LAYER_PREFIX)}(\d+)\.(.*)")
NON_LAYER_WEIGHT_KEYS = {
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
}


class ForwardImpl(StrEnum):
    LOOP = auto()
    SCAN_LAYER = auto()


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


def get_forward_impl(config: Config | PreTrainedConfig) -> ForwardImpl:
    forward_impl = config.additional_config.get("forward_impl", ForwardImpl.LOOP)
    try:
        return ForwardImpl(forward_impl)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(
            "Qwen-3 only supports forward_impl in {'loop', 'scan_layer'}."
        ) from exc


def get_rope_theta(config: Config) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if (
        not isinstance(rope_parameters, dict)
        or rope_parameters.get("rope_theta") is None
    ):
        raise TypeError(
            "Qwen-3 config must define `rope_parameters` as a dict with a `rope_theta` key "
            "(e.g. {'rope_theta': 1000000, ...})."
        )
    return float(rope_parameters["rope_theta"])


def split_layer_weights(
    weights: dict[str, Array],
    num_hidden_layers: int,
) -> tuple[dict[str, Array], dict[str, list[Array]]]:
    other_weights = {}
    layer_weights = defaultdict(lambda: [None] * num_hidden_layers)

    for key, value in weights.items():
        match = LAYER_PATTERN.fullmatch(key)
        if match is None:
            other_weights[key] = value
            continue

        layer_idx = int(match.group(1))
        inner_key = match.group(2)
        layer_weights[inner_key][layer_idx] = value

    for inner_key, values in layer_weights.items():
        missing = [idx for idx, value in enumerate(values) if value is None]
        if missing:
            raise KeyError(
                f"Missing layer weights for {inner_key!r} at indices {missing!r}."
            )

    return other_weights, dict(layer_weights)


def _with_prefix_sharding(sample: Array, value: Array) -> Array:
    try:
        sharding = sample.sharding
    except AttributeError:
        sharding = jax.typeof(sample).sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
        prefix_ndim = value.ndim - sample.ndim
        return jax.device_put(
            value,
            jax.sharding.NamedSharding(
                sharding.mesh,
                P(*((None,) * prefix_ndim), *sharding.spec),
            ),
        )
    return value


def zero_like_layer_value(value):
    from jaxformers.dispatch.lora import LoraArray

    if isinstance(value, LoraArray):
        return LoraArray(
            _w=jnp.zeros_like(value._w),
            a=jnp.zeros_like(value.a),
            b=jnp.zeros_like(value.b),
            alpha=value.alpha,
            allow_materialise=value.allow_materialise,
        )
    return jnp.zeros_like(value)


def stack_layer_values(values, leading_shape: tuple[int, ...]):
    from jaxformers.dispatch.lora import LoraArray

    sample = values[0]
    if isinstance(sample, LoraArray):
        return LoraArray(
            _w=stack_layer_values([value._w for value in values], leading_shape),
            a=stack_layer_values([value.a for value in values], leading_shape),
            b=stack_layer_values([value.b for value in values], leading_shape),
            alpha=sample.alpha,
            allow_materialise=sample.allow_materialise,
        )

    stacked = jnp.stack(values)
    stacked = stacked.reshape(*leading_shape, *sample.shape)
    return _with_prefix_sharding(sample, stacked)


def prepare_weights(
    config: Config,
    weights: dict[str, Array],
    forward_impl: ForwardImpl | str | None = None,
) -> dict[str, Array]:
    forward_impl = ForwardImpl(forward_impl or get_forward_impl(config))
    if forward_impl is ForwardImpl.LOOP:
        return weights

    if not any(LAYER_PATTERN.fullmatch(key) for key in weights):
        return weights

    num_hidden_layers = int(getattr(config, "num_hidden_layers"))
    other_weights, layer_weights = split_layer_weights(weights, num_hidden_layers)
    prepared_weights = dict(other_weights)

    for inner_key, values in layer_weights.items():
        prepared_weights[inner_key] = stack_layer_values(values, (num_hidden_layers,))

    return prepared_weights


def get_loop_layer_weights(
    weights: dict[str, Array],
    layer_idx: int,
) -> dict[str, Array]:
    prefix = f"{LAYER_PREFIX}{layer_idx}."
    return {
        key.removeprefix(prefix): value
        for key, value in weights.items()
        if key.startswith(prefix)
    }


def get_scannable_layer_weights(weights: dict[str, Array]) -> dict[str, Array]:
    return {
        key: value for key, value in weights.items() if key not in NON_LAYER_WEIGHT_KEYS
    }


def get_layer_metadata(config: Config) -> jax.Array:
    return jnp.arange(int(getattr(config, "num_hidden_layers")), dtype=jnp.int32)


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

    if key.endswith("norm.weight") or ".layernorm.weight" in key:
        return jax.nn.initializers.ones

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


def rms_norm(x: jax.Array, gamma: jax.Array, eps: float):
    rms = jnp.sqrt(jnp.square(x.astype(jnp.float32)).mean(-1, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def activation_out_sharding(rules, seqlen: int):
    axis_name = "context" if seqlen == 1 else "sequence"
    return logical_to_physical(("batch", axis_name, "none"), rules)


def forward_layer(
    config: Config,
    x: Float[Array, "B T D"],
    w: PyTree[Array, LayerWeights],
    layer_idx: Int[Array, ""] | int,
    kv=None,
    pos=0,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    del layer_idx, rngs
    bsz, seqlen, _hidden = x.shape
    rules = config.sharding_rules

    x_norm = rms_norm(x, w["input_layernorm.weight"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    q = einsum(
        "btd,md->btm",
        x_norm,
        w["self_attn.q_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )
    k = einsum(
        "btd,md->btm",
        x_norm,
        w["self_attn.k_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )
    v = einsum(
        "btd,md->btm",
        x_norm,
        w["self_attn.v_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )

    q = rearrange(q, "b t (n h) -> b t n h", n=config.num_attention_heads, h=config.head_dim)
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

    q = rms_norm(q, w["self_attn.q_norm.weight"], config.rms_norm_eps)
    k = rms_norm(k, w["self_attn.k_norm.weight"], config.rms_norm_eps)

    rope_theta = get_rope_theta(config)
    q = apply_rope(q, rope_theta, pos)
    k = apply_rope(k, rope_theta, pos)

    attention_mask = inputs["attention_mask"]
    if kv is not None:
        cache_k, cache_v, kv = update_kv_cache(k, v, kv, pos)
        if seqlen == 1:
            k = cache_k
            v = cache_v

    attn_impl = config.additional_config["attn_implementation"]
    attention_interface = ATTENTION_INTERFACE[attn_impl]
    q_sharding = jax.NamedSharding(
        jax.sharding.get_abstract_mesh(),
        logical_to_physical(("batch", "context", "model", "none"), rules),
    )
    attn_output = attention_interface(q, k, v, mask=attention_mask, q_sharding=q_sharding)

    attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
    attn_output = einsum(
        "btd,ed->bte",
        attn_output,
        w["self_attn.o_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=activation_out_sharding(rules, seqlen),
    )
    x = x + attn_output

    x_norm = rms_norm(x, w["post_attention_layernorm.weight"], config.rms_norm_eps)
    x_norm = reshard(x_norm, logical_to_physical(("batch", "context", "none"), rules))

    gate = jax.nn.silu(
        einsum(
            "btd,fd->btf",
            x_norm,
            w["mlp.gate_proj.weight"],
            preferred_element_type=jnp.float32,
            out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        )
    )
    up = einsum(
        "btd,fd->btf",
        x_norm,
        w["mlp.up_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
    )
    x = x + einsum(
        "btf,df->btd",
        gate * up,
        w["mlp.down_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=activation_out_sharding(rules, seqlen),
    )

    return (x, kv) if kv is not None else x


def make_mask(config, input_embeds, attention_mask=None, segment_ids=None, **kwargs):
    attn_implementation = config.additional_config["attn_implementation"]
    if attn_implementation not in ATTENTION_MASK_INTERFACE:
        return None

    kv = kwargs.get("kv")
    if kv is not None and input_embeds.shape[1] == 1:
        return make_decode_mask(
            batch_size=input_embeds.shape[0],
            q_length=input_embeds.shape[1],
            kv_length=kv[0][0].shape[1],
            pos=kwargs.get("pos", 0),
        )

    if attention_mask is not None:
        attention_mask = attention_mask.astype(jnp.bool_)

    return make_causal_mask(
        attn_implementation,
        input_embeds,
        attention_mask,
        segment_ids,
    )


def forward_loop(
    config: Config,
    x: Float[Array, "B T D"],
    weights: dict[str, Array],
    attention_mask,
    pos: int,
    kv=None,
):
    fwd = partial(forward_layer, config)
    if config.additional_config["remat_layer"] and kv is None:
        fwd = jax.remat(fwd)

    next_kv = [] if kv is not None else None
    for layer_idx in range(int(config.num_hidden_layers)):
        layer_cache = None if kv is None else kv[layer_idx]
        if layer_cache is None:
            x = fwd(
                x,
                get_loop_layer_weights(weights, layer_idx),
                layer_idx,
                pos=pos,
                attention_mask=attention_mask,
            )
        else:
            x, layer_cache = fwd(
                x,
                get_loop_layer_weights(weights, layer_idx),
                layer_idx,
                kv=layer_cache,
                pos=pos,
                attention_mask=attention_mask,
            )
            next_kv.append(layer_cache)
    return x, tuple(next_kv) if next_kv is not None else None


def forward_scan_layer(
    config: Config,
    x: Float[Array, "B T D"],
    weights: dict[str, Array],
    attention_mask,
    pos: int,
):
    fwd = partial(forward_layer, config)
    if config.additional_config["remat_layer"]:
        fwd = jax.remat(fwd)

    layer_idx = get_layer_metadata(config)
    scan_fwd = make_scan_fwd(
        fwd,
        int(layer_idx.shape[0]),
        argnums=0,
        argnames=("layer_idx",),
    )
    return scan_fwd(
        x,
        get_scannable_layer_weights(weights),
        layer_idx=layer_idx,
        pos=pos,
        attention_mask=attention_mask,
    )


def forward(
    config: Config,
    weights: PyTree[Array, ModelWeights],
    input_ids: Int[Array, "B T"],
    kv: PyTree | None = None,
    pos: int = 0,
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    del rngs
    rules = config.sharding_rules
    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    x = (
        weights["model.embed_tokens.weight"]
        .at[input_ids, :]
        .get(out_sharding=activation_out_sharding(rules, input_ids.shape[1]))
        .astype(dtype)
    )

    attention_mask = make_mask(config, x, kv=kv, pos=pos, **inputs)

    return_kv = kv is not None
    forward_impl = ForwardImpl.LOOP if return_kv else get_forward_impl(config)
    if forward_impl is not ForwardImpl.LOOP:
        weights = prepare_weights(config, weights, forward_impl)

    if forward_impl is ForwardImpl.LOOP:
        x, kv = forward_loop(config, x, weights, attention_mask, pos, kv=kv)
    elif forward_impl is ForwardImpl.SCAN_LAYER:
        x = forward_scan_layer(config, x, weights, attention_mask, pos)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported Qwen-3 forward implementation: {forward_impl!r}")

    x = rms_norm(x, weights["model.norm.weight"], config.rms_norm_eps)
    x = reshard(x, logical_to_physical(("batch", "context", "none"), rules))
    return (x, kv) if return_kv else x


def embed(
    config: Config,
    weights: PyTree[Array, ModelWeights],
    input_ids: Int[Array, "B T"],
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    del rngs, inputs
    rules = config.sharding_rules
    input_ids = reshard(input_ids, logical_to_physical(("batch", "context"), rules))
    return (
        weights["model.embed_tokens.weight"]
        .at[input_ids, :]
        .get(out_sharding=logical_to_physical(("batch", "sequence", "none"), rules))
        .astype(dtype)
    )


def unembed(
    config: Config,
    weights: PyTree[Array, ModelWeights],
    hidden_states: Float[Array, "B T D"],
    dtype: jnp.dtype = jnp.float32,
    *,
    rngs: PRNGKeyArray | None = None,
    **inputs,
):
    del dtype, rngs, inputs
    rules = config.sharding_rules
    out_embed = (
        weights["model.embed_tokens.weight"]
        if getattr(config, "tie_word_embeddings", True)
        else weights["lm_head.weight"]
    )
    return einsum(
        "btd,vd->btv",
        hidden_states,
        out_embed,
        out_sharding=logical_to_physical(("batch", "context", "model"), rules),
        preferred_element_type=jnp.float32,
    )


def save_safetensors(weights: PyTree[ModelWeights], path: str | Path):
    del weights, path


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
        return jax.device_put(arr, get_sharding(name, sharding_rules))

    hidden_size = int(getattr(config, "hidden_size"))
    intermediate_size = int(getattr(config, "intermediate_size"))
    num_hidden_layers = int(getattr(config, "num_hidden_layers"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    num_key_value_heads = int(getattr(config, "num_key_value_heads"))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
    vocab_size = int(getattr(config, "vocab_size"))

    weights: dict[str, Array] = {}
    weights["model.embed_tokens.weight"] = init_param(
        "model.embed_tokens.weight",
        (vocab_size, hidden_size),
    )
    if not getattr(config, "tie_word_embeddings", True):
        weights["lm_head.weight"] = init_param(
            "lm_head.weight",
            (vocab_size, hidden_size),
        )

    q_out = num_attention_heads * head_dim
    kv_out = num_key_value_heads * head_dim
    for layer_idx in range(num_hidden_layers):
        prefix = f"model.layers.{layer_idx}."
        weights[f"{prefix}input_layernorm.weight"] = init_param(
            f"{prefix}input_layernorm.weight",
            (hidden_size,),
        )
        weights[f"{prefix}post_attention_layernorm.weight"] = init_param(
            f"{prefix}post_attention_layernorm.weight",
            (hidden_size,),
        )
        weights[f"{prefix}self_attn.q_proj.weight"] = init_param(
            f"{prefix}self_attn.q_proj.weight",
            (q_out, hidden_size),
        )
        weights[f"{prefix}self_attn.k_proj.weight"] = init_param(
            f"{prefix}self_attn.k_proj.weight",
            (kv_out, hidden_size),
        )
        weights[f"{prefix}self_attn.v_proj.weight"] = init_param(
            f"{prefix}self_attn.v_proj.weight",
            (kv_out, hidden_size),
        )
        weights[f"{prefix}self_attn.o_proj.weight"] = init_param(
            f"{prefix}self_attn.o_proj.weight",
            (hidden_size, q_out),
        )
        weights[f"{prefix}self_attn.q_norm.weight"] = init_param(
            f"{prefix}self_attn.q_norm.weight",
            (head_dim,),
        )
        weights[f"{prefix}self_attn.k_norm.weight"] = init_param(
            f"{prefix}self_attn.k_norm.weight",
            (head_dim,),
        )
        weights[f"{prefix}mlp.gate_proj.weight"] = init_param(
            f"{prefix}mlp.gate_proj.weight",
            (intermediate_size, hidden_size),
        )
        weights[f"{prefix}mlp.up_proj.weight"] = init_param(
            f"{prefix}mlp.up_proj.weight",
            (intermediate_size, hidden_size),
        )
        weights[f"{prefix}mlp.down_proj.weight"] = init_param(
            f"{prefix}mlp.down_proj.weight",
            (hidden_size, intermediate_size),
        )

    weights["model.norm.weight"] = init_param("model.norm.weight", (hidden_size,))

    config.additional_config = additional_config
    config.parallel_dims = parallel_dims
    config.sharding_rules = sharding_rules
    config.mesh = mesh

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
        mesh=mesh,
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

    local_path = Path(local_dir).expanduser() if local_dir is not None else None
    if local_path is not None and (local_path / "config.json").exists():
        model_ckpt_dir = local_path
    else:
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

    config.additional_config = additional_config
    config.parallel_dims = parallel_dims
    config.sharding_rules = sharding_rules
    config.mesh = mesh

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
        mesh=mesh,
    )
