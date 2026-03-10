from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import AxisType
from jaxtyping import PRNGKeyArray
from safetensors import safe_open
from transformers import AutoTokenizer, Gemma3TextConfig

from jaxformers.distributed.parallel import drop_axis, ParallelDims
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
    make_sliding_window_causal_mask,
)
from jaxformers.modeling_utils import (
    AdditionalConfig,
    DEFAULT_ADDITIONAL_CONFIG,
    Model,
)
from jaxformers.models.transformers import (
    Attention,
    MLP,
    Transformers,
    TransformersLayer,
)
from jaxformers.module_utils import (
    EinsumLinear,
    Initializer,
    LinearEmbedding,
    Module,
    out_sharding,
    ParamTree,
    RMSNorm,
    ShardingConfig,
    unflatten_param_tree,
)

from ..attention_utils import ATTENTION_INTERFACE
from ..distributed import BATCH, CONTEXT, FSDP, MODEL, SEQ


Config: TypeAlias = Gemma3TextConfig


BASE_LOGICAL_TO_PHYSICAL = {
    "none": None,
    "batch": BATCH,
    "fsdp": FSDP,
    "model": MODEL,
    "sequence": SEQ,
    "context": CONTEXT,
}


def _match_first_initializer(
    key: str,
    initializers: dict[str, Initializer] | None,
) -> Initializer | None:
    if initializers is None:
        return None
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
    if key.endswith("norm.weight") or "layernorm.weight" in key:
        return jax.nn.initializers.zeros
    return jax.nn.initializers.truncated_normal(
        stddev=_default_initializer_range(config)
    )


def _resolve_initializer(
    config: Config,
    key: str,
    initializers: dict[str, Initializer] | None,
) -> Initializer:
    init_fn = _match_first_initializer(key, initializers)
    if init_fn is not None:
        return init_fn
    return _default_initializer_for_key(config, key)


def _logical_to_physical_mapping(parallel_dims: ParallelDims):
    rules = dict(BASE_LOGICAL_TO_PHYSICAL)
    for axis_name, axis_size in parallel_dims.items():
        if axis_size != 1:
            continue
        for key, value in list(rules.items()):
            rules[key] = drop_axis(value, axis_name)
    return rules


def _token_axis(parallel_dims: ParallelDims, sequence_parallelism: bool) -> str:
    if sequence_parallelism:
        return "sequence"
    if parallel_dims["cp"] > 1:
        return "context"
    return "none"


def make_gemma3_sharding_config(
    parallel_dims: ParallelDims,
    *,
    sequence_parallelism: bool = True,
) -> ShardingConfig:
    token_axis = _token_axis(parallel_dims, sequence_parallelism)
    context_axis = "context" if parallel_dims["cp"] > 1 else "none"
    activation_partition = ("batch", token_axis, "none")
    context_partition = ("batch", context_axis, "none")
    q_partition = ("batch", context_axis, "model", "none")
    kv_partition = ("batch", context_axis, "none", "none")
    ffn_partition = ("batch", context_axis, "model")
    logits_partition = ("batch", context_axis, "model")

    partition: dict[str, object | None] = {}

    def register(
        name: str,
        *,
        weights: tuple | None = None,
        inputs=None,
        outputs=None,
    ):
        partition[f"{name}.weights"] = weights
        partition[f"{name}.inputs"] = inputs
        partition[f"{name}.outputs"] = outputs

    register(
        "embed_tokens",
        weights=("model", "fsdp"),
        inputs=("batch", token_axis),
        outputs=activation_partition,
    )
    register("lm_head", weights=("model", "fsdp"), inputs=context_partition, outputs=logits_partition)
    register("norm", inputs=activation_partition, outputs=context_partition)
    register(
        "input_layernorm",
        inputs=activation_partition,
        outputs=context_partition,
    )
    register(
        "post_attention_layernorm",
        inputs=activation_partition,
        outputs=activation_partition,
    )
    register(
        "pre_feedforward_layernorm",
        inputs=activation_partition,
        outputs=context_partition,
    )
    register(
        "post_feedforward_layernorm",
        inputs=activation_partition,
        outputs=activation_partition,
    )
    register("q_proj", weights=("model", "fsdp"), inputs=context_partition, outputs=context_partition)
    register("k_proj", weights=("model", "fsdp"), inputs=context_partition, outputs=context_partition)
    register("v_proj", weights=("model", "fsdp"), inputs=context_partition, outputs=context_partition)
    register("o_proj", weights=("fsdp", "model"), inputs=context_partition, outputs=activation_partition)
    register(
        "self_attn",
        inputs=(
            (context_partition,),
            {"attention_mask": ("batch", "none", context_axis, "none")},
        ),
    )
    register("q_norm", inputs=q_partition, outputs=q_partition)
    register("k_norm", inputs=kv_partition, outputs=kv_partition)
    register("gate_proj", weights=("model", "fsdp"), inputs=context_partition, outputs=ffn_partition)
    register("up_proj", weights=("model", "fsdp"), inputs=context_partition, outputs=ffn_partition)
    register("down_proj", weights=("fsdp", "model"), inputs=context_partition, outputs=activation_partition)
    return ShardingConfig(
        partition=partition,
        logical_to_physical_mapping=_logical_to_physical_mapping(parallel_dims),
    )


def _param_sharding(key: str, sharding_config: ShardingConfig):
    parts = key.split(".")
    if len(parts) >= 2 and parts[-1] in {"weight", "bias", "lora_a", "lora_b"}:
        lookup_key = parts[-2]
    else:
        lookup_key = parts[-1]
    partition = sharding_config.partition.get(f"{lookup_key}.weights")
    partition = sharding_config.translate(partition)
    if callable(partition):
        raise TypeError("`*.weights` partitions must be concrete annotations, not callables.")
    return out_sharding(partition)


def apply_rope(x: jax.Array, theta: float, pos: int = 0):
    batch_size, seq_len, _heads, head_dim = x.shape
    positions = pos + jnp.broadcast_to(
        jnp.arange(seq_len)[None, :], [batch_size, seq_len]
    )
    freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    inp = jnp.einsum(
        "bt,h->bth",
        positions,
        freq,
        precision=jax.lax.Precision.HIGHEST,
    )
    x1, x2 = x[:, :, :, : head_dim // 2], x[:, :, :, head_dim // 2 :]
    sin = jnp.sin(inp).astype(x.dtype)[:, :, None, :]
    cos = jnp.cos(inp).astype(x.dtype)[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


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
        raise TypeError("Gemma-3 config must define `rope_parameters` as a dict.")
    attn_config = rope_parameters.get(attention_type)
    if not isinstance(attn_config, dict) or attn_config.get("rope_theta") is None:
        raise KeyError(f"Missing rope theta for attention type {attention_type!r}")
    return float(attn_config["rope_theta"])


def make_mask(config, input_embeds, attention_mask=None, segment_ids=None, **kwargs):
    attn_impl = config.additional_config["attn_implementation"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return {"full_attention": None, "sliding_attention": None}

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


@dataclass
class Gemma3Attention(Attention):
    layer_idx: int = 0
    config: Config | None = None
    sharding_config: ShardingConfig | None = None
    param_dtype: jnp.dtype = jnp.bfloat16
    initializers: dict[str, Initializer] | None = None

    q_proj: EinsumLinear | None = None
    k_proj: EinsumLinear | None = None
    v_proj: EinsumLinear | None = None
    o_proj: EinsumLinear | None = None
    q_norm: RMSNorm | None = None
    k_norm: RMSNorm | None = None

    def setup(self):
        assert self.config is not None
        assert self.sharding_config is not None
        attn_name = f"model.layers.{self.layer_idx}.self_attn"
        partition = self.sharding_config.partition
        hidden_size = int(self.config.hidden_size)
        head_dim = int(self.config.head_dim)
        q_out = int(self.config.num_attention_heads) * head_dim
        kv_out = int(self.config.num_key_value_heads) * head_dim
        use_bias = bool(getattr(self.config, "attention_bias", False))

        def init_fn(name: str):
            return _resolve_initializer(self.config, name, self.initializers)

        bias_term = "m" if use_bias else ""
        act_dtype = self.param_dtype
        self.q_proj = EinsumLinear(
            equation="...d,md->...m",
            weight_shape=(q_out, hidden_size),
            bias_term=bias_term,
            weight_init=init_fn(f"{attn_name}.q_proj.weight"),
            bias_init=init_fn(f"{attn_name}.q_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("q_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("q_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("q_proj.outputs")),
        )
        self.k_proj = EinsumLinear(
            equation="...d,md->...m",
            weight_shape=(kv_out, hidden_size),
            bias_term=bias_term,
            weight_init=init_fn(f"{attn_name}.k_proj.weight"),
            bias_init=init_fn(f"{attn_name}.k_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("k_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("k_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("k_proj.outputs")),
        )
        self.v_proj = EinsumLinear(
            equation="...d,md->...m",
            weight_shape=(kv_out, hidden_size),
            bias_term=bias_term,
            weight_init=init_fn(f"{attn_name}.v_proj.weight"),
            bias_init=init_fn(f"{attn_name}.v_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("v_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("v_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("v_proj.outputs")),
        )
        self.o_proj = EinsumLinear(
            equation="...d,ed->...e",
            weight_shape=(hidden_size, q_out),
            bias_term="e" if use_bias else "",
            weight_init=init_fn(f"{attn_name}.o_proj.weight"),
            bias_init=init_fn(f"{attn_name}.o_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("o_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("o_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("o_proj.outputs")),
        )
        self.q_norm = RMSNorm(
            hidden_size=head_dim,
            eps=float(self.config.rms_norm_eps),
            add_unit_offset=True,
            weight_init=init_fn(f"{attn_name}.q_norm.weight"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("q_norm.weights")),
            input_partition=self.sharding_config.translate(partition.get("q_norm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("q_norm.outputs")),
        )
        self.k_norm = RMSNorm(
            hidden_size=head_dim,
            eps=float(self.config.rms_norm_eps),
            add_unit_offset=True,
            weight_init=init_fn(f"{attn_name}.k_norm.weight"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("k_norm.weights")),
            input_partition=self.sharding_config.translate(partition.get("k_norm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("k_norm.outputs")),
        )
        self.input_partition = self.sharding_config.translate(
            partition.get("self_attn.inputs")
        )

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        q_key, k_key, v_key, o_key, qn_key, kn_key = jax.random.split(rngs, 6)
        return {
            "q_proj": self.q_proj.init(q_key),
            "k_proj": self.k_proj.init(k_key),
            "v_proj": self.v_proj.init(v_key),
            "o_proj": self.o_proj.init(o_key),
            "q_norm": self.q_norm.init(qn_key),
            "k_norm": self.k_norm.init(kn_key),
        }

    def forward(
        self,
        params: ParamTree,
        x: jax.Array,
        *,
        attention_mask,
        rope_theta: float,
        pos: int = 0,
    ):
        q = self.q_proj.apply(params["q_proj"], x)
        k = self.k_proj.apply(params["k_proj"], x)
        v = self.v_proj.apply(params["v_proj"], x)

        q = rearrange(
            q,
            "b t (n h) -> b t n h",
            n=int(self.config.num_attention_heads),
            h=int(self.config.head_dim),
        )
        k = rearrange(
            k,
            "b t (n h) -> b t n h",
            n=int(self.config.num_key_value_heads),
            h=int(self.config.head_dim),
        )
        v = rearrange(
            v,
            "b t (n h) -> b t n h",
            n=int(self.config.num_key_value_heads),
            h=int(self.config.head_dim),
        )

        q = self.q_norm.apply(params["q_norm"], q)
        k = self.k_norm.apply(params["k_norm"], k)

        query_pre_attn_scalar = float(
            getattr(self.config, "query_pre_attn_scalar", self.config.head_dim)
        )
        q = q * jnp.sqrt(
            jnp.asarray(self.config.head_dim / query_pre_attn_scalar, dtype=q.dtype)
        )
        q = apply_rope(q, rope_theta, pos)
        k = apply_rope(k, rope_theta, pos)

        q_sharding = jax.NamedSharding(
            jax.sharding.get_abstract_mesh(),
            out_sharding(
                self.sharding_config.translate(("batch", "context", "model", "none"))
            ),
        )
        attn_impl = self.config.additional_config["attn_implementation"]
        attention_interface = ATTENTION_INTERFACE[attn_impl]
        attn_output = attention_interface(
            q,
            k,
            v,
            mask=attention_mask,
            q_sharding=q_sharding,
        )
        attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
        return self.o_proj.apply(params["o_proj"], attn_output)


@dataclass
class Gemma3MLP(MLP):
    layer_idx: int = 0
    config: Config | None = None
    sharding_config: ShardingConfig | None = None
    param_dtype: jnp.dtype = jnp.bfloat16
    initializers: dict[str, Initializer] | None = None

    gate_proj: EinsumLinear | None = None
    up_proj: EinsumLinear | None = None
    down_proj: EinsumLinear | None = None

    def setup(self):
        assert self.config is not None
        assert self.sharding_config is not None
        mlp_name = f"model.layers.{self.layer_idx}.mlp"
        partition = self.sharding_config.partition
        hidden_size = int(self.config.hidden_size)
        intermediate_size = int(self.config.intermediate_size)
        use_bias = bool(getattr(self.config, "mlp_bias", False))

        def init_fn(name: str):
            return _resolve_initializer(self.config, name, self.initializers)

        self.gate_proj = EinsumLinear(
            equation="...d,fd->...f",
            weight_shape=(intermediate_size, hidden_size),
            bias_term="f" if use_bias else "",
            weight_init=init_fn(f"{mlp_name}.gate_proj.weight"),
            bias_init=init_fn(f"{mlp_name}.gate_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=self.param_dtype,
            weights_partition=self.sharding_config.translate(partition.get("gate_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("gate_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("gate_proj.outputs")),
        )
        self.up_proj = EinsumLinear(
            equation="...d,fd->...f",
            weight_shape=(intermediate_size, hidden_size),
            bias_term="f" if use_bias else "",
            weight_init=init_fn(f"{mlp_name}.up_proj.weight"),
            bias_init=init_fn(f"{mlp_name}.up_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=self.param_dtype,
            weights_partition=self.sharding_config.translate(partition.get("up_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("up_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("up_proj.outputs")),
        )
        self.down_proj = EinsumLinear(
            equation="...f,df->...d",
            weight_shape=(hidden_size, intermediate_size),
            bias_term="d" if use_bias else "",
            weight_init=init_fn(f"{mlp_name}.down_proj.weight"),
            bias_init=init_fn(f"{mlp_name}.down_proj.bias"),
            weight_dtype=self.param_dtype,
            activation_dtype=self.param_dtype,
            weights_partition=self.sharding_config.translate(partition.get("down_proj.weights")),
            input_partition=self.sharding_config.translate(partition.get("down_proj.inputs")),
            output_partition=self.sharding_config.translate(partition.get("down_proj.outputs")),
        )

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        gate_key, up_key, down_key = jax.random.split(rngs, 3)
        return {
            "gate_proj": self.gate_proj.init(gate_key),
            "up_proj": self.up_proj.init(up_key),
            "down_proj": self.down_proj.init(down_key),
        }

    def forward(self, params: ParamTree, x: jax.Array):
        act_fn = get_activation_fn(self.config.hidden_activation)
        gate = act_fn(self.gate_proj.apply(params["gate_proj"], x))
        up = self.up_proj.apply(params["up_proj"], x)
        return self.down_proj.apply(params["down_proj"], gate * up)


@dataclass
class Gemma3TransformerLayer(TransformersLayer):
    layer_idx: int = 0
    config: Config | None = None
    sharding_config: ShardingConfig | None = None
    param_dtype: jnp.dtype = jnp.bfloat16
    initializers: dict[str, Initializer] | None = None

    input_layernorm: RMSNorm | None = None
    post_attention_layernorm: RMSNorm | None = None
    pre_feedforward_layernorm: RMSNorm | None = None
    post_feedforward_layernorm: RMSNorm | None = None
    self_attn: Gemma3Attention | None = None
    mlp: Gemma3MLP | None = None

    def setup(self):
        assert self.config is not None
        assert self.sharding_config is not None
        layer_name = f"model.layers.{self.layer_idx}"
        partition = self.sharding_config.partition

        def init_fn(name: str):
            return _resolve_initializer(self.config, name, self.initializers)

        hidden_size = int(self.config.hidden_size)
        eps = float(self.config.rms_norm_eps)
        act_dtype = self.param_dtype
        self.input_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            add_unit_offset=True,
            weight_init=init_fn(f"{layer_name}.input_layernorm.weight"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("input_layernorm.weights")),
            input_partition=self.sharding_config.translate(partition.get("input_layernorm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("input_layernorm.outputs")),
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            add_unit_offset=True,
            weight_init=init_fn(f"{layer_name}.post_attention_layernorm.weight"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("post_attention_layernorm.weights")),
            input_partition=self.sharding_config.translate(partition.get("post_attention_layernorm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("post_attention_layernorm.outputs")),
        )
        self.pre_feedforward_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            add_unit_offset=True,
            weight_init=init_fn(f"{layer_name}.pre_feedforward_layernorm.weight"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("pre_feedforward_layernorm.weights")),
            input_partition=self.sharding_config.translate(partition.get("pre_feedforward_layernorm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("pre_feedforward_layernorm.outputs")),
        )
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            add_unit_offset=True,
            weight_init=init_fn(f"{layer_name}.post_feedforward_layernorm.weight"),
            weight_dtype=self.param_dtype,
            activation_dtype=act_dtype,
            weights_partition=self.sharding_config.translate(partition.get("post_feedforward_layernorm.weights")),
            input_partition=self.sharding_config.translate(partition.get("post_feedforward_layernorm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("post_feedforward_layernorm.outputs")),
        )
        self.self_attn = Gemma3Attention(
            layer_idx=self.layer_idx,
            config=self.config,
            sharding_config=self.sharding_config,
            param_dtype=self.param_dtype,
            initializers=self.initializers,
        )
        self.mlp = Gemma3MLP(
            layer_idx=self.layer_idx,
            config=self.config,
            sharding_config=self.sharding_config,
            param_dtype=self.param_dtype,
            initializers=self.initializers,
        )

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        keys = jax.random.split(rngs, 6)
        return {
            "input_layernorm": self.input_layernorm.init(keys[0]),
            "post_attention_layernorm": self.post_attention_layernorm.init(keys[1]),
            "pre_feedforward_layernorm": self.pre_feedforward_layernorm.init(keys[2]),
            "post_feedforward_layernorm": self.post_feedforward_layernorm.init(keys[3]),
            "self_attn": self.self_attn.init(keys[4]),
            "mlp": self.mlp.init(keys[5]),
        }

    def forward(
        self,
        params: ParamTree,
        x: jax.Array,
        *,
        attention_mask,
        rope_theta: float,
        pos: int = 0,
    ):
        residual = x
        x_norm = self.input_layernorm.apply(params["input_layernorm"], x)
        attn_output = self.self_attn.apply(
            params["self_attn"],
            x_norm,
            attention_mask=attention_mask,
            rope_theta=rope_theta,
            pos=pos,
        )
        attn_output = self.post_attention_layernorm.apply(
            params["post_attention_layernorm"],
            attn_output,
        )
        x = residual + attn_output

        residual = x
        x_norm = self.pre_feedforward_layernorm.apply(
            params["pre_feedforward_layernorm"],
            x,
        )
        ffw = self.mlp.apply(params["mlp"], x_norm)
        ffw = self.post_feedforward_layernorm.apply(
            params["post_feedforward_layernorm"],
            ffw,
        )
        return residual + ffw


@dataclass
class Gemma3Transformers(Transformers):
    config: Config | None = None
    sharding_config: ShardingConfig | None = None
    param_dtype: jnp.dtype = jnp.bfloat16
    initializers: dict[str, Initializer] | None = None

    norm: RMSNorm | None = None

    def setup(self):
        assert self.config is not None
        assert self.sharding_config is not None
        partition = self.sharding_config.partition
        self.layers = [
            Gemma3TransformerLayer(
                layer_idx=layer_idx,
                config=self.config,
                sharding_config=self.sharding_config,
                param_dtype=self.param_dtype,
                initializers=self.initializers,
            )
            for layer_idx in range(int(self.config.num_hidden_layers))
        ]
        self.norm = RMSNorm(
            hidden_size=int(self.config.hidden_size),
            eps=float(self.config.rms_norm_eps),
            add_unit_offset=True,
            weight_init=_resolve_initializer(
                self.config,
                "model.norm.weight",
                self.initializers,
            ),
            weight_dtype=self.param_dtype,
            activation_dtype=self.param_dtype,
            weights_partition=self.sharding_config.translate(partition.get("norm.weights")),
            input_partition=self.sharding_config.translate(partition.get("norm.inputs")),
            output_partition=self.sharding_config.translate(partition.get("norm.outputs")),
        )

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        params: ParamTree = {"layers": []}
        if not self.layers:
            params["norm"] = self.norm.init(rngs)
            return params
        keys = jax.random.split(rngs, len(self.layers) + 1)
        params["layers"] = [
            layer.init(key)
            for layer, key in zip(self.layers, keys[:-1], strict=True)
        ]
        params["norm"] = self.norm.init(keys[-1])
        return params

    def forward(
        self, params: ParamTree, hidden_states: jax.Array, *, pos: int = 0, **inputs
    ):
        layer_types = list(self.config.layer_types)
        for idx, layer in enumerate(self.layers):
            attention_type = layer_types[idx % len(layer_types)]
            hidden_states = layer.apply(
                params["layers"][idx],
                hidden_states,
                attention_mask=inputs["attention_mask"][attention_type],
                rope_theta=get_rope_theta(self.config, attention_type),
                pos=pos,
            )
        hidden_states = self.norm.apply(params["norm"], hidden_states)
        return hidden_states


@dataclass
class Gemma3LM(Module):
    config: Config | None = None
    sharding_config: ShardingConfig | None = None
    param_dtype: jnp.dtype = jnp.bfloat16
    initializers: dict[str, Initializer] | None = None

    embed_tokens: LinearEmbedding | None = None
    model: Gemma3Transformers | None = None
    lm_head: EinsumLinear | None = None

    def setup(self):
        assert self.config is not None
        assert self.sharding_config is not None
        hidden_size = int(self.config.hidden_size)
        vocab_size = int(self.config.vocab_size)
        tied = bool(getattr(self.config, "tie_word_embeddings", True))
        self.embed_tokens = LinearEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            use_lookup=False,
            weight_init=_resolve_initializer(
                self.config,
                "model.embed_tokens.weight",
                self.initializers,
            ),
            weight_dtype=self.param_dtype,
            activation_dtype=self.param_dtype,
            weights_partition=self.sharding_config.translate(self.sharding_config.partition.get("embed_tokens.weights")),
            input_partition=self.sharding_config.translate(self.sharding_config.partition.get("embed_tokens.inputs")),
            output_partition=self.sharding_config.translate(self.sharding_config.partition.get("embed_tokens.outputs")),
        )
        self.model = Gemma3Transformers(
            config=self.config,
            sharding_config=self.sharding_config,
            param_dtype=self.param_dtype,
            initializers=self.initializers,
        )
        lm_head_prefix = "model.embed_tokens" if tied else "lm_head"
        self.lm_head = EinsumLinear(
            equation="...d,vd->...v",
            weight_shape=(vocab_size, hidden_size),
            weight_init=_resolve_initializer(
                self.config,
                f"{lm_head_prefix}.weight",
                self.initializers,
            ),
            weight_dtype=self.param_dtype,
            activation_dtype=self.param_dtype,
            weights_partition=self.sharding_config.translate(self.sharding_config.partition.get("lm_head.weights")),
            input_partition=self.sharding_config.translate(self.sharding_config.partition.get("lm_head.inputs")),
            output_partition=self.sharding_config.translate(self.sharding_config.partition.get("lm_head.outputs")),
        )

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        embed_key, model_key, lm_head_key = jax.random.split(rngs, 3)
        params = {"model": self.model.init(model_key)}
        params["model"]["embed_tokens"] = self.embed_tokens.init(embed_key)
        if not getattr(self.config, "tie_word_embeddings", True):
            params["lm_head"] = self.lm_head.init(lm_head_key)
        return params

    def embed(self, params: ParamTree, input_ids: jax.Array):
        return self.embed_tokens.apply(params["model"]["embed_tokens"], input_ids)

    def unembed(self, params: ParamTree, hidden_states: jax.Array):
        if getattr(self.config, "tie_word_embeddings", True):
            lm_head_params = params["model"]["embed_tokens"]
        else:
            lm_head_params = params["lm_head"]
        hidden_states = jnp.asarray(hidden_states, dtype=self.lm_head.activation_dtype)
        weight = jnp.asarray(
            lm_head_params[self.lm_head.weight_name],
            dtype=self.lm_head.activation_dtype,
        )
        return jnp.einsum(
            self.lm_head.equation,
            hidden_states,
            weight,
            preferred_element_type=jnp.float32,
            out_sharding=out_sharding(self.lm_head.output_partition),
        )

    def forward_hidden(
        self,
        params: ParamTree,
        input_ids: jax.Array,
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        hidden_states = self.embed(params, input_ids).astype(dtype)
        mask_mapping = make_mask(self.config, hidden_states, **inputs)
        return self.model.apply(
            params["model"],
            hidden_states,
            attention_mask=mask_mapping,
            pos=pos,
            rngs=rngs,
        )

    def forward(
        self,
        params: ParamTree,
        input_ids: jax.Array,
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        logits_to_keep: int = 0,
        return_hidden_states: bool = False,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        hidden_states = self.forward_hidden(
            params,
            input_ids,
            pos=pos,
            dtype=dtype,
            rngs=rngs,
            **inputs,
        )
        if return_hidden_states:
            return hidden_states
        if logits_to_keep:
            hidden_states = hidden_states[:, -int(logits_to_keep) :, :]
        return self.unembed(params, hidden_states)


def _build_mesh(
    parallel_dims: ParallelDims,
    devices: list | None = None,
    multihost: bool = False,
):
    if multihost:
        jax.distributed.initialize()
    axis_shapes = tuple(parallel_dims.values())
    axis_names = tuple(parallel_dims.keys())
    axis_types = tuple(AxisType.Auto for _ in axis_names)
    mesh = jax.make_mesh(
        axis_shapes,
        axis_names,
        axis_types=axis_types,
        devices=devices,
    )
    jax.set_mesh(mesh)
    return mesh


def _build_runtime(
    config: Config,
    parallel_dims: ParallelDims,
    devices: list | None,
    multihost: bool,
    additional_config: AdditionalConfig | None,
):
    merged_additional_config = {
        **DEFAULT_ADDITIONAL_CONFIG,
        **(additional_config or {}),
    }
    _build_mesh(parallel_dims, devices, multihost)
    sharding_config = make_gemma3_sharding_config(
        parallel_dims,
        sequence_parallelism=merged_additional_config["sequence_parallelism"],
    )
    runtime_config = copy.deepcopy(config)
    runtime_config.additional_config = merged_additional_config
    runtime_config.parallel_dims = parallel_dims
    runtime_config.sharding_config = sharding_config
    runtime_config.sharding_rules = sharding_config.logical_to_physical_mapping
    return runtime_config, sharding_config


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
    runtime_config, sharding_config = _build_runtime(
        config,
        parallel_dims,
        devices,
        multihost,
        additional_config,
    )
    module = Gemma3LM(
        config=runtime_config,
        sharding_config=sharding_config,
        param_dtype=param_dtype,
        initializers=initializers,
    )
    weights = module.init(rngs)
    return Model(
        name=__name__,
        config=runtime_config,
        weights=weights,
        forward=module.forward,
        embed=module.embed,
        unembed=module.unembed,
        tokenizer=tokenizer,
        lm_head_key=(
            "model.embed_tokens.weight"
            if getattr(runtime_config, "tie_word_embeddings", True)
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
    model_path = Path(model_id)
    if model_path.exists():
        model_ckpt_dir = model_path
    else:
        try:
            model_ckpt_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    local_files_only=True,
                )
            )
        except Exception:
            model_ckpt_dir = Path(snapshot_download(repo_id=model_id, local_dir=local_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_dir, use_fast=True)
    config = Gemma3TextConfig.from_pretrained(model_ckpt_dir)
    runtime_config, sharding_config = _build_runtime(
        config,
        parallel_dims,
        devices,
        multihost,
        additional_config,
    )
    module = Gemma3LM(
        config=runtime_config,
        sharding_config=sharding_config,
        param_dtype=param_dtype,
    )

    weights: ParamTree = {}
    for file in model_ckpt_dir.glob("*.safetensors"):
        with safe_open(file, framework="numpy") as handle:
            for key in handle.keys():
                weights[key] = jax.device_put(
                    handle.get_tensor(key).astype(param_dtype),
                    _param_sharding(key, sharding_config),
                )

    if "model.embed_tokens.weight" not in weights:
        raise KeyError(
            "Expected Gemma-3 text checkpoint with `model.embed_tokens.weight`."
        )

    weights = unflatten_param_tree(weights)

    return Model(
        name=__name__,
        config=runtime_config,
        weights=weights,
        forward=module.forward,
        embed=module.embed,
        unembed=module.unembed,
        tokenizer=tokenizer,
        lm_head_key=(
            "model.embed_tokens.weight"
            if getattr(runtime_config, "tie_word_embeddings", True)
            else "lm_head.weight"
        ),
    )
