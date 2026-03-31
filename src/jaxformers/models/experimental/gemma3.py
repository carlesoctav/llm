import struct
from contextlib import ExitStack
from dataclasses import fields
from pathlib import Path
from typing import Callable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from safetensors import safe_open
from transformers import AutoConfig, PreTrainedConfig

from jaxformers.attention_utils import ATTENTION_INTERFACE
from jaxformers.dispatch.einsum import einsum
from jaxformers.distributed import from_logical_rules, get_logical_axis_rules
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
    make_sliding_window_causal_mask,
)
from jaxformers.modeling_utils import (
    AdditionalConfig,
    DEFAULT_ADDITIONAL_CONFIG,
    ForwardImpl,
    PreTrainedModel,
)
from jaxformers.module_utils import Stackable, StackModule
from jaxformers.print_utils import tree_pprint


Config: TypeAlias = PreTrainedConfig
default_init = jax.nn.initializers.variance_scaling(
    1 / 3.0, "fan_in", "uniform", in_axis=-1, out_axis=-2, batch_axis=()
)


def _reshard_logical(x, logical):
    return reshard(x, from_logical_rules(logical))


def get_layer_metadata(config: Config) -> tuple[jax.Array, jax.Array]:
    layer_types = config.layer_types
    rope_theta = jnp.asarray(
        [
            config.rope_parameters[attention_type]["rope_theta"]
            for attention_type in layer_types
        ],
        dtype=jnp.float32,
    )
    is_sliding = jnp.asarray(
        [attention_type == "sliding_attention" for attention_type in layer_types],
        dtype=jnp.bool_,
    )
    return rope_theta, is_sliding


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


def make_mask(config, input_embeds, attention_mask=None, segment_ids=None, **kwargs):
    attn_impl = config.additional_config["attn_impl"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return {
            "full_attention": None,
            "sliding_attention": None,
        }

    full_mask = make_causal_mask(attn_impl, input_embeds, attention_mask, segment_ids)
    window_size = config.sliding_window
    if window_size is None:
        sliding_mask = full_mask
    else:
        sliding_mask = make_sliding_window_causal_mask(
            attn_impl,
            input_embeds,
            window_size,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


def apply_rope(q, k, rope_theta: float, pos: int):
    bsz, seqlen, _nheads, head_dim = q.shape
    positions = pos + jnp.broadcast_to(jnp.arange(seqlen)[None, :], [bsz, seqlen])
    freq = 1.0 / (
        rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    inp = einsum(
        "bt,h->bth",
        positions,
        freq,
        precision=jax.lax.Precision.HIGHEST,
    )
    sin = jnp.sin(inp).astype(q.dtype)[:, :, None, :]
    cos = jnp.cos(inp).astype(q.dtype)[:, :, None, :]
    q1, q2 = q[:, :, :, : head_dim // 2], q[:, :, :, head_dim // 2 :]
    k1, k2 = k[:, :, :, : head_dim // 2], k[:, :, :, head_dim // 2 :]
    q = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
    k = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
    return q, k


def replace_module(module, **changes):
    updated = object.__new__(type(module))
    for field in fields(type(module)):
        value = (
            changes[field.name]
            if field.name in changes
            else getattr(module, field.name)
        )
        object.__setattr__(updated, field.name, value)
    return updated


class Linear(eqx.Module):
    weight: Array
    bias: Array | None

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    out_sharding: P | None = eqx.field(static=True)
    w_sharding: P | None = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        use_bias: bool,
        out_sharding: P | None,
        w_sharding: P | None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.out_sharding = out_sharding
        self.w_sharding = w_sharding
        weight_sharding = P() if self.w_sharding is None else self.w_sharding
        if use_bias:
            weight_rngs, bias_rngs = jax.random.split(rngs)
            bias_sharding = P() if self.w_sharding is None else P(self.w_sharding[0])
            self.bias = jax.device_put(
                jnp.zeros((out_features,), param_dtype), bias_sharding
            )
        else:
            weight_rngs = rngs
            self.bias = None
        self.weight = jax.device_put(
            default_init(weight_rngs, (out_features, in_features), param_dtype),
            weight_sharding,
        )

    def __call__(self, x):
        y = einsum(
            "btf,df->btd",
            x,
            self.weight,
            preferred_element_type=x.dtype,
            out_sharding=self.out_sharding,
        )
        if self.bias is not None:
            y = y + self.bias[None, None, :]
        return y


class Embedding(eqx.Module):
    weight: Array

    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    padding_idx: int | None = eqx.field(static=True)
    embed_scale: float = eqx.field(static=True)
    out_sharding: P | None = eqx.field(static=True)
    w_sharding: P | None = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        embed_scale: float,
        out_sharding: P | None,
        w_sharding: P | None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_scale = embed_scale
        self.out_sharding = out_sharding
        self.w_sharding = w_sharding
        weight_sharding = P() if self.w_sharding is None else self.w_sharding
        self.weight = jax.device_put(
            default_init(rngs, (num_embeddings, embedding_dim), param_dtype),
            weight_sharding,
        )

    def __call__(self, input_ids, *, dtype=jnp.float32):
        x = self.weight.at[input_ids, :].get(out_sharding=self.out_sharding)
        x = x.astype(dtype)
        return x * jnp.asarray(self.embed_scale, dtype=dtype)


class RMSNorm(eqx.Module):
    weight: Array

    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    w_sharding: P | None = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        eps: float,
        w_sharding: P | None = None,
    ):
        self.dim = dim
        self.eps = eps
        self.w_sharding = w_sharding
        weight_sharding = P() if self.w_sharding is None else self.w_sharding
        self.weight = jax.device_put(jnp.zeros((dim,), param_dtype), weight_sharding)

    def __call__(self, x):
        return gemma_rms_norm(x, self.weight, self.eps)


class Gemma3Attention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear

    q_norm: RMSNorm
    k_norm: RMSNorm

    head_dim: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    query_scale: float = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)

    def __init__(
        self,
        config: Config,
        *,
        attn_impl: str,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        q_out = config.num_attention_heads * head_dim
        kv_out = config.num_key_value_heads * head_dim
        (
            q_proj_rngs,
            k_proj_rngs,
            v_proj_rngs,
            o_proj_rngs,
            q_norm_rngs,
            k_norm_rngs,
        ) = jax.random.split(rngs, 6)

        self.q_proj = Linear(
            hidden_size,
            q_out,
            rngs=q_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", "none")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.k_proj = Linear(
            hidden_size,
            kv_out,
            rngs=k_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", "none")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.v_proj = Linear(
            hidden_size,
            kv_out,
            rngs=v_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", "none")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.o_proj = Linear(
            q_out,
            hidden_size,
            rngs=o_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "sequence", "none")),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.q_norm = RMSNorm(
            head_dim,
            rngs=q_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = RMSNorm(
            head_dim,
            rngs=k_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.head_dim = head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.query_scale = (config.head_dim / config.query_pre_attn_scalar) ** 0.5
        self.attn_impl = attn_impl

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        rope_theta: float,
        pos: int,
    ):
        q_sharding = jax.NamedSharding(
            jax.sharding.get_abstract_mesh(),
            from_logical_rules(("batch", "context", "model", "none")),
        )
        attention_interface = ATTENTION_INTERFACE[self.attn_impl]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(
            q,
            "b t (n h) -> b t n h",
            n=self.num_attention_heads,
            h=self.head_dim,
        )
        k = rearrange(
            k,
            "b t (n h) -> b t n h",
            n=self.num_key_value_heads,
            h=self.head_dim,
        )
        v = rearrange(
            v,
            "b t (n h) -> b t n h",
            n=self.num_key_value_heads,
            h=self.head_dim,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * jnp.asarray(self.query_scale, dtype=q.dtype)
        bsz, seqlen, _nheads, head_dim = q.shape
        positions = pos + jnp.broadcast_to(jnp.arange(seqlen)[None, :], [bsz, seqlen])
        freq = 1.0 / (
            rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
        )
        inp = einsum(
            "bt,h->bth",
            positions,
            freq,
            precision=jax.lax.Precision.HIGHEST,
        )
        sin = jnp.sin(inp).astype(q.dtype)[:, :, None, :]
        cos = jnp.cos(inp).astype(q.dtype)[:, :, None, :]
        q1, q2 = q[:, :, :, : head_dim // 2], q[:, :, :, head_dim // 2 :]
        k1, k2 = k[:, :, :, : head_dim // 2], k[:, :, :, head_dim // 2 :]
        q = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
        k = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

        attn_output = attention_interface(
            q,
            k,
            v,
            mask=attention_mask,
            q_sharding=q_sharding,
        )
        attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
        return self.o_proj(attn_output)


class Gemma3MLP(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    act_fn: str = eqx.field(static=True)

    def __init__(
        self,
        config: Config,
        *,
        act_fn: str,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        gate_proj_rngs, up_proj_rngs, down_proj_rngs = jax.random.split(rngs, 3)
        self.gate_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=gate_proj_rngs,
            param_dtype=param_dtype,
            use_bias=False,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.up_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=up_proj_rngs,
            param_dtype=param_dtype,
            use_bias=False,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.down_proj = Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=down_proj_rngs,
            param_dtype=param_dtype,
            use_bias=False,
            out_sharding=from_logical_rules(("batch", "sequence", "none")),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.act_fn = act_fn

    def __call__(self, x):
        gate = get_activation_fn(self.act_fn)(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Gemma3Layer(eqx.Module, Stackable):
    self_attn: Gemma3Attention
    mlp: Gemma3MLP

    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    pre_feedforward_layernorm: RMSNorm
    post_feedforward_layernorm: RMSNorm

    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(
        static=True, default=("rope_theta", "is_sliding")
    )
    in_axes: int = eqx.field(static=True, default=0)

    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: Config,
        *,
        additional_config: AdditionalConfig,
        layer_idx: int,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        (
            self_attn_rngs,
            mlp_rngs,
            input_layernorm_rngs,
            post_attention_layernorm_rngs,
            pre_feedforward_layernorm_rngs,
            post_feedforward_layernorm_rngs,
        ) = jax.random.split(rngs, 6)
        self.self_attn = Gemma3Attention(
            config,
            attn_impl=additional_config["attn_impl"],
            rngs=self_attn_rngs,
            param_dtype=param_dtype,
        )
        self.mlp = Gemma3MLP(
            config,
            act_fn=config.hidden_activation,
            rngs=mlp_rngs,
            param_dtype=param_dtype,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            rngs=input_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            rngs=post_attention_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            rngs=pre_feedforward_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            rngs=post_feedforward_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.remat = additional_config["remat_layer"]

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        rope_theta,
        attention_mask,
        is_sliding,
        pos: int,
    ):
        attention_mask = jax.lax.select(
            is_sliding,
            attention_mask["sliding_attention"],
            attention_mask["full_attention"],
        )

        residual = x
        x_norm = self.input_layernorm(x)
        x_norm = _reshard_logical(x_norm, ("batch", "context", "none"))
        attn_output = self.self_attn(
            x_norm,
            attention_mask=attention_mask,
            rope_theta=rope_theta,
            pos=pos,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        x = residual + attn_output

        residual = x
        x_norm = self.pre_feedforward_layernorm(x)
        x_norm = _reshard_logical(x_norm, ("batch", "context", "none"))
        ffw = self.mlp(x_norm)
        ffw = self.post_feedforward_layernorm(ffw)
        return residual + ffw


class Gemma3Model(eqx.Module):
    embed_tokens: Embedding
    layers: list[Gemma3Layer] | StackModule[Gemma3Layer]
    norm: RMSNorm

    config: Config = eqx.field(static=True)

    def __init__(
        self,
        config: Config,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        embed_tokens_rngs, layers_rngs, norm_rngs = jax.random.split(rngs, 3)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            rngs=embed_tokens_rngs,
            param_dtype=param_dtype,
            embed_scale=config.hidden_size**0.5,
            out_sharding=from_logical_rules(("batch", "sequence", "none")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        layer_rngs = jax.random.split(layers_rngs, config.num_hidden_layers)
        self.layers = [
            Gemma3Layer(
                config,
                additional_config=additional_config,
                layer_idx=layer_idx,
                rngs=layer_rngs[layer_idx],
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.config = config

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        del rngs
        input_ids = _reshard_logical(input_ids, ("batch", "context"))
        x = self.embed_tokens(input_ids, dtype=dtype)
        mask_mapping = make_mask(self.config, x, **inputs)
        tree_pprint(mask_mapping)

        forward_impl = self.config.additional_config["forward_impl"]

        if forward_impl not in tuple(ForwardImpl):
            raise ValueError(
                f"Unsupported Gemma-3 forward implementation: {forward_impl!r}"
            )

        if forward_impl == ForwardImpl.LOOP:
            layers = (
                self.layers.unstack()
                if isinstance(self.layers, StackModule)
                else self.layers
            )
            for layer, attention_type in zip(layers, self.config.layer_types):
                x = layer(
                    x,
                    rope_theta=self.config.rope_parameters[attention_type][
                        "rope_theta"
                    ],
                    attention_mask=mask_mapping,
                    is_sliding=attention_type == "sliding_attention",
                    pos=pos,
                )
        elif forward_impl == ForwardImpl.SCAN_LAYER:
            rope_theta, is_sliding = get_layer_metadata(self.config)
            layers = self.layers
            if isinstance(layers, list):
                layers = StackModule(
                    Gemma3Layer,
                    layers,
                    0,
                    argnames=("rope_theta", "is_sliding"),
                    remat=self.config.additional_config["remat_layer"],
                )
            x = layers(
                x,
                rope_theta=rope_theta,
                attention_mask=mask_mapping,
                is_sliding=is_sliding,
                pos=pos,
            )

        x = self.norm(x)
        return _reshard_logical(x, ("batch", "context", "none"))

    def embed(
        self,
        input_ids: Int[Array, "B T"],
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        del rngs, inputs
        input_ids = _reshard_logical(input_ids, ("batch", "context"))
        return self.embed_tokens(input_ids, dtype=dtype)


class Gemma3ForCausalLM(PreTrainedModel):
    model: Gemma3Model
    lm_head: Linear | None

    def __init__(
        self,
        config: Config,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        model_rngs, lm_head_rngs = jax.random.split(rngs)
        self.model = Gemma3Model(
            config,
            additional_config,
            rngs=model_rngs,
            param_dtype=param_dtype,
        )
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                config.hidden_size,
                config.vocab_size,
                rngs=lm_head_rngs,
                param_dtype=param_dtype,
                use_bias=False,
                out_sharding=from_logical_rules(("batch", "context", "model")),
                w_sharding=from_logical_rules(("model", "fsdp")),
            )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def embed(self, *args, **kwargs):
        return self.model.embed(*args, **kwargs)

    def unembed(
        self,
        hidden_states: Float[Array, "B T D"],
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        del dtype, rngs, inputs
        out_embed = (
            self.model.embed_tokens.weight
            if self.model.config.tie_word_embeddings
            else self.lm_head.weight
        )
        return einsum(
            "btd,vd->btv",
            hidden_states,
            out_embed,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            preferred_element_type=jnp.float32,
        )

    @classmethod
    def init(
        cls,
        config: Config | None = None,
        model_id: str | None = None,
        additional_config: AdditionalConfig | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: PRNGKeyArray,
    ) -> "Gemma3ForCausalLM":
        if (config is None) == (model_id is None):
            raise ValueError(
                "Exactly one of `config` or `model_id` must be provided to gemma3.init()."
            )

        if model_id is not None:
            config = AutoConfig.from_pretrained(model_id)

        if not isinstance(config, PreTrainedConfig):
            raise TypeError(f"Expected HF config, got {type(config)!r}")

        additional_config = {
            **DEFAULT_ADDITIONAL_CONFIG,
            **(additional_config or {}),
        }

        config.additional_config = additional_config
        config.sharding_rules = get_logical_axis_rules()

        return cls(
            config,
            additional_config,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        local_dir: str | None = None,
        additional_config: AdditionalConfig | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: PRNGKeyArray | None = None,
    ) -> "Gemma3ForCausalLM":
        if rngs is None:
            rngs = jax.random.key(0)

        additional_config = {
            **DEFAULT_ADDITIONAL_CONFIG,
            **(additional_config or {}),
        }

        model_ckpt_dir = Path(snapshot_download(repo_id=model_id, local_dir=local_dir))
        config = AutoConfig.from_pretrained(model_ckpt_dir)
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(f"Expected HF config, got {type(config)!r}")

        config.additional_config = additional_config
        config.sharding_rules = get_logical_axis_rules()

        load_additional_config = {
            **additional_config,
        }

        load_failures = {}
        with ExitStack() as stack:
            hf_index = {}
            for file in model_ckpt_dir.glob("*.safetensors"):
                opened = stack.enter_context(safe_open(file, framework="numpy"))
                for key in opened.keys():
                    hf_index[key] = opened

            abstract_model = jax.eval_shape(
                lambda: cls(
                    config,
                    load_additional_config,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
            )
            def load_leaf(path, leaf):
                key = jax.tree_util.keystr(path, simple=True, separator=".")
                if key not in hf_index:
                    load_failures[key] = "missing"
                    return leaf
                tensor = hf_index[key].get_tensor(key).astype(param_dtype)
                return jax.device_put(tensor, leaf.sharding.spec)

            model = jax.tree.map_with_path(
                load_leaf,
                abstract_model,
            )
        if load_failures:
            print("Gemma3 load failed tensors:")
            for key, reason in sorted(load_failures.items()):
                print(f"{key}: {reason}")

        return model
