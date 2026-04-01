from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from transformers import Gemma3Config

from jaxformers.attention_utils import ATTENTION_INTERFACE
from jaxformers.dispatch.einsum import einsum
from jaxformers.distributed import from_logical_rules
from jaxformers.masking_utils import (
    ATTENTION_MASK_INTERFACE,
    make_causal_mask,
    make_sliding_window_causal_mask,
)
from jaxformers.module_utils import (
    AbstractHuggingFacePreTrainedModel,
    AdditionalConfig,
    ForwardImpl,
    Stackable,
    StackModule,
)
from jaxformers.nn import Embedding, Linear


def get_layer_metadata(config: Gemma3Config) -> tuple[jax.Array, jax.Array]:
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


class Gemma3RMSNorm(eqx.Module):
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

    q_norm: Gemma3RMSNorm
    k_norm: Gemma3RMSNorm

    head_dim: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    query_scale: float = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)

    def __init__(
        self,
        config: Gemma3Config,
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
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.k_proj = Linear(
            hidden_size,
            kv_out,
            rngs=k_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.v_proj = Linear(
            hidden_size,
            kv_out,
            rngs=v_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.o_proj = Linear(
            q_out,
            hidden_size,
            rngs=o_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.q_norm = Gemma3RMSNorm(
            head_dim,
            rngs=q_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Gemma3RMSNorm(
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
            from_logical_rules(("batch", "context", "model", None)),
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
        config: Gemma3Config,
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
            out_sharding=from_logical_rules(("batch", "sequence", None)),
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

    input_layernorm: Gemma3RMSNorm
    post_attention_layernorm: Gemma3RMSNorm
    pre_feedforward_layernorm: Gemma3RMSNorm
    post_feedforward_layernorm: Gemma3RMSNorm

    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(
        static=True, default=("rope_theta", "is_sliding")
    )
    in_axes: int = eqx.field(static=True, default=0)

    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: Gemma3Config,
        *,
        additional_config: AdditionalConfig,
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
        self.input_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            rngs=input_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            rngs=post_attention_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size,
            rngs=pre_feedforward_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
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
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
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
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        ffw = self.mlp(x_norm)
        ffw = self.post_feedforward_layernorm(ffw)
        return residual + ffw


class Gemma3Model(AbstractHuggingFacePreTrainedModel):
    config: Gemma3Config = eqx.field(static=True)
    embed_tokens: Embedding
    layers: list[Gemma3Layer] | StackModule[Gemma3Layer]
    norm: Gemma3RMSNorm

    def __init__(
        self,
        config: Gemma3Config,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        store_config: bool = True,
    ):
        self.config = config
        embed_tokens_rngs, layers_rngs, norm_rngs = jax.random.split(rngs, 3)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            rngs=embed_tokens_rngs,
            param_dtype=param_dtype,
            embed_scale=config.hidden_size**0.5,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        layer_rngs = jax.random.split(layers_rngs, config.num_hidden_layers)
        self.layers = [
            Gemma3Layer(
                config,
                additional_config=additional_config,
                rngs=layer_rngs[layer_idx],
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Gemma3RMSNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        x = self.embed_tokens(input_ids, dtype=dtype)
        mask_mapping = make_mask(self.config, x, **inputs)

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
            fwd = jax.remat(Gemma3Layer.__call__) if layers[0].remat else Gemma3Layer.__call__
            for layer, attention_type in zip(layers, self.config.layer_types):
                x = fwd(
                    layer,
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
        return reshard(x, from_logical_rules(("batch", "context", None)))

    def embed(
        self,
        input_ids: Int[Array, "B T"],
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        input_ids = reshard(input_ids, from_logical_rules(("batch", "sequence")))
        return self.embed_tokens(input_ids, dtype=dtype)


class Gemma3ForCausalLM(AbstractHuggingFacePreTrainedModel):
    config: Gemma3Config = eqx.field(static=True)
    model: Gemma3Model
    lm_head: Linear | None

    def __init__(
        self,
        config: Gemma3Config,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
        store_config: bool = True,
    ):
        self.config = config
        model_rngs, lm_head_rngs = jax.random.split(rngs)
        self.model = Gemma3Model(
            config,
            additional_config,
            rngs=model_rngs,
            param_dtype=param_dtype,
            store_config=store_config,
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

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        return self.model(input_ids, pos, dtype, rngs = rngs, **inputs)

    @property
    def lm_head_w(self):
        return self.lm_head.weight if self.lm_head else self.model.embed_tokens.weight

    def unembed(
        self,
        hidden_states: Float[Array, "B T D"],
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        out_weights = (
            self.lm_head.weight if self.lm_head else self.model.embed_tokens.weight
        )
        return einsum(
            "btd,vd->btv",
            hidden_states,
            out_weights,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            preferred_element_type=jnp.float32,
        )
