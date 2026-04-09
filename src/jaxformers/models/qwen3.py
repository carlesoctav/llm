from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from jaxformers.attention_utils import ATTENTION_INTERFACE
from jaxformers.dispatch.einsum import einsum
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
from jaxformers.sampling_utils import make_kv_from_cache
from jaxformers.sharding_utils import from_logical_rules


def qwen3_rms_norm(x: jax.Array, weight: jax.Array, eps: float):
    x_fp32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.square(x_fp32).mean(-1, keepdims=True) + eps)
    out = (x_fp32 / rms) * weight.astype(jnp.float32)
    return out.astype(x.dtype)


def get_activation_fn(hidden_act: str) -> Callable[[jax.Array], jax.Array]:
    if hidden_act in ("gelu_pytorch_tanh", "gelu_new"):
        return lambda x: jax.nn.gelu(x, approximate=True)
    if hidden_act == "gelu":
        return lambda x: jax.nn.gelu(x, approximate=False)
    if hidden_act == "relu":
        return jax.nn.relu
    if hidden_act == "silu":
        return jax.nn.silu
    raise ValueError(f"Unsupported hidden activation {hidden_act!r}")


def get_rope_theta(config: Qwen3Config) -> float:
    rope_scaling = config.rope_scaling
    if not isinstance(rope_scaling, dict):
        raise TypeError("Qwen3 config must define rope_scaling")
    if rope_scaling["rope_type"] != "default":
        raise NotImplementedError(
            f"Unsupported Qwen3 rope_type: {rope_scaling['rope_type']!r}"
        )
    return rope_scaling["rope_theta"]


def make_mask(config, input_embeds, *, attention_mask=None, segment_ids=None):
    attn_impl = config.additional_config["attn_impl"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return {
            "full_attention": None,
            "sliding_attention": None,
        }

    full_mask = make_causal_mask(attn_impl, input_embeds, attention_mask, segment_ids)
    sliding_window = config.sliding_window
    if sliding_window is None:
        sliding_mask = full_mask
    else:
        sliding_mask = make_sliding_window_causal_mask(
            attn_impl,
            input_embeds,
            sliding_window,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


class Qwen3RMSNorm(eqx.Module):
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
        del rngs
        self.dim = dim
        self.eps = eps
        self.w_sharding = w_sharding
        weight_sharding = P() if self.w_sharding is None else self.w_sharding
        self.weight = jax.device_put(jnp.ones((dim,), param_dtype), weight_sharding)

    def __call__(self, x):
        return qwen3_rms_norm(x, self.weight, self.eps)


class Qwen3Attention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear

    q_norm: Qwen3RMSNorm
    k_norm: Qwen3RMSNorm

    head_dim: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)
    sliding_window: int | None = eqx.field(static=True)

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        attn_impl: str,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        (
            q_proj_rngs,
            k_proj_rngs,
            v_proj_rngs,
            o_proj_rngs,
            q_norm_rngs,
            k_norm_rngs,
        ) = jax.random.split(rngs, 6)
        q_out = config.num_attention_heads * config.head_dim
        kv_out = config.num_key_value_heads * config.head_dim
        self.q_proj = Linear(
            config.hidden_size,
            q_out,
            rngs=q_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.k_proj = Linear(
            config.hidden_size,
            kv_out,
            rngs=k_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.v_proj = Linear(
            config.hidden_size,
            kv_out,
            rngs=v_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.o_proj = Linear(
            q_out,
            config.hidden_size,
            rngs=o_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.q_norm = Qwen3RMSNorm(
            config.head_dim,
            rngs=q_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3RMSNorm(
            config.head_dim,
            rngs=k_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attn_impl = attn_impl
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        rope_theta: float,
        pos: int = 0,
        is_sliding: bool = False,
        segment_ids: Int[Array, "B T"] | None = None,
        decode_state: PyTree | None = None,
    ):
        do_decode = decode_state is not None
        extra_output = {} if do_decode else None
        q_sharding = jax.NamedSharding(
            jax.sharding.get_abstract_mesh(),
            from_logical_rules(("batch", "context", "model", None)),
        )
        if do_decode and self.attn_impl != "sdpa":
            attention_interface = ATTENTION_INTERFACE["sdpa"]
        else:
            attention_interface = ATTENTION_INTERFACE[self.attn_impl]

        q = rearrange(
            self.q_proj(x),
            "b t (n h) -> b t n h",
            n=self.num_attention_heads,
            h=self.head_dim,
        )
        k = rearrange(
            self.k_proj(x),
            "b t (n h) -> b t n h",
            n=self.num_key_value_heads,
            h=self.head_dim,
        )
        v = rearrange(
            self.v_proj(x),
            "b t (n h) -> b t n h",
            n=self.num_key_value_heads,
            h=self.head_dim,
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        batch_size, seq_len, _num_heads, head_dim = q.shape
        positions = pos + jnp.broadcast_to(
            jnp.arange(seq_len)[None, :], (batch_size, seq_len)
        )
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

        if do_decode:
            k, v, new_decode_state = make_kv_from_cache(k, v, pos, decode_state)
            extra_output["decode_state"] = new_decode_state

        attn_output = attention_interface(
            q,
            k,
            v,
            mask=attention_mask,
            q_sharding=q_sharding,
            segment_ids=segment_ids,
            is_sliding=is_sliding,
            window_size=self.sliding_window,
        )
        attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
        return self.o_proj(attn_output), extra_output


class Qwen3MLP(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    act_fn: str = eqx.field(static=True)

    def __init__(
        self,
        config: Qwen3Config,
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


class Qwen3DecoderLayer(eqx.Module, Stackable):
    self_attn: Qwen3Attention
    mlp: Qwen3MLP

    input_layernorm: Qwen3RMSNorm
    post_attention_layernorm: Qwen3RMSNorm

    attention_type: str = eqx.field(static=True)
    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(
        static=True,
        default=("attention_mask", "rope_theta", "is_sliding", "decode_state"),
    )
    in_axes: int = eqx.field(static=True, default=0)
    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
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
        ) = jax.random.split(rngs, 4)
        self.self_attn = Qwen3Attention(
            config,
            layer_idx,
            attn_impl=additional_config["attn_impl"],
            rngs=self_attn_rngs,
            param_dtype=param_dtype,
        )
        self.mlp = Qwen3MLP(
            config,
            act_fn=config.hidden_act,
            rngs=mlp_rngs,
            param_dtype=param_dtype,
        )
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            rngs=input_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            rngs=post_attention_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.attention_type = config.layer_types[layer_idx]
        self.remat = additional_config["remat_layer"]

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        rope_theta: float,
        pos: int,
        is_sliding: bool,
        segment_ids: Int[Array, "B T"] | None = None,
        decode_state: PyTree | None = None,
    ):
        residual = x
        x_norm = self.input_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        attn_output, extra_output = self.self_attn(
            x_norm,
            attention_mask=attention_mask,
            rope_theta=rope_theta,
            pos=pos,
            is_sliding=is_sliding,
            segment_ids=segment_ids,
            decode_state=decode_state,
        )
        x = residual + attn_output

        residual = x
        x_norm = self.post_attention_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        return residual + self.mlp(x_norm), extra_output


class Qwen3Model(AbstractHuggingFacePreTrainedModel):
    config: Qwen3Config = eqx.field(static=True)
    embed_tokens: Embedding
    layers: list[Qwen3DecoderLayer] | StackModule[Qwen3DecoderLayer]
    norm: Qwen3RMSNorm

    def __init__(
        self,
        config: Qwen3Config,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        self.config = config
        embed_tokens_rngs, layers_rngs, norm_rngs = jax.random.split(rngs, 3)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            rngs=embed_tokens_rngs,
            param_dtype=param_dtype,
            embed_scale=1.0,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        layer_rngs = jax.random.split(layers_rngs, config.num_hidden_layers)
        self.layers = [
            Qwen3DecoderLayer(
                config,
                layer_idx,
                additional_config=additional_config,
                rngs=layer_rngs[layer_idx],
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Array | None = None,
        pos: int = 0,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: PRNGKeyArray | None = None,
        decode_states: PyTree | None = None,
        forward_impl: ForwardImpl | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        **inputs,
    ):
        del rngs, inputs

        x = self.embed_tokens(input_ids, dtype=dtype)
        mask_mapping = make_mask(
            self.config,
            x,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        per_layer_masks = [
            mask_mapping[attention_type] for attention_type in self.config.layer_types
        ]
        return_decode_states = decode_states is not None
        decode_states = (
            [None] * self.config.num_hidden_layers
            if decode_states is None
            else decode_states
        )
        extra_output_list = []
        forward_impl = forward_impl or self.config.additional_config["forward_impl"]
        if forward_impl not in tuple(ForwardImpl):
            raise ValueError(
                f"Unsupported Qwen-3 forward implementation: {forward_impl!r}"
            )

        rope_theta_value = get_rope_theta(self.config)
        rope_theta = jnp.full(
            (self.config.num_hidden_layers,),
            rope_theta_value,
            dtype=jnp.float32,
        )
        is_sliding = jnp.asarray(
            [
                attention_type == "sliding_attention"
                for attention_type in self.config.layer_types
            ],
            dtype=jnp.bool_,
        )

        if forward_impl == ForwardImpl.LOOP:
            layers = (
                self.layers.unstack()
                if isinstance(self.layers, StackModule)
                else self.layers
            )
            fwd = (
                jax.remat(Qwen3DecoderLayer.__call__)
                if layers[0].remat
                else Qwen3DecoderLayer.__call__
            )
            for layer, layer_mask, layer_rope_theta, layer_is_sliding, decode_state in zip(
                layers,
                per_layer_masks,
                rope_theta,
                is_sliding,
                decode_states,
            ):
                x, extra_output = fwd(
                    layer,
                    x,
                    attention_mask=layer_mask,
                    rope_theta=layer_rope_theta,
                    pos=pos,
                    is_sliding=layer_is_sliding,
                    segment_ids=segment_ids,
                    decode_state=decode_state,
                )
                extra_output_list.append(extra_output)
        elif forward_impl == ForwardImpl.SCAN:
            layers = self.layers
            if isinstance(layers, list):
                layers = StackModule(
                    Qwen3DecoderLayer,
                    layers,
                    0,
                    argnames=("attention_mask", "rope_theta", "is_sliding", "decode_state"),
                    remat=self.config.additional_config["remat_layer"],
                )
            x, extra_output_list = layers(
                x,
                attention_mask=per_layer_masks,
                rope_theta=rope_theta,
                is_sliding=is_sliding,
                pos=pos,
                segment_ids=segment_ids,
                decode_state=decode_states,
            )

        if not return_decode_states and extra_output_list is None:
            extra_output_list = [None] * self.config.num_hidden_layers

        x = self.norm(x)
        return reshard(x, from_logical_rules(("batch", "context", None))), extra_output_list


class Qwen3ForCausalLM(AbstractHuggingFacePreTrainedModel):
    config: Qwen3Config = eqx.field(static=True)
    model: Qwen3Model
    lm_head: Linear | None

    def __init__(
        self,
        config: Qwen3Config,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        model_rngs, lm_head_rngs = jax.random.split(rngs)
        self.model = Qwen3Model(
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

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        attention_mask: Array | None = None,
        pos: int = 0,
        *,
        dtype: jnp.dtype = jnp.float32,
        return_hidden_states: bool = False,
        decode_states: PyTree | None = None,
        forward_impl: ForwardImpl | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        hidden_states, extra_outputs = self.model(
            input_ids,
            attention_mask,
            pos,
            dtype=dtype,
            rngs=rngs,
            decode_states=decode_states,
            forward_impl=forward_impl,
            segment_ids=segment_ids,
            **inputs,
        )
        if return_hidden_states:
            return hidden_states, extra_outputs

        return self.unembed(hidden_states), extra_outputs

    @property
    def lm_head_w(self):
        return self.lm_head.weight if self.lm_head else self.model.embed_tokens.weight

    def unembed(
        self,
        hidden_states: Float[Array, "B T D"],
        *,
        rngs: PRNGKeyArray | None = None,
        **inputs,
    ):
        del rngs, inputs
        return einsum(
            "btd,vd->btv",
            hidden_states,
            self.lm_head_w,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            preferred_element_type=jnp.float32,
        )
