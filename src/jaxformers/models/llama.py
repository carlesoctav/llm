from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from transformers.models.llama.configuration_llama import LlamaConfig

from jaxformers.attention_utils import ATTENTION_INTERFACE
from jaxformers.dispatch.einsum import einsum
from jaxformers.masking_utils import ATTENTION_MASK_INTERFACE, make_causal_mask
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


def llama_rms_norm(x: jax.Array, weight: jax.Array, eps: float):
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


def get_inv_freq(config: LlamaConfig):
    dim = config.head_dim
    rope_scaling = config.rope_scaling
    inv_freq = 1.0 / (
        rope_scaling["rope_theta"]
        ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )
    attention_factor = 1.0
    rope_type = rope_scaling["rope_type"]
    if rope_type == "default":
        return inv_freq, attention_factor
    if rope_type != "llama3":
        raise NotImplementedError(f"Unsupported Llama rope_type: {rope_type!r}")

    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * jnp.pi / inv_freq
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (
        old_context_len / wavelen - low_freq_factor
    ) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (
        (1 - smooth_factor) * inv_freq_llama / factor
        + smooth_factor * inv_freq_llama
    )
    is_medium_freq = jnp.logical_and(
        wavelen >= high_freq_wavelen,
        wavelen <= low_freq_wavelen,
    )
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, attention_factor


def make_rotary_embeddings(
    config: LlamaConfig,
    batch_size: int,
    seq_len: int,
    dtype: jnp.dtype,
    pos: int,
):
    inv_freq, attention_factor = get_inv_freq(config)
    positions = pos + jnp.broadcast_to(
        jnp.arange(seq_len)[None, :],
        (batch_size, seq_len),
    )
    freqs = einsum(
        "bt,h->bth",
        positions,
        inv_freq,
        precision=jax.lax.Precision.HIGHEST,
    )
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    cos = (jnp.cos(emb) * attention_factor).astype(dtype)
    sin = (jnp.sin(emb) * attention_factor).astype(dtype)
    return cos, sin


def rotate_half(x: jax.Array):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: Float[Array, "B T N H"],
    k: Float[Array, "B T K H"],
    cos: Float[Array, "B T H"],
    sin: Float[Array, "B T H"],
):
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def make_mask(config, input_embeds, *, attention_mask=None, segment_ids=None):
    attn_impl = config.additional_config["attn_impl"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return None
    return make_causal_mask(
        attn_impl,
        input_embeds,
        attention_mask=attention_mask,
        segment_ids=segment_ids,
    )


class LlamaRMSNorm(eqx.Module):
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
        return llama_rms_norm(x, self.weight, self.eps)


class LlamaAttention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear

    head_dim: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    num_key_value_heads: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)

    def __init__(
        self,
        config: LlamaConfig,
        *,
        attn_impl: str,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        q_proj_rngs, k_proj_rngs, v_proj_rngs, o_proj_rngs = jax.random.split(rngs, 4)
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
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attn_impl = attn_impl

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        cos: Float[Array, "B T H"],
        sin: Float[Array, "B T H"],
        pos: int = 0,
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
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if do_decode:
            k, v, new_decode_state = make_kv_from_cache(
                k,
                v,
                pos,
                decode_state,
                init=False,
            )
            extra_output["decode_state"] = new_decode_state

        attn_output = attention_interface(
            q,
            k,
            v,
            mask=attention_mask,
            q_sharding=q_sharding,
            # segment_ids=segment_ids,
        )
        attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
        return self.o_proj(attn_output), extra_output


class LlamaMLP(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    act_fn: str = eqx.field(static=True)

    def __init__(
        self,
        config: LlamaConfig,
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
            use_bias=config.mlp_bias,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.up_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=up_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.down_proj = Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=down_proj_rngs,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.act_fn = act_fn

    def __call__(self, x):
        gate = get_activation_fn(self.act_fn)(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(eqx.Module, Stackable):
    self_attn: LlamaAttention
    mlp: LlamaMLP

    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(static=True, default=("decode_state",))
    in_axes: int = eqx.field(static=True, default=0)
    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: LlamaConfig,
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
        self.self_attn = LlamaAttention(
            config,
            attn_impl=additional_config["attn_impl"],
            rngs=self_attn_rngs,
            param_dtype=param_dtype,
        )
        self.mlp = LlamaMLP(
            config,
            act_fn=config.hidden_act,
            rngs=mlp_rngs,
            param_dtype=param_dtype,
        )
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            rngs=input_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size,
            rngs=post_attention_layernorm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.remat = additional_config["remat_layer"]

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        cos: Float[Array, "B T H"],
        sin: Float[Array, "B T H"],
        pos: int = 0,
        segment_ids: Int[Array, "B T"] | None = None,
        decode_state: PyTree | None = None,
    ):
        residual = x
        x_norm = self.input_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        attn_output, extra_output = self.self_attn(
            x_norm,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            pos=pos,
            segment_ids=segment_ids,
            decode_state=decode_state,
        )
        x = residual + attn_output

        residual = x
        x_norm = self.post_attention_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        return residual + self.mlp(x_norm), extra_output


class LlamaModel(AbstractHuggingFacePreTrainedModel):
    config: LlamaConfig = eqx.field(static=True)
    embed_tokens: Embedding
    layers: list[LlamaDecoderLayer] | StackModule[LlamaDecoderLayer]
    norm: LlamaRMSNorm

    def __init__(
        self,
        config: LlamaConfig,
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
            LlamaDecoderLayer(
                config,
                additional_config=additional_config,
                rngs=layer_rngs[layer_idx],
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = LlamaRMSNorm(
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
        attention_mask = make_mask(
            self.config,
            x,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        cos, sin = make_rotary_embeddings(
            self.config,
            x.shape[0],
            x.shape[1],
            x.dtype,
            pos,
        )
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
                f"Unsupported Llama forward implementation: {forward_impl!r}"
            )

        if forward_impl == ForwardImpl.LOOP:
            layers = (
                self.layers.unstack()
                if isinstance(self.layers, StackModule)
                else self.layers
            )
            fwd = (
                jax.remat(LlamaDecoderLayer.__call__)
                if layers[0].remat
                else LlamaDecoderLayer.__call__
            )
            for layer, decode_state in zip(layers, decode_states):
                x, extra_output = fwd(
                    layer,
                    x,
                    attention_mask=attention_mask,
                    cos=cos,
                    sin=sin,
                    pos=pos,
                    segment_ids=segment_ids,
                    decode_state=decode_state,
                )
                extra_output_list.append(extra_output)
        elif forward_impl == ForwardImpl.SCAN:
            layers = self.layers
            if isinstance(layers, list):
                layers = StackModule(
                    LlamaDecoderLayer,
                    layers,
                    0,
                    argnames="decode_state",
                    remat=self.config.additional_config["remat_layer"],
                )
            x, extra_output_list = layers(
                x,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
                pos=pos,
                segment_ids=segment_ids,
                decode_state=decode_states,
            )

        if not return_decode_states and extra_output_list is None:
            extra_output_list = [None] * self.config.num_hidden_layers

        x = self.norm(x)
        return reshard(x, from_logical_rules(("batch", "context", None))), extra_output_list


class LlamaForCausalLM(AbstractHuggingFacePreTrainedModel):
    config: LlamaConfig = eqx.field(static=True)
    model: LlamaModel
    lm_head: Linear | None

    def __init__(
        self,
        config: LlamaConfig,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        model_rngs, lm_head_rngs = jax.random.split(rngs)
        self.model = LlamaModel(
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
