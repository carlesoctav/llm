from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from transformers import Gemma3TextConfig

from jaxformers.attention_utils import ATTENTION_INTERFACE, prepare_attention_kwargs
from jaxformers.dispatch.einsum import einsum
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


def get_layer_metadata(config: Gemma3TextConfig) -> tuple[jax.Array, jax.Array]:
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
        x_fp32 = jnp.asarray(x, jnp.float32)
        mean2 = jnp.mean(jax.lax.square(x_fp32), axis=-1, keepdims=True)
        y = jnp.asarray(x_fp32 * jax.lax.rsqrt(mean2 + self.eps), x.dtype)
        scale = jnp.asarray(1.0 + self.weight, dtype = x.dtype)
        return jnp.einsum("i...k,...k->i...k", y, scale, preferred_element_type= x.dtype)


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
    window_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        config: Gemma3TextConfig,
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
        self.query_scale = ( 1 / config.query_pre_attn_scalar) ** 0.5
        self.attn_impl = attn_impl
        self.window_size = config.sliding_window

    def __call__(
        self,
        x: Float[Array, "B T D"],
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        *,
        is_sliding,
        rope_theta: float,
        pos: int = 0,
        decode_state: PyTree | None = None,
    ):
        do_decode = True if decode_state is not None else False
        extra_output = {} if do_decode else None
        q_sharding = jax.sharding.NamedSharding(
            jax.sharding.get_abstract_mesh(),
            from_logical_rules(("batch", "context", "model", None)),
        )
        if do_decode and self.attn_impl != "sdpa":
            print(
                f"Decoding requested (decode_state provided), but attn_impl='{self.attn_impl}' is not 'sdpa'. Falling back to 'sdpa'."
            )
            active_attn_impl = "sdpa"
        else:
            active_attn_impl = self.attn_impl
        attention_interface = ATTENTION_INTERFACE[active_attn_impl]
        attention_args_kwargs = prepare_attention_kwargs(
            active_attn_impl,
            x,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            is_causal=True,
            is_sliding=is_sliding,
            is_mqa=self.num_key_value_heads == 1,
            window_size=self.window_size,
        )
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(
            q,
            "b t (n h) -> b t n h",  # (b, 1 , n, h) when decode
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

        # TODO: make this rope a layer
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
            scale=self.query_scale,
            q_sharding=q_sharding,
            **attention_args_kwargs,
        )
        attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
        return self.o_proj(attn_output), extra_output


class Gemma3MLP(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    act_fn: str = eqx.field(static=True)

    def __init__(
        self,
        config: Gemma3TextConfig,
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
        static=True, default=("rope_theta", "is_sliding", "decode_state")
    )
    in_axes: int = eqx.field(static=True, default=0)

    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: Gemma3TextConfig,
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
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        *,
        rope_theta,
        is_sliding,
        pos: int,
        decode_state: PyTree | None = None,
    ):
        residual = x
        x_norm = self.input_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        attn_output, extra_output = self.self_attn(
            x_norm,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            is_sliding=is_sliding,
            rope_theta=rope_theta,
            pos=pos,
            decode_state=decode_state,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        x = residual + attn_output

        residual = x
        x_norm = self.pre_feedforward_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        ffw = self.mlp(x_norm)
        ffw = self.post_feedforward_layernorm(ffw)
        return residual + ffw, extra_output


class Gemma3TextModel(AbstractHuggingFacePreTrainedModel):
    config: Gemma3TextConfig = eqx.field(static=True)
    embed_tokens: Embedding
    layers: list[Gemma3Layer] | StackModule[Gemma3Layer]
    norm: Gemma3RMSNorm

    def __init__(
        self,
        config: Gemma3TextConfig,
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
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        rngs: PRNGKeyArray | None = None,
        decode_states: PyTree | None = None,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        x = self.embed_tokens(input_ids, dtype=dtype)
        decode_states = (
            [None] * self.config.num_hidden_layers
            if decode_states is None
            else decode_states
        )
        extra_output_list = []
        forward_impl = forward_impl or self.config.additional_config["forward_impl"]

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
            fwd = (
                jax.remat(Gemma3Layer.__call__)
                if layers[0].remat
                else Gemma3Layer.__call__
            )
            for layer, attention_type, decode_state in zip(
                layers, self.config.layer_types, decode_states
            ):
                x, extra_output = fwd(
                    layer,
                    x,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    rope_theta=self.config.rope_parameters[attention_type][
                        "rope_theta"
                    ],
                    is_sliding=attention_type == "sliding_attention",
                    pos=pos,
                    decode_state=decode_state,
                )
                extra_output_list.append(extra_output)
        elif forward_impl == ForwardImpl.SCAN:
            rope_theta, is_sliding = get_layer_metadata(self.config)
            layers = self.layers
            if isinstance(layers, list):
                layers = StackModule(
                    Gemma3Layer,
                    layers,
                    0,
                    argnames=("rope_theta", "is_sliding", "decode_state"),
                    remat=self.config.additional_config["remat_layer"],
                )
            x, extra_output_list = layers(
                x,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                rope_theta=rope_theta,
                is_sliding=is_sliding,
                pos=pos,
                decode_state=decode_states,
            )

        x = self.norm(x)
        return reshard(
            x, from_logical_rules(("batch", "context", None))
        ), extra_output_list

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
    config: Gemma3TextConfig = eqx.field(static=True)
    model: Gemma3TextModel
    lm_head: Linear | None

    def __init__(
        self,
        config: Gemma3TextConfig,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
        store_config: bool = True,
    ):
        self.config = config
        model_rngs, lm_head_rngs = jax.random.split(rngs)
        self.model = Gemma3TextModel(
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
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        rngs: PRNGKeyArray | None = None,
        decode_states: PyTree | None = None,
        return_hidden_states=False,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        hidden_states, extra_outputs = self.model(
            input_ids,
            pos,
            dtype,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            rngs=rngs,
            decode_states=decode_states,
            forward_impl=forward_impl,
            **inputs,
        )

        if return_hidden_states:
            return hidden_states, extra_outputs

        out_weights = (
            self.lm_head.weight if self.lm_head else self.model.embed_tokens.weight
        )
        logits = einsum(
            "btd,vd->btv",
            hidden_states,
            out_weights,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            preferred_element_type=jnp.float32,
        )

        return logits, extra_outputs

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


class Gemma3ForSequenceClassification(AbstractHuggingFacePreTrainedModel):
    config: Gemma3TextConfig = eqx.field(static=True)
    model: Gemma3TextModel
    score: Linear
    num_labels: int = eqx.field(static=True)

    def __init__(
        self,
        config: Gemma3TextConfig,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        store_config: bool = True,
    ):
        self.config = config
        model_rngs, score_rngs = jax.random.split(rngs)
        self.num_labels = config.num_labels or additional_config.num_labels
        self.model = Gemma3TextModel(
            config,
            additional_config,
            rngs=model_rngs,
            param_dtype=param_dtype,
            store_config=store_config,
        )
        self.score = Linear(
            config.hidden_size,
            self.num_labels,
            use_bias=False,
            param_dtype=param_dtype,
            rngs=score_rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        *,
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        rngs: PRNGKeyArray | None = None,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        hidden_states = self.model(
            input_ids,
            pos,
            dtype,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            rngs=rngs,
            forward_impl=forward_impl,
            **inputs,
        )
        output = self.score(hidden_states)
        return output
