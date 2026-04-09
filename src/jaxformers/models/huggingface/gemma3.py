from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from transformers import Gemma3TextConfig

from jaxformers import tree_util
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
from jaxformers.sharding_utils import (
    from_logical_rules,
    logical_reshard,
)


GEMMA3_ATTENTION_PATTERN = (
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
)


def get_rope_theta(config: Gemma3TextConfig, attention_type: str) -> float:
    valid_layer_types = ("full_attention", "sliding_attention")
    if attention_type not in valid_layer_types:
        raise ValueError(f"Unsupported Gemma-3 attention type: {attention_type!r}")

    config_vars = vars(config)
    if "rope_parameters" in config_vars and config.rope_parameters is not None:
        return config.rope_parameters[attention_type]["rope_theta"]
    if attention_type == "full_attention":
        return config.rope_theta
    elif attention_type == "sliding_attention":
        return config.rope_local_base_freq


def get_attention_type(config: Gemma3TextConfig, layer_idx: int) -> str:
    attention_type = config.layer_types[layer_idx]
    valid_layer_types = ("full_attention", "sliding_attention")
    if attention_type not in valid_layer_types:
        raise ValueError(f"Unsupported Gemma-3 attention type: {attention_type!r}")
    return attention_type


def make_rotary_embeddings(
    rope_theta: float,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    dtype: jnp.dtype,
    pos: int,
):
    positions = pos + jnp.broadcast_to(
        jnp.arange(seq_len)[None, :],
        (batch_size, seq_len),
    )
    freq = 1.0 / (
        jnp.asarray(rope_theta, dtype=jnp.float32)
        ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    inp = einsum(
        "bt,h->bth",
        positions,
        freq,
        precision=jax.lax.Precision.HIGHEST,
    )
    sin = jnp.sin(inp).astype(dtype)[:, :, None, :]
    cos = jnp.cos(inp).astype(dtype)[:, :, None, :]
    return cos, sin


def apply_rotary_pos_emb(
    q: Float[Array, "B T N H"],
    k: Float[Array, "B T K H"],
    cos: Float[Array, "B T 1 H"],
    sin: Float[Array, "B T 1 H"],
):
    head_dim = q.shape[-1]
    q1, q2 = q[:, :, :, : head_dim // 2], q[:, :, :, head_dim // 2 :]
    k1, k2 = k[:, :, :, : head_dim // 2], k[:, :, :, head_dim // 2 :]
    q = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
    k = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)
    return q, k


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


def make_mask(config, input_embeds, *, attention_mask=None, segment_ids=None):
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
    attention_mask_key: str = eqx.field(static=True)

    def __init__(
        self,
        config: Gemma3TextConfig,
        *,
        attn_impl: str,
        attention_type: str,
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
        self.attention_mask_key = attention_type

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        cos: Float[Array, "B T 1 H"],
        sin: Float[Array, "B T 1 H"],
        pos: int = 0,
        decode_state: PyTree | None = None,
    ):
        do_decode = True if decode_state is not None else False
        extra_output = {} if do_decode else None
        q_sharding = jax.NamedSharding(
            jax.sharding.get_abstract_mesh(),
            from_logical_rules(("batch", "context", "model", None)),
        )
        if do_decode and self.attn_impl != "sdpa":
            print(
                f"Decoding requested (decode_state provided), but attn_impl='{self.attn_impl}' is not 'sdpa'. Falling back to 'sdpa'."
            )
            attention_interface = ATTENTION_INTERFACE["sdpa"]
        else:
            attention_interface = ATTENTION_INTERFACE[self.attn_impl]
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

        q = q * jnp.asarray(self.query_scale, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if do_decode:
            k, v, new_decode_state = make_kv_from_cache(k, v, pos, decode_state)
            extra_output["decode_state"] = new_decode_state

        attn_output = attention_interface(
            q,
            k,
            v,
            mask=attention_mask[self.attention_mask_key],
            q_sharding=q_sharding,
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

    argnums: tuple[int, ...] = eqx.field(static=True, default=())
    argnames: tuple[str, ...] = eqx.field(static=True, default=("decode_state",))
    in_axes: int = eqx.field(static=True, default=0)

    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: Gemma3TextConfig,
        *,
        attention_type: str,
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
            attention_type=attention_type,
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
        attention_mask,
        *,
        pos: int,
        full_cos: Float[Array, "B T 1 H"] | None = None,
        full_sin: Float[Array, "B T 1 H"] | None = None,
        sliding_cos: Float[Array, "B T 1 H"] | None = None,
        sliding_sin: Float[Array, "B T 1 H"] | None = None,
        decode_state: PyTree | None = None,
    ):
        residual = x
        x_norm = self.input_layernorm(x)
        x_norm = logical_reshard(
            x_norm, from_logical_rules(("batch", "context", None))
        )
        if (
            full_cos is None
            or full_sin is None
            or sliding_cos is None
            or sliding_sin is None
        ):
            raise ValueError(
                "Gemma3Layer requires precomputed full/sliding rotary embeddings."
            )
        if self.self_attn.attention_mask_key == "sliding_attention":
            cos, sin = sliding_cos, sliding_sin
        else:
            cos, sin = full_cos, full_sin
        attn_output, extra_output = self.self_attn(
            x_norm,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            pos=pos,
            decode_state=decode_state,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        x = residual + attn_output

        residual = x
        x_norm = self.pre_feedforward_layernorm(x)
        x_norm = logical_reshard(
            x_norm, from_logical_rules(("batch", "context", None))
        )
        ffw = self.mlp(x_norm)
        ffw = self.post_feedforward_layernorm(ffw)
        return residual + ffw, extra_output


class Gemma3Block(eqx.Module, Stackable):
    layers: list[Gemma3Layer]

    argnums: tuple[int, ...] = eqx.field(static=True, default=())
    argnames: tuple[str, ...] = eqx.field(static=True, default=("decode_state",))
    in_axes: int = eqx.field(static=True, default=0)
    remat: bool = eqx.field(static=True, default=True)

    def __init__(self, layers: list[Gemma3Layer]):
        self.layers = layers
        self.remat = layers[0].remat if layers else False

    def __call__(
        self,
        x: Float[Array, "B T D"],
        attention_mask,
        *,
        pos: int,
        full_cos: Float[Array, "B T 1 H"],
        full_sin: Float[Array, "B T 1 H"],
        sliding_cos: Float[Array, "B T 1 H"],
        sliding_sin: Float[Array, "B T 1 H"],
        decode_state: list[PyTree | None] | None = None,
    ):
        decode_states = (
            [None] * len(self.layers) if decode_state is None else decode_state
        )
        extra_output_list = []

        for layer, layer_decode_state in zip(self.layers, decode_states):
            x, extra_output = layer(
                x,
                attention_mask,
                pos=pos,
                full_cos=full_cos,
                full_sin=full_sin,
                sliding_cos=sliding_cos,
                sliding_sin=sliding_sin,
                decode_state=layer_decode_state,
            )
            extra_output_list.append(extra_output)

        return x, extra_output_list


class Gemma3TextModel(AbstractHuggingFacePreTrainedModel):
    config: Gemma3TextConfig = eqx.field(static=True)
    embed_tokens: Embedding
    layers: list[Gemma3Layer] | StackModule[Gemma3Layer] | StackModule[Gemma3Block]
    layers_remainder: Gemma3Block | None
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
                attention_type=get_attention_type(config, layer_idx),
                additional_config=additional_config,
                rngs=layer_rngs[layer_idx],
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.layers_remainder = None
        self.norm = Gemma3RMSNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )

    def stack_block(self):
        layers = (
            self.layers.unstack()
            if isinstance(self.layers, StackModule)
            else self.layers
        )
        if not layers:
            return self
        if isinstance(layers[0], Gemma3Block):
            return self

        block_size = len(GEMMA3_ATTENTION_PATTERN)
        num_blocks = len(layers) // block_size
        full_blocks = [
            Gemma3Block(layers[block_idx * block_size : (block_idx + 1) * block_size])
            for block_idx in range(num_blocks)
        ]
        remainder_layers = layers[num_blocks * block_size :]
        stacked_blocks = (
            StackModule(
                Gemma3Block,
                full_blocks,
                (),
                argnames="decode_state",
                remat=self.config.additional_config["remat_layer"],
            )
            if full_blocks
            else []
        )
        layers_remainder = (
            Gemma3Block(remainder_layers) if remainder_layers else None
        )
        return eqx.tree_at(
            lambda tree: (tree.layers, tree.layers_remainder),
            self,
            (stacked_blocks, layers_remainder),
        )

    def stack(self):
        return self.stack_block()

    def _flatten_scanned_block_outputs(self, block_outputs, num_blocks: int):
        if not block_outputs:
            return []
        if block_outputs[0] is None:
            return [None] * (num_blocks * len(block_outputs))

        layer_outputs = [tree_util.unstack(output) for output in block_outputs]
        extra_output_list = []
        for block_idx in range(num_blocks):
            for layer_output in layer_outputs:
                extra_output_list.append(layer_output[block_idx])
        return extra_output_list

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
        x = self.embed_tokens(input_ids, dtype=dtype)
        mask_mapping = make_mask(
            self.config,
            x,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        return_decode_states = decode_states is not None
        decode_states = (
            [None] * self.config.num_hidden_layers
            if decode_states is None
            else decode_states
        )
        extra_output_list = []
        forward_impl = forward_impl or self.config.additional_config["forward_impl"]
        full_cos, full_sin = make_rotary_embeddings(
            get_rope_theta(self.config, "full_attention"),
            x.shape[0],
            x.shape[1],
            self.config.head_dim,
            x.dtype,
            pos,
        )
        sliding_cos, sliding_sin = make_rotary_embeddings(
            get_rope_theta(self.config, "sliding_attention"),
            x.shape[0],
            x.shape[1],
            self.config.head_dim,
            x.dtype,
            pos,
        )

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
            if self.layers_remainder is not None or (
                layers and isinstance(layers[0], Gemma3Block)
            ):
                block_size = len(GEMMA3_ATTENTION_PATTERN)
                block_decode_states = [
                    decode_states[
                        block_idx * block_size : (block_idx + 1) * block_size
                    ]
                    for block_idx in range(len(layers))
                ]
                fwd = (
                    jax.remat(Gemma3Block.__call__)
                    if layers[0].remat
                    else Gemma3Block.__call__
                )
                for block, block_decode_state in zip(layers, block_decode_states):
                    x, block_extra_output = fwd(
                        block,
                        x,
                        mask_mapping,
                        pos=pos,
                        full_cos=full_cos,
                        full_sin=full_sin,
                        sliding_cos=sliding_cos,
                        sliding_sin=sliding_sin,
                        decode_state=block_decode_state,
                    )
                    extra_output_list.extend(block_extra_output)
                if self.layers_remainder is not None:
                    x, remainder_extra_output = self.layers_remainder(
                        x,
                        mask_mapping,
                        pos=pos,
                        full_cos=full_cos,
                        full_sin=full_sin,
                        sliding_cos=sliding_cos,
                        sliding_sin=sliding_sin,
                        decode_state=decode_states[len(layers) * block_size :],
                    )
                    extra_output_list.extend(remainder_extra_output)
            else:
                fwd = (
                    jax.remat(Gemma3Layer.__call__)
                    if layers[0].remat
                    else Gemma3Layer.__call__
                )
                for layer, decode_state in zip(layers, decode_states):
                    x, extra_output = fwd(
                        layer,
                        x,
                        mask_mapping,
                        pos=pos,
                        full_cos=full_cos,
                        full_sin=full_sin,
                        sliding_cos=sliding_cos,
                        sliding_sin=sliding_sin,
                        decode_state=decode_state,
                    )
                    extra_output_list.append(extra_output)
        elif forward_impl == ForwardImpl.SCAN:
            layers = self.layers
            if self.layers_remainder is not None or (
                isinstance(layers, StackModule) and layers.module == Gemma3Block
            ):
                block_size = len(GEMMA3_ATTENTION_PATTERN)
                num_blocks = layers.length if isinstance(layers, StackModule) else 0
                if num_blocks:
                    block_decode_states = [
                        decode_states[
                            block_idx * block_size : (block_idx + 1) * block_size
                        ]
                        for block_idx in range(num_blocks)
                    ]
                    x, block_outputs = layers(
                        x,
                        attention_mask=mask_mapping,
                        full_cos=full_cos,
                        full_sin=full_sin,
                        sliding_cos=sliding_cos,
                        sliding_sin=sliding_sin,
                        pos=pos,
                        decode_state=block_decode_states,
                    )
                    if return_decode_states:
                        extra_output_list = self._flatten_scanned_block_outputs(
                            block_outputs,
                            num_blocks,
                        )
                    else:
                        extra_output_list = [None] * (num_blocks * block_size)
                if self.layers_remainder is not None:
                    x, remainder_extra_output = self.layers_remainder(
                        x,
                        mask_mapping,
                        pos=pos,
                        full_cos=full_cos,
                        full_sin=full_sin,
                        sliding_cos=sliding_cos,
                        sliding_sin=sliding_sin,
                        decode_state=decode_states[num_blocks * block_size :],
                    )
                    if return_decode_states:
                        extra_output_list.extend(remainder_extra_output)
                    else:
                        extra_output_list.extend(
                            [None] * len(self.layers_remainder.layers)
                        )
            else:
                raise ValueError(
                    "Gemma-3 scan requires stack_block() or stack() before "
                    "forward_impl='scan'."
                )

        x = self.norm(x)
        return logical_reshard(
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
        input_ids = logical_reshard(
            input_ids, from_logical_rules(("batch", "sequence"))
        )
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
        attention_mask: Array | None = None,
        pos: int = 0,
        *,
        dtype: jnp.dtype = jnp.float32,
        return_hidden_states=False,
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
            decode_states=decode_states,
            forward_impl=forward_impl,
            segment_ids=segment_ids,
            dtype=dtype,
            rngs=rngs,
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

    def stack_block(self):
        return eqx.tree_at(
            lambda tree: tree.model,
            self,
            self.model.stack_block(),
        )

    def stack(self):
        return self.stack_block()

    def unembed(
        self,
        hidden_states: Float[Array, "B T D"],
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
        attention_mask: Array | None = None,
        pos: int = 0,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: PRNGKeyArray | None = None,
        forward_impl: ForwardImpl | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        **inputs,
    ):
        hidden_states, _ = self.model(
            input_ids,
            attention_mask,
            pos,
            forward_impl=forward_impl,
            segment_ids=segment_ids,
            dtype=dtype,
            rngs=rngs,
            **inputs,
        )
        output = self.score(hidden_states)
        return output

    def stack_block(self):
        return eqx.tree_at(
            lambda tree: tree.model,
            self,
            self.model.stack_block(),
        )

    def stack(self):
        return self.stack_block()
