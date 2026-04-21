from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from transformers import Qwen3Config

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
    ToVllmMappingAbstract,
    VllmMapping,
    VllmWeightLeaf,
    VllmWeightState,
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
    if hidden_act == "silu":
        return jax.nn.silu
    if hidden_act == "gelu":
        return lambda x: jax.nn.gelu(x, approximate=False)
    if hidden_act in ("gelu_new", "gelu_pytorch_tanh"):
        return lambda x: jax.nn.gelu(x, approximate=True)
    if hidden_act == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported hidden activation {hidden_act!r}")


def get_rope_theta(config: Qwen3Config) -> jax.Array:
    rope_parameters = config.rope_parameters
    if rope_parameters["rope_type"] != "default":
        raise ValueError(
            f"Unsupported Qwen3 rope_type {rope_parameters['rope_type']!r}"
        )
    return jnp.asarray(rope_parameters["rope_theta"], dtype=jnp.float32)


def make_mask(config, input_embeds, attention_mask=None, segment_ids=None, **kwargs):
    attn_impl = config.additional_config["attn_impl"]
    if attn_impl not in ATTENTION_MASK_INTERFACE:
        return {
            "full_attention": None,
            "sliding_attention": None,
        }

    full_mask = make_causal_mask(attn_impl, input_embeds, attention_mask, segment_ids)
    if config.sliding_window is None:
        sliding_mask = full_mask
    else:
        sliding_mask = make_sliding_window_causal_mask(
            attn_impl,
            input_embeds,
            config.sliding_window,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )

    return {
        "full_attention": reshard(
            full_mask, from_logical_rules(("batch", None, None, None))
        ),
        "sliding_attention": reshard(
            full_mask, from_logical_rules(("batch", None, None, None))
        ),
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
    scaling: float = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)

    def __init__(
        self,
        config: Qwen3Config,
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
        self.q_norm = Qwen3RMSNorm(
            head_dim,
            rngs=q_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3RMSNorm(
            head_dim,
            rngs=k_norm_rngs,
            param_dtype=param_dtype,
            eps=config.rms_norm_eps,
        )
        self.head_dim = head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = head_dim**-0.5
        self.attn_impl = attn_impl

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        attention_mask,
        rope_theta: jax.Array,
        pos: int = 0,
        decode_state: PyTree | None = None,
    ):
        do_decode = decode_state is not None
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
        q = q * jnp.asarray(self.scaling, dtype=q.dtype)

        if do_decode:
            k, v, new_decode_state = make_kv_from_cache(k, v, pos, decode_state)
            extra_output["decode_state"] = new_decode_state

        attn_output = attention_interface(
            q,
            k,
            v,
            mask=attention_mask,
            q_sharding=q_sharding,
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
        self.act_fn = config.hidden_act

    def __call__(self, x):
        gate = get_activation_fn(self.act_fn)(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(eqx.Module, Stackable):
    self_attn: Qwen3Attention
    mlp: Qwen3MLP

    input_layernorm: Qwen3RMSNorm
    post_attention_layernorm: Qwen3RMSNorm

    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(
        static=True, default=("rope_theta", "is_sliding", "decode_state")
    )
    in_axes: int = eqx.field(static=True, default=0)

    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: Qwen3Config,
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
            attn_impl=additional_config["attn_impl"],
            rngs=self_attn_rngs,
            param_dtype=param_dtype,
        )
        self.mlp = Qwen3MLP(
            config,
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
        self.remat = additional_config["remat_layer"]

    def __call__(
        self,
        x: Float[Array, "B T D"],
        *,
        rope_theta,
        attention_mask,
        is_sliding,
        pos: int,
        decode_state: PyTree | None = None,
    ):
        attention_mask = jax.lax.select(
            is_sliding,
            attention_mask["sliding_attention"],
            attention_mask["full_attention"],
        )

        residual = x
        x_norm = self.input_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        attn_output, extra_output = self.self_attn(
            x_norm,
            attention_mask=attention_mask,
            rope_theta=rope_theta,
            pos=pos,
            decode_state=decode_state,
        )
        x = residual + attn_output

        residual = x
        x_norm = self.post_attention_layernorm(x)
        x_norm = reshard(x_norm, from_logical_rules(("batch", "context", None)))
        ffw = self.mlp(x_norm)
        return residual + ffw, extra_output


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
            embed_scale=1.0,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        layer_rngs = jax.random.split(layers_rngs, config.num_hidden_layers)
        self.layers = [
            Qwen3DecoderLayer(
                config,
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
        pos: int = 0,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: PRNGKeyArray | None = None,
        decode_states: PyTree | None = None,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        x = self.embed_tokens(input_ids, dtype=dtype)
        mask_mapping = make_mask(self.config, x, **inputs)
        decode_states = (
            [None] * self.config.num_hidden_layers
            if decode_states is None
            else decode_states
        )
        extra_output_list = []
        forward_impl = forward_impl or self.config.additional_config["forward_impl"]

        if forward_impl not in tuple(ForwardImpl):
            raise ValueError(
                f"Unsupported Qwen3 forward implementation: {forward_impl!r}"
            )

        rope_theta = get_rope_theta(self.config)
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
            for layer, attention_type, decode_state in zip(
                layers, self.config.layer_types, decode_states
            ):
                x, extra_output = fwd(
                    layer,
                    x,
                    rope_theta=rope_theta,
                    attention_mask=mask_mapping,
                    is_sliding=attention_type == "sliding_attention",
                    pos=pos,
                    decode_state=decode_state,
                )
                extra_output_list.append(extra_output)
        elif forward_impl == ForwardImpl.SCAN:
            is_sliding = jnp.asarray(
                [
                    attention_type == "sliding_attention"
                    for attention_type in self.config.layer_types
                ],
                dtype=jnp.bool_,
            )
            layers = self.layers
            if isinstance(layers, list):
                layers = StackModule(
                    Qwen3DecoderLayer,
                    layers,
                    0,
                    argnames=("rope_theta", "is_sliding", "decode_state"),
                    remat=self.config.additional_config["remat_layer"],
                )
            rope_theta = jnp.full(
                (self.config.num_hidden_layers,), rope_theta, dtype=jnp.float32
            )
            x, extra_output_list = layers(
                x,
                rope_theta=rope_theta,
                attention_mask=mask_mapping,
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


class Qwen3ForCausalLM(AbstractHuggingFacePreTrainedModel, ToVllmMappingAbstract):
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
        store_config: bool = True,
    ):
        self.config = config
        model_rngs, lm_head_rngs = jax.random.split(rngs)
        self.model = Qwen3Model(
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
        decode_states: PyTree | None = None,
        return_hidden_states=False,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        hidden_states, extra_outputs = self.model(
            input_ids,
            pos,
            dtype,
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

    def to_vllm(self) -> VllmMapping:
        leaves: list[tuple[tuple[str, ...], VllmWeightLeaf]] = []
        mappings: dict[str, tuple[str, tuple[str, ...] | None]] = {}
        transpose_keys: dict[str, tuple[int, ...]] = {}

        def add(path: str, target_path: str, value: jax.Array):
            key = tuple(path.split("."))
            leaves.append((key, VllmWeightLeaf(value=value)))
            mappings[path] = (target_path, None)

        def add_transpose(path: str, axes: tuple[int, ...]):
            transpose_keys[path] = axes
            transpose_keys[path.split(".")[-1]] = axes

        add(
            "model.embed_tokens",
            "model.embed_tokens.weight",
            self.model.embed_tokens.weight,
        )

        layers = (
            self.model.layers.unstack()
            if isinstance(self.model.layers, StackModule)
            else self.model.layers
        )
        for layer_idx, layer in enumerate(layers):
            prefix = f"model.layers.{layer_idx}"
            add(
                f"{prefix}.input_layernorm",
                f"{prefix}.input_layernorm.weight",
                layer.input_layernorm.weight,
            )
            add(
                f"{prefix}.post_attention_layernorm",
                f"{prefix}.post_attention_layernorm.weight",
                layer.post_attention_layernorm.weight,
            )

            q_weight = rearrange(
                layer.self_attn.q_proj.weight,
                "(n h) d -> n h d",
                n=self.config.num_attention_heads,
                h=self.config.head_dim,
            )
            add(
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.q_proj.weight",
                q_weight,
            )
            add_transpose(f"{prefix}.self_attn.q_proj", (2, 0, 1))

            k_weight = rearrange(
                layer.self_attn.k_proj.weight,
                "(n h) d -> n h d",
                n=self.config.num_key_value_heads,
                h=self.config.head_dim,
            )
            add(
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.k_proj.weight",
                k_weight,
            )
            add_transpose(f"{prefix}.self_attn.k_proj", (2, 0, 1))

            v_weight = rearrange(
                layer.self_attn.v_proj.weight,
                "(n h) d -> n h d",
                n=self.config.num_key_value_heads,
                h=self.config.head_dim,
            )
            add(
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.v_proj.weight",
                v_weight,
            )
            add_transpose(f"{prefix}.self_attn.v_proj", (2, 0, 1))

            o_weight = rearrange(
                layer.self_attn.o_proj.weight,
                "d (n h) -> d n h",
                n=self.config.num_attention_heads,
                h=self.config.head_dim,
            )
            add(
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.self_attn.o_proj.weight",
                o_weight,
            )
            add_transpose(f"{prefix}.self_attn.o_proj", (1, 2, 0))

            if layer.self_attn.q_proj.bias is not None:
                add(
                    f"{prefix}.self_attn.q_proj_bias",
                    f"{prefix}.self_attn.q_proj.bias",
                    rearrange(
                        layer.self_attn.q_proj.bias,
                        "(n h) -> n h",
                        n=self.config.num_attention_heads,
                        h=self.config.head_dim,
                    ),
                )
            if layer.self_attn.k_proj.bias is not None:
                add(
                    f"{prefix}.self_attn.k_proj_bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    rearrange(
                        layer.self_attn.k_proj.bias,
                        "(n h) -> n h",
                        n=self.config.num_key_value_heads,
                        h=self.config.head_dim,
                    ),
                )
            if layer.self_attn.v_proj.bias is not None:
                add(
                    f"{prefix}.self_attn.v_proj_bias",
                    f"{prefix}.self_attn.v_proj.bias",
                    rearrange(
                        layer.self_attn.v_proj.bias,
                        "(n h) -> n h",
                        n=self.config.num_key_value_heads,
                        h=self.config.head_dim,
                    ),
                )
            if layer.self_attn.o_proj.bias is not None:
                add(
                    f"{prefix}.self_attn.o_proj_bias",
                    f"{prefix}.self_attn.o_proj.bias",
                    layer.self_attn.o_proj.bias,
                )

            add(
                f"{prefix}.self_attn.q_norm",
                f"{prefix}.self_attn.q_norm.weight",
                layer.self_attn.q_norm.weight,
            )
            add(
                f"{prefix}.self_attn.k_norm",
                f"{prefix}.self_attn.k_norm.weight",
                layer.self_attn.k_norm.weight,
            )

            add(
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.gate_proj.weight",
                layer.mlp.gate_proj.weight,
            )
            add_transpose(f"{prefix}.mlp.gate_proj", (1, 0))
            add(
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.up_proj.weight",
                layer.mlp.up_proj.weight,
            )
            add_transpose(f"{prefix}.mlp.up_proj", (1, 0))
            add(
                f"{prefix}.mlp.down_proj",
                f"{prefix}.mlp.down_proj.weight",
                layer.mlp.down_proj.weight,
            )
            add_transpose(f"{prefix}.mlp.down_proj", (1, 0))

        add(
            "model.norm",
            "model.norm.weight",
            self.model.norm.weight,
        )
        if self.lm_head is not None:
            add(
                "lm_head",
                "lm_head.weight",
                self.lm_head.weight,
            )
            add_transpose("lm_head", (1, 0))

        return VllmMapping(
            state=VllmWeightState(leaves=leaves),
            mappings=mappings,
            transpose_keys=transpose_keys,
        )
