from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from transformers import ModernBertConfig

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
from jaxformers.nn.linear import default_init
from jaxformers.sharding_utils import from_logical_rules


def get_layer_metadata(
    config: ModernBertConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    layer_types = config.layer_types
    rope_theta = jnp.asarray(
        [
            config.rope_parameters[layer_type]["rope_theta"]
            for layer_type in layer_types
        ],
        dtype=jnp.float32,
    )
    is_sliding = jnp.asarray(
        [layer_type == "sliding_attention" for layer_type in layer_types],
        dtype=jnp.bool_,
    )
    use_attn_norm = jnp.asarray(
        [layer_idx != 0 for layer_idx in range(config.num_hidden_layers)],
        dtype=jnp.bool_,
    )
    return rope_theta, is_sliding, use_attn_norm


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


class ModernBertLayerNorm(eqx.Module):
    weight: Array
    bias: Array | None

    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    w_sharding: P | None = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        eps: float,
        use_bias: bool,
        w_sharding: P | None = None,
    ):
        self.dim = dim
        self.eps = eps
        self.use_bias = use_bias
        self.w_sharding = w_sharding
        weight_sharding = P() if self.w_sharding is None else self.w_sharding
        self.weight = jax.device_put(jnp.ones((dim,), param_dtype), weight_sharding)
        if use_bias:
            bias_sharding = P() if self.w_sharding is None else P(self.w_sharding[0])
            self.bias = jax.device_put(jnp.zeros((dim,), param_dtype), bias_sharding)
        else:
            self.bias = None

    def __call__(self, x):
        x_fp32 = jnp.asarray(x, jnp.float32)
        mean = jnp.mean(x_fp32, axis=-1, keepdims=True)
        var = jnp.mean(jax.lax.square(x_fp32 - mean), axis=-1, keepdims=True)
        y = jnp.asarray((x_fp32 - mean) * jax.lax.rsqrt(var + self.eps), x.dtype)
        y = y * self.weight.astype(x.dtype)
        if self.bias is not None:
            y = y + self.bias.astype(x.dtype)
        return y


class ModernBertEmbeddings(eqx.Module):
    tok_embeddings: Embedding
    norm: ModernBertLayerNorm

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        tok_embeddings_rngs, norm_rngs = jax.random.split(rngs)
        self.tok_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            rngs=tok_embeddings_rngs,
            param_dtype=param_dtype,
            embed_scale=1.0,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.norm = ModernBertLayerNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.norm_eps,
            use_bias=config.norm_bias,
        )

    def __call__(self, input_ids: Int[Array, "B T"], dtype: jnp.dtype = jnp.float32):
        return self.norm(self.tok_embeddings(input_ids, dtype=dtype))


class ModernBertAttention(eqx.Module):
    Wqkv: Linear
    Wo: Linear

    head_dim: int = eqx.field(static=True)
    num_attention_heads: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)
    window_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        attn_impl: str,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        hidden_size = config.hidden_size
        head_dim = hidden_size // config.num_attention_heads
        qkv_out = 3 * config.num_attention_heads * head_dim
        Wqkv_rngs, Wo_rngs = jax.random.split(rngs)
        self.Wqkv = Linear(
            hidden_size,
            qkv_out,
            rngs=Wqkv_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "context", None)),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.Wo = Linear(
            hidden_size,
            hidden_size,
            rngs=Wo_rngs,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.head_dim = head_dim
        self.num_attention_heads = config.num_attention_heads
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
    ):
        q_sharding = jax.sharding.NamedSharding(
            jax.sharding.get_abstract_mesh(),
            from_logical_rules(("batch", "context", "model", None)),
        )
        attention_interface = ATTENTION_INTERFACE[self.attn_impl]
        attention_args_kwargs = prepare_attention_kwargs(
            self.attn_impl,
            x,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            is_causal=False,
            is_sliding=is_sliding,
            window_size=self.window_size,
        )
        qkv = self.Wqkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = rearrange(
            q,
            "b t (n h) -> b t n h",
            n=self.num_attention_heads,
            h=self.head_dim,
        )
        k = rearrange(
            k,
            "b t (n h) -> b t n h",
            n=self.num_attention_heads,
            h=self.head_dim,
        )
        v = rearrange(
            v,
            "b t (n h) -> b t n h",
            n=self.num_attention_heads,
            h=self.head_dim,
        )

        bsz, seqlen, _nheads, head_dim = q.shape
        positions = jnp.broadcast_to(jnp.arange(seqlen)[None, :], [bsz, seqlen])
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
            scale=self.head_dim**-0.5,
            q_sharding=q_sharding,
            **attention_args_kwargs,
        )
        attn_output = rearrange(attn_output, "b t n h -> b t (n h)")
        return self.Wo(attn_output)


class ModernBertMLP(eqx.Module):
    Wi: Linear
    Wo: Linear
    act_fn: str = eqx.field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        act_fn: str,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        Wi_rngs, Wo_rngs = jax.random.split(rngs, 2)
        self.Wi = Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            rngs=Wi_rngs,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            out_sharding=from_logical_rules(("batch", "context", "model")),
            w_sharding=from_logical_rules(("model", "fsdp")),
        )
        self.Wo = Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=Wo_rngs,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.act_fn = act_fn

    def __call__(self, x):
        x, gate = jnp.split(self.Wi(x), 2, axis=-1)
        return self.Wo(get_activation_fn(self.act_fn)(x) * gate)


class ModernBertLayer(eqx.Module, Stackable):
    attn_norm: ModernBertLayerNorm
    attn: ModernBertAttention
    mlp_norm: ModernBertLayerNorm
    mlp: ModernBertMLP

    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(
        static=True, default=("rope_theta", "is_sliding", "use_attn_norm")
    )
    in_axes: int = eqx.field(static=True, default=0)

    remat: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        additional_config: AdditionalConfig,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        attn_norm_rngs, attn_rngs, mlp_norm_rngs, mlp_rngs = jax.random.split(rngs, 4)
        self.attn_norm = ModernBertLayerNorm(
            config.hidden_size,
            rngs=attn_norm_rngs,
            param_dtype=param_dtype,
            eps=config.norm_eps,
            use_bias=config.norm_bias,
        )
        self.attn = ModernBertAttention(
            config,
            attn_impl=additional_config["attn_impl"],
            rngs=attn_rngs,
            param_dtype=param_dtype,
        )
        self.mlp_norm = ModernBertLayerNorm(
            config.hidden_size,
            rngs=mlp_norm_rngs,
            param_dtype=param_dtype,
            eps=config.norm_eps,
            use_bias=config.norm_bias,
        )
        self.mlp = ModernBertMLP(
            config,
            act_fn=config.hidden_activation,
            rngs=mlp_rngs,
            param_dtype=param_dtype,
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
        use_attn_norm,
    ):
        attn_input = jnp.where(use_attn_norm, self.attn_norm(x), x)
        attn_input = reshard(attn_input, from_logical_rules(("batch", "context", None)))
        x = x + self.attn(
            attn_input,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            is_sliding=is_sliding,
            rope_theta=rope_theta,
        )
        mlp_input = reshard(
            self.mlp_norm(x), from_logical_rules(("batch", "context", None))
        )
        x = x + self.mlp(mlp_input)
        return x, None


class ModernBertModel(AbstractHuggingFacePreTrainedModel):
    config: ModernBertConfig = eqx.field(static=True)
    embeddings: ModernBertEmbeddings
    layers: list[ModernBertLayer] | StackModule[ModernBertLayer]
    final_norm: ModernBertLayerNorm

    def __init__(
        self,
        config: ModernBertConfig,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
        store_config: bool = True,
    ):
        self.config = config
        embeddings_rngs, layers_rngs, norm_rngs = jax.random.split(rngs, 3)
        self.embeddings = ModernBertEmbeddings(
            config,
            rngs=embeddings_rngs,
            param_dtype=param_dtype,
        )
        layer_rngs = jax.random.split(layers_rngs, config.num_hidden_layers)
        self.layers = [
            ModernBertLayer(
                config,
                additional_config=additional_config,
                rngs=layer_rngs[layer_idx],
                param_dtype=param_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.final_norm = ModernBertLayerNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.norm_eps,
            use_bias=config.norm_bias,
        )

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
        dtype: jnp.dtype = jnp.float32,
        *,
        attention_mask: Bool[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        rngs: PRNGKeyArray | None = None,
        forward_impl: ForwardImpl | None = None,
        **inputs,
    ):
        x = self.embeddings(input_ids, dtype=dtype)
        forward_impl = forward_impl or self.config.additional_config["forward_impl"]

        if forward_impl not in tuple(ForwardImpl):
            raise ValueError(
                f"Unsupported ModernBert forward implementation: {forward_impl!r}"
            )

        if forward_impl == ForwardImpl.LOOP:
            layers = (
                self.layers.unstack()
                if isinstance(self.layers, StackModule)
                else self.layers
            )
            fwd = (
                jax.remat(ModernBertLayer.__call__)
                if layers[0].remat
                else ModernBertLayer.__call__
            )
            for layer_idx, (layer, attention_type) in enumerate(
                zip(layers, self.config.layer_types)
            ):
                x, _ = fwd(
                    layer,
                    x,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    rope_theta=self.config.rope_parameters[attention_type][
                        "rope_theta"
                    ],
                    is_sliding=attention_type == "sliding_attention",
                    use_attn_norm=layer_idx != 0,
                )
        elif forward_impl == ForwardImpl.SCAN:
            rope_theta, is_sliding, use_attn_norm = get_layer_metadata(self.config)
            layers = self.layers
            if isinstance(layers, list):
                layers = StackModule(
                    ModernBertLayer,
                    layers,
                    0,
                    argnames=("rope_theta", "is_sliding", "use_attn_norm"),
                    remat=self.config.additional_config["remat_layer"],
                )
            x, _ = layers(
                x,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                rope_theta=rope_theta,
                is_sliding=is_sliding,
                use_attn_norm=use_attn_norm,
            )

        x = self.final_norm(x)
        return reshard(x, from_logical_rules(("batch", "context", None)))


class ModernBertPredictionHead(eqx.Module):
    dense: Linear
    norm: ModernBertLayerNorm
    act_fn: str = eqx.field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        dense_rngs, norm_rngs = jax.random.split(rngs)
        self.dense = Linear(
            config.hidden_size,
            config.hidden_size,
            rngs=dense_rngs,
            param_dtype=param_dtype,
            use_bias=config.classifier_bias,
            out_sharding=from_logical_rules(("batch", "sequence", None)),
            w_sharding=from_logical_rules(("fsdp", "model")),
        )
        self.norm = ModernBertLayerNorm(
            config.hidden_size,
            rngs=norm_rngs,
            param_dtype=param_dtype,
            eps=config.norm_eps,
            use_bias=config.norm_bias,
        )
        self.act_fn = config.classifier_activation

    def __call__(self, x):
        return self.norm(get_activation_fn(self.act_fn)(self.dense(x)))


class ModernBertDecoder(eqx.Module):
    weight: Array | None
    bias: Array | None

    vocab_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    out_sharding: P | None = eqx.field(static=True)
    w_sharding: P | None = eqx.field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype,
    ):
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.out_sharding = from_logical_rules(("batch", "context", "model"))
        self.w_sharding = from_logical_rules(("model", "fsdp"))
        if config.tie_word_embeddings:
            self.weight = None
        else:
            self.weight = jax.device_put(
                default_init(
                    rngs, (config.vocab_size, config.hidden_size), param_dtype
                ),
                P() if self.w_sharding is None else self.w_sharding,
            )
        if config.decoder_bias:
            bias_sharding = P() if self.w_sharding is None else P(self.w_sharding[0])
            self.bias = jax.device_put(
                jnp.zeros((config.vocab_size,), param_dtype), bias_sharding
            )
        else:
            self.bias = None

    def __call__(self, x, tied_weight: Array | None = None):
        out_weights = tied_weight if self.weight is None else self.weight
        y = einsum(
            "btd,vd->btv",
            x,
            out_weights,
            out_sharding=self.out_sharding,
            preferred_element_type=jnp.float32,
        )
        if self.bias is not None:
            y = y + self.bias.astype(y.dtype)[None, None, :]
        return y


class ModernBertForMaskedLM(AbstractHuggingFacePreTrainedModel):
    config: ModernBertConfig = eqx.field(static=True)
    model: ModernBertModel
    head: ModernBertPredictionHead
    decoder: ModernBertDecoder

    def __init__(
        self,
        config: ModernBertConfig,
        additional_config: AdditionalConfig,
        *,
        rngs: PRNGKeyArray,
        param_dtype: jnp.dtype = jnp.bfloat16,
        store_config: bool = True,
    ):
        self.config = config
        model_rngs, head_rngs, decoder_rngs = jax.random.split(rngs, 3)
        self.model = ModernBertModel(
            config,
            additional_config,
            rngs=model_rngs,
            param_dtype=param_dtype,
            store_config=store_config,
        )
        self.head = ModernBertPredictionHead(
            config,
            rngs=head_rngs,
            param_dtype=param_dtype,
        )
        self.decoder = ModernBertDecoder(
            config,
            rngs=decoder_rngs,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        input_ids: Int[Array, "B T"],
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
            dtype,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            rngs=rngs,
            forward_impl=forward_impl,
            **inputs,
        )
        return self.decoder(
            self.head(hidden_states),
            tied_weight=self.model.embeddings.tok_embeddings.weight,
        )
