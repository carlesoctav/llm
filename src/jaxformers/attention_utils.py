from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import tokamax
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxformers.modeling_utils import logical_to_physical, make_mesh
from jaxformers.ops.attention import chunked_manual_dot_product_attention
from jaxformers.utils import GeneralInterface


class AttentionImpl(Protocol):
    def __call__(
        query: Float[Array, "B T N H"],
        key: Float[Array, "B S K H"],
        value: Float[Array, "B S K H"],
        bias: Array | None = None,
        mask: Bool[Array, " B #N T S"] | None = None,
        q_sharding: jax.NamedSharding | None = None,
        **kwargs,
    ):
        ...

def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, " B #N T S"] | None = None,
    *,
    dropout_rate: float = 0.0,
    dropout_rng: PRNGKeyArray | None = None,
    broadcast_dropout: bool = True,
    q_sharding: jax.NamedSharding | None = None,
    **kwargs,
) -> Float[Array, "B T N H"]:
    query = query / jnp.sqrt(query.shape[-1])

    B, T, N, H = query.shape
    Bk, S, K, Hk = key.shape
    Bv, Sv, Kv, Hv = value.shape

    if Hk != H or Hv != H:
        raise ValueError("Query, key, and value must share the same head dimension")

    if Kv != K:
        raise ValueError("Value tensor must share the key head axis for attention")

    if K != N:
        if K <= 0 or N % K != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )
        repeat_factor = N // K
        key = jnp.repeat(key, repeat_factor, axis=-2, out_sharding = q_sharding)
        value = jnp.repeat(value, repeat_factor, axis=-2, out_sharding = q_sharding)
        K = N

    scores = jnp.einsum(
        "btnh, bsnh -> bnts",
        query,
        key,
        preferred_element_type=query.dtype,
        precision=jax.lax.Precision.HIGHEST,
    )

    if bias is not None:
        scores = scores + bias

    if mask is not None:
        neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = jnp.where(mask, scores, neg_inf)

    dtype = jnp.result_type(scores.dtype, jnp.float32)

    weights = jax.nn.softmax(scores.astype(dtype), axis=-1).astype(scores.dtype)

    if dropout_rate > 0.0:
        if dropout_rng is None:
            raise TypeError("dropout_rate > 0 but no dropout_rng provided")
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
            keep = jnp.broadcast_to(keep, weights.shape)
        multiplier = keep.astype(weights.dtype) / keep_prob
        weights = weights * multiplier

    attn = jnp.einsum("bnts, bsnh -> btnh", weights, value, preferred_element_type=query.dtype)
    return attn

class AttentionInterface(GeneralInterface[str, AttentionImpl]):
    _global_mapping = {
        "eager": eager_dot_product_attention,
        "sdpa": partial(tokamax.dot_product_attention, precision = jax.lax.Precision.HIGHEST),
        # Historically, "xla_chunked" referred to Tokamax's chunked XLA attention.
        # In this codebase we instead map it to the manual chunked implementation,
        # because Tokamax's xla_chunked backward can have very large temp memory.
        "xla_chunked": chunked_manual_dot_product_attention,
        "chunked_manual": chunked_manual_dot_product_attention,
    }

ATTENTION_INTERFACE = AttentionInterface()


def _get_cache_sharding(config):
    rules = getattr(config, "sharding_rules", None)
    if rules is None:
        return None

    mesh = getattr(config, "mesh", None)
    if mesh is None:
        parallel_dims = getattr(config, "parallel_dims", None)
        if parallel_dims is None:
            return None
        mesh = make_mesh(parallel_dims)

    return jax.NamedSharding(
        mesh,
        logical_to_physical(("batch", "context", "model", "none"), rules),
    )


def init_kv_cache(
    config,
    *,
    batch_size: int,
    cache_len: int,
    dtype: jnp.dtype,
):
    num_layers = int(getattr(config, "num_hidden_layers"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(getattr(config, "hidden_size") // num_attention_heads)
    head_dim = int(head_dim)
    cache_shape = (batch_size, cache_len, num_kv_heads, head_dim)
    sharding = _get_cache_sharding(config)

    def make_layer():
        key_cache = jnp.zeros(cache_shape, dtype=dtype)
        value_cache = jnp.zeros(cache_shape, dtype=dtype)
        if sharding is not None:
            try:
                key_cache = jax.device_put(key_cache, sharding)
                value_cache = jax.device_put(value_cache, sharding)
            except ValueError:
                pass
        return (key_cache, value_cache)

    return tuple(make_layer() for _ in range(num_layers))


def update_kv_cache(key, value, cache, pos):
    cache_key, cache_value = cache
    try:
        key = jax.device_put(key, cache_key.sharding)
        value = jax.device_put(value, cache_value.sharding)
    except Exception:
        pass
    cache_key = jax.lax.dynamic_update_slice_in_dim(cache_key, key, pos, axis=1)
    cache_value = jax.lax.dynamic_update_slice_in_dim(cache_value, value, pos, axis=1)
    return cache_key, cache_value, (cache_key, cache_value)
