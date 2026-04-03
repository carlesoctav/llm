import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from .linear import default_init


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
        # [input_ids -> b, t ((dp_shard, dp), (tp, ))
        # embed = [vocab_size, hidden] (tp, None)
        # return [batch, t, hidden] with outsharding (None, tp, None)
        x = self.weight.at[input_ids, :].get(out_sharding=self.out_sharding)
        x = x.astype(dtype)
        return x * jnp.asarray(self.embed_scale, dtype=dtype)
