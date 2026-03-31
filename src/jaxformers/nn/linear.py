import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from jaxformers.dispatch.einsum import einsum


default_init = jax.nn.initializers.variance_scaling(
    1 / 3.0, "fan_in", "uniform", in_axis=-1, out_axis=-2, batch_axis=()
)


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
            bias_sharding = P() if self.w_sharding is None else P(self.w_sharding[0])
            self.bias = jax.device_put(
                jnp.zeros((out_features,), param_dtype), bias_sharding
            )
        else:
            self.bias = None
        self.weight = jax.device_put(
            default_init(rngs, (out_features, in_features), param_dtype),
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
