import jax.numpy as jnp
import quax

einsum = quax.quaxify(jnp.einsum)

