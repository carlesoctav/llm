import jax.numpy as jnp


scalar = jnp.asarray(0)
print("DEBUGPRINT {scalar}:", scalar)
print("DEBUGPRINT {scalar}:", type(scalar))
a = scalar.shape[0] == 12312
print("DEBUGPRINT {a}:", a)
