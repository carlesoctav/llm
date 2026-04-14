from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class State(eqx.Module):
    array: jax.Array = eqx.field()
    is_sliding: jax.Array = eqx.field()


@partial(jax.jit, static_argnums=1)
def f(a, b):
    def _f(carry, x):
        a, s,n = x
        print(s)
        print(n)
        return carry, None

    state = eqx.combine(a, b)
    is_sliding = np.asarray(state.is_sliding)
    not_sliding = (1, ) * 10
    print("DEBUGPRINT {state}:", type(np.asarray(state.is_sliding)))
    print("DEBUGPRINT {not_sliding}:", not_sliding)
    print("DEBUGPRINT {not_sliding}:", type(not_sliding))
    not_sliding = np.asarray(not_sliding)
    print("DEBUGPRINT {not_sliding}:", not_sliding)
    print("DEBUGPRINT {not_sliding}:", type(not_sliding))

    jax.lax.scan(_f, None, (state.array, is_sliding, not_sliding))


def partition_spec(path, x):
    pass


ones = jnp.ones((5,))
zeros = jnp.zeros((5,))
is_sliding = jnp.concat((ones, zeros))
is_sliding = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
is_sliding = (False, True, False, True, False, False, True, False, False, True)
state_dy = State(array=jnp.ones((10, 100)), is_sliding=None)
state_stat = State(array=None, is_sliding=is_sliding)
f(state_dy, state_stat)
