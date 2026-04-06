import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from jaxformers.modeling_utils import TrainState


BTNH = Float[Array, "B T N H"]


def make_new_decode_state(k, v):
    return {"k": k, "v": v}


def make_decode_state(l, b, t, n, h):
    decode_states = []
    for i in range(l):
        k = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
        v = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
        decode_states.append({"k": k, "v": v})

    return decode_states


def make_kv_from_cache(
    k: Float[Array, "B 1 N H"],
    v: Float[Array, "B 1 N H"],
    pos: int,
    decode_state: PyTree,
    init=True,
) -> tuple[BTNH, BTNH, BTNH]:

    k_cache = decode_state["k"]
    v_cache = decode_state["v"]
    k = jax.lax.dynamic_update_index_in_dim(k_cache, k, pos, axis=1)
    v = jax.lax.dynamic_update_index_in_dim(v_cache, v, pos, axis=1)
    new_decode_state = make_new_decode_state(k, v)
    return k, v, new_decode_state


def generate(
    train_state: TrainState,
    prompt_tokens,
    num_layers: int,
    num_heads: int,
    head_size: int,
    max_num_generation_tokens: int = 512,
    *,
    forward_impl="loop",
):
    B, T = prompt_tokens.shape[1]
    L, N, H = num_layers, num_heads, head_size
    decode_states = make_decode_state(L, B, T, N, H)

    @jax.jit
    def prefill(model):
        pass
