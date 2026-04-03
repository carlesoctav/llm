import jax
from jaxtyping import Array, Float, PyTree


BTNH = Float[Array, "B T N H"]


def make_new_decode_state(k,v):
    return {"k": k, "v": v}

def make_kv_from_cache(
    k: Float[Array, "B 1 N H"],
    v: Float[Array, "B 1 N H"],
    pos: int,
    decode_state: PyTree,
    init = True
) -> tuple[BTNH, BTNH, BTNH]:

    k_cache = decode_state["k"]
    v_cache = decode_state["v"]
    k = jax.lax.dynamic_update_index_in_dim(k_cache, k, pos, axis = 1)
    v = jax.lax.dynamic_update_index_in_dim(v_cache, v, pos, axis = 1)
    new_decode_state = make_new_decode_state(k, v)
    return k,v, new_decode_state





class LMInterface:
    def prefill_fn(self):
        pass

    def __init__(self, model, tokenizer):
        pass

    def generate(self, model, inputs):
        pass
