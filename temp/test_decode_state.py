from jaxformers.module_utils import DEFAULT_ADDITIONAL_CONFIG
from jax.extend.mlir.dialects.stablehlo import add
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from jaxformers.models.huggingface.gemma3 import Gemma3ForCausalLM
from jaxformers.print_utils import tree_pprint
from jaxformers.sharding_utils import (
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)


mesh_size = {"dp_shard": 1, "dp_replicate": 1, "tp": 1, "cp": 1}
mesh_size = {"dp_shard": 1, "dp_replicate": 1, "tp": 1, "cp": 1}
devices = jax.devices()[:1]
model_id = "google/gemma-3-1b-it"


def make_decode_state(l, b, t, n, h):
    decode_states = []
    for i in range(l):
        k = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
        v = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
        decode_states.append({"k": k, "v": v})

    return decode_states


mesh = make_mesh(mesh_size, devices)
rules = make_logical_axis_rules(mesh_size)
tokenizer = AutoTokenizer.from_pretrained(model_id)

with jax.set_mesh(mesh), with_logical_axis(rules):
    model = Gemma3ForCausalLM.from_pretrained(model_id, rngs=jax.random.key(0), param_dtype = jnp.bfloat16)

    inputs = tokenizer("hallo saya makan nasi goreng", return_tensors="np").data
    none_array = np.asarray([None, None, None])
    print("DEBUGPRINT {none_array}:", none_array)
    print("DEBUGPRINT {none_array}:", none_array.shape)
    print(type(inputs))
    tree_pprint(inputs)
    model = model.stack()
    decode_states = make_decode_state(
        model.config.num_hidden_layers,
        1,
        7,
        model.config.num_key_value_heads,
        model.config.head_dim,
    )

    @jax.jit
    def fwd(model, inputs, decode_states):
        output, extra_outputs = model(**inputs, dtype = jnp.bfloat16)
        return output, extra_outputs

    outputs, extra_outputs = fwd(model, inputs, decode_states)
    tree_pprint(extra_outputs)
