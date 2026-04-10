from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import os
from jax.sharding import AxisType

def print_compiled_memory_stats(compiled_stats):
    if compiled_stats is None:
        return

    def bytes_to_gb(num_bytes):
        return num_bytes / (1024**3)

    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    peak_gb = bytes_to_gb(compiled_stats.peak_memory_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb

    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB, Peak size: {peak_gb:.1f} GB.",
        f"Alias size: {alias_gb:.1f} GB",
    )

    return {
        "total_gb": round(total_gb, 1),
        "output_gb": round(output_gb, 1),
        "temp_gb": round(temp_gb, 1),
        "argument_gb": round(argument_gb, 1),
        "host_temp_gb": round(host_temp_gb, 1),
        "alias_gb": round(alias_gb, 1),
        "peak_gb": round(peak_gb, 1),
    }


Mesh, NamedSharding = jax.sharding.Mesh, jax.sharding.NamedSharding
P, with_sharding_constraint = jax.sharding.PartitionSpec, jax.lax.with_sharding_constraint

LOOP = False 
print(f"DEBUGPRINT[2]: a.py:41: LOOP={type(LOOP)}")
print(f"DEBUGPRINT[1]: a.py:10: LOOP={LOOP}")

batch, t = 64, 1024
num_layers, num_heads, head_size, embed_size  = 12, 48, 128, 9000

mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('data',), axis_types = AxisType.Explicit)
jax.set_mesh(mesh)
print(mesh)

if LOOP:
    print("using_loop")
    qkv_sharding = [NamedSharding(mesh, P(None, None, 'data')) for _ in range(num_layers)]
    o_sharding = [NamedSharding(mesh, P(None, None, 'data')) for _ in range(num_layers)]
else:
    print("not using loop")
    qkv_sharding = NamedSharding(mesh, P(None, None, None, 'data'))
    o_sharding = NamedSharding(mesh, P(None, None, None, 'data'))
x_sharding = NamedSharding(mesh, P('data', None, None))


@partial(jax.jit, out_shardings=(qkv_sharding, o_sharding, x_sharding))
def init():
    if LOOP:
        qkv = [jnp.ones((num_heads, head_size, embed_size), dtype = jnp.bfloat16) for _ in range(num_layers)]
        o = [jnp.ones((num_heads, head_size, embed_size), dtype=jnp.bfloat16) for _ in range(num_layers)]
    else:
        qkv = jnp.ones((num_layers, num_heads, head_size, embed_size), dtype = jnp.bfloat16)
        o = jnp.ones((num_layers, num_heads, head_size, embed_size), dtype=jnp.bfloat16)
    x = jnp.ones((batch, t, embed_size), dtype=jnp.bfloat16)
    return qkv, o, x


if LOOP:
    def forward(params, x):
        qkvs, os = params
        for qkv, o in zip(qkvs, os):
            y = jnp.einsum('bte,hde->bthd', x, qkv) 
            x = jnp.einsum('bthd,hde->bte', y, o, out_sharding = P('data', None, None))
        return jnp.mean(x**2)
else:
    def forward(params, x):
        def layer(_x, params):
            _qkv, _o = params
            y = jnp.einsum('bte,hde->bthd', _x, _qkv) 
            z = jnp.einsum('bthd,hde->bte', y, _o, out_sharding = P('data', None, None))
            return z, None
        out, _ = jax.lax.scan(layer, x, params)
        return jnp.mean(out**2)

key = jax.random.PRNGKey(0)
qkv, o, x = init()
params = (qkv, o)

backward = jax.jit(jax.grad(forward), out_shardings = (qkv_sharding, o_sharding)).lower(params, x).compile()
print_compiled_memory_stats(backward.memory_analysis())
out = backward(params, x)
