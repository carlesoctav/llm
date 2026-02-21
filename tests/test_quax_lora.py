import jax
import jax.numpy as jnp
import jax.random as jr
import quax

from jaxformers.dispatch.lora import LoraArray


def test_quax_lora_no_tracer_leak_under_nested_transforms():
    key = jr.PRNGKey(0)
    w = LoraArray(jnp.ones((8, 4), dtype=jnp.bfloat16), rank=2, alpha=2.0, key=key)
    x = jnp.ones((3, 4), dtype=jnp.bfloat16)

    @quax.quaxify
    def forward(x, w):
        # `einsum` uses an internal `jit`, which is where Quax+LoRA used to trigger
        # tracer leaks via `LoraArray.aval()`.
        return jnp.einsum("bd,md->bm", x, w, preferred_element_type=x.dtype)

    @jax.jit
    def step(w, x):
        def loss_fn(w):
            return jnp.sum(forward(x, w))

        return jax.value_and_grad(loss_fn)(w)

    loss, grad = step(w, x)
    assert loss.shape == ()
    assert type(grad) is LoraArray
    assert grad._w.shape == w._w.shape
    assert grad.a.shape == w.a.shape
    assert grad.b.shape == w.b.shape


def test_quax_lora_works_through_jax_remat():
    key = jr.PRNGKey(0)
    w = LoraArray(jnp.ones((8, 4), dtype=jnp.bfloat16), rank=2, alpha=2.0, key=key)
    x = jnp.ones((3, 4), dtype=jnp.bfloat16)

    @quax.quaxify
    def loss(x, w):
        def body(x, w):
            return jnp.einsum("bd,md->bm", x, w, preferred_element_type=x.dtype)

        y = jax.remat(body)(x, w)
        return jnp.sum(y)

    val = loss(x, w)
    grad = jax.grad(lambda ww: loss(x, ww))(w)
    assert val.shape == ()
    assert type(grad) is LoraArray
