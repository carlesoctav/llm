import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from jaxformers.inference.ops.sample import SamplingMetadata, sample


def test_sample_greedy():
    logits = jnp.asarray([[0.0, 1.0, -1.0], [2.0, 0.5, 1.5]], dtype=jnp.float32)
    metadata = SamplingMetadata(
        temperature=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        top_k=jnp.asarray([0, 0], dtype=jnp.int32),
        top_p=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        do_sample=False,
    )
    tokens, _ = sample(jax.random.PRNGKey(0), logits, metadata)
    np.testing.assert_array_equal(np.asarray(tokens), np.asarray([1, 0], dtype=np.int32))


def test_sample_topk_one_is_deterministic():
    logits = jnp.asarray([[0.0, 1.0, -1.0]], dtype=jnp.float32)
    metadata = SamplingMetadata(
        temperature=jnp.asarray([1.0], dtype=jnp.float32),
        top_k=jnp.asarray([1], dtype=jnp.int32),
        top_p=jnp.asarray([1.0], dtype=jnp.float32),
        do_sample=True,
    )
    tokens, _ = sample(jax.random.PRNGKey(0), logits, metadata)
    np.testing.assert_array_equal(np.asarray(tokens), np.asarray([1], dtype=np.int32))
