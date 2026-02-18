import jax.numpy as jnp
import numpy as np

from jaxformers.attention_utils import _normalize_mask


def test_normalize_mask_accepts_b1ts_and_broadcasts_heads():
    B, N, T, S = 2, 4, 3, 5
    mask = (jnp.arange(B * T * S).reshape(B, 1, T, S) % 2) == 0
    norm = _normalize_mask(mask, B, N, T, S)
    assert norm.shape == (B, N, T, S)
    a = np.asarray(norm)
    m = np.asarray(mask)
    assert np.array_equal(a[:, 0], m[:, 0])
    assert np.array_equal(a[:, 1], m[:, 0])


def test_normalize_mask_accepts_bnts():
    B, N, T, S = 2, 4, 3, 5
    mask = (jnp.arange(B * N * T * S).reshape(B, N, T, S) % 3) == 0
    norm = _normalize_mask(mask, B, N, T, S)
    assert norm.shape == (B, N, T, S)
    assert np.array_equal(np.asarray(norm), np.asarray(mask))
