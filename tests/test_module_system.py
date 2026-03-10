import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, PartitionSpec as P
from transformers import Gemma3TextConfig

from jaxformers.models import gemma3
from jaxformers.module_utils import flatten_param_tree


def _tiny_gemma3_config():
    return Gemma3TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=True,
        attention_bias=False,
        query_pre_attn_scalar=4,
    )


def test_gemma3_module_system_keeps_hf_keys_and_runs_forward():
    sharding_config = gemma3.make_gemma3_sharding_config(
        {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        sequence_parallelism=False,
    )
    lm = gemma3.Gemma3LM(
        config=_tiny_gemma3_config(),
        sharding_config=sharding_config,
        param_dtype=jnp.float32,
    )
    assert lm.embed_tokens is not None
    assert lm.model is not None
    assert lm.lm_head is not None

    model = gemma3.init(
        _tiny_gemma3_config(),
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=jax.devices()[:1],
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
        rngs=jax.random.key(0),
        tokenizer=None,
    )

    assert "model" in model.weights
    assert "layers" in model.weights["model"]

    flat_weights = flatten_param_tree(model.weights)
    assert "model.embed_tokens.weight" in flat_weights
    assert "model.layers.0.self_attn.q_proj.weight" in flat_weights
    assert "model.layers.0.mlp.down_proj.weight" in flat_weights
    assert "model.norm.weight" in flat_weights

    input_ids = jnp.arange(8, dtype=jnp.int32)[None, :]
    logits = model.forward(model.weights, input_ids)
    assert logits.shape == (1, 8, model.config.vocab_size)


def test_explicit_axes_ambiguous_einsum_requires_out_sharding_under_jit():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires at least 4 devices")

    mesh = jax.make_mesh(
        (2, 2),
        ("x", "y"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
        devices=devices[:4],
    )
    jax.set_mesh(mesh)

    weight = jnp.arange(2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8)
    act = jnp.arange(4 * 5 * 4 * 8, dtype=jnp.float32).reshape(4, 5, 4, 8)
    weight = jax.device_put(weight, P(None, "y", None))
    act = jax.device_put(act, P("x", None, "y", None))

    @jax.jit
    def post_reshard(lhs, rhs):
        out = jnp.einsum("ihd,bthd->bti", lhs, rhs)
        return jax.sharding.reshard(out, P("x", None, None))

    @jax.jit
    def with_out(lhs, rhs):
        return jnp.einsum(
            "ihd,bthd->bti",
            lhs,
            rhs,
            out_sharding=P("x", None, None),
        )

    with pytest.raises(Exception, match="output sharding"):
        post_reshard(weight, act)

    output = with_out(weight, act)
    assert output.shape == (4, 5, 2)
