from contextlib import contextmanager
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from transformers.models.llama.configuration_llama import LlamaConfig

from jaxformers.models.llama import LlamaForCausalLM
from jaxformers.sharding_utils import make_logical_axis_rules, make_mesh, with_logical_axis


PARALLEL_DIMS = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1}


@contextmanager
def model_context(devices):
    rule = make_logical_axis_rules(
        PARALLEL_DIMS,
        sequence_parallelism=False,
    )
    mesh = make_mesh(PARALLEL_DIMS, devices=devices)
    with jax.set_mesh(mesh), with_logical_axis(rule):
        yield


def make_config():
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
    )


def test_llama3_forward_impls_match_on_small_config():
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % 128
    attention_mask = jnp.ones_like(input_ids)

    with model_context(jax.devices("cpu")):
        loop_model = LlamaForCausalLM.init(
            config=deepcopy(make_config()),
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
                "forward_impl": "loop",
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )
        scan_model = LlamaForCausalLM.init(
            config=deepcopy(make_config()),
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
                "forward_impl": "scan",
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )

        loop_hidden, _ = loop_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=jnp.float32,
            return_hidden_states=True,
        )
        scan_hidden, _ = scan_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=jnp.float32,
            return_hidden_states=True,
        )

    np.testing.assert_allclose(loop_hidden, scan_hidden, atol=1e-5, rtol=1e-5)


def test_lm_head_property_resolves_tied_and_untied_heads():
    tied_config = make_config()
    untied_config = make_config()
    untied_config.tie_word_embeddings = False

    with model_context(jax.devices("cpu")):
        tied_model = LlamaForCausalLM.init(
            config=tied_config,
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )
        untied_model = LlamaForCausalLM.init(
            config=untied_config,
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(1),
        )

    assert tied_model.lm_head is None
    assert tied_model.lm_head_w is tied_model.model.embed_tokens.weight
    assert untied_model.lm_head_w is untied_model.lm_head.weight
