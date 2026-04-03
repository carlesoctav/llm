from contextlib import contextmanager
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import Gemma3TextConfig

import jaxformers.modeling_utils as modeling_utils
from jaxformers.distributed import make_logical_axis_rules, make_mesh, with_logical_axis
from jaxformers.models.huggingface.gemma3 import Gemma3ForCausalLM


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
    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=4,
        sliding_window_pattern=3,
    )


def test_gemma3_forward_impls_match_on_small_config():
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % 128
    attention_mask = jnp.ones_like(input_ids)

    with model_context(jax.devices("cpu")):
        loop_model = Gemma3ForCausalLM.init(
            config=deepcopy(make_config()),
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
                "forward_impl": "loop",
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )
        scan_model = Gemma3ForCausalLM.init(
            config=deepcopy(make_config()),
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
                "forward_impl": "scan_layer",
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )

        loop_hidden = loop_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=jnp.float32,
        )
        scan_hidden = scan_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=jnp.float32,
        )

    np.testing.assert_allclose(loop_hidden, scan_hidden, atol=1e-5, rtol=1e-5)


def test_lm_head_property_resolves_tied_and_untied_heads():
    tied_config = make_config()
    untied_config = make_config()
    untied_config.tie_word_embeddings = False

    with model_context(jax.devices("cpu")):
        tied_model = Gemma3ForCausalLM.init(
            config=tied_config,
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )
        untied_model = Gemma3ForCausalLM.init(
            config=untied_config,
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(1),
        )

    assert tied_model.lm_head is tied_model.model.embed_tokens
    assert untied_model.lm_head is untied_model._lm_head


def test_init_accepts_model_id_without_explicit_config(monkeypatch):
    config = make_config()
    called_with = []

    def fake_from_pretrained(model_source):
        called_with.append(model_source)
        return config

    monkeypatch.setattr(modeling_utils.AutoConfig, "from_pretrained", fake_from_pretrained)

    with model_context(jax.devices("cpu")):
        model = Gemma3ForCausalLM.init(
            model_id="google/gemma-3-1b-it",
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )

    assert called_with == ["google/gemma-3-1b-it"]
    assert model.model.config is config
    assert model.model.embed_tokens.weight.shape == (
        config.vocab_size,
        config.hidden_size,
    )


def test_init_rejects_config_and_model_id_together():
    with model_context(jax.devices("cpu")):
        with pytest.raises(ValueError, match="Exactly one of `config` or `model_id`"):
            Gemma3ForCausalLM.init(
                config=make_config(),
                model_id="google/gemma-3-1b-it",
                rngs=jax.random.key(0),
            )
