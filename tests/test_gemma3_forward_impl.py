from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import Gemma3TextConfig

from jaxformers.dispatch.lora import make_lora
from jaxformers.models import prepare_weights as prepare_model_weights
from jaxformers.models.huggingface import gemma3


def test_gemma3_forward_impls_match_on_small_config():
    config = Gemma3TextConfig(
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
    model = gemma3.init(
        config=config,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=jax.devices("cpu"),
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
            "forward_impl": "loop",
        },
        param_dtype=jnp.float32,
        rngs=jax.random.key(0),
        tokenizer=None,
    )

    input_ids = (jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % config.vocab_size)
    attention_mask = jnp.ones_like(input_ids)

    loop_hidden = model.forward(
        model.weights,
        input_ids=input_ids,
        attention_mask=attention_mask,
        dtype=jnp.float32,
    )

    for forward_impl in ("scan_layer", "scan_block"):
        impl_config = deepcopy(model.config)
        impl_config.additional_config = {
            **model.config.additional_config,
            "forward_impl": forward_impl,
        }
        impl_weights = gemma3.prepare_weights(
            impl_config,
            model.weights,
            forward_impl,
        )
        impl_hidden = gemma3.forward(
            impl_config,
            impl_weights,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(loop_hidden, impl_hidden, atol=1e-5, rtol=1e-5)


def test_scan_block_forward_handles_unstacked_lora_weights():
    config = Gemma3TextConfig(
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
    model = gemma3.init(
        config=config,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=jax.devices("cpu"),
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
            "forward_impl": "scan_block",
        },
        param_dtype=jnp.float32,
        rngs=jax.random.key(0),
        tokenizer=None,
    )

    lora_model = make_lora(
        model,
        "random",
        {
            "weights_path": ["self_attn.q_proj.weight"],
            "rank": 4,
            "alpha": 8,
        },
        rngs=jax.random.key(1),
    )
    lora_model = prepare_model_weights("huggingface.gemma3", lora_model)

    input_ids = (jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % config.vocab_size)
    attention_mask = jnp.ones_like(input_ids)
    hidden = lora_model.forward(
        lora_model.weights,
        input_ids=input_ids,
        attention_mask=attention_mask,
        dtype=jnp.float32,
    )

    assert lora_model.is_lora
    assert hidden.shape == (2, 8, config.hidden_size)


def test_init_accepts_model_id_without_explicit_config(monkeypatch):
    config = Gemma3TextConfig(
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
    called_with = []

    def fake_from_pretrained(model_source):
        called_with.append(model_source)
        return config

    monkeypatch.setattr(gemma3.AutoConfig, "from_pretrained", fake_from_pretrained)

    model = gemma3.init(
        model_id="google/gemma-3-1b-it",
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=jax.devices("cpu"),
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
        rngs=jax.random.key(0),
        tokenizer=None,
    )

    assert called_with == ["google/gemma-3-1b-it"]
    assert model.config is config
    assert model.weights["model.embed_tokens.weight"].shape == (
        config.vocab_size,
        config.hidden_size,
    )


def test_init_rejects_config_and_model_id_together():
    config = Gemma3TextConfig(
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

    with pytest.raises(ValueError, match="Exactly one of `config` or `model_id`"):
        gemma3.init(
            config=config,
            model_id="google/gemma-3-1b-it",
            parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
            rngs=jax.random.key(0),
        )
