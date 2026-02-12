import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from jaxformers.models import bert


MODEL_ID = "google-bert/bert-base-uncased"

def test_bert_base_cpu_parity():
    hf_model = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    devices = jax.devices("cpu")
    jax_model = bert.load(
        model_id=MODEL_ID,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=devices,
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
    )

    text = "jax bert parity check"
    hf_token = tokenizer(text, return_tensors="pt")
    jax_token = tokenizer(text, return_tensors="jax")

    with torch.no_grad():
        hf_output = hf_model(**hf_token)
        hf_hidden = hf_output.last_hidden_state.cpu().numpy().astype(np.float32)
        hf_pool = hf_output.pooler_output.cpu().numpy().astype(np.float32)

    jax_hidden, jax_pool = jax_model.forward(
        **jax_token,
        weights=jax_model.weights,
        return_pooled=True,
    )

    np.testing.assert_allclose(np.asarray(jax_hidden), hf_hidden, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(jax_pool), hf_pool, atol=1e-4, rtol=1e-4)


@pytest.mark.tpu_ci
def test_pass_bert_200M_tpu():
    try:
        tpu_devices = jax.devices("tpu")
    except RuntimeError:
        pytest.skip("TPU backend not available")

    if len(tpu_devices) < 4:
        pytest.skip("requires at least 4 TPU devices")

    jax_model = bert.load(
        model_id=MODEL_ID,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=[tpu_devices[0]],
        additional_config={
            "attn_implementation": "eager",
        },
        param_dtype=jnp.float32,
    )

    seq_len = 13
    input_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    token_type_ids = jnp.zeros((1, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((1, seq_len), dtype=jnp.int32)

    hidden, pooled = jax_model.forward(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        weights=jax_model.weights,
        return_pooled=True,
    )

    assert hidden.shape == (1, seq_len, jax_model.config["hidden_size"])
    assert pooled.shape == (1, jax_model.config["hidden_size"])
    assert bool(jnp.all(jnp.isfinite(hidden)))
    assert bool(jnp.all(jnp.isfinite(pooled)))
