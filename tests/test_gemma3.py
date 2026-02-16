import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jaxformers.models import gemma3


def test_gemma3_1b_it_cpu():
    model_id = "google/gemma-3-1b-it"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    devices = jax.devices("cpu")
    jax_model = gemma3.load(
        model_id=model_id,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=devices,
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = "hallo saya makan nasi goreng"
    hf_token = tokenizer(text, return_tensors="pt")
    jax_token = {k: jnp.asarray(v) for k, v in tokenizer(text, return_tensors="np").items()}

    with torch.no_grad():
        hf_logits = hf_model(**hf_token).logits.to(torch.float32).cpu().numpy()

    jax_logits = jax_model.forward(**jax_token, weights=jax_model.weights)
    np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-2, rtol=1e-2)


@pytest.mark.tpu_ci
def test_gemma3_1b_it_tpu_tp():
    model_id = "google/gemma-3-1b-it"
    tpu_devices = jax.devices("tpu")
    jax_model = gemma3.load(
        model_id=model_id,
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=tpu_devices,
        additional_config={
            "attn_implementation": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
    )
    jax_token = jnp.ones((1, 13), dtype=jnp.int32)
    jax_logits = jax_model.forward(jax_token, weights=jax_model.weights)

    assert jax_logits.ndim == 3
    assert jax_logits.shape == (1, 13, jax_model.config.vocab_size)
