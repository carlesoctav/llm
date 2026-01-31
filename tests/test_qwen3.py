import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from jaxformers.models import qwen3


def test_qwen3_0_6_b_cpu():
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", low_cpu_mem_usage=False, attn_implementation="sdpa"
    )

    devices = jax.devices("cpu")
    jax_model = qwen3.load(
        model_id="Qwen/Qwen3-0.6B",
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        devices=devices,
        param_dtype=jnp.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_str = "hallo saya makan nasi goreng"
    hf_token = tokenizer(test_str, return_tensors="pt")
    jax_token = tokenizer(test_str, return_tensors="jax")

    with torch.no_grad():
        hf_logits = hf_model(**hf_token).logits.cpu().numpy().astype(np.float32)

    jax_logits = jax_model.forward(**jax_token, weights=jax_model.weights)

    np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-3)


@pytest.mark.tpu_ci
def test_qwen3_0_6b_tpu_tp():
    jax_model = qwen3.load(
        model_id="Qwen/Qwen3-0.6B",
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4},
        param_dtype=jnp.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_str = "hallo saya makan nasi goreng"
    jax_token  = jnp.ones((1, 12), dtype = jnp.int32)
    jax_logits = jax_model.forward(jax_token, weights=jax_model.weights)

    assert jax_logits.ndim == 3
    assert jax_logits.shape == (1, 12, jax_model.config["vocab_size"])


if __name__ == "__main__":
    test_qwen3_0_6_b_cpu()
    test_qwen3_0_6b_tpu_tp()
    print("all good")
