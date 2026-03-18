import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jaxformers.models import qwen3


def test_correctness_qwen3_0_6_b_cpu():
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        low_cpu_mem_usage=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    devices = jax.devices("cpu")
    jax_model = qwen3.load(
        model_id="Qwen/Qwen3-0.6B",
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1},
        local_dir="~/hallo",
        devices=devices,
        param_dtype=jnp.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_str = "hallo saya makan nasi goreng"
    hf_token = tokenizer(test_str, return_tensors="pt")
    jax_token = {
        k: jnp.asarray(v) for k, v in tokenizer(test_str, return_tensors="np").items()
    }

    with torch.no_grad():
        hf_logits = hf_model(**hf_token).logits.to(torch.float32).cpu().numpy()

    jax_logits = jax_model.forward(**jax_token, weights=jax_model.weights)

    np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-3)


@pytest.mark.tpu_ci
def test_pass_qwen3_0_6b_tpu_tp():
    try:
        tpu_devices = jax.devices("tpu")
    except RuntimeError:
        pytest.skip("TPU backend not available")

    if len(tpu_devices) < 4:
        pytest.skip("requires at least 4 TPU devices")
    jax_model = qwen3.load(
        model_id="Qwen/Qwen3-0.6B",
        parallel_dims={"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4},
        devices=tpu_devices[:4],
        param_dtype=jnp.float32,
    )
    jax_token = jnp.ones((1, 12), dtype=jnp.int32)
    jax_logits = jax_model.forward(jax_token, weights=jax_model.weights)

    assert jax_logits.ndim == 3
    assert jax_logits.shape == (1, 12, jax_model.config.vocab_size)


if __name__ == "__main__":
    test_correctness_qwen3_0_6_b_cpu()
    test_pass_qwen3_0_6b_tpu_tp()
    print("all good")
