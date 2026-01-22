import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from jaxformers.models import qwen3


def test_qwen3_0_6_b_cpu():
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", low_cpu_mem_usage=False, attn_implementation="sdpa"
    )

    devices = jax.devices("cpu")
    devices = [jax.devices()[0]]
    jax_model = qwen3.load(model_id="Qwen/Qwen3-0.6B", tp_devices=1, devices=devices)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_str = "hallo saya makan nasi goreng"
    hf_token = tokenizer(test_str, return_tensors="pt")
    jax_token = tokenizer(test_str, return_tensors="jax")

    with torch.no_grad():
        hf_logits = hf_model(**hf_token).logits.cpu().numpy().astype(np.float32)


    jax_logits = jax_model.forward(**jax_token, weights=jax_model.weights)

    np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-1)


def test_qwen3_0_6_b_tpu():
    torchax.enable_globally()
    with torchax.disable_temporarily():
        hf_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", low_cpu_mem_usage=False, attn_implementation="sdpa"
        )

    mesh = jax.make_mesh((1, 1), ("data", "model"), devices=jax.devices("cpu"))
    with jax.set_mesh(mesh):
        hf_model = hf_model.to("jax")
        dtype = next(hf_model.parameters()).jax().dtype
        sharding = next(hf_model.parameters()).jax().sharding

    jax_model = qwen3.load(
        model_id="Qwen/Qwen3-0.6B", tp_devices=1, devices=[jax.devices()[0]]
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_str = "hallo saya makan nasi goreng"
    hf_token = tokenizer(test_str, return_tensors="pt")
    hf_token = {k: v.to("jax") for k, v in hf_token.items()}

    jax_token = tokenizer(test_str, return_tensors="jax")

    hf_logits = hf_model(**hf_token).logits


    jax_logits = jax_model.forward(**jax_token, weights=jax_model.weights)

    assert jnp.allclose(jax_logits, hf_logits.jax(), atol=1e-1)


def test_qwen3_0_6_bmix():
    hf_model_cpu = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", low_cpu_mem_usage=False, attn_implementation="sdpa"
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    test_str = "hallo saya makan nasi goreng"
    hf_token = tokenizer(test_str, return_tensors="pt")

    hf_logits_cpu = hf_model_cpu(**hf_token).logits.detach().numpy()

    torchax.enable_globally()

    hf_model_tpu = hf_model_cpu.to("jax")
    hf_token_tpu = {k: v.to("jax") for k, v in hf_token.items()}
    hf_logits_tpu = hf_model_tpu(**hf_token_tpu).logits

    np.testing.assert_allclose(hf_logits_cpu, hf_logits_tpu.jax(), atol=1e-1)


if __name__ == "__main__":
    test_qwen3_0_6_b_cpu()
    test_qwen3_0_6_bmix()
