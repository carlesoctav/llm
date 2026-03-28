import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers.models.gemma3 import Gemma3ForCausalLM

from jaxformers.models import gemma3


def stack_with_padding(arrays: list[np.ndarray], pad_value=0):
    max_size = max(array.shape[-1] for array in arrays)
    collect_before_pad = []
    collect_for_mask = []
    for array in arrays:
        arr_len = array.shape[-1]
        pad_amount = max_size - arr_len
        narray = np.pad(array, ((0, 0), (0, pad_amount)), constant_values=pad_value)
        mask = np.pad(
            np.ones((1, arr_len)), ((0, 0), (0, pad_amount)), constant_values=pad_value
        )
        collect_before_pad.append(narray)
        collect_for_mask.append(mask)

    return np.vstack(collect_before_pad), np.vstack(collect_for_mask)


def test_gemma3_1b_it_cpu():
    model_id = "google/gemma-3-1b-it"
    hf_model = Gemma3ForCausalLM.from_pretrained(
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
            "attn_impl": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
    )

    for_batching = []
    for i, num_token in enumerate([512, 1024, 8192]):
        input_ids = np.random.randint(0, hf_model.vocab_size, (1, num_token))
        with torch.no_grad():
            hf_logits = (
                hf_model(input_ids=torch.from_numpy(input_ids))
                .logits.to(torch.float32)
                .cpu()
                .numpy()
            )
        jax_logits = jax_model.forward(jax_model.weights, input_ids=input_ids)
        np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-2, rtol=1e-2)
        for_batching.append(input_ids)

    input_ids, attention_mask = stack_with_padding(for_batching)
    with torch.no_grad():
        hf_logits = (
            hf_model(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            )
            .logits.to(torch.float32)
            .cpu()
            .numpy()
        )

    jax_logits = jax_model.forward(
        jax_model.weights, input_ids=input_ids, attention_mask=attention_mask
    )
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
            "attn_impl": "eager",
            "sequence_parallelism": False,
        },
        param_dtype=jnp.float32,
    )
    jax_token = jnp.ones((1, 13), dtype=jnp.int32)
    jax_logits = jax_model.forward(jax_token, weights=jax_model.weights)

    assert jax_logits.ndim == 3
    assert jax_logits.shape == (1, 13, jax_model.config.vocab_size)


if __name__ == "__main__":
    test_gemma3_1b_it_cpu()
