import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers.models.gemma3 import Gemma3ForCausalLM as HFGemma3ForCausalLM

from jaxformers.distributed import make_logical_axis_rules, make_mesh, with_logical_axis
from jaxformers.models.huggingface.gemma3 import Gemma3ForCausalLM as JaxGemma3ForCausalLM


PARALLEL_DIMS = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1}


def make_context(devices):
    rule = make_logical_axis_rules(
        PARALLEL_DIMS,
        sequence_parallelism=False,
    )
    mesh = make_mesh(PARALLEL_DIMS, devices=devices)
    return mesh, rule


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


def get_jax_logits(model, mesh, rule, input_ids, attention_mask=None):
    with jax.set_mesh(mesh), with_logical_axis(rule):
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=jnp.float32,
        )
        return jax.device_get(model.unembed(hidden))


def load_jax_model(model_id: str, devices):
    mesh, rule = make_context(devices)
    with jax.set_mesh(mesh), with_logical_axis(rule):
        model = JaxGemma3ForCausalLM.from_pretrained(
            model_id=model_id,
            additional_config={
                "attn_impl": "eager",
                "sequence_parallelism": False,
            },
            param_dtype=jnp.float32,
        )
    return model, mesh, rule


def test_gemma3_1b_it_cpu():
    model_id = "google/gemma-3-1b-it"
    hf_model = HFGemma3ForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    cpu_devices = jax.devices("cpu")
    jax_model, mesh, rule = load_jax_model(model_id, cpu_devices)

    for_batching = []
    for num_token in [512, 1024, 8192]:
        input_ids = np.random.randint(0, hf_model.vocab_size, (1, num_token))
        with torch.no_grad():
            hf_logits = (
                hf_model(input_ids=torch.from_numpy(input_ids))
                .logits.to(torch.float32)
                .cpu()
                .numpy()
            )
        jax_logits = get_jax_logits(jax_model, mesh, rule, input_ids)
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

    jax_logits = get_jax_logits(jax_model, mesh, rule, input_ids, attention_mask)
    np.testing.assert_allclose(jax_logits, hf_logits, atol=1e-2, rtol=1e-2)


@pytest.mark.tpu_ci
def test_gemma3_1b_it_tpu_tp():
    model_id = "google/gemma-3-1b-it"
    tpu_devices = jax.devices("tpu")
    jax_model, mesh, rule = load_jax_model(model_id, tpu_devices)
    input_ids = jnp.ones((1, 13), dtype=jnp.int32)

    jax_logits = get_jax_logits(jax_model, mesh, rule, input_ids)

    assert jax_logits.ndim == 3
    assert jax_logits.shape == (1, 13, jax_model.model.config.vocab_size)


if __name__ == "__main__":
    test_gemma3_1b_it_cpu()
