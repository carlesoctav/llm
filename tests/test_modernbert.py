import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from jaxformers.models.huggingface.modernbert import ModernBertForMaskedLM
from jaxformers.sharding_utils import (
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)


MODEL_ID = "answerdotai/ModernBERT-base"
PARALLEL_DIMS = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1}


def test_modernbert_base_cpu_parity():
    texts = [
        "Hello, how are you?",
        "ModernBERT is a modernized bidirectional encoder-only Transformer model.",
        " ".join(["The quick brown fox jumps over the lazy dog."] * 40),
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_inputs = tokenizer(texts, padding=True, return_tensors="pt")
    np_inputs = tokenizer(texts, padding=True, return_tensors="np")

    hf_model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, attn_implementation="eager", dtype=torch.float32
    )
    hf_model.eval()
    with torch.no_grad():
        hf_hidden = (
            hf_model.model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
            )
            .last_hidden_state.float()
            .numpy()
        )
        hf_logits = (
            hf_model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs["attention_mask"],
            )
            .logits.float()
            .numpy()
        )

    devices = jax.devices("cpu")
    rule = make_logical_axis_rules(PARALLEL_DIMS, sequence_parallelism=False)
    mesh = make_mesh(PARALLEL_DIMS, devices=devices)
    with jax.set_mesh(mesh), with_logical_axis(rule):
        jax_model = ModernBertForMaskedLM.from_pretrained(
            model_id=MODEL_ID,
            additional_config={"attn_impl": "eager", "sequence_parallelism": False},
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )

    input_ids = jnp.asarray(np_inputs["input_ids"])
    attention_mask = jnp.asarray(np_inputs["attention_mask"])
    with jax.set_mesh(mesh), with_logical_axis(rule):
        jax_hidden_loop = np.asarray(
            jax_model.model(
                input_ids,
                attention_mask=attention_mask,
                dtype=jnp.float32,
                forward_impl="loop",
            )
        )
        jax_hidden_scan = np.asarray(
            jax_model.model(
                input_ids,
                attention_mask=attention_mask,
                dtype=jnp.float32,
                forward_impl="scan",
            )
        )
        jax_logits = np.asarray(
            jax_model(
                input_ids,
                attention_mask=attention_mask,
                dtype=jnp.float32,
                forward_impl="loop",
            )
        )

    real_tokens = np_inputs["attention_mask"].astype(bool)
    np.testing.assert_allclose(
        jax_hidden_loop[real_tokens], hf_hidden[real_tokens], atol=1e-2, rtol=1e-2
    )
    np.testing.assert_allclose(
        jax_logits[real_tokens], hf_logits[real_tokens], atol=5e-2, rtol=1e-2
    )
    np.testing.assert_allclose(
        jax_hidden_scan[real_tokens], jax_hidden_loop[real_tokens], atol=2e-2, rtol=1e-2
    )
