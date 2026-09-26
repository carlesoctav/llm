import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModel

from jaxformers.models.search.splade import SpladeModel
from jaxformers.sharding_utils import (
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)


MODEL_ID = "Linkup-Platform/linkup-sparseup-embed-v1"
PARALLEL_DIMS = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1}


def test_splade_base_cpu_parity():
    queries = ["England football player highest paid", "what is the capital of France?"]
    docs = [
        "Harry Kane is the England captain and among the highest-paid footballers.",
        " ".join(["Paris is the capital of France."] * 60),
    ]

    hf_model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    hf_model.eval()
    with torch.no_grad():
        q_ids, q_attn, q_pool, _ = hf_model._tokenize(
            queries, hf_model.config.query_prefix, hf_model.config.query_max_length
        )
        d_ids, d_attn, d_pool, _ = hf_model._tokenize(
            docs, hf_model.config.document_prefix, hf_model.config.doc_max_length
        )
        hf_q = hf_model(q_ids, q_attn, q_pool).float().numpy()
        hf_d = hf_model(d_ids, d_attn, d_pool).float().numpy()

    devices = jax.devices("cpu")
    rule = make_logical_axis_rules(PARALLEL_DIMS, sequence_parallelism=False)
    mesh = make_mesh(PARALLEL_DIMS, devices=devices)
    with jax.set_mesh(mesh), with_logical_axis(rule):
        jax_model = SpladeModel.from_pretrained(
            model_id=MODEL_ID,
            additional_config={"attn_impl": "eager", "sequence_parallelism": False},
            param_dtype=jnp.float32,
            rngs=jax.random.key(0),
        )
    np.testing.assert_array_equal(
        np.asarray(jax_model.vocab_fold_index),
        hf_model.vocab_fold_index.numpy(),
    )

    with jax.set_mesh(mesh), with_logical_axis(rule):
        jax_q = np.asarray(
            jax_model(
                jnp.asarray(q_ids.numpy()),
                attention_mask=jnp.asarray(q_attn.numpy()),
                pooling_mask=jnp.asarray(q_pool.numpy()),
                dtype=jnp.float32,
                forward_impl="loop",
            )
        )
        jax_d = np.asarray(
            jax_model(
                jnp.asarray(d_ids.numpy()),
                attention_mask=jnp.asarray(d_attn.numpy()),
                pooling_mask=jnp.asarray(d_pool.numpy()),
                dtype=jnp.float32,
                forward_impl="loop",
            )
        )
        jax_q_scan = np.asarray(
            jax_model(
                jnp.asarray(q_ids.numpy()),
                attention_mask=jnp.asarray(q_attn.numpy()),
                pooling_mask=jnp.asarray(q_pool.numpy()),
                dtype=jnp.float32,
                forward_impl="scan",
            )
        )

    np.testing.assert_allclose(jax_q, hf_q, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(jax_d, hf_d, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(jax_q_scan, jax_q, atol=1e-3, rtol=1e-3)
