from __future__ import annotations
from jax.sharding import Mesh

import asyncio
import os

from jaxformers.print_utils import debugtree


# os.environ["TPU_VISIBLE_CHIPS"] = "0"
# os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
# os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HOME"] = "/mnt/carles/.cache"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


import jax
import jax.numpy as jnp


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from jaxformers.inference.llm_client import (
    _flatten_vllm_mapping,
    SameProcessTPUInferenceClient,
)
from jaxformers.models.huggingface.gemma3 import Gemma3ForCausalLM
from jaxformers.models.huggingface.qwen3 import Qwen3ForCausalLM
from jaxformers.sharding_utils import (
    make_logical_axis_rules,
    make_mesh,
    with_logical_axis,
)


MESSAGES = [
    {
        "role": "system",
        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
    },
    {
        "role": "user",
        "content": (
            "Leah earned $28 working odd jobs around the neighborhood. She spent "
            "a seventh of it on a milkshake and put half of the rest in her savings "
            "account. She left the remaining money in her wallet. Her dog got ahold "
            "of her wallet and shredded all the money inside but $1. How many dollars "
            "did Leah lose?"
        ),
    },
]


def make_parallel():
    devices = jax.devices()
    trainer_devices = devices[:2]
    rollout_devices = devices[2:]

    return {
        "parallel_dims": {
            "dp_replicate": 1,
            "dp_shard": 2,
            "cp": 1,
            "tp": 1,
        },
        "devices": trainer_devices,
        "rollout_devices": rollout_devices,
        "sequence_parallelism": True,
    }


def make_model():
    parallel = make_parallel()
    mesh = make_mesh(**parallel)
    rule = make_logical_axis_rules(**parallel)
    with jax.set_mesh(mesh), with_logical_axis(rule):
        model = Qwen3ForCausalLM.from_pretrained(
            model_id="Qwen/Qwen3-0.6B",
            additional_config={
                "remat_layer": False,
                "attn_impl": "sdpa",
                "forward_impl": "loop",
                "sequence_parallelism": True,
            },
            rngs=jax.random.key(0),
            param_dtype=jnp.bfloat16,
        )
    return model, mesh


def make_client(mesh: Mesh):
    rollout_device = make_parallel()["rollout_devices"]
    rollout_id = [rollout_d.id for rollout_d in rollout_device]
    print("DEBUGPRINT {rollout_device}:", rollout_device)
    print("DEBUGPRINT {rollout_id}:", rollout_id)
    return SameProcessTPUInferenceClient(
        model="Qwen/Qwen3-0.6B",
        tokenizer="Qwen/Qwen3-0.6B",
        vllm_config={
            "load_format": "dummy",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": False,
            "max_num_seqs": 1,
            "max_model_len": 8192,
            "max_num_batched_tokens": 2048,
            "additional_config": {
                "sharding": {
                    "sharding_strategy": {
                        "device_indexes": rollout_id,
                    }
                }
            },
        },
    )


def generate_text(client) -> str:
    prompt_text = client.tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = client._make_sampling_params(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
        }
    )
    outputs = client.llm.generate(
        prompts=[prompt_text],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text


def block_state(state):
    jax.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf
        ),
        state,
    )


def assert_synced_state_matches_mapping(target_state, mapping):
    checked = 0
    for source_key, value in _flatten_vllm_mapping(mapping).items():
        target_key = mapping.mappings[source_key][0]
        if target_key not in target_state:
            prefixed_target_key = f"vllm_model.{target_key}"
            if prefixed_target_key not in target_state:
                raise KeyError(target_key)
            target_key = prefixed_target_key
        if source_key in mapping.transpose_keys:
            value = jnp.transpose(value, mapping.transpose_keys[source_key])
        target_value = target_state[target_key]
        if value.shape != target_value.shape:
            raise AssertionError(
                f"shape mismatch for {target_key}: {value.shape} != {target_value.shape}"
            )
        synced_value = jax.device_put(
            jnp.array(value, dtype=target_value.dtype, copy=True),
            target_value.sharding,
        )
        max_diff = jax.device_get(jnp.max(jnp.abs(synced_value - target_value))).item()
        if max_diff != 0.0:
            raise AssertionError(
                f"numeric mismatch for {target_key}: max_diff={max_diff}"
            )
        checked += 1
    return checked


def main():
    model, mesh = make_model()
    vllm_mapping = model.to_vllm()
    # debugtree("vllm_mapping", vllm_mapping.state)
    client = make_client(mesh)
    before = generate_text(client)
    print("before_sync:")
    print(before)
    model_runner = client.llm.llm_engine.model_executor.driver_worker.model_runner
    # debugtree("model_state", model_runner.state.flat_state())

    client.update_weights(model)
    block_state(model_runner.state)
    after = generate_text(client)
    print("after_sync:")
    print(after)
    checked = assert_synced_state_matches_mapping(model_runner.state, model.to_vllm())
    print(f"checked {checked} synced tensors")
    asyncio.run(client.close())


if __name__ == "__main__":
    main()
