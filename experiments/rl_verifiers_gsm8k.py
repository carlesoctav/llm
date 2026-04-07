import jax
import jax.numpy as jnp
import sws
import verifiers as vf


def get_config():
    config = sws.Config()

    config.project_name = "jaxformers"
    config.exp_name = "grpo-verifiers-gsm8k"
    config.seed = 42
    config.max_train_step = 100
    config.forward_dtype = lambda: jnp.bfloat16
    config.grad_accum = 1
    config.clip_epsilon = 0.2
    config.weights_impl = "stack"

    config.logger_name = "noop"
    config.logger.project = None
    config.logger.name = lambda: config.exp_name

    config.parallel.parallel_dims = {
        "dp_replicate": 1,
        "dp_shard": 1,
        "cp": 1,
        "tp": 1,
    }
    config.parallel.devices = lambda: jax.devices()
    config.parallel.multihost = False
    config.parallel.sequence_parallelism = True

    config.model_name = "huggingface.gemma3.Gemma3ForCausalLM.from_pretrained"
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = False
    config.model.additional_config.attn_impl = "sdpa"
    config.model.additional_config.forward_impl = "loop"
    config.model.additional_config.sequence_parallelism = True
    config.model.param_dtype = lambda: jnp.bfloat16

    config.data.source_name = "verifiers"
    config.data.source.envs = lambda: [
        vf.load_environment(
            env_id="gsm8k",
            num_train_examples=64,
        )
    ]
    config.data.source.env_names = ["gsm8k"]
    config.data.source.rollouts_per_example = 4
    config.data.source.sampling_args = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 64,
    }
    config.data.source.max_retries = 1

    config.data.loader.batch_size = 4
    config.data.loader.prefetch_buffer_size = 2

    config.inference.mode = "same_process"
    config.vllm.tensor_parallel_size = 1
    config.vllm.gpu_memory_utilization = 0.4
    config.vllm.enable_prefix_caching = False
    config.vllm.max_num_seqs = 1
    config.vllm.max_model_len = 512
    config.vllm.max_num_batched_tokens = 256

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.b1 = 0.9
    config.optimizer.b2 = 0.95
    config.optimizer.eps = 1e-8

    config.learning_rate = 1e-5
    config.lr_scheduler_name = "constant"

    return config
