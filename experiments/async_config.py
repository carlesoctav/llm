import jax
import jax.numpy as jnp
import sws


def get_config():
    config = sws.Config()

    config.project_name = "test-async"
    config.exp_name = "async-grpo-gsm8k"
    config.dir = "/mnt/carles/llm/ckptr"
    config.seed = 42
    config.max_train_step = 1000
    config.forward_dtype = lambda: jnp.bfloat16
    config.grad_accum = 8
    config.clip_epsilon = 0.2
    config.weights_impl = "free"

    config.logger_name = "wandb"
    config.logger.project = "grpo-test-again"
    config.logger.name = lambda: config.exp_name

    config.checkpoint.path = lambda: (
        f"{config.dir}/{config.project_name}/{config.exp_name}"
    )
    config.checkpoint.save_interval_steps = 100
    config.checkpoint.max_to_keep = None
    config.checkpoint.save_only_trainable = False

    config.parallel.parallel_dims = {
        "dp_replicate": 1,
        "dp_shard": 4,
        "cp": 1,
        "tp": 1,
    }
    config.parallel.devices = lambda: jax.devices()
    config.parallel.multihost = False
    config.parallel.sequence_parallelism = True

    config.model_name = "huggingface.qwen3.Qwen3ForCausalLM.from_pretrained"
    config.model.model_id = "Qwen/Qwen3-0.6B"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_impl = "eager"
    config.model.additional_config.forward_impl = "loop"
    config.model.additional_config.sequence_parallelism = True
    config.model.param_dtype = lambda: jnp.bfloat16

    config.data.source_name = "async_verifiers"
    config.data.source.envs = {
        "gsm8k": {
            "env_id": "gsm8k",
        }
    }
    config.data.source.rollouts_per_example = 4
    config.data.source.sampling_args = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 1024,
    }
    config.data.source.max_retries = 1

    config.data.loader.batch_size = 64
    config.data.loader.prefetch_buffer_size = 2

    config.inference.mode = "async_same_process"
    config.vllm.tensor_parallel_size = 4
    config.vllm.gpu_memory_utilization = 0.2
    config.vllm.enable_prefix_caching = True
    config.vllm.max_num_seqs = 128
    config.vllm.max_model_len = 4096

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.b1 = 0.9
    config.optimizer.b2 = 0.95
    config.optimizer.eps = 1e-8

    config.learning_rate = 1e-5
    config.lr_scheduler_name = "wsds"
    config.lr_scheduler.min_lr_ratio = 0.1
    config.lr_scheduler.warmup = 0.01
    config.lr_scheduler.decay = None
    config.lr_scheduler.rewarmup = 0.0
    config.lr_scheduler.cycle_length = 0.2
    config.lr_scheduler.cycles = None
    config.lr_scheduler.decay_schedule = "cosine"

    config.callback.log_grad_norm = {}
    config.callback.log_learning_rate = {}

    return config
