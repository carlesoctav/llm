import jax
import jax.numpy as jnp
import sws


def get_config():
    config = sws.Config()

    config.resume = False
    config.random_init = False
    config.skip_eval = True

    config.use_lora = False
    config.random_init_lora = False

    config.exp_name = ""
    config.project = ""
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.project}/{config.exp_name}"
    config.train_seed = 42
    config.max_train_step = 1000
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_implementation = "xla_chunked"

    config.model_name = "huggingface_gemma3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_implementation = "xla_chunked"
    config.model.additional_config.sequence_parallelism = True
    config.model.devices = lambda: jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    config.learning_rate = 3e-6
    config.lr_scheduler_name = None

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.grad_accum = 1

    config.data.load_kwargs = [
        {
            "path": "openai/gsm8k",
            "name": "main",
            "split": "train",
        }
    ]
    config.data.prompt_column = "question"
    config.data.answer_column = "answer"
    config.data.messages_column = None
    config.data.prompt_template = (
        "{prompt}\n\nRespond using <reasoning>...</reasoning><answer>...</answer>."
    )
    config.data.system_prompt = "Solve the math problem carefully."
    config.data.drop_last = True

    config.train_loader.global_batch_size = 2

    config.rollout.model_id = lambda: config.model.model_id
    config.rollout.tokenizer = lambda: config.model.model_id
    config.rollout.tensor_parallel_size = 4
    config.rollout.max_num_seqs = 8
    config.rollout.max_num_batched_tokens = 2048
    config.rollout.max_model_len = 1024
    config.rollout.max_tokens = 256
    config.rollout.gpu_memory_utilization = 0.9
    config.rollout.dtype = "bfloat16"
    config.rollout.temperature = 0.9
    config.rollout.top_p = 1.0
    config.rollout.top_k = 50
    config.rollout.num_generations = 2
    config.rollout.sync_every_n_steps = 1
    config.rollout.model_impl = "vllm"
    config.rollout.vllm_model_impl = "vllm"

    config.grpo.num_iterations = 1
    config.grpo.beta = 0.0
    config.grpo.epsilon = 0.2

    config.log.grad_norm = True
    config.log.learning_rate = True
    config.logger_name = "wandb"
    config.logger.project = lambda: config.project
    config.logger.name = lambda: config.exp_name
    config.logger.resume = "never"

    config.checkpoint_options.save_interval_steps = 100
    config.checkpoint_options.max_to_keep = 1

    return config
