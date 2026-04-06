import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


DEFAULT_LORA_PATHS = [
    "*.q_proj.weight",
    "*.k_proj.weight",
    "*.v_proj.weight",
    "*.o_proj.weight",
    "*.gate_proj.weight",
    "*.up_proj.weight",
    "*.down_proj.weight",
]


def get_config():
    config = sws.Config()

    config.exp_name = "gemma3-1b-it-lora-kl"
    config.project_name = "split-brain"
    config.dir = "gs://carles-git-good"
    config.checkpoint.path = lambda: (
        f"{config.dir}/{config.project_name}/{config.exp_name}"
    )
    config.seed = 42
    config.eval_every = None
    config.max_train_step = 1000
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_impl = "reference"
    config.grad_accum = 4

    config.loss_ratio.sft_loss = 1.0
    config.loss_ratio.kl_loss = 1.0

    config.logger_name = "noop"
    config.logger.project = lambda: config.project_name
    config.logger.name = lambda: config.exp_name

    config.checkpoint.save_interval_steps = 100
    config.checkpoint.max_to_keep = 1
    config.checkpoint.save_only_trainable = False

    config.init_lora = "random"
    config.weights_impl = "stack"
    config.model_name = "huggingface.gemma3.Gemma3ForCausalLM.from_pretrained"
    config.parallel.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.parallel.devices = lambda: jax.devices()
    config.parallel.multihost = False
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_impl = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.param_dtype = lambda: jnp.bfloat16

    config.lora.rank = 256
    config.lora.alpha = 512
    config.lora.weights_path = list(DEFAULT_LORA_PATHS)

    ds_name = "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages"

    def source_config(split):
        return {
            "load_kwargs": [
                {
                    "path": ds_name,
                    "split": split,
                }
            ],
            "streaming": False,
        }

    def transforms_config():
        return {
            "column": "messages",
            "max_length": 2048,
            "tokenizer": lambda: AutoTokenizer.from_pretrained(config.model.model_id),
            "data_type": "chat",
            "assistant_loss": True,
            "chat_template_path": "./temp/think.jinja",
            "packing": False,
            "packing_bins": 64,
        }

    config.data.loader.combine = "zip"
    config.data.loader.shard = False
    config.data.loader.seed = 42
    config.data.loader.num_workers = 8
    config.data.loader.num_threads = 1
    config.data.loader.prefetch_buffer_size = 500
    config.data.loader.per_worker_buffer_size = 1

    config.data.sft.source = source_config("train")
    config.data.kl.source = source_config("train")

    config.data.sft.transforms_name = "ntp"
    config.data.sft.transforms = transforms_config()

    config.data.kl.transforms_name = "kl"
    config.data.kl.transforms = transforms_config()

    config.data.sft.loader = {"batch_size": 32, "shuffle": False}
    config.data.kl.loader = {"batch_size": 32, "shuffle": False}

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

    return config
