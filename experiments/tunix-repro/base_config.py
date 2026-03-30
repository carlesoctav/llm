import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()

    config.project_name = ""
    config.exp_name = ""
    config.dir = "gs://carles-git-good"
    config.seed = 42
    config.eval_every = None
    config.max_train_step = 10_000
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_impl = "reference"
    config.grad_accum = 4

    config.logger_name = "wandb"
    config.checkpoint = None

    config.logger.project = lambda: config.project_name
    config.logger.name = lambda: config.exp_name

    config.callback_name = ["log_grad_norm", "log_learning_rate", "log_performance"]
    config.callback.log_performance.real_step_threshold = 0
    config.callback.log_performance.denom_keys = ["token"]


    config.init_model = "pretrained"
    # config.load_model.path =
    # config.load_model.target =
    # config.load_model.step =
    config.init_lora = None
    config.model_name = "experimental.gemma3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_impl = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.weights_impl = "stack"
    config.model.additional_config.forward_impl = "scan_layer"

    config.model.devices = lambda: jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    def ds_config(ds_name, split, streaming):
        return {
            "load_kwargs": [
                {
                    "path": ds_name,
                    "split": split,
                }
            ],
            "streaming": streaming,
        }

    def transforms_config():
        return {
            "column": "messages",
            "max_length": 2048,
            "tokenizer": lambda: AutoTokenizer.from_pretrained(config.model.model_id),
            "data_type": "chat",
            "assistant_loss": True,
            "chat_template_path": "./temp/think.jinja",
            "packing": True,
        }

    def loader_config(
        batch_size=32,
        shuffle=False,
        num_workers=0,
        num_threads=1,
        prefetch_buffer_size=500,
        per_worker_buffer_size=1,
        seed=0,
    ):
        return {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "num_threads": num_threads,
            "prefetch_buffer_size": prefetch_buffer_size,
            "per_worker_buffer_size": per_worker_buffer_size,
            "seed": seed,
        }

    ds_name = "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages"

    config.eval.minival.type = "simple"
    config.eval.minival.fn_name = "loss"
    config.eval.minival.transforms_name = "ntp"
    config.eval.minival.data = ds_config(ds_name, "train[:1%]", False)
    config.eval.minival.transforms = transforms_config()
    config.eval.minival.loader = loader_config(num_workers=0)

    config.data.train.transforms_name = "ntp"
    config.data.train.source = ds_config(ds_name, "train", False)
    config.data.train.transforms = transforms_config()

    config.data.loader.shard = False
    config.data.train.loader = loader_config(
        num_workers=0,
        shuffle=True,
        seed=lambda: config.seed,
    )

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.b1 = 0.9
    config.optimizer.b2 = 0.95
    config.optimizer.eps = 1e-8

    config.learning_rate = 1e-5
    config.lr_scheduler_name = "wsds"
    config.lr_scheduler.min_lr_ratio = 0.1
    config.lr_scheduler.warmup = 0.01
    config.lr_scheduler.decay = 0.5
    config.lr_scheduler.rewarmup = 0.0
    config.lr_scheduler.cycle_length = None
    config.lr_scheduler.cycles = 4
    config.lr_scheduler.decay_schedule = "cosine"

    return config
