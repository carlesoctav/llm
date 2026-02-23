import jax
import jax.numpy as jnp
import sws
from transformers import AutoConfig


def get_config() -> sws.Config:
    config = sws.Config()

    config.resume = False
    config.random_init = False
    config.skip_eval = True

    config.use_lora = False

    config.exp_name = "qwen4b_ntp_dummy"
    config.dir = "/tmp/jaxformers"
    config.ckpt_path = lambda: f"{config.dir}/{config.exp_name}"
    config.train_seed = 42
    config.eval_every = None
    config.max_train_step = 1
    config.forward_dtype = lambda: jnp.bfloat16

    config.loss_implementation = "tpu_pallas"

    config.model_name = "qwen3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
    config.model.model_id = "Qwen/Qwen3-4B-Base"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.remat_loss = True
    config.model.additional_config.attn_implementation = "xla_chunked"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.loss_parallel = False

    config.model.devices = jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    config.lr_scheduler_name = None
    config.learning_rate = 1e-5

    config.optimizer_name = "sgd"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.grad_accum = 1

    config.data_name = "dummy"
    config.data.load_kwargs = lambda: [
        {
            "seq_len": 2048,
            "vocab_size": int(getattr(AutoConfig.from_pretrained(config.model.model_id), "vocab_size")),
            "num_examples": 1024,
            "seed": 0,
        }
    ]

    config.train_loader.global_batch_size = 1
    config.train_loader.seed = 0
    config.train_loader.shuffle = False
    config.train_loader.worker_count = 0
    config.train_loader.drop_remainder = True

    config.log.grad_norm = False
    config.log.learning_rate = False

    config.checkpoint_options.save_interval_steps = 1000000
    config.checkpoint_options.max_to_keep = 1

    config.wandb.project = "jaxformers-oom-probe"
    config.wandb.name = lambda: config.exp_name
    config.wandb.mode = "disabled"

    return config
