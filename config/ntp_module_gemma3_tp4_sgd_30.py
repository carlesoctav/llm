import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()

    config.resume = False
    config.random_init = False
    config.skip_eval = True

    config.use_lora = False
    config.random_init_lora = False

    config.exp_name = "ntp_module_gemma3_tp4_sgd_30"
    config.project = "noop"
    config.dir = "/tmp/jaxformers_runs"
    config.ckpt_path = lambda: f"{config.dir}/{config.exp_name}"
    config.train_seed = 42
    config.eval_every = None
    config.max_train_step = 30
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_implementation = "reference"

    config.model_name = "gemma3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = False
    config.model.additional_config.attn_implementation = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.devices = lambda: jax.devices()[:4]
    config.model.param_dtype = lambda: jnp.bfloat16

    config.learning_rate = 1e-5
    config.lr_scheduler_name = None

    config.optimizer_name = "sgd"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.grad_accum = 4

    config.data_name = "huggingface"
    config.data.load_kwargs = [
        {
            "path": "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages",
            "split": "train",
            "streaming": False,
        }
    ]
    config.data.transforms.column = "messages"
    config.data.transforms.max_length = 2048
    config.data.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.transforms.assistant_loss = True
    config.data.transforms.is_chat = True
    config.data.transforms.is_tokenized = False
    config.data.transforms.chat_template_path = "./temp/think.jinja"
    config.data.transforms.packing = True
    config.data.transforms.packing_bins = 64

    config.train_loader.global_batch_size = 8
    config.train_loader.seed = 42
    config.train_loader.shuffle = False
    config.train_loader.drop_remainder = True

    config.log.grad_norm = False
    config.log.learning_rate = False
    config.logger_name = "noop"

    config.checkpoint_options.save_interval_steps = 30
    config.checkpoint_options.max_to_keep = 1

    return config
