from pathlib import Path

import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
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

    config.skip_eval = True

    config.exp_name = "gemma3-1b-it-lora-kl"
    config.project_name = "split-brain"
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.project_name}/{config.exp_name}"
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

    config.callback_name = []

    config.checkpoint.save_interval_steps = 100
    config.checkpoint.max_to_keep = 1
    config.checkpoint.save_only_trainable = False

    config.init_model = "pretrained"
    config.init_lora = "random"
    config.model_name = "huggingface.gemma3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_impl = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.weights_impl = "stack"
    config.model.additional_config.forward_impl = "scan_layer"
    config.model.devices = lambda: jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    config.lora.rank = 256
    config.lora.alpha = 512
    config.lora.weights_path = list(DEFAULT_LORA_PATHS)

    config.data.loader.combine = "zip"
    config.data.loader.shard = False
    config.data.loader.seed = 42

    config.data.sft.source.load_kwargs = [
        {
            "path": "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages",
            "split": "train",
        }
    ]
    config.data.sft.source.streaming = False
    config.data.kl.source.load_kwargs = [
        {
            "path": "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages",
            "split": "train",
        }
    ]
    config.data.kl.source.streaming = False

    config.data.sft.transforms_name = "ntp"
    config.data.sft.transforms.column = "messages"
    config.data.sft.transforms.max_length = 2048
    config.data.sft.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.sft.transforms.data_type = "chat"
    config.data.sft.transforms.assistant_loss = True
    config.data.sft.transforms.chat_template_path = str(ROOT / "temp/think.jinja")
    config.data.sft.transforms.packing = False
    config.data.sft.transforms.packing_bins = 64

    config.data.kl.transforms_name = "kl"
    config.data.kl.transforms.column = "messages"
    config.data.kl.transforms.max_length = 2048
    config.data.kl.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.kl.transforms.data_type = "chat"
    config.data.kl.transforms.assistant_loss = True
    config.data.kl.transforms.chat_template_path = str(ROOT / "temp/think.jinja")
    config.data.kl.transforms.packing = False
    config.data.kl.transforms.packing_bins = 64

    config.data.sft.loader.batch_size = 32
    config.data.sft.loader.shuffle = False
    config.data.kl.loader.batch_size = 32
    config.data.kl.loader.shuffle = False

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
