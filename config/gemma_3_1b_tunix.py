import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()

    config.skip_eval = True

    config.init_model = "pretrained"
    config.init_lora = None

    config.exp_name = ""
    config.project_name = ""
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.project_name}/{config.exp_name}"
    config.train_seed = 42
    config.eval_every = None
    config.max_train_step = 10_000
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_implementation = "reference"

    config.model_name = "huggingface.gemma3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = False
    config.model.additional_config.attn_implementation = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.forward_impl = "loop"

    config.model.devices = lambda: jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    config.lora.rank = 256
    config.lora.alpha = 512
    config.lora.weights_path = [
        "*.q_proj.weight",
        "*.k_proj.weight",
        "*.v_proj.weight",
        "*.o_proj.weight",
        "*.gate_proj.weight",
        "*.up_proj.weight",
        "*.down_proj.weight",
    ]

    config.learning_rate = 1e-5
    config.lr_scheduler_name = None
    # config.lr_scheduler.xx = xx

    config.optimizer_name = "adam"
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
    config.data.transforms.packing = False
    config.data.transforms.packing_bins = 64

    # total batch size is global_batch_size * config.optimizer.grad_accum
    config.train_loader.global_batch_size = 32
    config.train_loader.seed = 42

    # config.train_loader.pspec =
    # config.train_loader.mesh =
    # config.train_loader.num_epochs =
    # config.train_loader.dataset_weights =
    # config.train_loader.dataloading_host_index =
    # config.train_loader.dataloading_host_count =
    # config.train_loader.is_not_sharded =
    # config.train_loader.read_num_threads =
    # config.train_loader.read_prefetch_buffer_size =
    config.train_loader.shuffle = False
    # config.train_loader.shuffle_buffer_size =
    # config.train_loader.worker_count = 8
    # config.train_loader.worker_buffer_size = 100
    config.train_loader.drop_remainder = True

    config.log.grad_norm = True
    config.log.learning_rate = True
    config.logger_name = "wandb"

    # see orbax checkpointmanager options
    config.use_checkpoint = False
    config.checkpoint_options.save_interval_steps = 2500
    config.checkpoint_options.max_to_keep = 1

    config.logger.project = lambda: config.project_name
    config.logger.name = lambda: config.exp_name

    # config.logger.entity =
    # config.logger.dir =
    # config.logger.notes =
    # config.logger.tags =
    #
    #
    config.callback = [
        "log_grad_norm",
        "log_learning_rate",
        # ("log_performance", {"denom_keys": ["token", "batch"], "real_step_threshold": 0}),
    ]

    return config
