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
    # config.random_init_lora = False

    config.exp_name = ""
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.exp_name}"
    config.train_seed = 42
    config.eval_every = None
    config.max_train_step = 30
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_implementation = "xla_chunked"

    config.model_name = "qwen3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}
    config.model.model_id = "Qwen/Qwen3-0.6B"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_implementation = "xla_chunked"
    config.model.additional_config.sequence_parallelism = True

    config.model.devices = lambda: jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    # config.lora.rank = 64
    # config.lora.alpha = 1
    # config.lora.weights_path = [
    #     "*.q_proj.weight",
    #     "*.k_proj.weight",
    #     "*.v_proj.weight",
    #     "*.o_proj.weight",
    #     "*.gate_proj.weight",
    #     "*.up_proj.weight",
    #     "*.down_proj.weight",
    # ]

    config.lr_scheduler_name = None
    config.learning_rate = 1e-5

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.grad_accum = 1

    config.data_name = "huggingface"
    config.data.load_kwargs = [
        {
            "path": "allenai/Dolci-Instruct-SFT",
            "split": "train",
            "streaming": False,
        }
    ]
    config.data.transforms.column = "messages"
    config.data.transforms.max_length = 8192
    config.data.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.transforms.assistant_loss = False
    config.data.transforms.is_tokenized = False
    config.data.transforms.is_chat = True
    config.data.transforms.chat_template_path = None
    config.data.transforms.packing = True
    config.data.transforms.packing_bins = 64

    # total batch size is global_batch_size * config.optimizer.grad_accum
    config.train_loader.global_batch_size = 8
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
    config.train_loader.worker_count = 8
    config.train_loader.worker_buffer_size = 100
    config.train_loader.drop_remainder = True


    config.log.grad_norm = True
    config.log.learning_rate = True
    config.logger_name = "noop"

    # see orbax checkpointmanager options
    config.checkpoint_options.save_interval_steps = 10000
    config.checkpoint_options.max_to_keep = 1

    config.logger.project = "bench-loss-impl"
    config.logger.name = lambda: config.exp_name

    config.logger.resume = lambda: "allow" if config.resume else "never"

    # config.logger.entity =
    # config.logger.dir =
    # config.logger.notes =
    # config.logger.tags =

    return config

