import jax
import jax.numpy as jnp
import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()

    config.exp_name = "splade-msmarco"
    config.project_name = "splade"
    config.dir = "./checkpoints"
    config.checkpoint.path = lambda: (
        f"{config.dir}/{config.project_name}/{config.exp_name}"
    )
    config.seed = 42
    config.eval_every = None
    config.max_train_step = 10_000
    config.forward_dtype = lambda: jnp.bfloat16
    config.grad_accum = 1

    config.temperature = 6.0
    config.flops_query_weight = 1e-3
    config.flops_doc_weight = 5e-4

    config.logger_name = "noop"
    config.logger.project = lambda: config.project_name
    config.logger.name = lambda: config.exp_name

    config.checkpoint.save_interval_steps = 500
    config.checkpoint.max_to_keep = 3
    config.checkpoint.save_only_trainable = False

    config.weights_impl = "stack"
    config.model_name = "search.splade.SpladeModel.from_pretrained"
    config.parallel.parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 1}
    config.parallel.devices = lambda: jax.devices()
    config.parallel.multihost = False
    config.model.model_id = "Linkup-Platform/linkup-sparseup-embed-v1"
    config.model.additional_config.remat_layer = True
    config.model.additional_config.attn_impl = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.param_dtype = lambda: jnp.bfloat16

    ds_name = "bclavie/msmarco-500k-triplets"

    config.data.loader.shard = False
    config.data.loader.seed = 42
    config.data.loader.num_workers = 0
    config.data.loader.num_threads = 1
    config.data.loader.prefetch_buffer_size = 500
    config.data.loader.per_worker_buffer_size = 1

    config.data.train.source = {
        "load_kwargs": [
            {
                "path": ds_name,
                "split": "train",
            }
        ],
        "streaming": False,
    }
    config.data.train.transforms_name = "splade"
    config.data.train.transforms = {
        "tokenizer": lambda: AutoTokenizer.from_pretrained(config.model.model_id),
        "query_column": "query",
        "positive_column": "positive",
        "negatives_column": "negative",
        "num_negatives": 1,
        "query_prefix": "[Q] ",
        "document_prefix": "[D] ",
        "query_max_length": 64,
        "doc_max_length": 256,
    }
    config.data.train.loader = {"batch_size": 32, "shuffle": True}

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.b1 = 0.9
    config.optimizer.b2 = 0.95
    config.optimizer.eps = 1e-8

    config.learning_rate = 2e-5
    config.lr_scheduler_name = "wsds"
    config.lr_scheduler.min_lr_ratio = 0.1
    config.lr_scheduler.warmup = 0.01
    config.lr_scheduler.decay = None
    config.lr_scheduler.rewarmup = 0.0
    config.lr_scheduler.cycle_length = 0.2
    config.lr_scheduler.cycles = None
    config.lr_scheduler.decay_schedule = "cosine"

    return config
