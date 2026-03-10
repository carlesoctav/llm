import runpy
from pathlib import Path


def get_config():
    base_config_path = Path(__file__).with_name("gemma_3_1b_tunix.py")
    config = runpy.run_path(base_config_path)["get_config"]()

    config.logger_name = "noop"
    config.project = "microbatch_compare"
    config.exp_name = "test_gemma_scan_last_layer_lora"
    config.max_train_step = 1

    config.model_name = "huggingface_gemma3scan"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.model.additional_config.remat_layer = False
    config.model.additional_config.attn_implementation = "sdpa"

    config.loss_implementation = "reference"
    config.optimizer_name = "adam"
    config.optimizer.grad_accum = 4
    config.optimizer.microbatch_impl = "optax"
    config.train_loader.global_batch_size = 32
    config.data.transforms.packing = True

    config.use_lora = True
    config.random_init_lora = True
    config.lora.weights_path = [
        "model.layers.25.self_attn.q_proj.weight",
        "model.layers.25.self_attn.k_proj.weight",
        "model.layers.25.self_attn.v_proj.weight",
        "model.layers.25.self_attn.o_proj.weight",
        "model.layers.25.mlp.gate_proj.weight",
        "model.layers.25.mlp.up_proj.weight",
        "model.layers.25.mlp.down_proj.weight",
    ]

    return config
