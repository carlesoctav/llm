import jax
import jax.numpy as jnp
import sws


def get_config():
    config = sws.Config()

    config.init_model = "pretrained"
    config.init_lora = None

    config.model_name = "huggingface.gemma3"
    config.model.parallel_dims = {"dp_replicate": 1, "dp_shard": 4, "cp": 1, "tp": 1}
    config.model.model_id = "google/gemma-3-1b-it"
    config.model.additional_config.remat_layer = False
    config.model.additional_config.attn_implementation = "sdpa"
    config.model.additional_config.sequence_parallelism = True
    config.model.additional_config.forward_impl = "loop"

    config.model.devices = lambda: jax.devices()
    config.model.param_dtype = lambda: jnp.bfloat16

    return config
