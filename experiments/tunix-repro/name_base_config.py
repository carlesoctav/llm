import jax.numpy as jnp
import sws


def get_config():
    config = sws.Config()

    config.skip_eval = True

    config.exp_name = ""
    config.project_name = ""
    config.dir = "gs://carles-git-good"
    config.ckpt_path = lambda: f"{config.dir}/{config.project_name}/{config.exp_name}"
    config.seed = 42
    config.eval_every = None
    config.max_train_step = 10_000
    config.forward_dtype = lambda: jnp.bfloat16
    config.loss_implementation = "reference"
    config.grad_accum = 4

    config.logger_name = "wandb"

    config.use_checkpoint = False
    config.checkpoint_options.save_interval_steps = 2500
    config.checkpoint_options.max_to_keep = 1

    config.logger.project = lambda: config.project_name
    config.logger.name = lambda: config.exp_name

    config.callback_name = ["log_grad_norm", "log_learning_rate", "log_performance"]
    config.callback.log_performance.real_step_threshold = 0
    config.callback.log_performance.denom_keys = ["token"]

    return config
