import wandb


def make(logger_config):
    return wandb.init(**logger_config)
