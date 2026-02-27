import wandb


def make(config):
    log_config = config.to_dict()
    log_config.pop("logger", None)
    return wandb.init(**config.logger.to_dict(), config=log_config)
