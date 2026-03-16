import sws


def get_config():
    config = sws.Config()

    config.optimizer_name = "adam"
    config.optimizer.max_grad_norm = 1.0
    config.optimizer.b1 = 0.9
    config.optimizer.b2 = 0.95
    config.optimizer.eps = 1e-8

    # adamw-only config
    # config.optimizer.weights_decay = 1e-4
    # config.optimizer.weights_decay_path = [
    #     "*.q_proj.weight",
    #     "*.k_proj.weight",
    #     "*.v_proj.weight",
    #     "*.o_proj.weight",
    #     "*.gate_proj.weight",
    #     "*.up_proj.weight",
    #     "*.down_proj.weight",
    # ]

    # sgd-only config
    # config.optimizer.momentum = 0.0
    # config.optimizer.nesterov = False

    return config
