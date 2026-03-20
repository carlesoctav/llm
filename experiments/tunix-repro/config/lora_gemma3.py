import sws


def get_config():
    config = sws.Config()

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

    return config
