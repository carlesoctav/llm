from pathlib import Path

from jaxformers.sws_utils import merge_config_builders as merge_config_builder


CONFIG_PATHS = [
    str(Path(__file__).parent / "name_base_config.py"),
    str(Path(__file__).parent / "model_base_config.py"),
    str(Path(__file__).parent / "data_base_config.py"),
    str(Path(__file__).parent / "optimizer_base_config.py"),
    str(Path(__file__).parent / "scheduler_base_config.py"),
]


def get_config():
    config = merge_config_builder(CONFIG_PATHS)

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
