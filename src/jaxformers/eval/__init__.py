import importlib
from pathlib import Path
from typing import Callable


EVAL_DIR = Path(__file__).parent


def make(eval_config: dict[str, dict], predict_fns: dict[str, Callable]):
    evaluation = {}
    for name, config in eval_config.items():
        eval_type = config["type"] if "type" in config else "simple"
        pred_key = config["fn"] if "fn" in config else "predict"
        data_config = config["data"] if "data" in config else {}

        module_path = EVAL_DIR.joinpath(*eval_type.split(".")).with_suffix(".py")
        if module_path.is_file():
            eval_module = importlib.import_module(f"jaxformers.eval.{eval_type}")
            evaluation[name] = eval_module.make(data_config, predict_fns[pred_key])
    return evaluation
