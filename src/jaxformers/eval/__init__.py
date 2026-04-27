import importlib
from pathlib import Path
from typing import Callable

from jaxformers.benchmark_utils import print_timing


EVAL_DIR = Path(__file__).parent


@print_timing
def make_eval(eval_config: dict[str, dict], predict_fns: dict[str, Callable]):
    evaluation = {}
    for name, config in eval_config.items():
        eval_type = config["type"] if "type" in config else "simple"
        pred_key = config["fn_name"] if "fn_name" in config else "predict"
        data_config = {
            "source": config["data"],
            "transforms": config["transforms"],
            "loader": config["loader"],
        }
        if "source_name" in config:
            data_config["source_name"] = config["source_name"]
        if "transforms_name" in config:
            data_config["transforms_name"] = config["transforms_name"]

        module_path = EVAL_DIR.joinpath(*eval_type.split(".")).with_suffix(".py")
        if module_path.is_file():
            eval_module = importlib.import_module(f"jaxformers.eval.{eval_type}")
            evaluation[name] = eval_module.make(
                name, data_config, predict_fns[pred_key]
            )
    return evaluation
