import importlib
from pathlib import Path


EVAL_DIR = Path(__file__).parent


def make_eval(eval_config: dict, predict_fns):
    for name, config in eval_config.items():
        module_path = EVAL_DIR.joinpath(*name.split(".")).with_suffix(".py")

        if module_path.is_file():
            eval_module = importlib.import_module(f"jaxformers.eval.{name}")
