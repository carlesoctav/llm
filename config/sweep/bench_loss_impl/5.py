import sws

from jaxformers.sweep_utils import load_config_builder

BASE_CONFIG_PATH = '/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py'
OVERRIDES = [('optimizer_name', 'sgd'), ('data.transforms.max_length', 4096), ('loss_implementation', 'reference')]

def get_config() -> sws.Config:
    c = load_config_builder(BASE_CONFIG_PATH)
    for key, value in OVERRIDES:
        c[key] = value
    return c
