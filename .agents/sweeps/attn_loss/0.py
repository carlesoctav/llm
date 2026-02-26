import sws

from jaxformers.bench.sweep_utils import load_config_builder

BASE_CONFIG_PATH = '/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py'
OVERRIDES = [('model.additional_config.attn_implementation', 'xla_chunked'), ('loss_implementation', 'xla_chunked')]

def get_config() -> sws.Config:
    c = load_config_builder(BASE_CONFIG_PATH)
    for key, value in OVERRIDES:
        c[key] = value
    return c
