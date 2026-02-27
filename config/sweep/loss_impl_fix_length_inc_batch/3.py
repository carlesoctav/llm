import sws

from jaxformers.sweep_utils import load_config_builder

BASE_CONFIG_PATH = '/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py'
OVERRIDES = [('data.transforms.max_length', 2048), ('optimizer_name', 'sgd'), ('loss_implementation', 'xla_chunked'), ('train_loader.global_batch_size', 64)]

def get_config() -> sws.Config:
    c = load_config_builder(BASE_CONFIG_PATH)
    for key, value in OVERRIDES:
        c[key] = value
    return c
