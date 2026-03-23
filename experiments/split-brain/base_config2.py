from pathlib import Path

import sws
from transformers import AutoTokenizer

from jaxformers.data.transforms.kl import make as make_kl_transforms
from jaxformers.sws_utils import load_config_builder


CONFIG_PATH = str(Path(__file__).with_name("base_config.py"))
ROOT = Path(__file__).resolve().parents[2]


def get_config() -> sws.Config:
    config = load_config_builder(CONFIG_PATH)
    config.data.kl.transforms = lambda: make_kl_transforms(
        column="messages",
        max_length=2048,
        tokenizer=AutoTokenizer.from_pretrained(config.model.model_id),
        data_type="chat",
        chat_template_path=str(ROOT / "temp/think.jinja"),
        assistant_loss=True,
        packing=False,
        packing_bins=64,
    )
    return config
