import importlib

from .huggingface import HuggingFaceSourceIterDataset, HuggingFaceSourceMapDataset
from .training import make_dataloader
from .next_token_prediction import transforms


def make_dataset(
    data_name: str,
    load_data_config,
    transforms_config: dict,
    train_loader_config: dict,
):
    data_module = importlib.import_module(f"jaxformers.data.{data_name}")
    transforms_fn = transforms(**transforms_config)

    train_dataset = data_module.make(load_data_config)
    return make_dataloader(
        train_dataset,
        transforms_fn,
        **train_loader_config,
    )
