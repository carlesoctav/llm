import importlib

from .huggingface import HuggingFaceSourceIterDataset, HuggingFaceSourceMapDataset
from .verifiers import VerifiersDataset, VerifiersSourceIterDataset


__all__ = [
    "HuggingFaceSourceIterDataset",
    "HuggingFaceSourceMapDataset",
    "VerifiersDataset",
    "VerifiersSourceIterDataset",
    "make_source",
]


def make_source(
    source_name: str,
    source_config: dict,
):
    source_module = importlib.import_module(f"jaxformers.data.source.{source_name}")
    return source_module.make(**source_config)
