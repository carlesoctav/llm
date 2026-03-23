import importlib

from .huggingface import HuggingFaceSourceIterDataset, HuggingFaceSourceMapDataset


__all__ = [
    "HuggingFaceSourceIterDataset",
    "HuggingFaceSourceMapDataset",
    "make_source",
]


def make_source(
    source_name: str,
    source_config: dict,
    *,
    streaming: bool = False,
):
    source_module = importlib.import_module(f"jaxformers.data.source.{source_name}")
    return source_module.make(streaming=streaming, **source_config)
