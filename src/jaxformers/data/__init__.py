import importlib

from .source.huggingface import (
    HuggingFaceSourceIterDataset,
    HuggingFaceSourceMapDataset,
)


__all__ = [
    "HuggingFaceSourceIterDataset",
    "HuggingFaceSourceMapDataset",
    "make_source",
    "make_transforms",
    "make_loader",
    "make_dataset",
]


def make_source(source_name: str, source_config: dict):
    source_module = importlib.import_module(f"jaxformers.data.source.{source_name}")
    factory = getattr(source_module, "make", None)
    if not callable(factory):
        raise ValueError(f"jaxformers.data.source.{source_name} must define make()")
    return factory(**source_config)


def make_transforms(transforms_name: str, transforms_config: dict):
    transforms_module = importlib.import_module(
        f"jaxformers.data.transforms.{transforms_name}"
    )
    factory = getattr(transforms_module, "make", None)
    if not callable(factory):
        raise ValueError(
            f"jaxformers.data.transforms.{transforms_name} must define make()"
        )
    return factory(**transforms_config)


def make_loader(
    loader_name: str,
    datasets,
    transforms,
    loader_config: dict,
):
    loader_module = importlib.import_module(f"jaxformers.data.loader.{loader_name}")
    factory = getattr(loader_module, "make", None)
    if not callable(factory):
        raise ValueError(f"jaxformers.data.loader.{loader_name} must define make()")
    return factory(
        datasets=datasets,
        transforms=transforms,
        **loader_config,
    )


def make_dataset(
    source_name: str,
    source_config: dict,
    transforms_name: str,
    transforms_config: dict,
    loader_name: str,
    loader_config: dict,
):
    datasets = make_source(source_name, source_config)
    transforms = make_transforms(transforms_name, transforms_config)
    return make_loader(loader_name, datasets, transforms, loader_config)
