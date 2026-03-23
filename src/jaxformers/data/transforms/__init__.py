import importlib

from .base import DatasetTransforms


__all__ = ["DatasetTransforms", "make_transforms"]


def make_transforms(transforms_name: str, transforms_config: dict):
    transforms_module = importlib.import_module(
        f"jaxformers.data.transforms.{transforms_name}"
    )
    return transforms_module.make(**transforms_config)
