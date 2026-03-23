import importlib


__all__ = ["make_loader"]


def make_loader(
    loader_name: str,
    datasets,
    transforms,
    loader_config: dict,
    *,
    mesh=None,
):
    loader_module = importlib.import_module(f"jaxformers.data.loader.{loader_name}")
    return loader_module.make(datasets, transforms, mesh, **loader_config)
