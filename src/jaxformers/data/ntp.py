from jax.sharding import Mesh

from jaxformers.data.loader.simple import make_simple_loader
from jaxformers.data.source.huggingface import make_huggingface_datasets
from jaxformers.data.transforms.ntp import make_ntp_transforms


def make_ntp_data(
    source: dict,
    transforms_config: dict,
    loader_config: dict,
    *,
    streaming: bool = False,
    mesh: Mesh | None = None,
):
    datasets = make_huggingface_datasets(source["load_kwargs"], streaming=streaming)
    transforms = make_ntp_transforms(**transforms_config)
    data_loader = make_simple_loader(datasets, transforms, mesh, **loader_config)
    return data_loader
