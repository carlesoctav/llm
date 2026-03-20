from jax.sharding import Mesh

from jaxformers.data.loader.zip import make as make_zip_loader
from jaxformers.data.source.huggingface import make_huggingface_datasets
from jaxformers.data.transforms.kl import make as make_kl_transforms
from jaxformers.data.transforms.ntp import make_ntp_transforms


def make(
    sft_source: dict,
    sft_transforms_config: dict,
    kl_source: dict,
    kl_transforms_config: dict,
    loader_config: dict,
    *,
    streaming: bool = False,
    mesh: Mesh | None = None,
):
    sft_datasets = make_huggingface_datasets(
        sft_source["load_kwargs"], streaming=streaming
    )
    kl_datasets = make_huggingface_datasets(kl_source["load_kwargs"], streaming=streaming)
    sft_transforms = make_ntp_transforms(**sft_transforms_config)
    kl_transforms = make_kl_transforms(**kl_transforms_config)
    return make_zip_loader(
        [sft_datasets, kl_datasets],
        [sft_transforms, kl_transforms],
        mesh,
        **loader_config,
    )
