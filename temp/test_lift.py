import jax
from jaxformers.data.source.huggingface import make_huggingface_datasets


make_huggingface_datasets
lift_data = [
    {
        "path": "carlesoctav/4b-generated-dolci-instruct-sft-no-tools-messages",
        "split": "train",
        "streaming": False,
    },
    {
        "path": "carlesoctav/4b-generated-dolci-instruct-sft-no-tools-messages",
        "split": "train",
        "streaming": False,
    },
]


def lift(func):
    is_leaf = lambda x: isinstance(x, (tuple, list, dict))
    def _lift(*args, **kwargs):
        if args and kwargs:
            return jax.tree.map(func, args, kwargs, is_leaf = is_leaf)
        elif args:
            print("DEBUGPRINT {args}:", args)
            return jax.tree.map(func, args, is_leaf = is_leaf)
        elif kwargs:
            return jax.tree.map(func, kwargs, is_leaf = is_leaf)

    return _lift



make_lift = lift(make_huggingface_datasets)
a = make_lift(lift_data)
