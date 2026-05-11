import abc
import dataclasses
from contextlib import ExitStack
from enum import auto, StrEnum
from pathlib import Path
from typing import Any, Generic, Self, TypedDict, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from huggingface_hub import snapshot_download
from jaxtyping import PRNGKeyArray
from safetensors import safe_open
from transformers import AutoConfig, PreTrainedConfig

from jaxformers import tree_util
from jaxformers.scan_utils import make_scan_fwd
from jaxformers.sharding_utils import get_logical_axis_rules


default_init = jax.nn.initializers.variance_scaling(
    1 / 3.0, "fan_in", "uniform", in_axis=-1, out_axis=-2, batch_axis=()
)


M = TypeVar("M", bound=eqx.Module)


@dataclasses.dataclass
class VllmWeightLeaf:
    value: jax.Array


@dataclasses.dataclass
class VllmWeightState:
    leaves: list[tuple[tuple[str, ...], VllmWeightLeaf]]

    def flat_state(self):
        return self.leaves

    def from_flat_path(self, _flat_state):
        return self


@dataclasses.dataclass
class VllmMapping:
    state: VllmWeightState
    mappings: dict[str, tuple[str, tuple[str, ...] | None]]
    transpose_keys: dict[str, tuple[int, ...]]


class ForwardImpl(StrEnum):
    LOOP = auto()
    SCAN = auto()


class StackImpl(StrEnum):
    STACK = auto()
    FREE = auto()


class AdditionalConfig(TypedDict):
    remat_layer: bool

    attn_impl: str = "sdpa"
    sequence_parallelism: bool = True
    weights_impl: str = "stack"
    forward_impl: str = "scan"


DEFAULT_ADDITIONAL_CONFIG = {
    "remat_layer": False,
    "attn_impl": "sdpa",
    "sequence_parallelism": True,
    "forward_impl": "scan",
}


def module_replace(module, **kwargs):
    "dataclasses replace for eqx.Module"


@dataclasses.dataclass
class Stackable:
    argnums: tuple[int, ...] = eqx.field(static=True, default=0)
    argnames: tuple[str, ...] = eqx.field(static=True, default=())
    in_axes: int = eqx.field(static=True, default=0)

    weights_impl: StackImpl = eqx.field(static=True, default="free")


class StackModule(eqx.Module, Generic[M]):
    layers: M
    module: type[M] = eqx.field(static=True)

    argnums: tuple[int, ...] = eqx.field(static=True)
    argnames: tuple[str, ...] = eqx.field(static=True)
    in_axes: tuple[int, ...] = eqx.field(static=True)
    length: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(
        self,
        module: type[M],
        layers: list[M],
        argnums: int | tuple[int, ...],
        *,
        argnames: str | tuple[str, ...] = (),
        in_axes: int | tuple[int, ...] = 0,
        remat: bool = False,
    ):
        def _stack(*leaf):
            if leaf[0] is None:
                return None
            return jnp.stack(leaf)

        self.module = module
        self.argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        self.argnames = (argnames,) if isinstance(argnames, str) else tuple(argnames)
        if isinstance(in_axes, int):
            self.in_axes = (in_axes,) * (len(self.argnums) + len(self.argnames))
        else:
            self.in_axes = tuple(in_axes)
        self.remat = remat

        if isinstance(layers, list):
            self.length = len(layers)
            self.layers = jax.tree.map(_stack, *layers, is_leaf=lambda x: x is None)
            return

        self.layers = layers
        self.length = next(
            leaf.shape[0]
            for leaf in jax.tree.leaves(layers, is_leaf=lambda x: x is None)
            if leaf is not None
        )

    def unstack(self):
        def _unstack_leaf(leaf):
            if leaf is None:
                return None
            return jnp.unstack(leaf)

        trees = jax.tree.map(_unstack_leaf, self.layers, is_leaf=lambda x: x is None)
        length = None
        for leaf in jax.tree.leaves(trees, is_leaf=lambda x: isinstance(x, tuple)):
            if isinstance(leaf, tuple):
                length = len(leaf)
                break
        if length is None:
            raise ValueError("Cannot unstack a layer tree without array leaves.")

        return [
            jax.tree.map(
                lambda leaf: None if leaf is None else leaf[layer_idx],
                trees,
                is_leaf=lambda x: x is None or isinstance(x, tuple),
            )
            for layer_idx in range(length)
        ]

    def __call__(self, *args, **kwargs):
        module_call = (
            jax.remat(self.module.__call__) if self.remat else self.module.__call__
        )

        def stack_fwd(carry, layer, *fwd_args, **fwd_kwargs):
            return module_call(layer, carry, *fwd_args, **fwd_kwargs)

        return make_scan_fwd(
            stack_fwd,
            self.length,
            self.argnums,
            argnames=self.argnames,
            in_axes=self.in_axes,
        )(*args, self.layers, **kwargs)


class AbstractModel(eqx.Module):
    config: eqx.AbstractVar[Any]

    @abc.abstractmethod
    def get_config(self): ...

    def stack(self):
        _is_leaf = lambda x: isinstance(x, list)

        def f(path, leaf):
            if isinstance(leaf, list):
                l0 = leaf[0]
                if isinstance(l0, Stackable):
                    print(
                        f"{jtu.keystr(path, simple=True, separator='.')} is a stackable list, converthing to stack"
                    )
                    return StackModule(
                        type(l0),
                        leaf,
                        l0.argnums,
                        argnames=l0.argnames,
                        in_axes=l0.in_axes,
                        remat=l0.remat,
                    )
                else:
                    return leaf
            else:
                return leaf

        return jax.tree.map_with_path(f, self, is_leaf=_is_leaf)


class ToVllmMappingAbstract(abc.ABC):
    @abc.abstractmethod
    def to_vllm(self) -> VllmMapping: ...


class AbstractHuggingFacePreTrainedModel(AbstractModel):
    config: eqx.AbstractVar[PreTrainedConfig]

    def get_config(self):
        return self.config.to_diff_dict()

    @classmethod
    def init(
        cls: type[Self],
        config: PreTrainedConfig | None = None,
        model_id: str | None = None,
        additional_config: AdditionalConfig | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs,
    ) -> Self:
        if (config is None) == (model_id is None):
            raise ValueError(
                f"Exactly one of `config` or `model_id` must be provided to {cls.__name__}.init()."
            )

        if model_id is not None:
            config = AutoConfig.from_pretrained(model_id)

        if not isinstance(config, PreTrainedConfig):
            raise TypeError(f"Expected HF config, got {type(config)!r}")

        additional_config = {
            **DEFAULT_ADDITIONAL_CONFIG,
            **(additional_config or {}),
        }
        config.sharding_rules = get_logical_axis_rules()
        config.additional_config = additional_config
        return cls(
            config,
            additional_config,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    @classmethod
    def from_pretrained(
        cls: type[Self],
        model_id: str,
        local_dir: str | None = None,
        additional_config: AdditionalConfig | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs,
    ) -> Self:
        model_rngs, missing_rngs = jax.random.split(rngs)
        additional_config = {
            **DEFAULT_ADDITIONAL_CONFIG,
            **(additional_config or {}),
        }

        model_ckpt_dir = Path(snapshot_download(repo_id=model_id, local_dir=local_dir))
        config = AutoConfig.from_pretrained(model_ckpt_dir)
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(f"Expected HF config, got {type(config)!r}")
        config.sharding_rules = get_logical_axis_rules()
        config.additional_config = additional_config
        with ExitStack() as stack:
            missing_key = set()
            used_key = set()
            state_dict = {}
            for file in model_ckpt_dir.glob("*.safetensors"):
                file_pointer = stack.enter_context(safe_open(file, framework="numpy"))
                for key in file_pointer.keys():
                    state_dict[key] = file_pointer.get_slice(key)

            safetensor_key = set(state_dict.keys())
            abstract_model = jax.eval_shape(
                lambda: cls(
                    config,
                    additional_config,
                    rngs=model_rngs,
                    param_dtype=param_dtype,
                )
            )
            needed_model_key = set(tree_util.flatten(abstract_model).keys())

            def load_leaf(path, leaf):
                key = jax.tree_util.keystr(path, simple=True, separator=".")
                if key not in state_dict:
                    return leaf
                tensor = state_dict[key][:].astype(param_dtype)
                used_key.add(key)
                return jax.device_put(tensor, leaf.sharding.spec)

            model = jax.tree.map_with_path(
                load_leaf,
                abstract_model,
            )

        if missing_key := needed_model_key - used_key:
            print(
                "Warning: The following required keys are missing from the safetensors archive and will be "
                "left as default-initialized:",
                *sorted(missing_key),
            )
            model = init_missing_module(model, missing_key, rngs=missing_rngs)

        if not_used_key := safetensor_key - used_key:
            print(
                f"Some keys are present in the safetensors archive but are not required by the model {cls}. "
                "This can be expected if the safetensors weights were derived from a different model variant "
                "(for example, one with an added classification head). Please review whether this is an expected outcome:",
                *not_used_key,
            )

        return model


def init_missing_module(model, missing_key, *, rngs: PRNGKeyArray):
    counter = 0

    def f(path, leaf):
        nonlocal counter
        if isinstance(leaf, jax.ShapeDtypeStruct):
            array = default_init(
                jax.random.fold_in(rngs, counter),
                leaf.shape,
                out_sharding=leaf.sharding.spec,
                dtype=leaf.dtype,
            )
            counter += 1
            return array
        else:
            return leaf

    return jax.tree.map_with_path(f, model)
