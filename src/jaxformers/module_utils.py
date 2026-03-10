from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeAlias

import einops
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import PRNGKeyArray


AxisName: TypeAlias = str | tuple[str, ...] | None
PartitionTree: TypeAlias = Any
Initializer: TypeAlias = Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], jax.Array]
ParamTree: TypeAlias = dict[str, Any]


def flatten_param_tree(tree: ParamTree, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _flatten(node: Any, path: tuple[str, ...]):
        if isinstance(node, Mapping):
            for key, value in node.items():
                _flatten(value, (*path, str(key)))
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                _flatten(value, (*path, str(idx)))
            return
        flat[".".join(path)] = node

    initial_path = (prefix,) if prefix else ()
    _flatten(tree, initial_path)
    return flat


def unflatten_param_tree(flat: Mapping[str, Any]) -> ParamTree:
    tree: ParamTree = {}

    for key, value in flat.items():
        cursor = tree
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if child is None:
                child = {}
                cursor[part] = child
            elif not isinstance(child, dict):
                raise KeyError(f"Duplicate parameter key {key!r}")
            cursor = child
        if parts[-1] in cursor:
            raise KeyError(f"Duplicate parameter key {key!r}")
        cursor[parts[-1]] = value

    def _convert(node: Any):
        if isinstance(node, dict):
            converted = {key: _convert(value) for key, value in node.items()}
            if converted and all(key.isdigit() for key in converted):
                indices = sorted(int(key) for key in converted)
                if indices == list(range(len(indices))):
                    return [converted[str(idx)] for idx in indices]
            return converted
        if isinstance(node, list):
            return [_convert(value) for value in node]
        return node

    return _convert(tree)


def partition_spec(partition: PartitionTree | None) -> P:
    if isinstance(partition, P):
        return partition
    if partition is None:
        return P()
    if isinstance(partition, str):
        return P(partition)
    if isinstance(partition, tuple):
        return P(*partition)
    if isinstance(partition, list):
        return P(*partition)
    raise TypeError(f"Unsupported partition annotation: {type(partition)!r}")


def out_sharding(partition: PartitionTree | None):
    if partition is None:
        return None
    if isinstance(partition, NamedSharding):
        return partition
    return partition_spec(partition)


def _reshard_leaf(x: Any, partition: PartitionTree | None):
    if partition is None:
        return x
    if not hasattr(x, "shape"):
        return x
    if isinstance(partition, NamedSharding):
        return jax.sharding.reshard(x, partition)
    return jax.sharding.reshard(x, partition_spec(partition))


def _is_axis_entry(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return True
    if isinstance(x, tuple):
        return all(isinstance(item, str) for item in x)
    return False


def _is_partition_leaf(x: Any) -> bool:
    if callable(x):
        return True
    if isinstance(x, (NamedSharding, P)):
        return True
    if _is_axis_entry(x):
        return True
    if isinstance(x, list):
        return all(_is_axis_entry(item) for item in x)
    if isinstance(x, tuple):
        return all(_is_axis_entry(item) for item in x)
    return False


@dataclass(frozen=True)
class _PartitionMask:
    partition: PartitionTree | None


def _materialize_partition_tree(value: Any, partition: PartitionTree | None):
    if callable(partition):
        return _materialize_partition_tree(value, jtu.tree_map(partition, value))

    if isinstance(value, Mapping):
        if isinstance(partition, Mapping):
            return {
                key: _materialize_partition_tree(child, partition.get(key))
                for key, child in value.items()
            }
        return {
            key: _materialize_partition_tree(child, partition)
            for key, child in value.items()
        }

    if isinstance(value, tuple):
        if isinstance(partition, tuple) and not _is_partition_leaf(partition):
            return tuple(
                _materialize_partition_tree(
                    child,
                    partition[idx] if idx < len(partition) else None,
                )
                for idx, child in enumerate(value)
            )
        return tuple(_materialize_partition_tree(child, partition) for child in value)

    if isinstance(value, list):
        if isinstance(partition, list) and not _is_partition_leaf(partition):
            return [
                _materialize_partition_tree(
                    child,
                    partition[idx] if idx < len(partition) else None,
                )
                for idx, child in enumerate(value)
            ]
        return [_materialize_partition_tree(child, partition) for child in value]

    return _PartitionMask(partition)


def _apply_partition_tree(value: Any, partition: PartitionTree | None):
    if partition is None:
        return value
    partition_tree = _materialize_partition_tree(value, partition)
    return jtu.tree_map(
        lambda x, p: _reshard_leaf(x, p.partition) if p.partition is not None else x,
        value,
        partition_tree,
        is_leaf=lambda x: x is None or isinstance(x, _PartitionMask),
    )


def maybe_reshard_input(*args, partition: PartitionTree | None = None, **kwargs):
    if partition is None:
        return args, kwargs
    reshared_args, reshared_kwargs = _apply_partition_tree(
        (args, kwargs),
        partition,
    )
    return tuple(reshared_args), dict(reshared_kwargs)


def maybe_reshard_output(output: Any, partition: PartitionTree | None):
    return _apply_partition_tree(output, partition)


def logical_to_physical(
    partition: PartitionTree | None,
    mapping: Mapping[str, AxisName],
) -> PartitionTree | None:
    if partition is None:
        return None
    if callable(partition):
        return lambda leaf: logical_to_physical(partition(leaf), mapping)
    if isinstance(partition, (NamedSharding, P)):
        return partition
    if isinstance(partition, str):
        return mapping.get(partition, partition)
    return jtu.tree_map(lambda axis: logical_to_physical(axis, mapping), partition)


@dataclass(frozen=True)
class ShardingConfig:
    partition: Mapping[str, PartitionTree | None]
    logical_to_physical_mapping: Mapping[str, AxisName]

    def translate(self, partition: PartitionTree | None):
        return logical_to_physical(partition, self.logical_to_physical_mapping)

@dataclass
class Module:
    input_partition: PartitionTree | None = None
    output_partition: PartitionTree | None = None
    weights_partition: PartitionTree | None = None

    def __post_init__(self):
        self.setup()

    def setup(self):
        pass

    def apply(self, params: ParamTree, *input_args, **input_kwargs):
        if self.input_partition is None:
            output = self.forward(params, *input_args, **input_kwargs)
        else:
            input_args, input_kwargs = maybe_reshard_input(
                *input_args,
                partition=self.input_partition,
                **input_kwargs,
            )
            output = self.forward(params, *input_args, **input_kwargs)

        if self.output_partition is None:
            return output
        return maybe_reshard_output(output, self.output_partition)

    def forward(self, params: ParamTree, *input_args, **input_kwargs):
        raise NotImplementedError

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        raise NotImplementedError


def _parse_einsum_eqn(eqn: str) -> tuple[str, str, str]:
    eqn = eqn.replace(" ", "")
    if "->" not in eqn:
        raise ValueError("`eqn` must include `->`.")
    operands, output_term = eqn.split("->")
    if operands.count(",") != 1:
        raise ValueError("`eqn` must have exactly two operands.")
    input_term, weight_term = operands.split(",")
    if "..." in weight_term:
        raise ValueError("Weight term cannot contain ellipsis.")
    return input_term, weight_term, output_term


def _create_char_dict(term: str, seq) -> dict[str, Any]:
    if seq is None:
        return {char: None for char in term.replace("...", "")}
    if "..." not in term:
        return {char: seq[idx] for idx, char in enumerate(term)}

    prefix, suffix = term.split("...")
    result = {}
    for idx, char in enumerate(prefix):
        result[char] = seq[idx]
    for idx, char in enumerate(suffix):
        result[char] = seq[len(seq) - len(suffix) + idx]
    return result


def _reshape_bias(
    bias: jax.Array, *, output_term: str, bias_term: str, output_shape: tuple[int, ...]
) -> jax.Array:
    if "..." in output_term:
        prefix, suffix = output_term.split("...")
    else:
        prefix, suffix = output_term, ""

    def _mapped(string: str) -> str:
        chars = []
        for char in string:
            chars.append(char if char in bias_term else "1")
        return "".join(chars)

    target_term = _mapped(prefix)
    target_term += "1" * (len(output_shape) - len(prefix + suffix))
    target_term += _mapped(suffix)
    return einops.rearrange(
        bias,
        f"{' '.join(bias_term)} -> {' '.join(target_term)}",
    )


def _infer_bias_partition(
    output_term: str,
    output_partition: PartitionTree | None,
    bias_term: str,
):
    if output_partition is None:
        return None
    output_chars = _create_char_dict(output_term, output_partition)
    return tuple(output_chars[char] for char in bias_term)


@dataclass
class EinsumLinear(Module):
    equation: str = ""
    weight_shape: tuple[int, ...] = ()
    weight_init: Initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
    bias_init: Initializer = jax.nn.initializers.zeros
    weight_dtype: jnp.dtype = jnp.bfloat16
    activation_dtype: jnp.dtype = jnp.bfloat16
    bias_term: str = ""
    weight_name: str = "weight"
    bias_name: str = "bias"

    def setup(self):
        self.input_term, self.weight_term, self.output_term = _parse_einsum_eqn(
            self.equation
        )
        if len(self.weight_term) != len(self.weight_shape):
            raise ValueError(
                f"Weight shape {self.weight_shape} does not match equation {self.equation!r}"
            )
        if self.bias_term:
            char_map = _create_char_dict(self.weight_term, self.weight_shape)
            self.bias_shape = tuple(char_map[char] for char in self.bias_term)
            self.bias_partition = _infer_bias_partition(
                self.output_term,
                self.output_partition,
                self.bias_term,
            )
        else:
            self.bias_shape = ()
            self.bias_partition = None

    def _weight_partition(self):
        if isinstance(self.weights_partition, Mapping):
            return self.weights_partition.get(self.weight_name)
        return self.weights_partition

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        weight_key, bias_key = jax.random.split(rngs)
        weight = self.weight_init(weight_key, self.weight_shape, self.weight_dtype)
        weight_partition = self._weight_partition()
        if weight_partition is not None:
            weight = jax.device_put(weight, out_sharding(weight_partition))

        params: ParamTree = {self.weight_name: weight}
        if self.bias_term:
            bias = self.bias_init(bias_key, self.bias_shape, self.weight_dtype)
            if self.bias_partition is not None:
                bias = jax.device_put(bias, out_sharding(self.bias_partition))
            params[self.bias_name] = bias
        return params

    def forward(self, params: ParamTree, x: jax.Array):
        x = jnp.asarray(x, dtype=self.activation_dtype)
        weight = jnp.asarray(params[self.weight_name], dtype=self.activation_dtype)
        output = jnp.einsum(
            self.equation,
            x,
            weight,
            preferred_element_type=x.dtype,
        )
        if self.bias_term:
            bias = jnp.asarray(params[self.bias_name], dtype=self.activation_dtype)
            output = output + _reshape_bias(
                bias,
                output_term=self.output_term,
                bias_term=self.bias_term,
                output_shape=output.shape,
            )
        return output


@dataclass
class LinearEmbedding(Module):
    vocab_size: int = 0
    hidden_size: int = 0
    weight_init: Initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
    weight_dtype: jnp.dtype = jnp.bfloat16
    activation_dtype: jnp.dtype = jnp.bfloat16
    scale_by_sqrt_hidden: bool = True
    use_lookup: bool = True
    weight_name: str = "weight"

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        weight = self.weight_init(
            rngs, (self.vocab_size, self.hidden_size), self.weight_dtype
        )
        if self.weights_partition is not None:
            weight = jax.device_put(weight, out_sharding(self.weights_partition))
        return {self.weight_name: weight}

    def forward(self, params: ParamTree, input_ids: jax.Array):
        weight = jnp.asarray(params[self.weight_name], dtype=self.activation_dtype)
        if self.use_lookup:
            output = weight.at[input_ids, :].get()
        else:
            one_hot = jax.nn.one_hot(
                input_ids,
                self.vocab_size,
                dtype=self.activation_dtype,
            )
            output = jnp.einsum(
                "...v,vd->...d",
                one_hot,
                weight,
                preferred_element_type=self.activation_dtype,
            )
        if self.scale_by_sqrt_hidden:
            output = output * jnp.sqrt(
                jnp.asarray(self.hidden_size, dtype=self.activation_dtype)
            )
        return output


@dataclass
class EinsumLinearLora(EinsumLinear):
    rank: int = 0
    alpha: float = 1.0
    lora_a_init: Initializer = jax.nn.initializers.he_normal(
        in_axis=-1,
        out_axis=-2,
    )
    lora_b_init: Initializer = jax.nn.initializers.zeros
    allow_materialise: bool = False
    lora_a_name: str = "lora_a"
    lora_b_name: str = "lora_b"

    def _lora_partitions(self):
        weight_partition = self._weight_partition()
        if weight_partition is None:
            return None, None
        if isinstance(weight_partition, Mapping):
            weight_partition = weight_partition.get(self.weight_name)
        spec = tuple(partition_spec(weight_partition))
        if len(spec) < 2:
            return None, None
        a_partition = P(spec[-2], None)
        b_partition = P(*spec[:-2], None, spec[-1])
        return a_partition, b_partition

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        if self.rank <= 0:
            return super().init(rngs)

        base_key, a_key, b_key = jax.random.split(rngs, 3)
        params = super().init(base_key)
        a_partition, b_partition = self._lora_partitions()
        a = self.lora_a_init(
            a_key,
            (self.weight_shape[-2], self.rank),
            self.weight_dtype,
        )
        b = self.lora_b_init(
            b_key,
            (*self.weight_shape[:-2], self.rank, self.weight_shape[-1]),
            self.weight_dtype,
        )
        if a_partition is not None:
            a = jax.device_put(a, a_partition)
        if b_partition is not None:
            b = jax.device_put(b, b_partition)
        params[self.lora_a_name] = a
        params[self.lora_b_name] = b
        return params

    def forward(self, params: ParamTree, x: jax.Array):
        x = jnp.asarray(x, dtype=self.activation_dtype)
        weight = jnp.asarray(params[self.weight_name], dtype=self.activation_dtype)
        if self.rank > 0:
            a = jnp.asarray(params[self.lora_a_name], dtype=self.activation_dtype)
            b = jnp.asarray(params[self.lora_b_name], dtype=self.activation_dtype)
            scaling = jnp.asarray(self.alpha / self.rank, dtype=self.activation_dtype)
            delta = scaling * jnp.einsum(
                "ir,...rj->...ij",
                a,
                b,
                preferred_element_type=x.dtype,
            )
            weight = weight + delta
        output = jnp.einsum(
            self.equation,
            x,
            weight,
            preferred_element_type=x.dtype,
        )
        if self.bias_term:
            bias = jnp.asarray(params[self.bias_name], dtype=self.activation_dtype)
            output = output + _reshape_bias(
                bias,
                output_term=self.output_term,
                bias_term=self.bias_term,
                output_shape=output.shape,
            )
        return output


@dataclass
class RMSNorm(Module):
    hidden_size: int = 0
    eps: float = 1e-6
    add_unit_offset: bool = False
    weight_init: Initializer | None = None
    weight_dtype: jnp.dtype = jnp.bfloat16
    activation_dtype: jnp.dtype = jnp.bfloat16
    weight_name: str = "weight"

    def init(self, rngs: PRNGKeyArray) -> ParamTree:
        if self.weight_init is None:
            if self.add_unit_offset:
                weight = jnp.zeros((self.hidden_size,), dtype=self.weight_dtype)
            else:
                weight = jnp.ones((self.hidden_size,), dtype=self.weight_dtype)
        else:
            weight = self.weight_init(rngs, (self.hidden_size,), self.weight_dtype)
        if self.weights_partition is not None:
            weight = jax.device_put(weight, out_sharding(self.weights_partition))
        return {self.weight_name: weight}

    def forward(self, params: ParamTree, x: jax.Array):
        weight = jnp.asarray(
            params[self.weight_name],
            dtype=jnp.float32,
        )
        x_fp32 = jnp.asarray(x, dtype=jnp.float32)
        rms = jnp.sqrt(jnp.square(x_fp32).mean(axis=-1, keepdims=True) + self.eps)
        if self.add_unit_offset:
            output = (x_fp32 / rms) * (1.0 + weight)
        else:
            output = (x_fp32 / rms) * weight
        return output.astype(self.activation_dtype)
