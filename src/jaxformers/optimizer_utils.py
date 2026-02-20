import equinox
from dataclasses import replace
import optax

import jax.numpy as jnp
import jax.tree_util as jtu

from jaxformers.dispatch.lora import LoraArray


def find_apply_every_count(opt_state):
    """Find optax.apply_every state's `count` inside a nested opt_state PyTree."""

    if opt_state is None:
        return None
    # optax.apply_every state has fields `count` and `grad_acc`.
    if hasattr(opt_state, "count") and hasattr(opt_state, "grad_acc"):
        return opt_state.count
    # optax.masked wrapper state.
    if hasattr(opt_state, "inner_state"):
        return find_apply_every_count(opt_state.inner_state)
    # optax.partition wrapper state.
    if hasattr(opt_state, "inner_states"):
        try:
            values = opt_state.inner_states.values()
        except Exception:
            values = ()
        for v in values:
            found = find_apply_every_count(v)
            if found is not None:
                return found
        return None
    if isinstance(opt_state, (tuple, list)):
        for v in opt_state:
            found = find_apply_every_count(v)
            if found is not None:
                return found
        return None
    if isinstance(opt_state, dict):
        for v in opt_state.values():
            found = find_apply_every_count(v)
            if found is not None:
                return found
        return None
    return None


def lora_only_param_labels_v2(params):
    """Label LoRA parameters for optax.partition.

    Train only `LoraArray.a` and `LoraArray.b`. Everything else is frozen.
    This prevents allocating Adam moments / grad accumulators for the full
    base model weights.
    """

    is_lora_array = lambda x: isinstance(x, LoraArray)

    def label(leaf):
        if isinstance(leaf, LoraArray):
            return replace(leaf, _w=False, a=True, b=True)
        return False

    return jtu.tree_map(label, params, is_leaf=is_lora_array)

def lora_only_param_labels(params):
    """Label LoRA parameters for optax.partition.

    Train only `LoraArray.a` and `LoraArray.b`. Everything else is frozen.
    This prevents allocating Adam moments / grad accumulators for the full
    base model weights.
    """

    def label(path, leaf):
        if isinstance(path[-1], jtu.GetAttrKey) and path[-1].name in ('a', 'b'):
            return True
        return False

    mask = jtu.tree_map_with_path(label, params)
    return mask




def _freeze_non_accum_states(emit, new_state, old_state):
    """Freeze downstream optimizer states (e.g. Adam moments) on non-emit microsteps.

    Assumes the optimizer is an optax.chain where the first two transforms are
    `apply_every` and `divide_every`.
    """

    if isinstance(new_state, tuple) and isinstance(old_state, tuple):
        if len(new_state) < 2 or len(old_state) < 2:
            return new_state
        tail = jtu.tree_map(
            lambda ns, os: jnp.where(emit, ns, os), new_state[2:], old_state[2:]
        )
        return new_state[:2] + tail

    if hasattr(new_state, "inner_states") and hasattr(old_state, "inner_states"):
        inner_new = dict(new_state.inner_states)
        inner_old = old_state.inner_states

        for k, v_new in list(inner_new.items()):
            v_old = inner_old.get(k, None) if isinstance(inner_old, dict) else None
            if v_old is None:
                continue
            if not (hasattr(v_new, "inner_state") and hasattr(v_old, "inner_state")):
                continue
            s_new = v_new.inner_state
            s_old = v_old.inner_state
            if not (isinstance(s_new, tuple) and isinstance(s_old, tuple)):
                continue
            if len(s_new) < 2 or len(s_old) < 2:
                continue
            tail = jtu.tree_map(
                lambda ns, os: jnp.where(emit, ns, os), s_new[2:], s_old[2:]
            )
            inner_new[k] = v_new._replace(inner_state=s_new[:2] + tail)

        return new_state._replace(inner_states=inner_new)

    return new_state

optax.partition
