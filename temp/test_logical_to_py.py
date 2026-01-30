from dataclasses import dataclass
import jax.tree_util as jtu
import jax

@dataclass
class Rules:
    fsdp: tuple[str, ...] = ("dp_shard", "cp")
    tp: tuple[str, ...] = ("tp")


def logical_to_physical(logicals, rules):
    spec = [getattr(rules, logical) for logical in logicals]
    print("DEBUGPRINT {spec}:", spec)
    leave = jtu.tree_leaves(spec)
    print("DEBUGPRINT {leave}:", leave)
    if len(set(leave)) != len(leave):
        raise ValueError("soemthing when wrong")



logical_to_physical(("fsdp", "tp"), Rules())
