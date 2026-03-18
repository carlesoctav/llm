from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclass(frozen=True)
class ShardCheck:
    name: str
    shape: tuple[int, ...]
    axis_dim: int
    axis_size: int


def is_shardable(shape: tuple[int, ...], axis_dim: int, axis_size: int) -> bool:
    if axis_dim < 0 or axis_dim >= len(shape):
        raise ValueError(f"axis_dim {axis_dim} out of range for shape {shape}")
    return shape[axis_dim] % axis_size == 0


def describe_check(check: ShardCheck) -> str:
    ok = is_shardable(check.shape, check.axis_dim, check.axis_size)
    status = "OK" if ok else "NOT DIVISIBLE"
    return (
        f"{check.name}: shape={check.shape}, axis_dim={check.axis_dim}, "
        f"axis_size={check.axis_size} -> {status}"
    )


def main() -> None:
    checks = [
        ShardCheck("q_proj", (4096, 4096), 0, 8),
        ShardCheck("o_proj", (4096, 4096), 1, 8),
        ShardCheck("up_proj", (11008, 4096), 0, 7),
        ShardCheck("down_proj", (4096, 11008), 1, 7),
        ShardCheck("embed_tokens", (32000, 4096), 0, 8),
    ]

    print("Sharding divisibility checks")
    for check in checks:
        print(describe_check(check))

    print("\nJAX device_put sharding checks")
    devices = jax.devices()
    if len(devices) < 2:
        print(f"Only {len(devices)} device(s) available; need 2 to demo error.")
        return

    mesh = Mesh(np.array(devices[:2]).reshape((2,)), ("model",))
    sharding = NamedSharding(mesh, P("model"))

    ok_array = jax.numpy.zeros((4, 4), dtype=jax.numpy.float32)
    bad_array = jax.numpy.zeros((3, 4), dtype=jax.numpy.float32)

    with mesh:
        ok_result = jax.device_put(ok_array, sharding)
        print("OK device_put shape (4, 4)", ok_result.sharding)

        try:
            jax.device_put(bad_array, sharding)
            print("Unexpected: bad shape succeeded")
        except Exception as exc:  # noqa: BLE001 - demo script
            print("Expected error for shape (3, 4):", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
