from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = """\
from __future__ import annotations

import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def main():
    proc_id = int(os.environ["PROC_ID"])
    jax.distributed.initialize(
        coordinator_address=os.environ["COORDINATOR"],
        num_processes=int(os.environ["NUM_PROCESSES"]),
        process_id=proc_id,
        local_device_ids=[int(x) for x in os.environ["LOCAL_DEVICE_IDS"].split(",")],
        cluster_detection_method="deactivated",
    )

    print(f"proc {proc_id}: local_devices={jax.local_devices()}", flush=True)
    print(f"proc {proc_id}: global_devices={jax.devices()}", flush=True)

    mesh = Mesh(np.array(jax.devices()), ("model",))
    sharding = NamedSharding(mesh, P("model"))
    local_data = np.arange(proc_id * 4, proc_id * 4 + 4, dtype=np.int32)
    array = jax.make_array_from_process_local_data(sharding, local_data)

    print(
        f"proc {proc_id}: shape={array.shape} is_fully_addressable={array.is_fully_addressable}",
        flush=True,
    )
    print(
        "proc"
        f" {proc_id}: addressable_shards="
        f"{[(shard.device, np.asarray(shard.data).tolist()) for shard in array.addressable_shards]}",
        flush=True,
    )

    target_mesh = Mesh(np.array(jax.devices())[::-1], ("model",))
    target_sharding = NamedSharding(target_mesh, P("model"))
    try:
        target_array = jax.device_put(array, target_sharding)
        print(
            f"proc {proc_id}: device_put(array, target_sharding) ok "
            f"fully_addressable={target_array.is_fully_addressable} "
            f"devices={target_array.devices()}",
            flush=True,
        )
        print(
            "proc"
            f" {proc_id}: target_addressable_shards="
            f"{[(shard.device, np.asarray(shard.data).tolist()) for shard in target_array.addressable_shards]}",
            flush=True,
        )
    except Exception as err:
        print(
            "proc"
            f" {proc_id}: device_put(array, target_sharding) ERROR -> "
            f"{type(err).__name__}: {err}",
            flush=True,
        )

    array.copy_to_host_async()
    print(f"proc {proc_id}: copy_to_host_async ok", flush=True)

    try:
        host_value = jax.device_get(array)
        print(f"proc {proc_id}: device_get ok -> {host_value.tolist()}", flush=True)
    except Exception as err:
        print(
            f"proc {proc_id}: device_get ERROR -> {type(err).__name__}: {err}",
            flush=True,
        )

    try:
        host_value = np.asarray(array)
        print(f"proc {proc_id}: np.asarray ok -> {host_value.tolist()}", flush=True)
    except Exception as err:
        print(
            f"proc {proc_id}: np.asarray ERROR -> {type(err).__name__}: {err}",
            flush=True,
        )

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
"""


def main():
    worker_path = Path("/tmp/jax_mproc_cpu_worker.py")
    worker_path.write_text(SCRIPT)

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    env["NUM_PROCESSES"] = "2"
    env["COORDINATOR"] = "127.0.0.1:30111"

    procs = []
    for proc_id in ("0", "1"):
        proc_env = env.copy()
        proc_env["PROC_ID"] = proc_id
        proc_env["LOCAL_DEVICE_IDS"] = "0"
        procs.append(
            subprocess.Popen(
                [sys.executable, str(worker_path)],
                env=proc_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    for index, proc in enumerate(procs):
        stdout, stderr = proc.communicate(timeout=90)
        print(f"=== PROC {index} RET {proc.returncode} STDOUT ===")
        print(stdout, end="")
        print(f"=== PROC {index} STDERR ===")
        print(stderr, end="")


if __name__ == "__main__":
    main()
