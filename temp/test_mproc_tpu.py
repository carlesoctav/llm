from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


WORKER = """\
from __future__ import annotations

import os

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P


def main():
    proc_id = int(os.environ["PROC_ID"])
    print(
        f"proc {proc_id}: visible={os.environ['TPU_VISIBLE_CHIPS']} "
        f"process_bounds={os.environ['TPU_PROCESS_BOUNDS']} "
        f"chips_per_process={os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS']}",
        flush=True,
    )

    mesh = Mesh(
        np.array(jax.devices()),
        ("model",),
        axis_types=(AxisType.Auto,),
    )
    sharding = NamedSharding(mesh, P("model"))
    local_data = np.arange(proc_id * 4, proc_id * 4 + 4, dtype=np.int32)
    array = jax.make_array_from_process_local_data(sharding, local_data)

    print(f"proc {proc_id}: local_devices={jax.local_devices()}", flush=True)
    print(f"proc {proc_id}: global_devices={jax.devices()}", flush=True)
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


if __name__ == "__main__":
    main()
"""


def main():
    worker_path = Path("/tmp/jax_mproc_tpu_worker.py")
    worker_path.write_text(WORKER)

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "tpu"
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,2,1"
    env["TPU_PROCESS_BOUNDS"] = "1,2,1"
    env["TPU_PROCESS_ADDRESSES"] = "127.0.0.1:30211,127.0.0.1:30212"

    chip_groups = {"0": "0,1", "1": "2,3"}
    proc_ports = {"0": "30211", "1": "30212"}
    procs = []
    for proc_id in ("0", "1"):
        proc_env = env.copy()
        proc_env["PROC_ID"] = proc_id
        proc_env["CLOUD_TPU_TASK_ID"] = proc_id
        proc_env["TPU_PROCESS_PORT"] = proc_ports[proc_id]
        proc_env["TPU_VISIBLE_CHIPS"] = chip_groups[proc_id]
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
        stdout, stderr = proc.communicate(timeout=180)
        print(f"=== PROC {index} RET {proc.returncode} STDOUT ===")
        print(stdout, end="")
        print(f"=== PROC {index} STDERR ===")
        print(stderr, end="")


if __name__ == "__main__":
    main()
