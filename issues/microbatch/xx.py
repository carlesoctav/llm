import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclasses.dataclass(frozen=True)
class ReproConfig:
    num_layers: int = 26
    hidden_size: int = 1152
    intermediate_size: int = 6912
    microbatch_size: int = 8
    accum_steps: int = 4
    seq_len: int = 2048
    learning_rate: float = 1e-4
    mesh_devices: int = 4
    dump_hlo: bool = True

    @property
    def logical_batch_size(self) -> int:
        return self.microbatch_size * self.accum_steps


def infer_shardings(tree):
    def leaf_sharding(leaf):
        if isinstance(leaf, jax.Array):
            return leaf.sharding
        return None

    return jtu.tree_map(leaf_sharding, tree)


def make_state_shardings(params, opt_state, mesh: Mesh):
    replicated = NamedSharding(mesh, P())
    param_shardings = infer_shardings(params)
    shape_to_sharding = {}
    for param_leaf, sharding_leaf in zip(
        jtu.tree_leaves(params), jtu.tree_leaves(param_shardings), strict=True
    ):
        if isinstance(param_leaf, jax.Array):
            shape_to_sharding.setdefault(tuple(param_leaf.shape), sharding_leaf)

    def opt_leaf_sharding(leaf):
        if not isinstance(leaf, jax.Array):
            return None
        return shape_to_sharding.get(tuple(leaf.shape), replicated)

    opt_state_shardings = jtu.tree_map(opt_leaf_sharding, opt_state)
    return (param_shardings, opt_state_shardings)


def print_compiled_memory_stats(compiled_stats):
    if compiled_stats is None:
        return None

    def bytes_to_gb(num_bytes):
        return num_bytes / (1024**3)

    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb

    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
    )

    return {
        "total_gb": round(total_gb, 1),
        "output_gb": round(output_gb, 1),
        "temp_gb": round(temp_gb, 1),
        "argument_gb": round(argument_gb, 1),
        "host_temp_gb": round(host_temp_gb, 1),
    }


def print_flops(compiled_stats):
    tflops = compiled_stats.get("flops") / 1e12
    print(f"estimated tflops per step: {tflops}")
    return {"tflops": tflops}


def validate_mode_config(mode: str, config: ReproConfig):
    if mode == "single":
        return
    if config.logical_batch_size % config.mesh_devices != 0:
        raise ValueError(
            "`logical_batch_size` must be divisible by `mesh_devices` for dp/fsdp."
        )
    if config.microbatch_size % config.mesh_devices != 0:
        raise ValueError(
            "`microbatch_size` must be divisible by `mesh_devices` for dp/fsdp."
        )


def select_devices(mode: str, mesh_devices: int):
    devices = jax.devices()
    if mode == "single":
        return devices[:1]
    if mesh_devices < 2:
        raise ValueError("`mesh_devices` must be >= 2 for dp/fsdp modes.")
    if len(devices) < mesh_devices:
        raise ValueError(
            f"Requested {mesh_devices} devices, but only found {len(devices)}."
        )
    return devices[:mesh_devices]


def make_mesh(mode: str, mesh_devices: int):
    devices = np.asarray(select_devices(mode, mesh_devices))
    return Mesh(devices, ("data",))


def make_shardings(mode: str, mesh: Mesh):
    replicated = NamedSharding(mesh, P())
    batch_sharded = NamedSharding(mesh, P("data", None, None))
    fsdp_up = NamedSharding(mesh, P(None, "data"))
    fsdp_down = NamedSharding(mesh, P("data", None))

    if mode == "single":
        return {
            "batch": replicated,
            "gate": replicated,
            "up": replicated,
            "down": replicated,
        }
    if mode == "dp":
        return {
            "batch": batch_sharded,
            "gate": replicated,
            "up": replicated,
            "down": replicated,
        }
    if mode == "fsdp":
        return {
            "batch": batch_sharded,
            "gate": fsdp_up,
            "up": fsdp_up,
            "down": fsdp_down,
        }
    raise ValueError(f"Unknown mode: {mode}")


def make_params(config: ReproConfig, shardings):
    params = []
    for _ in range(config.num_layers):
        gate = jnp.full(
            (config.intermediate_size, config.hidden_size),
            1e-2,
            dtype=jnp.bfloat16,
        )
        up = jnp.full(
            (config.intermediate_size, config.hidden_size),
            1e-2,
            dtype=jnp.bfloat16,
        )
        down = jnp.full(
            (config.hidden_size, config.intermediate_size),
            1e-2,
            dtype=jnp.bfloat16,
        )
        params.append(
            (
                jax.device_put(gate, shardings["gate"]),
                jax.device_put(up, shardings["up"]),
                jax.device_put(down, shardings["down"]),
            )
        )
    return tuple(params)


def make_batch(batch_size: int, config: ReproConfig, batch_sharding):
    x = jnp.full(
        (batch_size, config.seq_len, config.hidden_size),
        1e-2,
        dtype=jnp.bfloat16,
    )
    return jax.device_put(x, batch_sharding)


def mlp_forward(params, x):
    h = x
    for gate_w, up_w, down_w in params:
        gate = jnp.einsum("bth,fh->btf", h, gate_w)
        up = jnp.einsum("bth,fh->btf", h, up_w)
        hidden = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
        out = jnp.einsum("btf,hf->bth", hidden.astype(h.dtype), down_w)
        h = h + out
    return h


def loss_fn(params, batch):
    y = mlp_forward(params, batch)
    return jnp.mean(jnp.square(y.astype(jnp.float32)))


def make_microbatch_train_step(config: ReproConfig):
    tx = optax.sgd(config.learning_rate)

    def train_step(state, batch):
        params, opt_state = state
        grad_fn = optax.microbatch(
            jax.value_and_grad(loss_fn),
            argnums=1,
            microbatch_size=config.microbatch_size,
        )
        loss, grads = grad_fn(params, batch)
        grads = jtu.tree_map(lambda g: g / config.accum_steps, grads)
        loss = loss / config.accum_steps
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    return train_step, tx


def make_multistep_train_step(config: ReproConfig):
    tx = optax.MultiSteps(
        optax.sgd(config.learning_rate),
        every_k_schedule=config.accum_steps,
        use_grad_mean=True,
    )

    def train_step(state, batch):
        params, opt_state = state
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    return train_step, tx


def compile_and_report(label, train_step, state, batch, dump_path: Path | None):
    print(f"{label}:")
    step_jit = jax.jit(
        train_step,
        donate_argnums=(0,),
        in_shardings=(infer_shardings(state), infer_shardings(batch)),
        out_shardings=(infer_shardings(state), None),
    )
    lower = step_jit.lower(state, batch)
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(lower.as_text())
    compiled = lower.compile()
    print_compiled_memory_stats(compiled.memory_analysis())
    print_flops(compiled.cost_analysis())
    state, loss = compiled(state, batch)
    loss.block_until_ready()
    return state, float(loss)


def run_microbatch(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_step, tx = make_microbatch_train_step(config)
        opt_state = tx.init(params)
        state = (params, opt_state)
        # state = jax.device_put((params, opt_state), make_state_shardings(params, opt_state, mesh))
        batch = make_batch(config.logical_batch_size, config, shardings["batch"])
        hlo_path = (
            Path("issues/microbatch/hlo") / f"{mode}_microbatch.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/microbatch",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/microbatch loss={loss:.6f}")


def run_multistep(mode: str, config: ReproConfig):
    validate_mode_config(mode, config)
    mesh = make_mesh(mode, config.mesh_devices)
    shardings = make_shardings(mode, mesh)
    with mesh:
        params = make_params(config, shardings)
        train_step, tx = make_multistep_train_step(config)
        opt_state = tx.init(params)
        state = jax.device_put((params, opt_state), make_state_shardings(params, opt_state, mesh))
        batch = make_batch(config.microbatch_size, config, shardings["batch"])
        hlo_path = (
            Path("issues/microbatch/hlo") / f"{mode}_multistep.stablehlo"
            if config.dump_hlo
            else None
        )
        _, loss = compile_and_report(
            f"{mode}/multistep",
            train_step,
            state,
            batch,
            hlo_path,
        )
        print(f"{mode}/multistep loss={loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single", "dp", "fsdp"],
        choices=["single", "dp", "fsdp"],
    )
    parser.add_argument("--num-layers", type=int, default=26)
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--intermediate-size", type=int, default=6912)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--mesh-devices", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-dump-hlo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ReproConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        seq_len=args.seq_len,
        microbatch_size=args.microbatch_size,
        accum_steps=args.accum_steps,
        learning_rate=args.learning_rate,
        mesh_devices=args.mesh_devices,
        dump_hlo=not args.no_dump_hlo,
    )

    print(
        "config:",
        {
            "num_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "seq_len": config.seq_len,
            "microbatch_size": config.microbatch_size,
            "accum_steps": config.accum_steps,
            "logical_batch_size": config.logical_batch_size,
            "mesh_devices": config.mesh_devices,
            "available_devices": len(jax.devices()),
        },
    )

    for mode in args.modes:
        print(f"\n=== mode={mode} ===")
        run_microbatch(mode, config)
        run_multistep(mode, config)


if __name__ == "__main__":
    main()
