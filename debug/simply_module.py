#!/usr/bin/env python3
"""Reproduce explicit-TP failures in the full `simply` TransformerLM.

Run this with the `simply` virtualenv, for example:

  UV_CACHE_DIR=/mnt/carles/.cache /mnt/carles/simply/.venv/bin/python \
    /mnt/carles/llm/debug/simply_module.py --case all
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import traceback

DEFAULT_SIMPLY_ROOT = pathlib.Path("/mnt/carles/simply")


def _import_simply(simply_root: pathlib.Path):
    sys.path.insert(0, str(simply_root))
    from simply import config_lib, model_lib  # pylint: disable=import-error

    return config_lib, model_lib


def _base_config(config_lib, sharding_config=None):
    config = dataclasses.replace(
        config_lib.BaseExperimentConfig(),
        model_dim=16,
        per_head_dim=4,
        n_heads=8,
        n_layers=2,
        expand_factor=4,
        use_scan=False,
        use_flash_attention=False,
        activation_dtype_name="float32",
        vocab_size=64,
        seq_len=8,
        batch_size=2,
    )
    if sharding_config is not None:
        config = dataclasses.replace(config, sharding_config=sharding_config)
    return config


def _o_proj_sharding(config_lib):
    return dataclasses.replace(
        config_lib.gspmd_sharding(),
        ffn0_partition=(None, "model"),
        ffn1_partition=("model", None),
        attn_qkv_partition=(None, "model", None),
        attn_o_partition=(None, "model", None),
        embed_partition=(None, None),
        attn_activation_partition=(None, None, "model", None),
        activation_partition=(None, None, None),
        ffn0_activation_partition=(None, None, "model"),
        logits_partition=(None, None, None),
        data_partition=(None, None),
    )


def _ffn1_sharding(config_lib):
    return dataclasses.replace(
        _o_proj_sharding(config_lib),
        attn_o_partition=(None, None, None),
    )


def _make_mesh(jax):
    from jax.sharding import AxisType

    if len(jax.devices()) < 4:
        raise RuntimeError(
            f"Need at least 4 devices for explicit tp=4, got {len(jax.devices())}."
        )
    return jax.make_mesh(
        (1, 1, 4),
        ("replica", "data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        devices=jax.devices()[:4],
    )


def _run_model(config, model_lib, jax, jnp, js):
    model = model_lib.TransformerLM(config)
    params_key = jax.random.key(0)
    batch = jnp.arange(config.batch_size * config.seq_len, dtype=jnp.int32)
    batch = batch.reshape(config.batch_size, config.seq_len) % config.vocab_size
    mesh = _make_mesh(jax)

    with js.set_mesh(mesh):
        params = model.init(params_key)

        @jax.jit
        def run(params, batch):
            logits, _ = model.apply(params, batch)
            return logits

        logits = run(params, batch)
        logits.block_until_ready()
        return logits


def _print_case_header(name: str):
    print(f"\n=== {name} ===")


def _run_case(case: str, config, model_lib, jax, jnp, js):
    _print_case_header(case)
    try:
        logits = _run_model(config, model_lib, jax, jnp, js)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FAILED: {type(exc).__name__}")
        print(exc)
        print()
        print(traceback.format_exc())
        return False

    print("OK")
    print(f"shape={logits.shape}")
    print(f"sharding={logits.sharding}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simply-root",
        type=pathlib.Path,
        default=DEFAULT_SIMPLY_ROOT,
        help="Path to the local /mnt/carles/simply checkout.",
    )
    parser.add_argument(
        "--case",
        choices=("stock", "o_proj", "ffn1", "all"),
        default="all",
        help="Which full-model repro to run.",
    )
    args = parser.parse_args()

    config_lib, model_lib = _import_simply(args.simply_root)

    import jax
    import jax.numpy as jnp
    import jax.sharding as js

    jax.config.update("jax_threefry_partitionable", False)

    print(f"jax={jax.__version__}")
    print(f"simply_root={args.simply_root}")

    cases = {
        "stock": _base_config(config_lib),
        "o_proj": _base_config(config_lib, _o_proj_sharding(config_lib)),
        "ffn1": _base_config(config_lib, _ffn1_sharding(config_lib)),
    }

    if args.case == "all":
        selected = ("stock", "o_proj", "ffn1")
    else:
        selected = (args.case,)

    ok = True
    for name in selected:
        ok = _run_case(name, cases[name], model_lib, jax, jnp, js) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
