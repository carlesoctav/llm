import pytest

from jaxformers.sweep_utils import (
    build_sweep_space,
    SweepConfigError,
)


def test_build_sweep_space_keeps_singleton_dimension():
    overrides = {
        "learning_rate": 1e-5,
        "loss_implementation": ["xla_chunked", "reference"],
    }
    space = build_sweep_space(overrides=overrides, group=[])

    assert len(space) == 2
    runs = [run for _, run in space]
    assert all(run["learning_rate"] == 1e-5 for run in runs)
    assert {run["loss_implementation"] for run in runs} == {"xla_chunked", "reference"}


def test_build_sweep_space_group_multiplies_by_free_dimensions():
    space = build_sweep_space(
        overrides={"loss_implementation": ["xla_chunked", "reference"]},
        group=[
            {"learning_rate": 1e-4, "optimizer_name": "adam"},
            {"learning_rate": 1e-5, "optimizer_name": "sgd"},
        ],
    )

    assert len(space) == 4
    groups = [group_idx for group_idx, _ in space]
    assert groups.count(0) == 2
    assert groups.count(1) == 2


def test_build_sweep_space_allows_any_key_name():
    space = build_sweep_space(
        overrides={"optmizer.b1": [0.9, 0.95]},
        group=[],
    )

    values = [run["optmizer.b1"] for _, run in space]
    assert values == [0.9, 0.95]


def test_build_sweep_space_rejects_overlap_between_group_and_free_dims():
    with pytest.raises(SweepConfigError):
        build_sweep_space(
            overrides={"learning_rate": [1e-5, 1e-4]},
            group=[{"learning_rate": 1e-5, "optimizer_name": "adam"}],
        )


def test_build_sweep_space_accepts_c_prefix():
    space = build_sweep_space(
        overrides={"c.learning_rate": 1e-5},
        group=[{"c.optimizer.b1": 0.8}],
    )

    assert len(space) == 1
    group_idx, run = space[0]
    assert group_idx == 0
    assert run["c.learning_rate"] == 1e-5
    assert run["c.optimizer.b1"] == 0.8

def test_build_sweep_space_list_values_are_dimensions():
    space = build_sweep_space(
        overrides={"lora.weights_path": ["*.q_proj.weight", "*.k_proj.weight"]},
        group=[],
    )

    assert len(space) == 2
    values = [run["lora.weights_path"] for _, run in space]
    assert values == ["*.q_proj.weight", "*.k_proj.weight"]
