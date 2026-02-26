import pytest

from jaxformers.bench.sweep_utils import (
    build_sweep_space,
    SweepConfigError,
)


def test_build_sweep_space_keeps_singleton_dimension():
    overrides = {
        "learning_rate": 1e-5,
        "loss_implementation": ["xla_chunked", "reference"],
    }
    space = build_sweep_space(overrides=overrides, group=[])

    assert space.dimensions["learning_rate"] == [1e-5]
    assert space.dimensions["loss_implementation"] == ["xla_chunked", "reference"]
    assert space.run_count == 2


def test_build_sweep_space_group_multiplies_by_free_dimensions():
    space = build_sweep_space(
        overrides={"loss_implementation": ["xla_chunked", "reference"]},
        group=[
            {"learning_rate": 1e-4, "optimizer_name": "adam"},
            {"learning_rate": 1e-5, "optimizer_name": "sgd"},
        ],
    )

    assert space.run_count == 4
    runs = list(space.iter_runs())
    assert len(runs) == 4


def test_build_sweep_space_allows_any_key_name():
    space = build_sweep_space(
        overrides={"optmizer.b1": [0.9, 0.95]},
        group=[],
    )

    assert space.dimensions["optmizer.b1"] == [0.9, 0.95]


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

    assert space.dimensions["c.learning_rate"] == [1e-5]
    assert space.groups == [{"c.optimizer.b1": 0.8}]

def test_build_sweep_space_list_values_are_dimensions():
    space = build_sweep_space(
        overrides={"lora.weights_path": ["*.q_proj.weight", "*.k_proj.weight"]},
        group=[],
    )

    assert "lora.weights_path" in space.dimensions
    assert space.dimensions["lora.weights_path"] == [
        "*.q_proj.weight",
        "*.k_proj.weight",
    ]
