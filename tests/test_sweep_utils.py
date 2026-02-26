import pytest

from jaxformers.bench.sweep_utils import (
    build_sweep_space,
    flatten_overrides,
    parse_maybe_expr,
    SweepConfigError,
)


def test_flatten_overrides_nested_mapping():
    overrides = {
        "learning_rate": 1e-4,
        "optimizer": {"max_grad_norm": 1.0, "grad_accum": 2},
    }
    assert flatten_overrides(overrides) == {
        "learning_rate": 1e-4,
        "optimizer.max_grad_norm": 1.0,
        "optimizer.grad_accum": 2,
    }


def test_parse_maybe_expr_range_with_floats():
    assert parse_maybe_expr("range(1e-5, 1e-4, 5e-5)") == [1e-5, 6e-5]


def test_parse_maybe_expr_range_object():
    assert parse_maybe_expr(range(512, 1025, 512)) == [512, 1024]


def test_build_sweep_space_keeps_singleton_dimension():
    base_store = {
        "learning_rate": 1e-5,
        "loss_implementation": "xla_chunked",
    }
    overrides = {
        "learning_rate": 1e-5,
        "loss_implementation": ["xla_chunked", "reference"],
    }
    space = build_sweep_space(base_store=base_store, overrides=overrides, group=[])

    assert space.fixed["learning_rate"] == 1e-5
    assert space.dimensions["loss_implementation"] == ["xla_chunked", "reference"]
    assert space.run_count == 2


def test_build_sweep_space_group_multiplies_by_free_dimensions():
    base_store = {
        "learning_rate": 1e-5,
        "optimizer_name": "adam",
        "loss_implementation": "xla_chunked",
    }
    space = build_sweep_space(
        base_store=base_store,
        overrides={"loss_implementation": ["xla_chunked", "reference"]},
        group=[
            {"learning_rate": 1e-4, "optimizer_name": "adam"},
            {"learning_rate": 1e-5, "optimizer_name": "sgd"},
        ],
    )

    assert space.run_count == 4
    runs = list(space.iter_runs())
    assert len(runs) == 4


def test_build_sweep_space_requires_exact_free_keys():
    base_store = {"learning_rate": 1e-5}
    with pytest.raises(SweepConfigError):
        build_sweep_space(
            base_store=base_store,
            overrides={"leanring_rate": [1e-5, 1e-4]},
            group=[],
        )


def test_build_sweep_space_requires_exact_group_keys():
    base_store = {"optimizer.b1": 0.9}
    with pytest.raises(SweepConfigError):
        build_sweep_space(
            base_store=base_store,
            overrides={},
            group=[{"optmizer.b1": 0.9}],
        )


def test_build_sweep_space_allows_new_leaf_under_existing_parent():
    base_store = {"optimizer.max_grad_norm": 1.0}
    space = build_sweep_space(
        base_store=base_store,
        overrides={"optimizer.b1": [0.9, 0.95]},
        group=[],
    )

    assert space.dimensions["optimizer.b1"] == [0.9, 0.95]


def test_build_sweep_space_rejects_overlap_between_group_and_free_dims():
    base_store = {"learning_rate": 1e-5, "optimizer_name": "adam"}
    with pytest.raises(SweepConfigError):
        build_sweep_space(
            base_store=base_store,
            overrides={"learning_rate": [1e-5, 1e-4]},
            group=[{"learning_rate": 1e-5, "optimizer_name": "adam"}],
        )


def test_build_sweep_space_accepts_c_prefix():
    base_store = {"optimizer.b1": 0.9, "learning_rate": 1e-5}
    space = build_sweep_space(
        base_store=base_store,
        overrides={"c.learning_rate": "range(1e-5,1e-4,5e-5)"},
        group=[{"c.optimizer.b1": 0.8}],
    )

    assert space.dimensions["learning_rate"] == [1e-5, 6e-5]
    assert space.groups == [{"optimizer.b1": 0.8}]


def test_build_sweep_space_list_base_is_fixed_when_list_value():
    base_store = {"data.load_kwargs": [{"path": "x"}]}
    space = build_sweep_space(
        base_store=base_store,
        overrides={"data.load_kwargs": [{"path": "a"}, {"path": "b"}]},
        group=[],
    )

    assert "data.load_kwargs" in space.fixed
    assert "data.load_kwargs" not in space.dimensions


def test_build_sweep_space_list_base_sweeps_on_list_of_lists():
    base_store = {"lora.weights_path": ["*.q_proj.weight"]}
    space = build_sweep_space(
        base_store=base_store,
        overrides={"lora.weights_path": [["*.q_proj.weight"], ["*.k_proj.weight"]]},
        group=[],
    )

    assert "lora.weights_path" in space.dimensions
    assert space.dimensions["lora.weights_path"] == [
        ["*.q_proj.weight"],
        ["*.k_proj.weight"],
    ]
