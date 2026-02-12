from jaxformers.distributed.parallel import mutate_sharding_rule_parallel_dims


def test_disable_sequence_parallelism_removes_tp_from_context():
    rules = {"context": ("tp", "cp"), "sequence": ("tp", "cp")}
    parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 2, "tp": 4}

    mutated = mutate_sharding_rule_parallel_dims(
        rules,
        parallel_dims,
        sequence_parallelism=False,
    )

    assert mutated["context"] == ("cp",)
    assert mutated["sequence"] == ("cp",)


def test_disable_sequence_parallelism_with_cp1_sets_context_to_none():
    rules = {"context": ("tp", "cp"), "sequence": ("tp", "cp")}
    parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 1, "tp": 4}

    mutated = mutate_sharding_rule_parallel_dims(
        rules,
        parallel_dims,
        sequence_parallelism=False,
    )

    assert mutated["context"] is None
    assert mutated["sequence"] == ("cp",)


def test_sequence_parallelism_enabled_keeps_context_tp_axis():
    rules = {"context": ("tp", "cp"), "sequence": ("tp", "cp")}
    parallel_dims = {"dp_replicate": 1, "dp_shard": 1, "cp": 2, "tp": 4}

    mutated = mutate_sharding_rule_parallel_dims(
        rules,
        parallel_dims,
        sequence_parallelism=True,
    )

    assert mutated["context"] == ("tp", "cp")
    assert mutated["sequence"] == ("tp", "cp")
