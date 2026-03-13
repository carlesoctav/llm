from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from jaxformers.scheduler import make_scheduler


def test_make_scheduler_constant_passthrough():
    scheduler = make_scheduler(None, 1e-3, 1000)

    assert scheduler == 1e-3


def test_wsds_warmup_and_cosine_decay():
    scheduler = make_scheduler(
        "wsds",
        1e-2,
        1000,
        scheduler_config={
            "warmup": 0.1,
            "min_lr_ratio": 0.1,
            "lr_schedule": "cosine",
        },
    )

    assert np.isclose(scheduler(0), 0.0)
    assert np.isclose(scheduler(50), 0.5e-2)
    assert np.isclose(scheduler(100), 1e-2)
    assert np.isclose(scheduler(999), 1e-3, atol=1e-5)


def test_wsds_schedule_with_cycles():
    scheduler = make_scheduler(
        "wsds",
        1e-3,
        1000,
        scheduler_config={
            "warmup": 0.0,
            "decay": 0.1,
            "min_lr_ratio": 0.1,
            "lr_schedule": "cosine",
            "cycles": [300, 700],
        },
    )

    assert np.isclose(scheduler(0), 1e-3)
    assert np.isclose(scheduler(269), 1e-3)
    assert scheduler(271) < 1e-3
    assert np.isclose(scheduler(300), 1e-3)
    assert np.isclose(scheduler(659), 1e-3)
    assert scheduler(661) < 1e-3
    assert np.isclose(scheduler(701), 1e-3)
    assert np.isclose(scheduler(969), 1e-3)
    assert scheduler(971) < 1e-3


def test_wsds_schedule_with_rewarmup():
    scheduler = make_scheduler(
        "wsds",
        1e-2,
        1000,
        scheduler_config={
            "warmup": 0.2,
            "min_lr_ratio": 0.2,
            "lr_schedule": "linear",
            "cycles": 2,
            "rewarmup": 0.05,
        },
    )

    assert np.isclose(scheduler(0), 0.0)
    assert np.isclose(scheduler(100), 1e-2)
    assert np.isclose(scheduler(300), 0.6e-2)
    assert np.isclose(scheduler(500), 0.2e-2, atol=1e-5)

    rewarmup_start = 500
    rewarmup_end = rewarmup_start + int(0.05 * 500)
    assert np.isclose(scheduler(rewarmup_start), 0.2e-2, atol=1e-5)
    assert np.isclose(scheduler(rewarmup_end), 1e-2)
    assert scheduler(rewarmup_end - 1) < scheduler(rewarmup_end)
    assert scheduler(rewarmup_end + 1) < scheduler(rewarmup_end)
    assert scheduler(998) > scheduler(999)
    assert np.isclose(scheduler(999), 0.2e-2, atol=1e-4)
