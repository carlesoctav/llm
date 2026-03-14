from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np


def reward_identity(
    rewards: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    epsilon: float = 1e-6,
    center: bool = False,
    scale: bool = False,
) -> np.ndarray:
    del group_ids, epsilon, center, scale
    return rewards.astype(np.float32, copy=False)


def reward_on_policy(
    rewards: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    epsilon: float = 1e-6,
    center: bool = True,
    scale: bool = True,
) -> np.ndarray:
    del group_ids
    normalized = rewards.astype(np.float32, copy=False)
    if center:
        normalized = normalized - normalized.mean()
    if scale:
        normalized = normalized / max(float(normalized.std()), epsilon)
    return normalized


def reward_grpo(
    rewards: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    epsilon: float = 1e-6,
    center: bool = True,
    scale: bool = True,
) -> np.ndarray:
    if group_ids is None:
        raise ValueError("GRPO reward processing requires group_ids.")

    normalized = rewards.astype(np.float32, copy=True)
    for group_id in np.unique(group_ids):
        group_mask = group_ids == group_id
        group_values = normalized[group_mask]
        if center:
            group_values = group_values - group_values.mean()
        if scale:
            group_values = group_values / max(float(group_values.std()), epsilon)
        normalized[group_mask] = group_values
    return normalized


def make_reward_postprocessor(
    reward_name: str | None,
    reward_config: dict | None = None,
) -> Callable[..., np.ndarray]:
    reward_config = reward_config or {}
    normalized_name = (reward_name or "none").lower()

    def _filter_kwargs(fn: Callable[..., np.ndarray]) -> dict:
        accepted = inspect.signature(fn).parameters
        return {
            key: value
            for key, value in reward_config.items()
            if key in accepted
        }

    if normalized_name in {"none", "identity"}:
        return lambda rewards, *, group_ids=None: reward_identity(
            rewards,
            group_ids=group_ids,
            **_filter_kwargs(reward_identity),
        )
    if normalized_name in {"on_policy", "onpolicy"}:
        return lambda rewards, *, group_ids=None: reward_on_policy(
            rewards,
            group_ids=group_ids,
            **_filter_kwargs(reward_on_policy),
        )
    if normalized_name == "grpo":
        return lambda rewards, *, group_ids=None: reward_grpo(
            rewards,
            group_ids=group_ids,
            **_filter_kwargs(reward_grpo),
        )

    raise ValueError(f"Unsupported reward_name: {reward_name!r}")
