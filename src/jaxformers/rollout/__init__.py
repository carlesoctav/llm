import importlib


def make_rollout_engine(
    rollout_name: str,
    *,
    model,
    rollout_config: dict,
    params=None,
    forward_dtype=None,
):
    rollout_module = importlib.import_module(f"jaxformers.rollout.{rollout_name}")
    return rollout_module.make(
        model=model,
        rollout_config=rollout_config,
        params=params,
        forward_dtype=forward_dtype,
    )
