from typing import Any

import orbax.checkpoint.experimental.v1 as ocp

from jaxformers import tree_util
from jaxformers.modeling_utils import Model


def _make_save_decision_policy(save_every_steps: int, save_on_steps) -> Any:
    policies = [
        ocp.training.save_decision_policies.FixedIntervalPolicy(save_every_steps),
        ocp.training.save_decision_policies.PreemptionCheckpointingPolicy(),
        ocp.training.save_decision_policies.InitialSavePolicy(),
    ]
    if save_on_steps:
        policies.append(
            ocp.training.save_decision_policies.SpecificStepsPolicy(save_on_steps)
        )
    return ocp.training.save_decision_policies.AnySavePolicy(policies)


def _make_preservation_policy(max_to_keep, keep_period, keep_time_interval) -> Any:
    policies = []
    if max_to_keep is not None:
        policies.append(ocp.training.preservation_policies.LatestN(max_to_keep))
    if keep_period is not None:
        policies.append(ocp.training.preservation_policies.EveryNSteps(keep_period))
    if keep_time_interval is not None:
        interval_secs = int(keep_time_interval.total_seconds())
        policies.append(
            ocp.training.preservation_policies.EveryNSeconds(interval_secs)
        )
    if not policies:
        return ocp.training.preservation_policies.PreserveAll()
    if len(policies) == 1:
        return policies[0]
    return ocp.training.preservation_policies.AnyPreservationPolicy(policies)


def make_checkpointer(config) -> ocp.training.Checkpointer:
    checkpoint_options = config.checkpoint_options.to_dict()
    save_every_steps = config.eval_every
    configured_save_every = checkpoint_options["save_interval_steps"]

    if save_every_steps is None:
        raise ValueError("checkpoint saving requires config.eval_every to be set")
    if configured_save_every != save_every_steps:
        raise ValueError(
            "config.checkpoint_options.save_interval_steps must equal config.eval_every"
        )

    save_decision_policy = _make_save_decision_policy(
        save_every_steps,
        checkpoint_options.get("save_on_steps"),
    )
    preservation_policy = _make_preservation_policy(
        checkpoint_options.get("max_to_keep"),
        checkpoint_options.get("keep_period"),
        checkpoint_options.get("keep_time_interval"),
    )
    return ocp.training.Checkpointer(
        config.ckpt_path,
        save_decision_policy=save_decision_policy,
        preservation_policy=preservation_policy,
    )


def _weights_to_save(model: Model, save_trainable: bool):
    if save_trainable and model.train_mask is not None:
        weights, _ = tree_util.partition(model.weights, model.train_mask)
        return weights
    return model.weights


def save_checkpoint(
    checkpointer: ocp.training.Checkpointer,
    step: int,
    model: Model,
    *,
    to_save: tuple[str, ...] = ("weights", "opt_state"),
    save_trainable: bool = True,
):
    checkpointables = {}
    for item_name in to_save:
        if item_name == "weights":
            checkpointables["weights"] = _weights_to_save(model, save_trainable)
        elif item_name == "opt_state":
            checkpointables["opt_state"] = model.opt_state
        else:
            raise ValueError(f"Unsupported checkpoint item: {item_name}")
    return checkpointer.save_checkpointables_async(step, checkpointables)
