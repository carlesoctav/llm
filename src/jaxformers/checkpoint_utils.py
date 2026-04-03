import dataclasses
from typing import Any, Iterator

import jax
import orbax.checkpoint.experimental.v1 as ocp
from etils import epath

from jaxformers import tree_util
from jaxformers.modeling_utils import TrainState
from jaxformers.print_utils import tree_pformat, tree_pprint


def _make_save_decision_policy(save_every_steps: int) -> Any:
    return ocp.training.save_decision_policies.AnySavePolicy(
        [
            ocp.training.save_decision_policies.FixedIntervalPolicy(save_every_steps),
            ocp.training.save_decision_policies.PreemptionCheckpointingPolicy(),
            ocp.training.save_decision_policies.InitialSavePolicy(),
        ]
    )


def _make_preservation_policy(max_to_keep: int | None) -> Any:
    if max_to_keep is None:
        return ocp.training.preservation_policies.PreserveAll()
    return ocp.training.preservation_policies.LatestN(max_to_keep)


def _write_config_json(ckpt_path: str, config_json: str):
    if jax.process_index() != 0:
        return

    ckpt_dir = epath.Path(ckpt_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(config_json)


@dataclasses.dataclass
class CheckpointerWithInfo:
    ckptr: ocp.training.Checkpointer
    save_only_trainable: bool

    def close(self):
        self.ckptr.close()

    def load_checkpoint(self, model: TrainState) -> TrainState:
        if self.ckptr.latest is None:
            print("This is a new checkpoint; nothing will be loaded.")
            return model

        abstract = {}
        checkpoint = None
        # target = self.ckptr.checkpointables_metadata().metadata.keys()
        target = ["model", "opt_state"]
        info = self.ckptr.root_metadata().custom_metadata
        if info.get("save_only_trainable"):
            print("Checkpoint metadata indicates 'save_only_trainable=True'.")
            print(
                "Loading static (non-trainable) weights from step 0 and trainable weights from latest step."
            )
            for t in target:
                abstract[t] = tree_util.to_abstract(getattr(model, t))
            checkpoint_0 = self.ckptr.load_checkpointables(0, abstract)

            for t in target:
                abstract[t] = tree_util.to_abstract(getattr(model, t))
            checkpoint_N = self.ckptr.load_checkpointables(None, abstract)
            checkpoint = tree_util.combine(checkpoint_N, checkpoint_0)
        else:
            for t in target:
                abstract[t] = tree_util.to_abstract(getattr(model, t))
            checkpoint = self.ckptr.load_checkpointables(None, abstract)

        print(
            f"Replacing attributes {target} on the model with values loaded from the checkpoint."
        )

        return dataclasses.replace(model, **checkpoint, step=self.ckptr.latest.step)

    def save_checkpoint(
        self,
        step: int,
        train_state: TrainState,
        data: Iterator | None = None,
    ):
        if step in [c.step for c in self.ckptr.checkpoints]:
            print(
                f"Step {step} already exists; this is expected when resuming from a checkpoint (with config.checkpoint). Skipping save for step {step}."
            )
            return

        if step == 0:
            checkpointables = {
                "model": train_state.model,
                "opt_state": train_state.opt_state,
                "train_mask": train_state.train_mask,
            }

        else:
            if self.save_only_trainable:
                trainable, _ = tree_util.partition(
                    train_state.model, train_state.train_mask
                )
                checkpointables = {
                    "model": trainable,
                    "opt_state": train_state.opt_state,
                    "train_mask": train_state.train_mask,
                }
            else:
                checkpointables = {
                    "model": train_state.model,
                    "opt_state": train_state.opt_state,
                    "train_mask": train_state.train_mask,
                }

        return self.ckptr.save_checkpointables_async(step, checkpointables)


def make_checkpointer(
    train_state: TrainState,
    path: str,
    save_interval_steps: int,
    max_to_keep: int | None = None,
    save_only_trainable: bool = True,
) -> CheckpointerWithInfo | None:

    # im tsill not really sure about how to make a check for this train_mask
    things_to_check_pytree = ["train_mask"]
    things_to_check_args = ["max_to_keep", "save_only_trainable"]

    if is_used_checkpoint(path):
        old_ocp = ocp.training.Checkpointer(path)
        old_config = old_ocp.root_metadata().custom_metadata
        for item in things_to_check_args:
            existing = old_config.get(item)
            requested = locals().get(item)
            if existing != requested:
                raise ValueError(
                    f"Existing checkpoint config at {path} has {item}={existing}, "
                    f"which does not match the requested {item}={requested}. "
                    "Please use a matching configuration or remove the existing checkpoint."
                )
        for item in things_to_check_pytree:
            existing, existing_treedef = jax.tree.flatten(
                old_ocp.load_checkpointables(
                    0, {item: tree_util.to_abstract(getattr(train_state, item))}
                )[item]
            )
            requested, requested_treedef = jax.tree.flatten(getattr(train_state, item))
            if requested_treedef != existing_treedef:
                raise ValueError(
                    f"Structure mismatch for checkpoint item '{item}' in the existing checkpoint at {path}."
                    f"Checkpoint treedef {existing_treedef} does not match model treedef {requested_treedef}. "
                    "Please use a model with the same parameter structure or remove the existing checkpoint."
                )
            equal = jax.tree.all(jax.tree.map(lambda x, y: x == y, existing, requested))
            if not equal:
                raise ValueError(
                    f"Existing checkpoint item at {path} has {item} leaves={tree_pformat(existing)}, "
                    f"which does not match the requested {item} leaves={tree_pformat(requested)}. "
                    "Please use a matching configuration or remove the existing checkpoint."
                )
    if train_state.train_mask is None and save_only_trainable:
        raise ValueError(
            "save_only_trainable=True was requested but the model has no train_mask. "
            "Provide a valid model.train_mask indicating which parameters are trainable, "
            "or set save_only_trainable=False."
        )

    if train_state.train_mask is not None and save_only_trainable:
        leave = jax.tree.leaves(train_state.train_mask)
        trainable_size = sum(leave)
        print(f"save only {trainable_size} / {len(leave)} params")
        if max_to_keep:
            raise ValueError(
                "max_to_keep must be None when save_only_trainable is True. "
                "If only trainable parameters are saved, the static (non-trainable) weights are stored separately and only once no step 0, "
                "so a retention policy (max_to_keep) cannot be applied."
            )

    ckpt_config = {
        "path": path,
        "save_internal_steps": save_interval_steps,
        "max_to_keep": max_to_keep,
        "save_only_trainable": save_only_trainable,
        "config": train_state.model.get_config(),
    }

    ckptr = ocp.training.Checkpointer(
        path,
        save_decision_policy=_make_save_decision_policy(save_interval_steps),
        preservation_policy=_make_preservation_policy(max_to_keep),
        custom_metadata=ckpt_config,
    )

    return CheckpointerWithInfo(
        ckptr=ckptr,
        save_only_trainable=save_only_trainable,
    )


def is_used_checkpoint(path):
    ckptr = ocp.training.Checkpointer(path)
    if ckptr.latest:
        return True
    return False


def load_checkpoint_from_path(
    train_state: TrainState,
    path: str,
    target: list[str] = ["model", "opt_state"],
    step: int | None = None,
) -> TrainState:
    abstract = {}
    ckptr = ocp.training.Checkpointer(path)
    info = ckptr.root_metadata().custom_metadata
    if info.get("save_only_trainable"):
        print("Checkpoint metadata indicates 'save_only_trainable=True'.")
        print(
            f"Loading static (non-trainable) weights from step 0 and trainable weights from step {step if step is not None else ckptr.latest.step} from the checkpoint at {path}."
        )
        for t in target:
            abstract[t] = tree_util.to_abstract(getattr(train_state, t))
        checkpoint_0 = ckptr.load_checkpointables(0, abstract)
        for t in target:
            abstract[t] = tree_util.to_abstract(getattr(train_state, t))

        checkpoint_N = ckptr.load_checkpointables(step, abstract)
        checkpoint = tree_util.combine(checkpoint_N, checkpoint_0)
    else:
        print(
            f"Loading {target} from step {step if step is not None else ckptr.latest.step} at {path}"
        )
        for t in target:
            abstract[t] = tree_util.to_abstract(getattr(train_state, t))
        checkpoint = ckptr.load_checkpointables(step, abstract)

    print(
        f"Replacing attributes {target} on the model with values loaded from the checkpoint."
    )
    train_state = dataclasses.replace(train_state, **checkpoint)
    return train_state
