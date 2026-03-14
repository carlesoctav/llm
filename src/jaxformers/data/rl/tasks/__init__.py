import importlib

from .base import DatasetWithReward


def make_task(task_name: str, **task_config) -> DatasetWithReward:
    task_module = importlib.import_module(f"jaxformers.data.rl.tasks.{task_name}")
    return task_module.make(**task_config)
