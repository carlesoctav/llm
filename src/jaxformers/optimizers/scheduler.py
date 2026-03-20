def make_scheduler(scheduler_name: str | None, learning_rate):
    if scheduler_name in (None, "constant"):
        return learning_rate

    raise NotImplementedError(
        f"Unsupported lr scheduler {scheduler_name!r}; only constant is supported."
    )
