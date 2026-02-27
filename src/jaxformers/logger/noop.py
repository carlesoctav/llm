from typing import Any


class NoopLoggerConfig:
    def update(self, values: dict[str, Any]) -> None:
        _ = values


class NoopLogger:
    def __init__(self):
        self.config = NoopLoggerConfig()

    def log(self, values: dict[str, Any], *, step: int) -> None:
        _ = (values, step)

    def finish(self) -> None:
        return


def make(config):
    _ = config
    return NoopLogger()
