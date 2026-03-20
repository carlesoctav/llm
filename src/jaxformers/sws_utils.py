from __future__ import annotations

import functools
import inspect
import keyword
import os
import runpy
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sws


class ConfigLoadError(RuntimeError):
    pass


def load_config_builder(
    config_path: str, *, default_func: str = "get_config"
) -> sws.Config:
    path = config_path
    func_name = default_func
    if ":" in path:
        path, func_name = path.split(":", 1)

    path = os.path.abspath(path)
    factory = runpy.run_path(path).get(func_name)
    if not callable(factory):
        raise ConfigLoadError(f"Function {func_name!r} not found in {path}")
    builder = factory()
    if not isinstance(builder, sws.Config):
        raise ConfigLoadError(f"Config factory in {path} must return a sws.Config")
    return builder


def _make_cell(value: Any):
    return (lambda x: lambda: x)(value).__closure__[0]


def _view_for_prefix(config: sws.Config, prefix: str) -> sws.Config:
    if not prefix:
        return config

    view = config
    for part in prefix.removesuffix(".").split("."):
        view = view[part]
    return view


def _rebind_value(value: Any, *, source: sws.Config, target: sws.Config) -> Any:
    if isinstance(value, sws.Config) and value._store is source._store:
        return _view_for_prefix(target, value._prefix)

    if isinstance(value, sws.Fn):
        return sws.Fn(_rebind_value(value.fn, source=source, target=target))

    if isinstance(value, types.FunctionType):
        return _rebind_function(value, source=source, target=target)

    if isinstance(value, functools.partial):
        return functools.partial(
            _rebind_value(value.func, source=source, target=target),
            *[_rebind_value(item, source=source, target=target) for item in value.args],
            **{
                key: _rebind_value(item, source=source, target=target)
                for key, item in (value.keywords or {}).items()
            },
        )

    if isinstance(value, tuple):
        return tuple(
            _rebind_value(item, source=source, target=target) for item in value
        )

    if isinstance(value, list):
        return [_rebind_value(item, source=source, target=target) for item in value]

    if isinstance(value, dict):
        return {
            key: _rebind_value(item, source=source, target=target)
            for key, item in value.items()
        }

    if isinstance(value, set):
        return {_rebind_value(item, source=source, target=target) for item in value}

    return value


def _rebind_function(fn: types.FunctionType, *, source: sws.Config, target: sws.Config):
    closure = fn.__closure__ or ()
    new_closure = []
    changed = False

    for cell in closure:
        try:
            current = cell.cell_contents
        except ValueError:
            new_closure.append(cell)
            continue

        rebound = _rebind_value(current, source=source, target=target)
        if rebound is not current:
            changed = True
            new_closure.append(_make_cell(rebound))
        else:
            new_closure.append(cell)

    defaults = fn.__defaults__
    rebound_defaults = None
    if defaults is not None:
        rebound_defaults = tuple(
            _rebind_value(item, source=source, target=target) for item in defaults
        )
        changed = changed or rebound_defaults != defaults

    kwdefaults = fn.__kwdefaults__
    rebound_kwdefaults = None
    if kwdefaults is not None:
        rebound_kwdefaults = {
            key: _rebind_value(item, source=source, target=target)
            for key, item in kwdefaults.items()
        }
        changed = changed or rebound_kwdefaults != kwdefaults

    if not changed:
        return fn

    rebound = types.FunctionType(
        fn.__code__,
        fn.__globals__,
        name=fn.__name__,
        argdefs=rebound_defaults,
        closure=tuple(new_closure),
    )
    rebound.__kwdefaults__ = rebound_kwdefaults
    rebound.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rebound.__dict__.update(getattr(fn, "__dict__", {}))
    rebound.__doc__ = fn.__doc__
    rebound.__module__ = fn.__module__
    rebound.__qualname__ = fn.__qualname__
    return rebound


def merge_config_builders(
    config_paths: Sequence[str], *, default_func: str = "get_config"
) -> sws.Config:
    merged = sws.Config()
    for config_path in config_paths:
        source = load_config_builder(config_path, default_func=default_func)
        for key, value in source.to_flat_dict().items():
            merged[key] = _rebind_value(value, source=source, target=merged)
    return merged


def clone_builder_value(source: sws.Config, target: sws.Config, key: str) -> Any:
    return _rebind_value(source._store[key], source=source, target=target)


def _format_config_target(root_name: str, key: str) -> str:
    expr = root_name
    for part in key.split("."):
        if part.isidentifier() and not keyword.iskeyword(part):
            expr += f".{part}"
        else:
            expr += f"[{part!r}]"
    return expr


def combine_and_write(
    write_path: str | os.PathLike[str],
    *config_paths: str | os.PathLike[str],
    default_func: str = "get_config",
) -> str:
    if not config_paths:
        raise ValueError("combine_and_write requires at least one config path")

    resolved_paths = [os.path.abspath(os.fspath(path)) for path in config_paths]
    output_path = Path(os.path.abspath(os.fspath(write_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "from __future__ import annotations",
        "",
        "import sws",
        "",
        "from jaxformers.sws_utils import clone_builder_value, load_config_builder",
        "",
        f"CONFIG_PATHS = {resolved_paths!r}",
        "",
        "def get_config() -> sws.Config:",
        f"    config = load_config_builder(CONFIG_PATHS[0], default_func={default_func!r})",
    ]

    for idx, config_path in enumerate(resolved_paths[1:], start=1):
        source = load_config_builder(config_path, default_func=default_func)
        source_var = f"_config_{idx}"
        lines.append(
            f"    {source_var} = load_config_builder(CONFIG_PATHS[{idx}], default_func={default_func!r})"
        )
        for key in source.to_flat_dict():
            target = _format_config_target("config", key)
            lines.append(
                f"    {target} = clone_builder_value({source_var}, config, {key!r})"
            )

    lines.extend(
        [
            "    return config",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    return str(output_path)


def _looks_like_config_path(token: str) -> bool:
    path = token.split(":", 1)[0]
    return path.endswith(".py") or path.startswith("gs://")


def _split_config_args(
    argv: Sequence[str], *, config_flag: str
) -> tuple[list[str], list[str]]:
    config_paths: list[str] = []
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        token = argv[i]
        if token == config_flag:
            i += 1
            start = len(config_paths)
            while i < len(argv):
                next_token = argv[i]
                if next_token == config_flag or next_token.startswith(
                    config_flag + "="
                ):
                    break
                if next_token == "--" or "=" in next_token:
                    break
                if not _looks_like_config_path(next_token):
                    break
                config_paths.append(next_token)
                i += 1
            if len(config_paths) == start:
                raise ValueError(f"{config_flag} requires at least one config path")
            continue

        if token.startswith(config_flag + "="):
            raw_value = token.split("=", 1)[1]
            parts = [part.strip() for part in raw_value.split(",") if part.strip()]
            if not parts:
                raise ValueError(f"{config_flag} requires at least one config path")
            config_paths.extend(parts)
            i += 1
            continue

        remaining.append(token)
        i += 1

    return config_paths, remaining


def _load_default_builder(caller_file: str, *, default_func: str) -> sws.Config:
    if caller_file and os.path.exists(caller_file):
        factory = runpy.run_path(caller_file).get(default_func, lambda: sws.Config())
        if not callable(factory):
            raise AttributeError(
                f"Function {default_func!r} not found in {caller_file}"
            )
        builder = factory()
        if not isinstance(builder, sws.Config):
            raise TypeError("Config factory must return a sws.Config")
        return builder

    return sws.Config()


def run(
    main,
    *,
    argv: Sequence[str] | None = None,
    config_flag: str = "--config",
    default_func: str = "get_config",
    forward_extras: bool = False,
):
    args = list(sys.argv[1:] if argv is None else argv)
    config_paths, args = _split_config_args(args, config_flag=config_flag)

    if config_paths:
        builder = merge_config_builders(config_paths, default_func=default_func)
    else:
        caller_file = inspect.stack()[1].filename
        builder = _load_default_builder(caller_file, default_func=default_func)

    final, unused = builder.finalize(args, return_unused_argv=True)
    if forward_extras:
        return main(final, unused)
    if unused:
        raise ValueError(f"Unused extra arguments: {unused}")
    return main(final)
