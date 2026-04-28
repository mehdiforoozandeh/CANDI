"""YAML + strict `SandboxConfig` merge; dotted-path CLI helpers."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, MutableMapping, Optional, Tuple, Type, get_args, get_origin, get_type_hints

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from sandbox.config_types import SandboxConfig, _coerce_container, _coerce_scalar, _strip_optional, config_from_dict


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for sandbox configs (`pip install pyyaml`).")
    with Path(path).open("r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    if out is None:
        return {}
    if not isinstance(out, dict):
        raise ValueError(f"Config root must be a mapping, got {type(out)}")
    return dict(out)


def dump_yaml(cfg: SandboxConfig, path: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, default_flow_style=False)


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def merge_yaml_files(paths: List[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for p in paths:
        merged = deep_merge(merged, load_yaml(p))
    return merged


def load_sandbox_config(
    *yaml_paths: Path,
    cli_overrides: Optional[Mapping[str, Any]] = None,
) -> SandboxConfig:
    """default.yaml → overlays → optional flat/dotted CLI overrides dict."""
    defaults = asdict(SandboxConfig())
    merged = defaults
    for p in yaml_paths:
        merged = deep_merge(merged, load_yaml(Path(p)))
    if cli_overrides:
        merged = deep_merge(merged, dict(cli_overrides))
    return config_from_dict(SandboxConfig, merged)


def _get_in(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for p in path:
        cur = cur[p]
    return cur


def _field_type(dc: Type[Any], path: Tuple[str, ...]):
    """Return (parent dataclass, Field object with resolved type) for a dotted path."""
    cur_cls = dc
    for i, name in enumerate(path):
        flds = {f.name: f for f in fields(cur_cls)}
        if name not in flds:
            raise KeyError(name)
        f = flds[name]
        hints = get_type_hints(cur_cls)
        rtype = hints[f.name]
        inner = _strip_optional(rtype)
        if i == len(path) - 1:
            return (cur_cls, f)
        if not is_dataclass(inner):
            raise TypeError(f"path {path}: {name} is not a nested dataclass")
        cur_cls = inner
    raise RuntimeError("unreachable")


def add_sandbox_cli_args(parser: argparse.ArgumentParser, defaults: SandboxConfig) -> None:
    """Register `--group.subfield` for every leaf in `SandboxConfig` (plan §10.2)."""
    d = asdict(defaults)

    def _is_literal(tp: Any) -> bool:
        o = get_origin(_strip_optional(tp))
        return o is Literal or (o is not None and getattr(o, "__name__", "") == "Literal")

    def walk(prefix: Tuple[str, ...], dc_cls: Type[Any]) -> None:
        hints = get_type_hints(dc_cls)
        for f in fields(dc_cls):
            p = prefix + (f.name,)
            dest = ".".join(p)
            dest_attr = dest.replace(".", "__")
            inner = _strip_optional(hints[f.name])
            if is_dataclass(inner):
                walk(p, inner)
                continue
            cur = _get_in(d, p)
            flag = f"--{dest}"
            if inner is bool:
                parser.add_argument(
                    flag,
                    dest=dest_attr,
                    choices=("true", "false"),
                    default=argparse.SUPPRESS,
                    help=f"bool (default {cur})",
                )
            elif get_origin(hints[f.name]) is list or get_origin(inner) is list:
                parser.add_argument(
                    flag,
                    dest=dest_attr,
                    default=argparse.SUPPRESS,
                    help=f"comma-separated list (default {cur})",
                )
            elif _is_literal(hints[f.name]):
                choices = list(get_args(_strip_optional(hints[f.name])))
                parser.add_argument(
                    flag,
                    dest=dest_attr,
                    choices=[str(c) for c in choices],
                    default=argparse.SUPPRESS,
                    help=f"(default {cur!r})",
                )
            elif inner is int:
                parser.add_argument(flag, dest=dest_attr, type=int, default=argparse.SUPPRESS)
            elif inner is float:
                parser.add_argument(flag, dest=dest_attr, type=float, default=argparse.SUPPRESS)
            elif inner is str:
                parser.add_argument(flag, dest=dest_attr, type=str, default=argparse.SUPPRESS)
            else:
                parser.add_argument(flag, dest=dest_attr, type=str, default=argparse.SUPPRESS)

    walk((), SandboxConfig)


def overrides_from_parsed_args(args: argparse.Namespace) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}

    def set_path(path: Tuple[str, ...], value: Any) -> None:
        root = tree
        for p in path[:-1]:
            root = root.setdefault(p, {})
        root[path[-1]] = value

    for k, v in vars(args).items():
        if "__" not in k or v is None:
            continue
        path = tuple(k.split("__"))
        parent, fld = _field_type(SandboxConfig, path)
        hints = get_type_hints(parent)
        ft = hints[fld.name]
        inner = _strip_optional(ft)
        if inner is bool:
            val = str(v).lower() in ("true", "1", "yes")
        elif get_origin(ft) is list or get_origin(inner) is list:
            val = _coerce_container(ft, v)
        else:
            val = _coerce_scalar(ft, v)
        set_path(path, val)
    return tree


# --- legacy shallow merge (used by prepare_h5 / tests) ---
def merge_dict(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out
