"""Auto-generate argparse arguments from dataclass fields.

Instead of manually writing hundreds of ``add_argument`` calls, annotate each
``@dataclass`` config with a ``CLI_META`` class variable and call
``populate_from_dataclass`` once per config class.
"""
from __future__ import annotations

import argparse
import dataclasses
from typing import Any, Dict


def _to_flag(name: str) -> str:
    """field_name → --field-name."""
    return "--" + name.replace("_", "-")


def populate_from_dataclass(
    parser: argparse.ArgumentParser,
    cls: type,
    group_name: str = "",
    cli_meta: dict | None = None,
) -> None:
    """Auto-generate argparse args from dataclass *cls*.

    Parameters
    ----------
    parser : ArgumentParser
    cls : a ``@dataclass`` class
    group_name : if given, wrap args in ``add_argument_group``
    cli_meta : mapping field name → help string **or** dict with keys:

        help, choices, aliases, nargs, metavar, flag, skip, flag_only,
        no_flag, no_help, mutex, action, default_override, type_override.

        Falls back to ``cls.CLI_META`` when *cli_meta* is None.
    """
    target = parser.add_argument_group(group_name) if group_name else parser
    meta: dict = cli_meta if cli_meta is not None else getattr(cls, "CLI_META", {})
    mutex_groups: Dict[str, argparse._MutuallyExclusiveGroup] = {}
    deferred_defaults: Dict[str, Any] = {}

    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue

        raw = meta.get(f.name, {})
        info: dict = {"help": raw} if isinstance(raw, str) else dict(raw)
        if info.get("skip"):
            continue

        help_text = info.get("help", f.name.replace("_", " "))
        flag = info.get("flag", _to_flag(f.name))
        aliases: list = list(info.get("aliases", []))
        flags = [flag] + aliases

        # Add underscore-style alias for backward compatibility
        underscore_flag = "--" + f.name
        if underscore_flag not in flags:
            flags.append(underscore_flag)

        # Target (mutex or normal group).
        mx = info.get("mutex")
        if mx:
            if mx not in mutex_groups:
                mutex_groups[mx] = target.add_mutually_exclusive_group()
            dest = mutex_groups[mx]
        else:
            dest = target

        # Resolve default.
        if "default_override" in info:
            default = info["default_override"]
        elif f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            continue  # sub-config / container field — skip
        else:
            default = None  # required field with no default

        # Detect booleans.
        type_str = str(f.type) if f.type else ""
        is_bool = isinstance(default, bool)
        is_optional_bool = default is None and "bool" in type_str

        if is_bool or is_optional_bool:
            dest.add_argument(
                *flags, dest=f.name, action="store_true", help=help_text,
            )
            if not info.get("flag_only"):
                nf = info.get("no_flag", "--no-" + flag.lstrip("-"))
                nh = info.get("no_help", f"Disable {f.name.replace('_', ' ')}.")
                dest.add_argument(nf, dest=f.name, action="store_false", help=nh)
            deferred_defaults[f.name] = default
        else:
            kw: Dict[str, Any] = {"dest": f.name, "help": help_text, "default": default}
            if "choices" in info:
                kw["choices"] = info["choices"]
            if "nargs" in info:
                kw["nargs"] = info["nargs"]
            if "metavar" in info:
                kw["metavar"] = info["metavar"]
            if "action" in info:
                kw["action"] = info["action"]
                # action=append doesn't take type
            elif "type_override" in info:
                kw["type"] = info["type_override"]
            elif isinstance(default, int):
                kw["type"] = int
            elif isinstance(default, float):
                kw["type"] = float
            elif default is None:
                if "float" in type_str:
                    kw["type"] = float
                elif "int" in type_str:
                    kw["type"] = int
            dest.add_argument(*flags, **kw)

    if deferred_defaults:
        parser.set_defaults(**deferred_defaults)
