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

        # Resolve the concrete default. This is the SINGLE SOURCE for the
        # dataclass field's value — but it is used only for help text and type
        # inference, never as the argparse default. Every flag defaults to None
        # (the "unset" sentinel) so that TrainingConfig.from_args can tell an
        # unspecified flag from an explicit one and let the dataclass supply the
        # value. This is what makes ``train --config x.yaml`` respect YAML for
        # flags the user did not pass (instead of clobbering them with CLI
        # defaults).
        if "default_override" in info:
            resolved_default = info["default_override"]
        elif f.default is not dataclasses.MISSING:
            resolved_default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            continue  # sub-config / container field — skip
        else:
            resolved_default = None  # required field with no default

        # Surface the real default in --help (argparse's own default is None).
        if (
            help_text is not None
            and help_text is not argparse.SUPPRESS
            and resolved_default not in (None, "")
        ):
            help_text = f"{help_text} (default: {resolved_default})"

        # Detect booleans.
        type_str = str(f.type) if f.type else ""
        is_bool = isinstance(resolved_default, bool)
        is_optional_bool = resolved_default is None and "bool" in type_str

        if is_bool or is_optional_bool:
            dest.add_argument(
                *flags, dest=f.name, action="store_true", help=help_text,
            )
            if not info.get("flag_only"):
                nf = info.get("no_flag", "--no-" + flag.lstrip("-"))
                nh = info.get("no_help", f"Disable {f.name.replace('_', ' ')}.")
                dest.add_argument(nf, dest=f.name, action="store_false", help=nh)
            # Unset sentinel: neither --x nor --no-x passed → None (not the
            # concrete default), so from_args falls back to the dataclass value.
            deferred_defaults[f.name] = None
        else:
            kw: Dict[str, Any] = {"dest": f.name, "help": help_text, "default": None}
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
            elif isinstance(resolved_default, bool):
                pass  # handled above
            elif isinstance(resolved_default, int):
                kw["type"] = int
            elif isinstance(resolved_default, float):
                kw["type"] = float
            elif resolved_default is None:
                if "float" in type_str:
                    kw["type"] = float
                elif "int" in type_str:
                    kw["type"] = int
            dest.add_argument(*flags, **kw)

    if deferred_defaults:
        parser.set_defaults(**deferred_defaults)
