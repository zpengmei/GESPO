"""Utility helpers shared across training and sampling.

Keep imports lightweight: some utility modules depend on optional heavy deps
(e.g., OpenMM). Import submodules lazily via attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["config", "constants", "endpoints", "geometry", "openmm_utils", "selection"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name not in __all__:
        raise AttributeError(name)
    if name == "openmm_utils":
        try:
            mod = importlib.import_module(f"{__name__}.openmm_utils")
        except Exception:
            mod = None
        globals()[name] = mod
        return mod
    mod = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = mod
    return mod
