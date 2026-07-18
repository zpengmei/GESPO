"""GESPO – modularized generative enhanced sampling toolkit.

Curated public API. Everything below is imported lazily (PEP 562 ``__getattr__``),
so ``import gespo`` stays cheap and does not pull in OpenMM/torch until you
actually touch one of these names.

Typical usage::

    import gespo

    # Train from a YAML file, a dict, or a TrainingConfig
    gespo.train("config.yaml")

    # Pretrain a TorsionAR policy from trajectory data
    gespo.pretrain_torsion_ar(
        source_path="history_dcds/",
        out_path="checkpoints/r0.pt",
        pdb_path="endpoint_A.pdb",
    )

    # Build/inspect configuration
    cfg = gespo.load_config("config.yaml")
    print(gespo.generate_template("generative"))
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "train",
    "pretrain_torsion_ar",
    "TrainingConfig",
    "load_config",
    "save_config",
    "generate_template",
    "__version__",
]

# Map each public name to the submodule that defines it. Kept as (module, attr)
# so ``__getattr__`` can import the submodule only on first access.
_LAZY_EXPORTS = {
    "train": ("gespo.api", "train"),
    "pretrain_torsion_ar": ("gespo.api", "pretrain_torsion_ar"),
    "TrainingConfig": ("gespo.config", "TrainingConfig"),
    "load_config": ("gespo.config", "load_config"),
    "save_config": ("gespo.config", "save_config"),
    "generate_template": ("gespo.config", "generate_template"),
}


def _resolve_version() -> str:
    """Best-effort package version, falling back to the pinned value."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("gespo")
        except PackageNotFoundError:
            return "0.1.0"
    except Exception:
        return "0.1.0"


def __getattr__(name: str):
    """Lazily resolve curated public names (PEP 562)."""
    if name == "__version__":
        value = _resolve_version()
        globals()["__version__"] = value  # cache
        return value
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'gespo' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value  # cache so repeated access skips the import machinery
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    # Give static analyzers / IDEs the real symbols without triggering the
    # heavy imports at runtime.
    from gespo.api import pretrain_torsion_ar, train
    from gespo.config import (
        TrainingConfig,
        generate_template,
        load_config,
        save_config,
    )
