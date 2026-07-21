"""Seed-trajectory generation for GESPO.

The initial-data stage of the pipeline: short biased/high-temperature MD that
produces the trajectories used to (1) pretrain the TorsionAR policy and (2) seed
the replay pool for ``gespo train``. Three methods, one shared system-building
core, explicit or implicit solvent:

- ``metad``     — well-tempered metadynamics (CV sets: holo / apo / folding)
- ``high-temp`` — staged-heat high-temperature MD
- ``rmsd-bias`` — RMSD-biased MD (adapter over ``gespo.bootstrap_cli``)

CLI:  ``gespo seed {metad,high-temp,rmsd-bias} --help``
API:  ``gespo.generate_seeds(method="metad", pdb=..., out_dir=..., ...)``

Imports here stay dependency-light (no OpenMM at import time); the MD stack is
pulled in lazily when a method actually runs.
"""
from __future__ import annotations

from typing import List

SEED_METHODS = ("metad", "high-temp", "rmsd-bias")
WALKER_METHODS = ("metad", "high-temp")


def generate_seeds(
    method: str,
    *,
    out_dir: str,
    n_walkers: int = 1,
    device: str = "cuda:0",
    bias_dir: str = "",
    **method_kwargs,
) -> List[str]:
    """Generate seed trajectories programmatically; return the walker DCD paths.

    Supports the walker methods ``metad`` and ``high-temp``. ``method_kwargs`` are
    forwarded to the method (e.g. ``pdb=``, ``cv_set=``, ``ligand_smiles=``,
    ``explicit=``, ``production_ns=``). For ``rmsd-bias`` use the CLI
    (``gespo seed rmsd-bias``) — it is argparse-driven.
    """
    m = str(method).strip().lower()
    if m not in WALKER_METHODS:
        raise ValueError(
            f"generate_seeds supports {WALKER_METHODS}; got {method!r}. "
            "For rmsd-bias use `gespo seed rmsd-bias` on the CLI."
        )
    from . import walkers
    return walkers.run_method(
        m, out_dir=out_dir, n_walkers=n_walkers, device=device,
        bias_dir=bias_dir, method_kwargs=method_kwargs)
