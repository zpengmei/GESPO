"""Dependency-light helpers shared across seed methods.

Deliberately free of OpenMM/mdtraj imports so the CLI, walker fan-out, and unit
tests can use it without the scientific stack installed. The MD-heavy builders
live in ``system.py`` / ``restraints.py`` / ``cvs.py``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List


# Residue names treated as solvent/ions (stripped before re-solvation) and as
# candidate ligand residues. Mirrors the constants in the proven seed scripts and
# (for later consolidation) gespo/sampling/env.py.
SOLVENT_NAMES = frozenset({
    "HOH", "WAT", "TIP3", "TIP3P", "SOL", "NA", "CL", "K", "CA2", "MG", "ZN",
})
LIGAND_NAMES = frozenset({"UNK", "BEN", "LIG"})
AA_NAMES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS",
    "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "ACE", "NME",
})


def ns_to_steps(ns: float, dt_fs: float) -> int:
    """Convert a duration in nanoseconds to an integer MD step count.

    Identical to the ``_steps`` helper in the source seed scripts, so ported
    protocols run for exactly the same number of steps.
    """
    return int(round(float(ns) * 1_000_000.0 / float(dt_fs)))


def log(msg: str) -> None:
    """Timestamped progress line (matches the seed scripts' ``_log``)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def walker_dir(out_dir, index: int) -> Path:
    """Per-walker output directory ``<out_dir>/walker_<index>`` (1-indexed)."""
    return Path(out_dir) / f"walker_{index}"


def walker_dirs(out_dir, n_walkers: int) -> List[Path]:
    """The list of per-walker directories a run will populate."""
    return [walker_dir(out_dir, i) for i in range(1, int(n_walkers) + 1)]


def production_dcd(walker_dir_path) -> Path:
    """The DCD each walker writes — the file pretrain/train consume."""
    return Path(walker_dir_path) / "production.dcd"


def resolve_devices(device_spec: str, n_walkers: int) -> List[str]:
    """Expand a device spec into one device string per walker.

    Accepts a single device (``cuda:0`` → round-robins all walkers onto it) or a
    comma-separated list (``cuda:0,cuda:1`` → walkers cycle across the list).
    """
    spec = str(device_spec or "cuda:0").strip()
    devices = [d.strip() for d in spec.split(",") if d.strip()] or ["cuda:0"]
    return [devices[i % len(devices)] for i in range(int(n_walkers))]
