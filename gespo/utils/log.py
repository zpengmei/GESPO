"""Lightweight logging seam for GESPO training loops.

Progress and warnings are routed through ``print`` so that the stdout tee set
up in ``cli.py`` still captures everything to the run log. This module is a
single choke point: a later switch to the ``logging`` module (with a
``StreamHandler(sys.stdout)`` to preserve the tee) need only change this file,
not the hundreds of call sites.

The epoch banners give a run a legible stage narrative — before this, only
"Phase B" and "Phase E" were ever labeled among the loop's ~230 prints.
"""
from __future__ import annotations

# Human-readable names for the six epoch phases (see BaseTrainingLoop).
PHASE_NAMES = {
    "A": "Refresh projector",
    "B": "Sample starts",
    "C": "Run MD trajectories",
    "D": "Compute rewards",
    "E": "Policy update",
    "F": "Log / checkpoint",
}


def stage(msg: str = "") -> None:
    """Emit a stage/progress line."""
    print(msg, flush=True)


def warn(msg: str = "") -> None:
    """Emit a warning/diagnostic line."""
    print(msg, flush=True)


def epoch_phase(epoch: int, total: int, phase: str, msg: str = "") -> None:
    """Announce entry into an epoch phase, e.g. ``[Epoch 3/20][Phase C] ...``."""
    label = PHASE_NAMES.get(phase, "")
    detail = msg or label
    line = f"[Epoch {epoch}/{total}][Phase {phase}]"
    if detail:
        line += f" {detail}"
    print(line, flush=True)
