"""Collective-variable registry for metadynamics seeding.

Two layers:

- a **pure** spec layer (``CV_SETS`` / ``resolve_cv_set``) with no OpenMM import,
  so the CLI and unit tests can reason about CV sets; and
- ``build_bias_variables`` which turns specs into OpenMM ``BiasVariable`` objects
  exactly as the proven ``metad_seeds_holo.py`` / ``metad_seeds_apo.py`` scripts do.

Named CV sets (each metadynamics run uses two CVs):

- ``holo``    — pocket Cα RMSD  +  ligand↔pocket COM distance   (bound-complex poses)
- ``apo``     — pocket Cα RMSD  +  global Cα RMSD                (loop + global motion)
- ``folding`` — global Cα RMSD  +  end-to-end distance           (fold/unfold)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CVSpec:
    """One metadynamics CV: an OpenMM force kind plus its bias-grid bounds."""
    kind: str          # pocket_rmsd | global_rmsd | lig_pocket_dist | e2e
    min_nm: float
    max_nm: float
    width_nm: float
    periodic: bool = False


# Per-kind default bounds/width, taken from the source seed scripts.
_CV_DEFAULTS: Dict[str, CVSpec] = {
    "pocket_rmsd":     CVSpec("pocket_rmsd",     0.0, 0.40, 0.020),
    "global_rmsd":     CVSpec("global_rmsd",     0.0, 0.50, 0.020),
    "lig_pocket_dist": CVSpec("lig_pocket_dist", 0.15, 1.0, 0.030),
    "e2e":             CVSpec("e2e",             0.5, 4.0, 0.050),
}

# Named two-CV sets. Order is (CV1, CV2).
CV_SETS: Dict[str, Tuple[str, str]] = {
    "holo":    ("pocket_rmsd", "lig_pocket_dist"),
    "apo":     ("pocket_rmsd", "global_rmsd"),
    "folding": ("global_rmsd", "e2e"),
}

# Which CV kinds need which inputs (drives CLI validation / auto-detection).
_KINDS_NEED_POCKET = frozenset({"pocket_rmsd", "lig_pocket_dist"})
_KINDS_NEED_LIGAND = frozenset({"lig_pocket_dist"})


def resolve_cv_set(
    name: str,
    *,
    cv1_max_nm: float | None = None,
    cv2_min_nm: float | None = None,
    cv2_max_nm: float | None = None,
) -> List[CVSpec]:
    """Return the two ``CVSpec`` for a named set, applying optional bound overrides.

    ``cv1_max_nm`` overrides CV1's upper bound; ``cv2_min_nm`` / ``cv2_max_nm``
    override CV2's bounds — matching the ``--cv1-max-nm`` / ``--cv2-*`` knobs of
    the original scripts.
    """
    key = str(name or "").strip().lower()
    if key not in CV_SETS:
        valid = ", ".join(sorted(CV_SETS))
        raise ValueError(f"Unknown --cv-set '{name}'. Valid CV sets: {valid}")
    cv1, cv2 = (_CV_DEFAULTS[k] for k in CV_SETS[key])
    if cv1_max_nm is not None:
        cv1 = replace(cv1, max_nm=float(cv1_max_nm))
    if cv2_min_nm is not None:
        cv2 = replace(cv2, min_nm=float(cv2_min_nm))
    if cv2_max_nm is not None:
        cv2 = replace(cv2, max_nm=float(cv2_max_nm))
    return [cv1, cv2]


def cv_set_needs_pocket(name: str) -> bool:
    return any(k in _KINDS_NEED_POCKET for k in CV_SETS.get(str(name).lower(), ()))


def cv_set_needs_ligand(name: str) -> bool:
    return any(k in _KINDS_NEED_LIGAND for k in CV_SETS.get(str(name).lower(), ()))


def build_bias_variables(specs, *, pos_ref, all_ca, pocket_ca, lig_heavy, periodic_box: bool):
    """Turn ``CVSpec`` into OpenMM ``BiasVariable`` objects (lazy OpenMM import).

    ``pos_ref`` is the post-minimization reference positions for RMSD CVs.
    ``all_ca`` / ``pocket_ca`` / ``lig_heavy`` are atom-index lists in the built
    system. ``periodic_box`` toggles PBC-aware distance CVs (explicit solvent).
    Distance/RMSD forces are *not* added to the System here — OpenMM's
    ``Metadynamics`` wraps and adds them itself, exactly as the source scripts do.
    """
    import openmm as mm
    from openmm import app, unit

    _NM = unit.nanometer
    variables = []
    for spec in specs:
        if spec.kind == "pocket_rmsd":
            force = mm.RMSDForce(pos_ref, pocket_ca)
        elif spec.kind == "global_rmsd":
            force = mm.RMSDForce(pos_ref, all_ca)
        elif spec.kind == "lig_pocket_dist":
            force = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
            force.addGroup(lig_heavy)
            force.addGroup(pocket_ca)
            force.addBond([0, 1])
            force.setUsesPeriodicBoundaryConditions(bool(periodic_box))
        elif spec.kind == "e2e":
            # End-to-end distance: first Cα ↔ last Cα.
            force = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
            force.addGroup([int(all_ca[0])])
            force.addGroup([int(all_ca[-1])])
            force.addBond([0, 1])
            force.setUsesPeriodicBoundaryConditions(bool(periodic_box))
        else:  # pragma: no cover - guarded by resolve_cv_set
            raise ValueError(f"Unknown CV kind '{spec.kind}'")
        variables.append(app.metadynamics.BiasVariable(
            force,
            minValue=spec.min_nm * _NM,
            maxValue=spec.max_nm * _NM,
            biasWidth=spec.width_nm * _NM,
            periodic=bool(spec.periodic),
        ))
    return variables
