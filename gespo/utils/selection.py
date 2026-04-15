"""Topology/structure selection helpers (protein vs solvent/ions/lipids).

These utilities exist to make GESPO robust to "full system" PDB/DCD inputs that
contain water, ions, lipids, and other non-protein residues (e.g., membrane
systems). Most learning components in this repo assume a protein(-like) solute
atom set.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# Common residue name aliases seen in PDBs / forcefields.
WATER_RESNAMES = {
    "HOH",
    "WAT",
    "SOL",
    "TIP3",
    "TIP3P",
    "TIP4",
    "TIP4P",
    "SPC",
    "SPCE",
}

ION_RESNAMES = {
    # Alkali/halides
    "NA",
    "NA+",
    "SOD",
    "CL",
    "CL-",
    "CLA",
    "K",
    "K+",
    "LI",
    "LI+",
    "RB",
    "RB+",
    "CS",
    "CS+",
    # Common divalent / metal ions
    "MG",
    "MG2+",
    "CA",
    "CA2+",
    "ZN",
    "ZN2+",
    "FE",
    "FE2+",
    "FE3+",
    "CU",
    "CU2+",
    "MN",
    "MN2+",
}

# Minimal-but-practical lipid set; expand as new systems are encountered.
LIPID_RESNAMES = {
    # Observed in civsd/init_data/civsd-0-initial.pdb
    "POP",
    # Common 3-letter lipid residue names (CHARMM/AMBER/Slipids variants)
    "POPC",
    "POPE",
    "POPG",
    "POPS",
    "DPPC",
    "DPPE",
    "DOPC",
    "DOPE",
    "DLPC",
    "DLPE",
    "DSPC",
    "DSPA",
    "DPPG",
    "CHOL",
    "CHL1",
    "CHL",
    "CER",
    "CER1",
    "SM",
}

CAP_RESNAMES = {
    "ACE",
    "NME",
    "NH2",
    "NHE",
    "NHE2",
}


def _resname(obj) -> str:
    try:
        return str(getattr(obj, "name", "") or "").strip().upper()
    except Exception:
        return ""


def _iter_atoms(residue) -> Iterable:
    atoms_attr = getattr(residue, "atoms", None)
    if atoms_attr is None:
        return ()
    try:
        # openmm.app.Topology.Residue exposes atoms() -> iterator
        if callable(atoms_attr):
            return atoms_attr()
    except Exception:
        pass
    # mdtraj residue exposes atoms -> list/iterator
    try:
        return iter(atoms_attr)
    except TypeError:
        return ()


def residue_has_backbone(residue) -> bool:
    """Heuristic protein-like residue test: contains N, CA, and C atom names."""

    try:
        names = {str(getattr(atom, "name", "") or "").strip().upper() for atom in _iter_atoms(residue)}
    except Exception:
        return False
    return {"N", "CA", "C"}.issubset(names)


def residue_is_solvent_like(residue) -> bool:
    """True for water/ion/lipid residues by name (best-effort)."""

    rn = _resname(residue)
    return rn in WATER_RESNAMES or rn in ION_RESNAMES or rn in LIPID_RESNAMES


def protein_like_residue_filter(*, include_caps: bool = True):
    """Return a predicate that keeps protein(-like) residues and optional caps."""

    def _keep(residue) -> bool:
        rn = _resname(residue)
        if rn in WATER_RESNAMES or rn in ION_RESNAMES or rn in LIPID_RESNAMES:
            return False
        if include_caps and rn in CAP_RESNAMES:
            return True
        return residue_has_backbone(residue)

    return _keep


def infer_protein_atom_indices(
    topology,
    *,
    include_caps: bool = True,
    allow_empty: bool = False,
) -> np.ndarray:
    """Best-effort protein(-like) atom selection for an mdtraj.Topology.

    Strategy:
    1) Use MDTraj's selection language when available (fast, standard).
    2) Union with a backbone-based heuristic to recover nonstandard residues.
    3) If still empty, optionally fall back to all atoms.
    """

    idx_sel = np.zeros((0,), dtype=np.int64)

    select = getattr(topology, "select", None)
    if callable(select):
        expr = "protein"
        if include_caps:
            expr = "protein or resname ACE or resname NME or resname NH2 or resname NHE or resname NHE2"
        try:
            idx = select(expr)
            if idx is not None:
                idx_sel = np.asarray(idx, dtype=np.int64).reshape(-1)
        except Exception:
            idx_sel = np.zeros((0,), dtype=np.int64)

    # Heuristic residue-based fallback (also used to recover nonstandard protein residues).
    keep_res = protein_like_residue_filter(include_caps=include_caps)
    idx_heur: list[int] = []
    try:
        for atom in getattr(topology, "atoms", []):
            if keep_res(getattr(atom, "residue", None)):
                idx_heur.append(int(getattr(atom, "index", -1)))
    except Exception:
        idx_heur = []
    idx_heur_arr = np.asarray([i for i in idx_heur if i >= 0], dtype=np.int64)

    if idx_sel.size == 0:
        idx = idx_heur_arr
    elif idx_heur_arr.size == 0:
        idx = idx_sel
    else:
        idx = np.unique(np.concatenate([idx_sel, idx_heur_arr], axis=0))

    if idx.size == 0 and not allow_empty:
        try:
            n_atoms = int(getattr(topology, "n_atoms", 0) or 0)
            idx = np.arange(n_atoms, dtype=np.int64)
        except Exception:
            idx = np.zeros((0,), dtype=np.int64)
    return idx.astype(np.int64, copy=False)


def is_subset_selection(selection: Optional[Sequence[int]], total_atoms: int) -> bool:
    if selection is None:
        return False
    try:
        return 0 < int(len(selection)) < int(total_atoms)
    except Exception:
        return False

