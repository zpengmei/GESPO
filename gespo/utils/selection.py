"""Topology/structure selection helpers (protein vs solvent/ions/lipids).

These utilities exist to make GESPO robust to "full system" PDB/DCD inputs that
contain water, ions, lipids, and other non-protein residues (e.g., membrane
systems). Most learning components in this repo assume a protein(-like) solute
atom set.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple

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

# Schuyler's edits (with Claude Code) for constructing TMEM63 metrics
# ---------------------------------------------------------------------------
# Residue-number -> atom-index resolution, at a configurable atom-selection
# granularity. This is the shared "build once from a topology" layer that
# per-system global-token metric files (adk_metrics.py, tmem63_metrics.py,
# ...) resolve their residue-based groups/pairs through, instead of each
# hand-rolling its own (previously identical, copy-pasted) version of these
# functions.
# ---------------------------------------------------------------------------

class AtomSelectionMode(str, Enum):
    """How much of a matched residue's atoms to include in a selection."""

    BACKBONE = "backbone"        # N, CA, C, O (+ OXT)
    BACKBONE_CB = "backbone_cb"  # backbone + CB (legacy ADK/TMEM63 default)
    ALL_HEAVY = "all_heavy"      # every non-hydrogen atom
    ALL_ATOM = "all_atom"        # every atom, including hydrogens

_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "OXT"}

def _iter_topology_atoms(topology) -> Iterable:
    atoms_attr = getattr(topology, "atoms", None)
    if atoms_attr is None:
        return ()
    if callable(atoms_attr):
        try:
            return atoms_attr()
        except Exception:
            return ()
    try:
        return iter(atoms_attr)
    except TypeError:
        return ()

def _iter_topology_residues(topology) -> Iterable:
    residues_attr = getattr(topology, "residues", None)
    if residues_attr is None:
        return ()
    if callable(residues_attr):
        try:
            return residues_attr()
        except Exception:
            return ()
    try:
        return iter(residues_attr)
    except TypeError:
        return ()

def resolve_residue_number(res, *, use_resseq: bool) -> Optional[int]:
    """Return a residue's number as int, preferring PDB resSeq when requested,
    falling back to the residue's 0-based topology index."""
    if use_resseq:
        val = getattr(res, "resSeq", None)
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(res, "index"))
    except (TypeError, ValueError, AttributeError):
        return None

def atom_is_backbone(atom) -> bool:
    try:
        return bool(getattr(atom, "is_backbone"))
    except Exception:
        return str(getattr(atom, "name", "") or "").strip().upper() in _BACKBONE_ATOM_NAMES

def atom_is_hydrogen(atom) -> bool:
    element = getattr(atom, "element", None)
    symbol = getattr(element, "symbol", None)
    if symbol is not None:
        return str(symbol) == "H"
    # Fall back to name-based heuristic when element info isn't resolved (CHARMM-GUI
    # hydrogen names are H-prefixed, e.g. HN1, HB2, HG1 -- see step5_input.psf).
    return str(getattr(atom, "name", "") or "").strip().upper().startswith("H")

def atom_matches_selection_mode(atom, mode: AtomSelectionMode) -> bool:
    """Whether `atom` belongs to the given AtomSelectionMode."""
    if mode == AtomSelectionMode.ALL_ATOM:
        return True
    if mode == AtomSelectionMode.ALL_HEAVY:
        return not atom_is_hydrogen(atom)
    is_bb = atom_is_backbone(atom)
    if mode == AtomSelectionMode.BACKBONE:
        return is_bb
    if mode == AtomSelectionMode.BACKBONE_CB:
        name = str(getattr(atom, "name", "") or "").strip().upper()
        return is_bb or name == "CB"
    raise ValueError(f"unknown AtomSelectionMode: {mode!r}")

def expand_residue_range(start: int, end: int) -> Tuple[int, ...]:
    """Turn an inclusive (start, end) residue range into an explicit tuple of
    residue numbers, so downstream selection code only ever handles one shape
    of input (a set of residue numbers) instead of both ranges and lists."""
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    return tuple(range(start, end + 1))

def select_atoms_for_residues(
    topology,
    residues: Sequence[int],
    mode: AtomSelectionMode,
    *,
    use_resseq: bool,
) -> np.ndarray:
    """Select atom indices for an explicit set of residue numbers, at a given
    atom-selection granularity.

    Args:
        topology: MDTraj-like Topology (.atoms, .residues; each residue has
            .resSeq and/or .index; each atom has .name and, ideally, .element).
        residues: explicit residue numbers to include. Use
            expand_residue_range() first if you have a contiguous range.
        mode: which atoms within each matched residue to keep.
        use_resseq: interpret `residues` as PDB resSeq values if True, else as
            0-based residue index.

    Returns:
        Sorted, deduplicated int64 array of global atom indices. Empty if no
        residue in `residues` was found in the topology, or none of their
        atoms satisfied `mode`. Callers building a required group should
        treat an empty result as a configuration error (see
        build_tmem63_token_indices's completeness checks) rather than silently
        proceeding with a group that resolves to nothing.
    """
    residue_set = {int(r) for r in residues}
    out: list[int] = []
    for atom in _iter_topology_atoms(topology):
        res = getattr(atom, "residue", None)
        if res is None:
            continue
        resnum = resolve_residue_number(res, use_resseq=use_resseq)
        if resnum is None or resnum not in residue_set:
            continue
        if atom_matches_selection_mode(atom, mode):
            out.append(int(getattr(atom, "index")))
    return np.asarray(sorted(set(out)), dtype=np.int64)

def find_ca_atom(topology, *, residue_num: int, use_resseq: bool) -> Optional[int]:
    """Find the CA atom index for a single residue, by resSeq or 0-based index.

    Intentionally does not fall back to "first atom in the residue" when CA is
    missing: in explicit-solvent systems, resSeq values can coincide with
    water/ion residues too, and a silent fallback would treat e.g. a water
    oxygen as "CA" and produce a meaningless metric with no warning.
    """
    target = int(residue_num)
    for res in _iter_topology_residues(topology):
        resnum = resolve_residue_number(res, use_resseq=use_resseq)
        if resnum is None or resnum != target:
            continue
        for atom in getattr(res, "atoms", ()):
            if str(getattr(atom, "name", "") or "").strip().upper() == "CA":
                return int(getattr(atom, "index"))
        return None
    return None
