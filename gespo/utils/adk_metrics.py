"""ADK-specific global metrics (domain angles/distances) for global tokens.

This module provides utilities to compute:
  - Two hinge angles (NMP/LID) from centroids of residue groups
  - Two inter-residue distances (CA-CA)
  - RMSD to a reference (typically CA RMSD, computed elsewhere via indices)

All distances are in nm; angles are in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .geometry import calculate_rmsd


ADK_GLOBAL_TOKEN_NAMES: Tuple[str, ...] = (
    "rmsd_nm",
    "theta_nmp_rad",
    "theta_lid_rad",
    "dist_145_52_nm",
    "dist_169_55_nm",
)

# ADK domain-only globals (no RMSD). Useful when you want purely geometric hinge/dist proxies
# without tying the policy/reward to endpoint RMSD.
ADK_DOMAIN_TOKEN_NAMES: Tuple[str, ...] = ADK_GLOBAL_TOKEN_NAMES[1:]

_BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}


@dataclass(frozen=True)
class ADKTokenIndices:
    """Precomputed atom indices (local to protein coords arrays)."""

    # Atom indices for centroid groups (local indices into protein-only coords).
    group_atoms: Dict[str, np.ndarray]
    # Pair atom indices for distances (local indices into protein-only coords).
    pairs: Dict[str, Tuple[int, int]]


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _residue_number(res, *, use_resseq: bool) -> Optional[int]:
    """Return residue number as int (prefers resSeq when present)."""
    if use_resseq:
        val = getattr(res, "resSeq", None)
        out = _safe_int(val)
        if out is not None:
            return out
    return _safe_int(getattr(res, "index", None))


def _atom_is_backbone(atom) -> bool:
    try:
        # MDTraj Atom has is_backbone
        return bool(getattr(atom, "is_backbone"))
    except Exception:
        return str(getattr(atom, "name", "")).upper() in _BACKBONE_NAMES


def _select_group_atoms_global(
    topology,
    *,
    res_start: int,
    res_end: int,
    use_resseq: bool,
) -> np.ndarray:
    """Select atom indices for (backbone OR CB) within a residue-number range (inclusive)."""
    start = int(res_start)
    end = int(res_end)
    if end < start:
        start, end = end, start
    out: list[int] = []
    for atom in topology.atoms:
        resnum = _residue_number(atom.residue, use_resseq=use_resseq)
        if resnum is None or not (start <= resnum <= end):
            continue
        name = str(getattr(atom, "name", "")).upper()
        if _atom_is_backbone(atom) or name == "CB":
            out.append(int(atom.index))
    return np.asarray(out, dtype=np.int64)


def _find_ca_atom_global(topology, *, residue_number: int, use_resseq: bool) -> Optional[int]:
    """Find the CA atom index for a residue (by resSeq or by 0-based index).

    Note: we intentionally **do not** fall back to the first atom in the residue when
    CA is missing. In explicit-solvent systems, residue numbers (resSeq) can match
    water/ion residues too; falling back would then spuriously treat (e.g.) a water
    oxygen as "CA" and create meaningless ADK distance metrics/steering forces.
    """
    target = int(residue_number)
    for res in topology.residues:
        resnum = _residue_number(res, use_resseq=use_resseq)
        if resnum is None or resnum != target:
            continue
        for atom in res.atoms:
            if str(getattr(atom, "name", "")).upper() == "CA":
                return int(atom.index)
        return None
    return None


def build_adk_token_indices(
    topology,
    *,
    protein_indices: Optional[np.ndarray] = None,
    prefer_resseq: bool = True,
) -> ADKTokenIndices:
    """Build local atom indices for the ADK hinge-angle/distance metrics.

    Args:
        topology: MDTraj-like Topology (must have .atoms and .residues).
        protein_indices: Optional global atom indices defining the "protein-only"
            coordinate arrays (as used by env.step()); when provided, returned indices
            are mapped into this local indexing.
        prefer_resseq: Interpret residue numbers as PDB resSeq when possible; fall
            back to 0-based residue index if no matches are found.
    """
    if protein_indices is None:
        protein_indices = np.arange(int(getattr(topology, "n_atoms", 0) or 0), dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    idx_to_local = {int(a): int(i) for i, a in enumerate(protein_indices.tolist())}

    def _map_to_local(global_idx: np.ndarray) -> np.ndarray:
        if global_idx.size == 0:
            return np.zeros((0,), dtype=np.int64)
        loc = np.asarray([idx_to_local.get(int(i), -1) for i in global_idx.tolist()], dtype=np.int64)
        loc = loc[loc >= 0]
        return loc

    # Try resSeq first; if any key selection fails (returns empty), fall back to residue index.
    use_resseq = bool(prefer_resseq)

    def _group(name: str, start: int, end: int) -> np.ndarray:
        nonlocal use_resseq
        g = _select_group_atoms_global(topology, res_start=start, res_end=end, use_resseq=use_resseq)
        if g.size == 0 and use_resseq:
            # Fall back to 0-based residue indices
            use_resseq = False
            g = _select_group_atoms_global(topology, res_start=start, res_end=end, use_resseq=use_resseq)
        return _map_to_local(g)

    group_atoms = {
        "core_lid": _group("core_lid", 115, 125),
        "core1": _group("core1", 90, 100),
        "nmp": _group("nmp", 35, 55),
        "core2": _group("core2", 179, 185),
        "lid": _group("lid", 125, 153),
    }

    def _pair(res_a: int, res_b: int, label: str) -> Optional[Tuple[int, int]]:
        idx_a = _find_ca_atom_global(topology, residue_number=res_a, use_resseq=use_resseq)
        idx_b = _find_ca_atom_global(topology, residue_number=res_b, use_resseq=use_resseq)
        if idx_a is None or idx_b is None:
            return None
        a_loc = idx_to_local.get(int(idx_a), -1)
        b_loc = idx_to_local.get(int(idx_b), -1)
        if a_loc < 0 or b_loc < 0 or a_loc == b_loc:
            return None
        return (int(a_loc), int(b_loc))

    pairs: Dict[str, Tuple[int, int]] = {}
    p = _pair(145, 52, "dist_145_52_nm")
    if p is not None:
        pairs["dist_145_52_nm"] = p
    p = _pair(169, 55, "dist_169_55_nm")
    if p is not None:
        pairs["dist_169_55_nm"] = p

    return ADKTokenIndices(group_atoms=group_atoms, pairs=pairs)


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ∠ABC in radians for batched points a,b,c of shape [F,3]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(c, dtype=np.float64).reshape(-1, 3)
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    cosang = np.sum(v1 * v2, axis=1) / denom_safe
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    ang = np.where(denom > 1e-12, ang, 0.0)
    return ang.astype(np.float32, copy=False)


def _centroid_from_frames(frames_nm: Sequence[np.ndarray], atom_idx: np.ndarray) -> np.ndarray:
    if atom_idx.size == 0:
        return np.zeros((len(frames_nm), 3), dtype=np.float32)
    xyz = np.asarray([np.asarray(f, dtype=np.float64)[atom_idx] for f in frames_nm], dtype=np.float64)
    return xyz.mean(axis=1).astype(np.float32, copy=False)


def _centroid_from_xyz(xyz_nm: np.ndarray, atom_idx: np.ndarray) -> np.ndarray:
    xyz_nm = np.asarray(xyz_nm, dtype=np.float64)
    atom_idx = np.asarray(atom_idx, dtype=np.int64).reshape(-1)
    if xyz_nm.ndim != 3:
        raise ValueError(f"xyz_nm must be [F,N,3], got {xyz_nm.shape}")
    if atom_idx.size == 0:
        return np.zeros((xyz_nm.shape[0], 3), dtype=np.float32)
    sel = xyz_nm[:, atom_idx, :]
    return sel.mean(axis=1).astype(np.float32, copy=False)


def _distance_from_frames(frames_nm: Sequence[np.ndarray], a: int, b: int) -> np.ndarray:
    pa = np.asarray([np.asarray(f, dtype=np.float64)[int(a)] for f in frames_nm], dtype=np.float64)
    pb = np.asarray([np.asarray(f, dtype=np.float64)[int(b)] for f in frames_nm], dtype=np.float64)
    return np.linalg.norm(pa - pb, axis=1).astype(np.float32, copy=False)


def _distance_from_xyz(xyz_nm: np.ndarray, a: int, b: int) -> np.ndarray:
    xyz_nm = np.asarray(xyz_nm, dtype=np.float64)
    if xyz_nm.ndim != 3:
        raise ValueError(f"xyz_nm must be [F,N,3], got {xyz_nm.shape}")
    pa = xyz_nm[:, int(a), :]
    pb = xyz_nm[:, int(b), :]
    return np.linalg.norm(pa - pb, axis=1).astype(np.float32, copy=False)


def compute_adk_global_values_from_xyz(
    xyz_nm: np.ndarray,
    *,
    ca_indices: np.ndarray,
    ref_ca_nm: np.ndarray,
    idx: ADKTokenIndices,
) -> np.ndarray:
    """Compute ADK global token values from a packed coordinate array (nm).

    Returns:
        values: [F, 5] float32 with columns matching `ADK_GLOBAL_TOKEN_NAMES`.
    """
    xyz_nm = np.asarray(xyz_nm, dtype=np.float64)
    if xyz_nm.ndim != 3 or xyz_nm.shape[-1] != 3:
        raise ValueError(f"xyz_nm must be [F,N,3], got {xyz_nm.shape}")
    n = int(xyz_nm.shape[0])
    out = np.zeros((n, len(ADK_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
    if n == 0:
        return out

    ca_idx = np.asarray(ca_indices, dtype=np.int64).reshape(-1)
    ref_ca_nm = np.asarray(ref_ca_nm, dtype=np.float64).reshape(-1, 3)
    if ca_idx.size >= 3 and ref_ca_nm.shape[0] == ca_idx.size:
        try:
            coords_ca = xyz_nm[:, ca_idx, :]
            rmsd = calculate_rmsd(coords_ca, ref_ca_nm, atom_indices=None).astype(np.float32, copy=False)
            out[:, 0] = rmsd
        except Exception:
            pass

    try:
        core_lid = _centroid_from_xyz(xyz_nm, idx.group_atoms.get("core_lid", np.zeros((0,), dtype=np.int64)))
        core1 = _centroid_from_xyz(xyz_nm, idx.group_atoms.get("core1", np.zeros((0,), dtype=np.int64)))
        nmp = _centroid_from_xyz(xyz_nm, idx.group_atoms.get("nmp", np.zeros((0,), dtype=np.int64)))
        core2 = _centroid_from_xyz(xyz_nm, idx.group_atoms.get("core2", np.zeros((0,), dtype=np.int64)))
        lid = _centroid_from_xyz(xyz_nm, idx.group_atoms.get("lid", np.zeros((0,), dtype=np.int64)))
        out[:, 1] = calc_angle(core_lid, core1, nmp)
        out[:, 2] = calc_angle(core2, core_lid, lid)
    except Exception:
        pass

    pair = idx.pairs.get("dist_145_52_nm")
    if pair is not None:
        try:
            out[:, 3] = _distance_from_xyz(xyz_nm, pair[0], pair[1])
        except Exception:
            pass

    pair = idx.pairs.get("dist_169_55_nm")
    if pair is not None:
        try:
            out[:, 4] = _distance_from_xyz(xyz_nm, pair[0], pair[1])
        except Exception:
            pass

    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    out = np.clip(out, 0.0, None)
    return out


def compute_adk_global_values_from_frames(
    frames_nm: Sequence[np.ndarray],
    *,
    ca_local: np.ndarray,
    ref_ca_nm: np.ndarray,
    idx: ADKTokenIndices,
) -> np.ndarray:
    """Compute ADK global token values for a sequence of protein coordinate frames (nm).

    Returns:
        values: [F, 5] float32 with columns matching `ADK_GLOBAL_TOKEN_NAMES`.
    """
    n = len(frames_nm)
    out = np.zeros((n, len(ADK_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
    if n == 0:
        return out

    ca_local = np.asarray(ca_local, dtype=np.int64).reshape(-1)
    ref_ca_nm = np.asarray(ref_ca_nm, dtype=np.float64).reshape(-1, 3)
    if ca_local.size >= 3 and ref_ca_nm.shape[0] == ca_local.size:
        try:
            coords_ca = np.asarray([np.asarray(f, dtype=np.float64)[ca_local] for f in frames_nm], dtype=np.float64)
            rmsd = calculate_rmsd(coords_ca, ref_ca_nm, atom_indices=None).astype(np.float32, copy=False)
            out[:, 0] = rmsd
        except Exception:
            pass

    try:
        core_lid = _centroid_from_frames(frames_nm, idx.group_atoms.get("core_lid", np.zeros((0,), dtype=np.int64)))
        core1 = _centroid_from_frames(frames_nm, idx.group_atoms.get("core1", np.zeros((0,), dtype=np.int64)))
        nmp = _centroid_from_frames(frames_nm, idx.group_atoms.get("nmp", np.zeros((0,), dtype=np.int64)))
        core2 = _centroid_from_frames(frames_nm, idx.group_atoms.get("core2", np.zeros((0,), dtype=np.int64)))
        lid = _centroid_from_frames(frames_nm, idx.group_atoms.get("lid", np.zeros((0,), dtype=np.int64)))

        out[:, 1] = calc_angle(core_lid, core1, nmp)  # theta_nmp
        out[:, 2] = calc_angle(core2, core_lid, lid)  # theta_lid
    except Exception:
        pass

    pair = idx.pairs.get("dist_145_52_nm")
    if pair is not None:
        try:
            out[:, 3] = _distance_from_frames(frames_nm, pair[0], pair[1])
        except Exception:
            pass

    pair = idx.pairs.get("dist_169_55_nm")
    if pair is not None:
        try:
            out[:, 4] = _distance_from_frames(frames_nm, pair[0], pair[1])
        except Exception:
            pass

    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    out = np.clip(out, 0.0, None)  # enforce nonnegativity for log1p-based token modeling
    return out
