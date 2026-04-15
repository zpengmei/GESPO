"""
Vinculin H5-focused global token computations.

Token names:
  h5_bundle_dist_nm   — COM distance between H5 and the rest of the bundle (H1–H4)
  h5_h1_angle_deg     — interhelix angle between H5 and H1 PCA-1 axes (0–90°)
  h4h5_linker_e2e_nm  — CA distance from last H4 residue to first H5 residue
  q_h5_bundle         — fraction of native CA-CA contacts between H5 and H1–H4
  h5_helicity         — helicity proxy: fraction of H5 residues with CA(i)-CA(i+3) < 0.62 nm

All indices are 0-based CA residue indices within a 180-residue vinculin Vt chain.
"""

import numpy as np
from typing import List, Optional, Tuple

# ── Helix definitions (0-based CA residue indices) ─────────────────────────
H1 = list(range(10, 28))     # GLU892–ALA909
H2 = list(range(36, 56))     # ASP918–VAL937
H3 = list(range(59, 90))     # SER941–GLN971
H4 = list(range(93, 124))    # LYS975–MET1005
H5 = list(range(131, 163))   # ASP1013–ALA1044
BUNDLE = H1 + H2 + H3 + H4  # everything except H5

# H4–H5 linker endpoints (CA indices)
H4_LAST = H4[-1]   # 123
H5_FIRST = H5[0]   # 131

VINCULIN_H5_TOKEN_NAMES = [
    "h5_bundle_dist_nm",
    "h5_h1_angle_deg",
    "h4h5_linker_e2e_nm",
    "q_h5_bundle",
    "h5_helicity",
]

# Helicity proxy: CA(i)-CA(i+3) distance cutoff for alpha helix
_HELIX_CA_I_I3_CUTOFF_NM = 0.62

# ── Pre-computed native contact pairs (set once from reference) ────────────
_native_pairs: Optional[np.ndarray] = None
_native_cutoff: float = 0.8  # nm


def set_native_contacts_from_ref(ref_ca_xyz_nm: np.ndarray, cutoff_nm: float = 0.8) -> None:
    """
    Compute H5-vs-bundle native contact pairs from a reference CA structure.

    Parameters
    ----------
    ref_ca_xyz_nm : np.ndarray, shape (180, 3)
        Reference CA coordinates in nm.
    cutoff_nm : float
        Distance cutoff for native contacts.
    """
    global _native_pairs, _native_cutoff
    _native_cutoff = cutoff_nm

    pairs = []
    for i in H5:
        for j in BUNDLE:
            if abs(i - j) >= 4:
                pairs.append((i, j))
    pairs = np.array(pairs, dtype=np.int64)

    if pairs.size == 0:
        _native_pairs = np.zeros((0, 2), dtype=np.int64)
        return

    # Compute reference distances
    d = np.linalg.norm(
        ref_ca_xyz_nm[pairs[:, 0]] - ref_ca_xyz_nm[pairs[:, 1]], axis=1
    )
    _native_pairs = pairs[d < cutoff_nm]


def _helix_axis(coords: np.ndarray, indices: list) -> np.ndarray:
    """PCA-1 axis of selected CA positions. Returns unit vector (3,)."""
    pts = coords[indices]
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    # Orient N-term -> C-term
    if np.dot(axis, pts[-1] - pts[0]) < 0:
        axis = -axis
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 1e-12 else axis


def _angle_between(a1: np.ndarray, a2: np.ndarray) -> float:
    """Angle in degrees between two unit vectors (0–90, parallel/antiparallel symmetric)."""
    cos_a = np.clip(np.dot(a1, a2), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_a))))


def _ca_helicity_proxy(coords: np.ndarray, indices: list) -> float:
    """
    Helicity proxy: fraction of residues i in `indices` where
    dist(CA_i, CA_{i+3}) < cutoff (characteristic of alpha helix).
    """
    count = 0
    total = 0
    for idx in indices:
        if idx + 3 < coords.shape[0]:
            d = np.linalg.norm(coords[idx] - coords[idx + 3])
            if d < _HELIX_CA_I_I3_CUTOFF_NM:
                count += 1
            total += 1
    return count / max(total, 1)


def compute_vinculin_h5_globals(ca_coords: np.ndarray) -> np.ndarray:
    """
    Compute all 5 vinculin H5 global tokens from CA coordinates.

    Parameters
    ----------
    ca_coords : np.ndarray, shape (N, 180, 3)
        CA coordinates in nm for N frames.

    Returns
    -------
    np.ndarray, shape (N, 5)
        Columns: h5_bundle_dist_nm, h5_h1_angle_deg, h4h5_linker_e2e_nm,
                 q_h5_bundle, h5_helicity
    """
    n_frames = ca_coords.shape[0]
    out = np.zeros((n_frames, 5), dtype=np.float32)

    h5_idx = np.array(H5, dtype=np.int64)
    bundle_idx = np.array(BUNDLE, dtype=np.int64)

    # 1. H5–bundle COM distance
    h5_com = ca_coords[:, h5_idx, :].mean(axis=1)       # (N, 3)
    bundle_com = ca_coords[:, bundle_idx, :].mean(axis=1)  # (N, 3)
    out[:, 0] = np.linalg.norm(h5_com - bundle_com, axis=1)

    # 2. H5–H1 interhelix angle + 3. linker e2e + 5. helicity (per-frame loop)
    for f in range(n_frames):
        coords_f = ca_coords[f]  # (180, 3)

        # H5-H1 angle
        ax_h5 = _helix_axis(coords_f, H5)
        ax_h1 = _helix_axis(coords_f, H1)
        out[f, 1] = _angle_between(ax_h5, ax_h1)

        # H5 helicity proxy
        out[f, 4] = _ca_helicity_proxy(coords_f, H5)

    # 3. H4–H5 linker e2e distance (vectorized)
    out[:, 2] = np.linalg.norm(
        ca_coords[:, H5_FIRST, :] - ca_coords[:, H4_LAST, :], axis=1
    )

    # 4. Q (H5 vs bundle) native contacts
    if _native_pairs is not None and _native_pairs.shape[0] > 0:
        d = np.linalg.norm(
            ca_coords[:, _native_pairs[:, 0], :] - ca_coords[:, _native_pairs[:, 1], :],
            axis=2,
        )  # (N, n_pairs)
        out[:, 3] = (d < _native_cutoff).mean(axis=1)
    else:
        out[:, 3] = 1.0  # no pairs defined

    return out


def compute_vinculin_h5_globals_from_frames(
    coords_list: List[np.ndarray],
    ca_local: np.ndarray,
) -> np.ndarray:
    """
    Compute H5 globals from a list of full-protein coordinate arrays.

    Parameters
    ----------
    coords_list : List[np.ndarray]
        Each element is shape (n_protein_atoms, 3) in nm.
    ca_local : np.ndarray, shape (n_ca,)
        Indices of CA atoms within the protein coordinate array.

    Returns
    -------
    np.ndarray, shape (N, 5)
    """
    if not coords_list or ca_local.size < 180:
        return np.zeros((max(len(coords_list), 0), 5), dtype=np.float32)

    ca_coords = np.asarray(
        [np.asarray(c, dtype=np.float64)[ca_local[:180]] for c in coords_list],
        dtype=np.float64,
    )
    return compute_vinculin_h5_globals(ca_coords)
