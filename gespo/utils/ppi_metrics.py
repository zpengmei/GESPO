"""PPI global token computations for two-chain protein complexes.

Token names:
  chain_com_dist_nm        — CA COM distance between chain A (scaffold) and chain B (ligand), nm
  min_interface_ca_dist_nm — minimum CA–CA distance between any chain A and chain B residue, nm
  q_interface              — fraction of native inter-chain CA–CA contacts within cutoff

Chain assignment is determined by MDTraj chain ordering (chain 0 = scaffold, chain 1 = ligand
by default; configurable via build_ppi_token_indices).

All coordinates are in nm, over protein-only atom arrays (as used by GESPO env.step()).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


PPI_GLOBAL_TOKEN_NAMES: Tuple[str, ...] = (
    "chain_com_dist_nm",
    "min_interface_ca_dist_nm",
    "q_interface",
)


@dataclass(frozen=True)
class PPITokenIndices:
    """Precomputed CA atom indices for each chain (local to protein-only coords).

    Attributes
    ----------
    chain_a_ca : np.ndarray, shape (n_a,) int64
        Local CA indices for chain A (scaffold) within the protein-only coordinate array.
    chain_b_ca : np.ndarray, shape (n_b,) int64
        Local CA indices for chain B (ligand) within the protein-only coordinate array.
    native_pairs : np.ndarray, shape (n_pairs, 2) int64
        Pairs (a_idx, b_idx) of local CA indices that form native inter-chain contacts.
        Populated by set_native_contacts(); empty by default.
    native_cutoff_nm : float
        Distance cutoff used to define Q_interface (nm).
    """

    chain_a_ca: np.ndarray
    chain_b_ca: np.ndarray
    native_pairs: np.ndarray
    native_cutoff_nm: float = 0.8


def build_ppi_token_indices(
    topology,
    *,
    protein_indices: Optional[np.ndarray] = None,
    chain_a_id: int = 0,
    chain_b_id: int = 1,
) -> PPITokenIndices:
    """Build local CA index arrays for each chain from an MDTraj topology.

    Parameters
    ----------
    topology : MDTraj Topology
        Full or protein-only topology with .chains, .atoms attributes.
    protein_indices : array-like, optional
        Global atom indices that define the "protein-only" coordinate slice used by
        GESPO (as in env.step()).  When provided, returned indices are mapped into
        this local (0-based within protein slice) indexing.
    chain_a_id : int
        0-based chain index for chain A (scaffold), default 0.
    chain_b_id : int
        0-based chain index for chain B (ligand), default 1.

    Returns
    -------
    PPITokenIndices with empty native_pairs (call set_native_contacts() to populate).
    """
    if protein_indices is None:
        n = int(getattr(topology, "n_atoms", 0) or 0)
        protein_indices = np.arange(n, dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    idx_to_local = {int(g): int(i) for i, g in enumerate(protein_indices.tolist())}

    chains = list(topology.chains)

    def _ca_for_chain(chain_id: int) -> np.ndarray:
        if chain_id < 0 or chain_id >= len(chains):
            return np.zeros((0,), dtype=np.int64)
        out: List[int] = []
        for atom in chains[chain_id].atoms:
            if str(getattr(atom, "name", "")).upper() == "CA":
                local = idx_to_local.get(int(atom.index), -1)
                if local >= 0:
                    out.append(local)
        return np.array(out, dtype=np.int64)

    return PPITokenIndices(
        chain_a_ca=_ca_for_chain(chain_a_id),
        chain_b_ca=_ca_for_chain(chain_b_id),
        native_pairs=np.zeros((0, 2), dtype=np.int64),
        native_cutoff_nm=0.8,
    )


def set_native_contacts(
    idx: PPITokenIndices,
    ref_ca_xyz_nm: np.ndarray,
    cutoff_nm: float = 0.8,
) -> PPITokenIndices:
    """Return a new PPITokenIndices with native_pairs populated from a reference structure.

    Parameters
    ----------
    idx : PPITokenIndices
        From build_ppi_token_indices(); chain_a_ca / chain_b_ca must be set.
    ref_ca_xyz_nm : np.ndarray, shape (n_ca_total, 3)
        Reference CA coordinates in nm indexed by local (protein-only) CA order.
        Typically the AF3/equilibrated structure.
    cutoff_nm : float
        CA–CA distance cutoff for native contacts (default 0.8 nm).

    Returns
    -------
    New PPITokenIndices with native_pairs set.
    """
    a_idx = idx.chain_a_ca
    b_idx = idx.chain_b_ca
    empty = PPITokenIndices(
        chain_a_ca=a_idx,
        chain_b_ca=b_idx,
        native_pairs=np.zeros((0, 2), dtype=np.int64),
        native_cutoff_nm=cutoff_nm,
    )
    if a_idx.size == 0 or b_idx.size == 0:
        return empty

    ref = np.asarray(ref_ca_xyz_nm, dtype=np.float64)

    # Build all cross-chain CA pairs and compute reference distances.
    grid_a, grid_b = np.meshgrid(a_idx, b_idx, indexing="ij")  # (n_a, n_b)
    pa = ref[grid_a.ravel()]  # (n_a*n_b, 3)
    pb = ref[grid_b.ravel()]
    d = np.linalg.norm(pa - pb, axis=1)
    mask = d < float(cutoff_nm)
    pairs = np.stack([grid_a.ravel()[mask], grid_b.ravel()[mask]], axis=1).astype(np.int64)

    return PPITokenIndices(
        chain_a_ca=a_idx,
        chain_b_ca=b_idx,
        native_pairs=pairs,
        native_cutoff_nm=cutoff_nm,
    )


# ---------------------------------------------------------------------------
# Core computation kernels
# ---------------------------------------------------------------------------

def _compute_ppi_from_xyz(xyz_nm: np.ndarray, idx: PPITokenIndices) -> np.ndarray:
    """Internal: compute PPI tokens from [F, N, 3] array in nm.

    Returns [F, 3] float32 columns: [chain_com_dist, min_interface_ca_dist, q_interface].
    """
    n = int(xyz_nm.shape[0])
    out = np.zeros((n, len(PPI_GLOBAL_TOKEN_NAMES)), dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    a_idx = idx.chain_a_ca
    b_idx = idx.chain_b_ca

    if a_idx.size == 0 or b_idx.size == 0:
        return out.astype(np.float32)

    xa = xyz_nm[:, a_idx, :]   # [F, n_a, 3]
    xb = xyz_nm[:, b_idx, :]   # [F, n_b, 3]

    # (0) Chain COM distance
    com_a = xa.mean(axis=1)    # [F, 3]
    com_b = xb.mean(axis=1)
    out[:, 0] = np.linalg.norm(com_a - com_b, axis=1)

    # (1) Minimum inter-chain CA distance (vectorised via expanded L2)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    aa = np.einsum("fia,fia->fi", xa, xa)      # [F, n_a]
    bb = np.einsum("fjb,fjb->fj", xb, xb)     # [F, n_b]
    ab = np.einsum("fia,fja->fij", xa, xb)    # [F, n_a, n_b]
    d2 = aa[:, :, None] + bb[:, None, :] - 2.0 * ab  # [F, n_a, n_b]
    d2 = np.maximum(d2, 0.0)
    out[:, 1] = np.sqrt(d2.reshape(n, -1).min(axis=1))

    # (2) Q_interface — fraction of native pairs within cutoff
    pairs = idx.native_pairs
    if pairs.shape[0] > 0:
        cutoff = float(idx.native_cutoff_nm)
        pa = xyz_nm[:, pairs[:, 0], :]   # [F, n_pairs, 3]
        pb = xyz_nm[:, pairs[:, 1], :]
        dp = np.linalg.norm(pa - pb, axis=-1)   # [F, n_pairs]
        out[:, 2] = (dp < cutoff).mean(axis=1)

    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, None)
    return out.astype(np.float32)


def compute_ppi_globals_from_xyz(
    xyz_nm: np.ndarray,
    idx: PPITokenIndices,
) -> np.ndarray:
    """Compute PPI global tokens from a packed coordinate array.

    Parameters
    ----------
    xyz_nm : np.ndarray, shape (F, N, 3)
        Protein-only coordinates in nm; N = number of protein atoms.
    idx : PPITokenIndices

    Returns
    -------
    np.ndarray, shape (F, 3) float32, columns matching PPI_GLOBAL_TOKEN_NAMES.
    """
    xyz_nm = np.asarray(xyz_nm, dtype=np.float64)
    if xyz_nm.ndim != 3 or xyz_nm.shape[-1] != 3:
        raise ValueError(f"xyz_nm must be [F, N, 3], got {xyz_nm.shape}")
    return _compute_ppi_from_xyz(xyz_nm, idx)


def compute_ppi_globals_from_frames(
    frames_nm: Sequence[np.ndarray],
    idx: PPITokenIndices,
) -> np.ndarray:
    """Compute PPI global tokens from a list of per-frame protein coordinate arrays.

    Parameters
    ----------
    frames_nm : sequence of np.ndarray, each shape (N, 3)
        Protein-only coordinate frames in nm.
    idx : PPITokenIndices

    Returns
    -------
    np.ndarray, shape (F, 3) float32, columns matching PPI_GLOBAL_TOKEN_NAMES.
    """
    if len(frames_nm) == 0:
        return np.zeros((0, len(PPI_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
    xyz_nm = np.asarray(
        [np.asarray(f, dtype=np.float64) for f in frames_nm], dtype=np.float64
    )
    return _compute_ppi_from_xyz(xyz_nm, idx)
