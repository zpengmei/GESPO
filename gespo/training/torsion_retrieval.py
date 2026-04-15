from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CandidatePool:
    indices: np.ndarray  # [M] indices into flattened history/buffer
    emb: np.ndarray      # [M, D] float32
    support_d2_threshold: float  # scalar squared-distance threshold for d_min guard


def phi_psi_mask_4l(phi_exists: np.ndarray, psi_exists: np.ndarray) -> np.ndarray:
    """
    Build a [4L] mask for (sinφ, cosφ, sinψ, cosψ) embedding channels.
    """
    phi_exists = np.asarray(phi_exists, dtype=bool).reshape(-1)
    psi_exists = np.asarray(psi_exists, dtype=bool).reshape(-1)
    if phi_exists.shape != psi_exists.shape:
        raise ValueError(f"phi_exists and psi_exists must have same shape, got {phi_exists.shape} vs {psi_exists.shape}")
    l = int(phi_exists.shape[0])
    m = np.zeros((l, 4), dtype=np.float32)
    m[:, 0] = phi_exists.astype(np.float32)
    m[:, 1] = phi_exists.astype(np.float32)
    m[:, 2] = psi_exists.astype(np.float32)
    m[:, 3] = psi_exists.astype(np.float32)
    return m.reshape(-1)


def embed_phi_psi_from_torsions(
    torsions: torch.Tensor,
    phi_exists: torch.Tensor,
    psi_exists: torch.Tensor,
) -> np.ndarray:
    """
    torsions: [B, L, 3] (radians)
    returns:  [B, 4L] float32 embedding for φ/ψ only, masked.
    """
    if torsions.ndim != 3 or torsions.shape[-1] < 2:
        raise ValueError(f"torsions must be [B,L,>=2], got {tuple(torsions.shape)}")
    phi = torsions[..., 0]
    psi = torsions[..., 1]
    emb = torch.stack([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi)], dim=-1)  # [B,L,4]

    phi_exists = phi_exists.to(dtype=torch.float32, device=emb.device).view(1, -1, 1)  # [1,L,1]
    psi_exists = psi_exists.to(dtype=torch.float32, device=emb.device).view(1, -1, 1)  # [1,L,1]
    mask = torch.cat([phi_exists, phi_exists, psi_exists, psi_exists], dim=-1)  # [1,L,4]
    emb = emb * mask
    return emb.reshape(emb.shape[0], -1).detach().cpu().numpy().astype(np.float32)

def embed_sincos_from_torsions_np(
    torsions: np.ndarray,
    angle_exists: np.ndarray,
    *,
    angle_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Build a masked sin/cos embedding for arbitrary torsion tensors.

    Args:
        torsions: [N, L, A] radians.
        angle_exists: [L, A] bool, True where angle is defined.
        angle_indices: optional subset of angle indices to include (default: all).

    Returns:
        emb: [N, 2*L*len(angle_indices)] float32
    """
    torsions = np.asarray(torsions, dtype=np.float32)
    if torsions.ndim != 3:
        raise ValueError(f"torsions must be [N,L,A], got {torsions.shape}")
    n, l, a = torsions.shape

    angle_exists = np.asarray(angle_exists, dtype=bool)
    if angle_exists.shape != (l, a):
        raise ValueError(f"angle_exists must have shape {(l, a)}, got {angle_exists.shape}")

    if angle_indices is None:
        angle_indices_arr = np.arange(a, dtype=np.int64)
    else:
        angle_indices_arr = np.asarray(list(angle_indices), dtype=np.int64).reshape(-1)
        if angle_indices_arr.size == 0:
            return np.zeros((n, 0), dtype=np.float32)
        if np.any((angle_indices_arr < 0) | (angle_indices_arr >= a)):
            raise ValueError(f"angle_indices out of range for A={a}: {angle_indices_arr.tolist()}")

    tors_sel = torsions[:, :, angle_indices_arr]  # [N,L,A']
    emb = np.stack([np.sin(tors_sel), np.cos(tors_sel)], axis=-1)  # [N,L,A',2]
    m = angle_exists[:, angle_indices_arr].astype(np.float32)  # [L,A']
    emb *= m[None, :, :, None]
    return emb.reshape(n, -1).astype(np.float32, copy=False)


def dihedral_batch(coords_nm: np.ndarray, torsion_atoms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Vectorized dihedral angles for a batch of frames.

    Args:
        coords_nm: [N, n_atoms, 3] float32 positions.
        torsion_atoms: [T, 4] int atom indices into axis-1 of coords_nm.

    Returns:
        angles: [N, T] float32 radians in [-pi, pi].
    """
    coords = np.asarray(coords_nm, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"coords_nm must be [N,n_atoms,3], got {coords.shape}")
    tors = np.asarray(torsion_atoms, dtype=np.int64).reshape(-1, 4)
    if tors.size == 0:
        return np.zeros((coords.shape[0], 0), dtype=np.float32)

    a = tors[:, 0]
    b = tors[:, 1]
    c = tors[:, 2]
    d = tors[:, 3]
    if np.any(a < 0) or np.any(b < 0) or np.any(c < 0) or np.any(d < 0):
        raise ValueError("torsion_atoms contains negative indices")
    if np.any(a >= coords.shape[1]) or np.any(b >= coords.shape[1]) or np.any(c >= coords.shape[1]) or np.any(d >= coords.shape[1]):
        raise ValueError("torsion_atoms indices out of bounds for coords_nm")

    r0 = coords[:, a, :]  # [N,T,3]
    r1 = coords[:, b, :]
    r2 = coords[:, c, :]
    r3 = coords[:, d, :]

    b0 = r0 - r1
    b1 = r2 - r1
    b2 = r3 - r2

    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_u = b1 / (b1_norm + float(eps))

    v = b0 - np.sum(b0 * b1_u, axis=-1, keepdims=True) * b1_u
    w = b2 - np.sum(b2 * b1_u, axis=-1, keepdims=True) * b1_u

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_u, v) * w, axis=-1)
    ang = np.arctan2(y, x).astype(np.float32, copy=False)
    return ang


def build_residue_torsions_from_cv_and_coords(
    cv_batch: np.ndarray,
    coords_batch_nm: Optional[np.ndarray],
    *,
    cv_residue_slots: np.ndarray,
    cv_is_phi: np.ndarray,
    n_residues: int,
    n_angles: int,
    chi_torsion_atoms: Optional[np.ndarray] = None,
    chi_residue_idx: Optional[np.ndarray] = None,
    chi_order_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a per-residue torsion tensor consistent with the TorsionAR layout:
      [phi, psi, omega, chi1, chi2, ...]

    Omega is left at 0.0 (and should be masked off downstream unless explicitly filled).
    Sidechain chis are filled from coordinates when `n_angles > 3` and chi_* arrays are provided.

    Args:
        cv_batch: [N, n_cvs] float32 (phi/psi CV values).
        coords_batch_nm: [N, n_atoms, 3] float32 protein coords; required for chis.
        cv_residue_slots: [n_cvs] int mapping CV -> residue slot.
        cv_is_phi: [n_cvs] bool, True for phi CVs, False for psi.
        n_residues: number of residue slots L.
        n_angles: total torsion channels A.
        chi_torsion_atoms: [T,4] int torsion atom indices (coords axis-1).
        chi_residue_idx: [T] int residue slot for each torsion.
        chi_order_idx: [T] int chi order index (0=chi1, 1=chi2, ...).

    Returns:
        torsions: [N, L, A] float32
    """
    cv_batch = np.asarray(cv_batch, dtype=np.float32)
    if cv_batch.ndim != 2:
        raise ValueError(f"cv_batch must be [N,n_cvs], got {cv_batch.shape}")
    n = int(cv_batch.shape[0])
    l = int(n_residues)
    a = int(n_angles)
    if a < 2:
        raise ValueError(f"n_angles must be >= 2, got {a}")

    cv_residue_slots = np.asarray(cv_residue_slots, dtype=np.int64).reshape(-1)
    cv_is_phi = np.asarray(cv_is_phi, dtype=bool).reshape(-1)
    if cv_residue_slots.shape[0] != cv_batch.shape[1] or cv_is_phi.shape[0] != cv_batch.shape[1]:
        raise ValueError("cv_residue_slots/cv_is_phi must match cv_batch.shape[1]")

    phi = np.zeros((n, l), dtype=np.float32)
    psi = np.zeros((n, l), dtype=np.float32)
    for cv_idx in range(int(cv_batch.shape[1])):
        slot = int(cv_residue_slots[cv_idx])
        if slot < 0 or slot >= l:
            continue
        if bool(cv_is_phi[cv_idx]):
            phi[:, slot] = cv_batch[:, cv_idx]
        else:
            psi[:, slot] = cv_batch[:, cv_idx]

    pi = float(np.pi)
    phi = ((phi + pi) % (2.0 * pi)) - pi
    psi = ((psi + pi) % (2.0 * pi)) - pi

    torsions = np.zeros((n, l, a), dtype=np.float32)
    torsions[:, :, 0] = phi
    torsions[:, :, 1] = psi

    if a > 3 and coords_batch_nm is not None and chi_torsion_atoms is not None and chi_residue_idx is not None and chi_order_idx is not None:
        coords_batch_nm = np.asarray(coords_batch_nm, dtype=np.float32)
        if coords_batch_nm.shape[:1] != (n,) or coords_batch_nm.ndim != 3 or coords_batch_nm.shape[2] != 3:
            raise ValueError(f"coords_batch_nm must be [N,n_atoms,3], got {coords_batch_nm.shape}")

        chi_torsion_atoms = np.asarray(chi_torsion_atoms, dtype=np.int64).reshape(-1, 4)
        chi_residue_idx = np.asarray(chi_residue_idx, dtype=np.int64).reshape(-1)
        chi_order_idx = np.asarray(chi_order_idx, dtype=np.int64).reshape(-1)
        if chi_torsion_atoms.shape[0] != chi_residue_idx.shape[0] or chi_residue_idx.shape[0] != chi_order_idx.shape[0]:
            raise ValueError("chi_* arrays must have consistent first dimension")

        n_chi = max(0, a - 3)
        if chi_torsion_atoms.shape[0] > 0 and n_chi > 0:
            ang = dihedral_batch(coords_batch_nm, chi_torsion_atoms)  # [N,T]
            for t in range(int(chi_torsion_atoms.shape[0])):
                res = int(chi_residue_idx[t])
                order = int(chi_order_idx[t])
                if res < 0 or res >= l:
                    continue
                if order < 0 or order >= n_chi:
                    continue
                torsions[:, res, 3 + order] = ang[:, t]

    return torsions


def embed_phi_psi_from_cv_batch(
    cv_batch: np.ndarray,
    cv_residue_slots: np.ndarray,
    cv_is_phi: np.ndarray,
    phi_exists: np.ndarray,
    psi_exists: np.ndarray,
) -> np.ndarray:
    """
    cv_batch: [N, n_cvs] float32
    returns:  [N, 4L] float32 embedding for φ/ψ only, masked.
    """
    cv_batch = np.asarray(cv_batch, dtype=np.float32)
    if cv_batch.ndim != 2:
        raise ValueError(f"cv_batch must be 2D [N,n_cvs], got {cv_batch.shape}")

    cv_residue_slots = np.asarray(cv_residue_slots, dtype=np.int64).reshape(-1)
    cv_is_phi = np.asarray(cv_is_phi, dtype=bool).reshape(-1)
    if cv_residue_slots.shape[0] != cv_batch.shape[1] or cv_is_phi.shape[0] != cv_batch.shape[1]:
        raise ValueError("cv_residue_slots/cv_is_phi must match n_cvs")

    l = int(np.asarray(phi_exists).reshape(-1).shape[0])
    n = int(cv_batch.shape[0])

    phi = np.zeros((n, l), dtype=np.float32)
    psi = np.zeros((n, l), dtype=np.float32)
    for cv_idx in range(cv_batch.shape[1]):
        slot = int(cv_residue_slots[cv_idx])
        if slot < 0 or slot >= l:
            continue
        if bool(cv_is_phi[cv_idx]):
            phi[:, slot] = cv_batch[:, cv_idx]
        else:
            psi[:, slot] = cv_batch[:, cv_idx]

    emb = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1).reshape(n, -1).astype(np.float32)
    m = phi_psi_mask_4l(phi_exists=phi_exists, psi_exists=psi_exists)
    emb *= m[None, :]
    return emb


def pairwise_sq_l2(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute squared L2 distances between q and x:
      q: [B,D], x: [M,D] -> [B,M]
    """
    q = np.asarray(q, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    q_norm = (q * q).sum(axis=1, keepdims=True)  # [B,1]
    x_norm = (x * x).sum(axis=1, keepdims=False)  # [M]
    d2 = q_norm + x_norm[None, :] - 2.0 * (q @ x.T)
    return np.maximum(d2, 0.0)


def compute_support_threshold(
    cand_emb: np.ndarray,
    rng: np.random.Generator,
    sample_size: int = 256,
    quantile: float = 0.99,
) -> float:
    """
    Estimate a reasonable d_min threshold from candidate self-nearest-neighbor distances.
    """
    cand_emb = np.asarray(cand_emb, dtype=np.float32)
    m = int(cand_emb.shape[0])
    if m <= 2:
        return float("inf")

    s = min(int(sample_size), m)
    picks = rng.choice(m, size=s, replace=False)
    q = cand_emb[picks]
    d2 = pairwise_sq_l2(q, cand_emb)  # [s,m]
    # exclude self-match when possible
    for row, idx in enumerate(picks):
        d2[row, int(idx)] = np.inf
    nn = np.min(d2, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return float("inf")
    return float(np.quantile(nn, float(quantile)))


def balanced_candidate_indices(
    labels: np.ndarray,
    n_clusters: int,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    DeepTICA-balanced candidate indices with NO backfill.
    Returns <= m indices.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(labels.shape[0])
    if n == 0 or m <= 0:
        return np.zeros((0,), dtype=np.int64)

    quota = max(1, int(m) // int(n_clusters))
    picks: list[int] = []
    for k in range(int(n_clusters)):
        idxs = np.where(labels == k)[0]
        if idxs.size == 0:
            continue
        take = min(int(quota), int(idxs.size))
        sel = rng.choice(idxs, size=take, replace=False)
        picks.extend([int(x) for x in np.asarray(sel).reshape(-1)])
    return np.asarray(picks, dtype=np.int64)


def sample_from_topk(
    topk_idx: np.ndarray,
    topk_d2: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
) -> int:
    """
    Soft sampling among top-K nearest candidates using exp(-d2 / temperature).
    If temperature <= 0, falls back to argmin.
    """
    topk_idx = np.asarray(topk_idx, dtype=np.int64).reshape(-1)
    topk_d2 = np.asarray(topk_d2, dtype=np.float32).reshape(-1)
    if topk_idx.size == 0:
        raise ValueError("topk is empty")
    if temperature <= 0 or not np.isfinite(temperature):
        return int(topk_idx[int(np.argmin(topk_d2))])
    # stabilize
    z = -topk_d2 / float(temperature)
    z = z - float(np.max(z))
    w = np.exp(z).astype(np.float64)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return int(topk_idx[int(np.argmin(topk_d2))])
    p = (w / s).astype(np.float64)
    j = int(rng.choice(topk_idx.size, p=p))
    return int(topk_idx[j])
