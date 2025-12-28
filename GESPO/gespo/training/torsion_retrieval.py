from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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

