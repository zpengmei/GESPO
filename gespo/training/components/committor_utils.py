"""Pure committor utility functions (no state, no nonlocal captures)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ...utils.geometry import calculate_rmsd


def committor_from_rmsd(rmsd_a: np.ndarray, rmsd_b: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """RMSD-based proxy committor: q = rmsd_a^2 / (rmsd_a^2 + rmsd_b^2 + eps)."""
    da2 = np.asarray(rmsd_a, dtype=np.float64) ** 2
    db2 = np.asarray(rmsd_b, dtype=np.float64) ** 2
    denom = da2 + db2 + float(eps)
    q = da2 / denom
    q = np.where(np.isfinite(q), q, 0.0)
    return np.clip(q, 0.0, 1.0).astype(np.float32, copy=False)


def rmsd_state_labels_2d(
    rmsd_a: np.ndarray,
    rmsd_b: np.ndarray,
    *,
    bins: int,
    max_nm: float,
) -> np.ndarray:
    """Discretize (rmsd_a, rmsd_b) into a 2D grid label in [0, bins^2)."""
    bins_i = int(bins)
    if bins_i <= 1:
        n = int(min(np.asarray(rmsd_a).reshape(-1).shape[0], np.asarray(rmsd_b).reshape(-1).shape[0]))
        return np.zeros((max(0, n),), dtype=np.int64)
    max_nm_f = float(max_nm)
    if not np.isfinite(max_nm_f) or max_nm_f <= 0.0:
        max_nm_f = 1e-6
    ra = np.asarray(rmsd_a, dtype=np.float64).reshape(-1)
    rb = np.asarray(rmsd_b, dtype=np.float64).reshape(-1)
    n = int(min(int(ra.shape[0]), int(rb.shape[0])))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    ra = ra[:n]
    rb = rb[:n]
    ra = np.where(np.isfinite(ra), ra, max_nm_f)
    rb = np.where(np.isfinite(rb), rb, max_nm_f)
    ra = np.clip(ra, 0.0, max_nm_f)
    rb = np.clip(rb, 0.0, max_nm_f)
    ba = np.floor(float(bins_i) * (ra / max_nm_f)).astype(np.int64, copy=False)
    bb = np.floor(float(bins_i) * (rb / max_nm_f)).astype(np.int64, copy=False)
    ba = np.clip(ba, 0, int(bins_i) - 1)
    bb = np.clip(bb, 0, int(bins_i) - 1)
    return (ba * int(bins_i) + bb).astype(np.int64, copy=False)


def lift_center_for_delay(x: np.ndarray, *, n_embed: int, lag: int, target_len: int = 0) -> np.ndarray:
    """Lift a per-frame array to the center of a delay-embedding window."""
    x_arr = np.asarray(x).reshape(-1)
    n_embed_i = int(n_embed)
    if n_embed_i <= 0:
        return x_arr
    lag_i = int(lag)
    if lag_i <= 0:
        lag_i = 1
    total = int(lag_i) * int(n_embed_i)
    out_T = int(x_arr.shape[0]) - int(total)
    if out_T <= 0:
        return np.zeros((0,), dtype=x_arr.dtype)
    pad_left = int(total // 2)
    pad_right = int(total - pad_left)
    lifted = x_arr[pad_left: int(x_arr.shape[0]) - pad_right]
    if int(target_len) > 0 and int(lifted.shape[0]) != int(target_len):
        m = int(min(int(lifted.shape[0]), int(target_len)))
        lifted = lifted[:m]
    return lifted


def rmsd_to_endpoints(
    ca_coords: np.ndarray,
    *,
    endpoint_ca_ref_a: Optional[np.ndarray],
    endpoint_ca_ref_b: Optional[np.ndarray],
    endpoint_token_mask: Optional[np.ndarray],
    _units_warned: list,  # mutable container for warning state: [bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame RMSD to endpoints A and B from CA coordinates.

    Handles Angstrom/nm unit detection automatically.

    Parameters
    ----------
    ca_coords : ndarray
        CA coordinates, shape ``(T, n_tokens, 3)``.
    endpoint_ca_ref_a, endpoint_ca_ref_b : ndarray or None
        Reference CA coordinates (nm) for endpoints A and B.
    endpoint_token_mask : ndarray or None
        Boolean mask of valid tokens.
    _units_warned : list[bool]
        Mutable container to track whether unit warning has been issued.
    """
    ca_np = ca_coords
    try:
        if isinstance(ca_coords, torch.Tensor):
            ca_np = ca_coords.detach().cpu().numpy()
    except Exception:
        ca_np = ca_coords

    ca_raw = np.asarray(ca_np, dtype=np.float64)
    ca_use = ca_raw
    try:
        mask_u = np.ones((ca_raw.shape[1],), dtype=bool)
        if endpoint_token_mask is not None and int(endpoint_token_mask.size) == int(ca_raw.shape[1]):
            mask_u = np.asarray(endpoint_token_mask, dtype=bool)
        coords_u = ca_raw[:, mask_u, :]
        if coords_u.ndim == 3 and int(coords_u.shape[1]) >= 3:
            d = np.linalg.norm(coords_u[:, 1:, :] - coords_u[:, :-1, :], axis=2).reshape(-1)
            d = d[np.isfinite(d) & (d > 0)]
            med = float(np.median(d)) if d.size else float("nan")
            if np.isfinite(med) and med < 1.0:
                ca_use = ca_raw
                if not _units_warned[0]:
                    _units_warned[0] = True
                    print(
                        f"[Path Boundaries] Warning: CA coords appear to be in nm already (median_adj={med:.3f}); "
                        "skipping A\u0327\u2192nm conversion in RMSD-to-endpoints.",
                        flush=True,
                    )
            else:
                ca_use = ca_raw * 0.1
        else:
            ca_use = ca_raw * 0.1
    except Exception:
        ca_use = ca_raw * 0.1

    ca_coords_nm = ca_use
    if ca_coords_nm.ndim != 3 or ca_coords_nm.shape[-1] != 3:
        raise ValueError(f"ca_coords must be (T, tokens, 3), got {ca_coords_nm.shape}")
    T = int(ca_coords_nm.shape[0])
    if endpoint_ca_ref_a is None or endpoint_ca_ref_b is None:
        return np.full((T,), np.inf, dtype=np.float64), np.full((T,), np.inf, dtype=np.float64)
    mask = np.ones((ca_coords_nm.shape[1],), dtype=bool)
    if endpoint_token_mask is not None and int(endpoint_token_mask.size) == int(ca_coords_nm.shape[1]):
        mask = np.asarray(endpoint_token_mask, dtype=bool)
    ca_masked = ca_coords_nm[:, mask, :]
    ref_a = np.asarray(endpoint_ca_ref_a, dtype=np.float64)[mask, :]
    ref_b = np.asarray(endpoint_ca_ref_b, dtype=np.float64)[mask, :]
    if ca_masked.shape[1] != ref_a.shape[0] or ca_masked.shape[1] != ref_b.shape[0]:
        raise ValueError("Endpoint CA reference shape mismatch (token mask / topology mismatch).")
    rmsd_a = calculate_rmsd(ca_masked, ref_a, atom_indices=None).astype(np.float64, copy=False)
    rmsd_b = calculate_rmsd(ca_masked, ref_b, atom_indices=None).astype(np.float64, copy=False)
    return rmsd_a, rmsd_b
