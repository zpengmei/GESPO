"""Global metric computation from protein coordinates.

Computes per-frame RMSD, end-to-end distance, radius of gyration,
ADK domain metrics, vinculin H5 metrics, and PPI inter-chain metrics
aligned to a list of global token names.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from gespo.utils.geometry import calculate_rmsd
from gespo.utils.log import warn as log_warn
from gespo.utils.adk_metrics import (
    ADK_GLOBAL_TOKEN_NAMES,
    compute_adk_global_values_from_frames,
)
from gespo.utils.tmem63_metrics import (
    TMEM63_GLOBAL_TOKEN_NAMES,
    compute_tmem63_global_values_from_frames,
)
from gespo.utils.ppi_metrics import (
    PPI_GLOBAL_TOKEN_NAMES,
    PPITokenIndices,
    compute_ppi_globals_from_frames,
)


def compute_globals_from_coords(
    coords_list: List[np.ndarray],
    *,
    global_dim: int,
    global_token_names: List[str],
    ca_local: np.ndarray,
    ref_ca_nm: np.ndarray,
    ref_ca_a_nm: Optional[np.ndarray] = None,
    ref_ca_b_nm: Optional[np.ndarray] = None,
    adk_token_indices: Optional[Any] = None,
    tmem63_token_indices: Optional[Any] = None,
    tmem63_ref_rmsd_nm: Optional[np.ndarray] = None,
    ppi_token_indices: Optional[PPITokenIndices] = None,
    pocket_ca_local: Optional[np.ndarray] = None,
    pocket_ref_ca_nm: Optional[np.ndarray] = None,
    prot_heavy_local: Optional[np.ndarray] = None,
    lig_heavy_local: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-frame global metrics for a list of protein coordinate frames (nm).

    Returns ``[N, global_dim]`` float32 array aligned to *global_token_names*;
    falls back to zeros on failure for any individual metric.

    Parameters
    ----------
    coords_list : list[ndarray]
        Per-frame protein coordinates in nm, each shape ``(n_protein_atoms, 3)``.
    global_dim : int
        Number of global tokens (columns in the output).
    global_token_names : list[str]
        Names of each global token (determines which metrics to compute).
    ca_local : ndarray
        Indices mapping valid CA atoms to the protein-only coordinate array.
    ref_ca_nm : ndarray
        Reference CA positions (nm) for generic RMSD, shape ``(n_ca, 3)``.
    ref_ca_a_nm, ref_ca_b_nm : ndarray or None
        Endpoint A/B CA positions (nm) for RMSD_A / RMSD_B tokens.
    adk_token_indices : Any or None
        ADK token index object from ``build_adk_token_indices``.
    tmem63_token_indices : Any or None
        TMEM63 token index object from ``build_tmem63_token_indices``.
    tmem63_ref_rmsd_nm : ndarray or None
        Reference coordinates in the order of the TMEM63 RMSD atom indices.
    ppi_token_indices : PPITokenIndices or None
        PPI token index object from ``build_ppi_token_indices``.
    """
    if not coords_list:
        return np.zeros((0, max(0, global_dim)), dtype=np.float32)
    if global_dim <= 0:
        return np.zeros((len(coords_list), 0), dtype=np.float32)

    out = np.zeros((len(coords_list), global_dim), dtype=np.float32)

    # Determine which structural metrics are needed based on token names.
    need_rmsd_any = any(str(n).lower().startswith("rmsd") for n in global_token_names)
    need_rmsd_a = any(str(n).lower().startswith(("rmsd_a", "rmsda")) for n in global_token_names)
    need_rmsd_b = any(str(n).lower().startswith(("rmsd_b", "rmsdb")) for n in global_token_names)
    need_rmsd_ref = any(
        (str(n).lower().startswith("rmsd") and not str(n).lower().startswith(("rmsd_a", "rmsda", "rmsd_b", "rmsdb")))
        for n in global_token_names
    )
    need_e2e = any(str(n).lower().startswith("e2e") for n in global_token_names)
    need_rg = any(str(n).lower().startswith("rg") for n in global_token_names)

    rmsd_nm = None
    rmsd_a_nm = None
    rmsd_b_nm = None
    e2e_nm = None
    rg_nm = None

    if (need_rmsd_any or need_e2e or need_rg) and ca_local.size >= 2 and ref_ca_nm.shape[0] == ca_local.size:
        try:
            coords_traj_ca = np.asarray(
                [np.asarray(c, dtype=np.float64)[ca_local] for c in coords_list], dtype=np.float64
            )
            if need_rmsd_a and ref_ca_a_nm is not None and ref_ca_a_nm.shape[0] == coords_traj_ca.shape[1]:
                rmsd_a_nm = calculate_rmsd(coords_traj_ca, ref_ca_a_nm, atom_indices=None).astype(
                    np.float32, copy=False
                )
            if need_rmsd_b and ref_ca_b_nm is not None and ref_ca_b_nm.shape[0] == coords_traj_ca.shape[1]:
                rmsd_b_nm = calculate_rmsd(coords_traj_ca, ref_ca_b_nm, atom_indices=None).astype(
                    np.float32, copy=False
                )
            if need_rmsd_ref:
                rmsd_nm = calculate_rmsd(coords_traj_ca, ref_ca_nm, atom_indices=None).astype(
                    np.float32, copy=False
                )
            if need_e2e:
                e2e_nm = np.linalg.norm(coords_traj_ca[:, 0, :] - coords_traj_ca[:, -1, :], axis=1).astype(
                    np.float32, copy=False
                )
            if need_rg:
                centered = coords_traj_ca - coords_traj_ca.mean(axis=1, keepdims=True)
                rg_nm = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1)).astype(np.float32, copy=False)
        except Exception:
            rmsd_nm = None
            e2e_nm = None
            rg_nm = None

    # ADK hinge/distance metrics.
    adk_vals = None
    if adk_token_indices is not None and any(n in set(ADK_GLOBAL_TOKEN_NAMES) for n in global_token_names):
        try:
            adk_vals = compute_adk_global_values_from_frames(
                coords_list,
                ca_local=ca_local,
                ref_ca_nm=ref_ca_nm,
                idx=adk_token_indices,
            )
        except Exception:
            adk_vals = None

    # TMEM63 RMSD can select atoms beyond CA, so it uses its own aligned reference.
    tmem63_vals = None
    if tmem63_token_indices is not None and any(n in set(TMEM63_GLOBAL_TOKEN_NAMES) for n in global_token_names):
        try:
            tmem63_vals = compute_tmem63_global_values_from_frames(
                coords_list,
                idx=tmem63_token_indices,
                ref_rmsd_nm=tmem63_ref_rmsd_nm,
            )
        except Exception as exc:
            log_warn(
                f"[analysis.metrics] compute_tmem63_global_values_from_frames failed, "
                f"TMEM63 global tokens will be all-zero this batch: {type(exc).__name__}: {exc}"
            )
            tmem63_vals = None

    # Vinculin H5 metrics.
    vinculin_h5_vals = None
    from gespo.utils.vinculin_h5_metrics import VINCULIN_H5_TOKEN_NAMES
    _vinculin_h5_names_set = set(VINCULIN_H5_TOKEN_NAMES)
    if any(n in _vinculin_h5_names_set for n in global_token_names):
        try:
            from gespo.utils.vinculin_h5_metrics import compute_vinculin_h5_globals_from_frames
            vinculin_h5_vals = compute_vinculin_h5_globals_from_frames(
                coords_list, ca_local=ca_local,
            )
        except Exception:
            vinculin_h5_vals = None

    # PPI inter-chain metrics.
    ppi_vals = None
    _ppi_names_set = set(PPI_GLOBAL_TOKEN_NAMES)
    if ppi_token_indices is not None and any(n in _ppi_names_set for n in global_token_names):
        try:
            ppi_vals = compute_ppi_globals_from_frames(coords_list, idx=ppi_token_indices)
        except Exception:
            ppi_vals = None

    # Assemble output columns by token name.
    try:
        adk_name_to_col = {name: i for i, name in enumerate(ADK_GLOBAL_TOKEN_NAMES)}
        tmem63_name_to_col = {name: i for i, name in enumerate(TMEM63_GLOBAL_TOKEN_NAMES)}
        vinculin_h5_name_to_col = {name: i for i, name in enumerate(VINCULIN_H5_TOKEN_NAMES)}
        ppi_name_to_col = {name: i for i, name in enumerate(PPI_GLOBAL_TOKEN_NAMES)}
        for j, name in enumerate(global_token_names):
            key = str(name)
            key_l = key.lower()
            if ppi_vals is not None and key in ppi_name_to_col:
                out[:, j] = ppi_vals[:, ppi_name_to_col[key]]
            elif vinculin_h5_vals is not None and key in vinculin_h5_name_to_col:
                out[:, j] = vinculin_h5_vals[:, vinculin_h5_name_to_col[key]]
            elif key_l.startswith(("rmsd_a", "rmsda")):
                if rmsd_a_nm is not None:
                    out[:, j] = rmsd_a_nm
                elif adk_vals is not None:
                    out[:, j] = adk_vals[:, 0]
                elif rmsd_nm is not None:
                    out[:, j] = rmsd_nm
            elif key_l.startswith(("rmsd_b", "rmsdb")):
                if rmsd_b_nm is not None:
                    out[:, j] = rmsd_b_nm
                elif adk_vals is not None:
                    out[:, j] = adk_vals[:, 0]
                elif rmsd_nm is not None:
                    out[:, j] = rmsd_nm
            elif key_l.startswith("rmsd"):
                if adk_vals is not None:
                    out[:, j] = adk_vals[:, 0]
                elif rmsd_nm is not None:
                    out[:, j] = rmsd_nm
            elif key_l.startswith("e2e"):
                if e2e_nm is not None:
                    out[:, j] = e2e_nm
            elif key_l.startswith("rg"):
                if rg_nm is not None:
                    out[:, j] = rg_nm
            elif key_l == "pocket_rmsd_nm":
                if (pocket_ca_local is not None and pocket_ref_ca_nm is not None
                        and pocket_ca_local.size >= 2):
                    try:
                        pocket_ca_traj = np.asarray(
                            [np.asarray(c, dtype=np.float64)[pocket_ca_local] for c in coords_list], dtype=np.float64)
                        out[:, j] = calculate_rmsd(pocket_ca_traj, pocket_ref_ca_nm, atom_indices=None).astype(np.float32)
                    except Exception:
                        pass
            elif key_l in ("n_contacts_scaled", "n_contacts"):
                if (prot_heavy_local is not None and lig_heavy_local is not None
                        and prot_heavy_local.size > 0 and lig_heavy_local.size > 0):
                    try:
                        for fi, c in enumerate(coords_list):
                            c = np.asarray(c, dtype=np.float64)
                            dists = np.linalg.norm(
                                c[prot_heavy_local, :][:, None, :] - c[lig_heavy_local, :][None, :, :], axis=-1)
                            out[fi, j] = float((dists < 0.4).sum()) / 100.0
                    except Exception:
                        pass
            elif key_l == "lig_pocket_dist_nm":
                if (pocket_ca_local is not None and lig_heavy_local is not None
                        and pocket_ca_local.size > 0 and lig_heavy_local.size > 0):
                    try:
                        for fi, c in enumerate(coords_list):
                            c = np.asarray(c, dtype=np.float64)
                            lig_com    = c[lig_heavy_local].mean(axis=0)
                            pocket_com = c[pocket_ca_local].mean(axis=0)
                            out[fi, j] = float(np.linalg.norm(lig_com - pocket_com))
                    except Exception:
                        pass
            elif adk_vals is not None and key in adk_name_to_col:
                out[:, j] = adk_vals[:, adk_name_to_col[key]]
            elif tmem63_vals is not None and key in tmem63_name_to_col:
                out[:, j] = tmem63_vals[:, tmem63_name_to_col[key]]
    except Exception:
        pass

    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    out = np.clip(out, 0.0, None)
    return out
