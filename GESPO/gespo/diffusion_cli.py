"""Command-line workflow for preparing diffusion data and training checkpoints."""
from __future__ import annotations

import argparse
import math
import os
import glob as _glob
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ._imports import md
from .diffusion import create_diffusion
from .models.diffusion import ConditionalHierarchicalCartesianDiffusion
from .utils.geometry import build_residue_maps

try:
    import yaml
except ImportError:
    yaml = None


def _compute_segment_counts(topology) -> np.ndarray:
    counts: List[int] = []
    for residue in topology.residues:
        heavy = sum(1 for atom in residue.atoms if atom.element.symbol != "H")
        counts.append(heavy)
    return np.asarray(counts, dtype=np.int64)


def compute_pairwise_distances_per_residue(traj, step=10, min_separation=10):
    """
    Compute interleaved sparse pairwise CA-CA distances for diffusion conditioning.
    
    This function creates a structured sparse pattern where each residue i predicts
    distances to a fixed set of downstream partners: {i+min_separation, i+min_separation+step, ...}.
    This pattern avoids trivial nearest-neighbor distances and focuses on non-local contacts
    while maintaining computational efficiency.
    
    CRITICAL: The returned partner_residues array MUST be saved in the checkpoint
    and used consistently during GESPO training to ensure the policy, generator,
    and environment compute pairwise features using the same interleaved pattern.
    
    Args:
        traj: MDTraj trajectory
        step: Residue spacing for interleaved pattern (default: 10)
        min_separation: Minimum residue separation to skip local contacts (default: 10)
    
    Returns:
        pair_distances: [n_frames, n_residues, max_partners] array of distances (Angstrom)
        pair_mask: [n_residues, max_partners] boolean mask indicating valid partners
        partner_residues: [n_residues, max_partners] index array defining the pattern
        
    Example pattern (20 residues, step=10, min_separation=10):
        Residue 0 → [10, 20, ...]  (partners at indices 10, 20)
        Residue 5 → [15, 25, ...]  (partners at indices 15, 25)
        Residue 10 → [20, ...]      (only partner at index 20)
    """
    ca_indices = traj.top.select("name CA")
    n_residues = len(ca_indices)
    partners_per_res = []
    for i in range(n_residues):
        partners = list(range(i + min_separation, n_residues, step))
        partners_per_res.append(partners)
    max_partners = max((len(p) for p in partners_per_res), default=0)
    pair_distances = np.zeros((traj.n_frames, n_residues, max_partners), dtype=np.float32)
    pair_mask = np.zeros((n_residues, max_partners), dtype=bool)
    partner_residues = np.zeros((n_residues, max_partners), dtype=np.int32)

    all_pairs = []
    slice_indices = [0]
    for i in range(n_residues):
        partners = partners_per_res[i]
        n_partners = len(partners)
        if n_partners > 0:
            all_pairs.extend([[ca_indices[i], ca_indices[j]] for j in partners])
            partner_residues[i, :n_partners] = partners
            pair_mask[i, :n_partners] = True
        slice_indices.append(slice_indices[-1] + n_partners)

    if not all_pairs:
        return pair_distances, pair_mask, partner_residues

    all_pairs = np.asarray(all_pairs, dtype=np.int32)
    all_distances = md.compute_distances(traj, all_pairs) * 10.0
    for i in range(n_residues):
        start_idx = slice_indices[i]
        end_idx = slice_indices[i + 1]
        n_partners = end_idx - start_idx
        if n_partners > 0:
            pair_distances[:, i, :n_partners] = all_distances[:, start_idx:end_idx]
    return pair_distances.astype(np.float32), pair_mask, partner_residues


def _torsion_residue_indices(reference, torsion_indices):
    indices = []
    for torsion in torsion_indices:
        atom = reference.top.atom(int(torsion[1]))
        indices.append(atom.residue.index)
    return np.asarray(indices, dtype=np.int32)


def build_per_residue_torsion_features(
    phi_sin, phi_cos, psi_sin, psi_cos, phi_res_idx, psi_res_idx, n_residues
):
    feats = np.zeros((phi_sin.shape[0], n_residues, 4), dtype=np.float32)
    feats[:, :, 1] = 1.0  # default cos(phi)=1
    feats[:, :, 3] = 1.0  # default cos(psi)=1
    if phi_sin.size > 0 and phi_res_idx is not None:
        phi_res_idx = np.asarray(phi_res_idx, dtype=np.int64).reshape(-1)
        for col, res_idx in enumerate(phi_res_idx):
            feats[:, res_idx, 0] = phi_sin[:, col]
            feats[:, res_idx, 1] = phi_cos[:, col]
    if psi_sin.size > 0 and psi_res_idx is not None:
        psi_res_idx = np.asarray(psi_res_idx, dtype=np.int64).reshape(-1)
        for col, res_idx in enumerate(psi_res_idx):
            feats[:, res_idx, 2] = psi_sin[:, col]
            feats[:, res_idx, 3] = psi_cos[:, col]
    return feats


def _load_traj(path: Path, pdb_path: str, reference: md.Trajectory, align: bool) -> md.Trajectory:
    traj = md.load(str(path), top=pdb_path)
    if align:
        traj.superpose(reference, atom_indices=[atom.index for atom in reference.topology.atoms])
    return traj


def prepare_dataset(
    pdb_path: str,
    traj_dir: str,
    traj_pattern: str,
    output_path: str,
    align: bool = True,
    frame_stride: int = 1,
    traj_chunk_frames: int = 500,
    max_frames: int | None = None,
    conditioning_mode: str = "torsion",
    pair_step: int = 10,
    pair_min_separation: int = 10,
    cond_injection_mode: str = "global",
    # Optional PCA-balanced subsampling
    pca_balance: bool = False,
    pca_balance_bins: int = 50,
    pca_balance_max_per_bin: int = 0,
    pca_balance_fit_cap: int = 50000,
    pca_balance_plot: str | None = None,
    tica_balance: bool = False,
    tica_balance_bins: int = 50,
    tica_balance_max_per_bin: int = 0,
    tica_balance_plot: str | None = None,
    tica_balance_model: str | None = None,
    tica_balance_lagtime: int = 10,
) -> Path:
    traj_dir_path = Path(traj_dir)
    patterns = traj_pattern if isinstance(traj_pattern, (list, tuple)) else [traj_pattern]
    traj_files = []
    # Support both relative patterns (from traj_dir) and absolute patterns.
    for pattern in patterns:
        pattern_str = str(pattern)
        if os.path.isabs(pattern_str):
            matches = _glob.glob(pattern_str, recursive=True)
        else:
            combined = str(traj_dir_path / pattern_str)
            matches = _glob.glob(combined, recursive=True)
        traj_files.extend(Path(p) for p in matches)
    traj_files = sorted({path for path in traj_files})
    if not traj_files:
        raise FileNotFoundError(f"No trajectories matching {patterns} in {traj_dir}")

    reference = md.load(pdb_path)
    topology = reference.topology
    atomic_numbers = np.array([atom.element.atomic_number for atom in topology.atoms], dtype=np.int64)
    segment_counts = _compute_segment_counts(topology)

    use_torsion = conditioning_mode in ("torsion", "hybrid")
    use_pairwise = conditioning_mode in ("pairwise", "hybrid")

    positions_list: List[np.ndarray] = []
    phi_list: List[np.ndarray] = []
    psi_list: List[np.ndarray] = []
    pair_list: List[np.ndarray] = []
    pair_mask = None
    partner_residues = None
    collected = 0
    chunk_size = max(1, int(traj_chunk_frames))
    align_indices = [atom.index for atom in reference.topology.atoms]
    stop_loading = False

    for traj_file in traj_files:
        if stop_loading:
            break
        try:
            for traj_chunk in md.iterload(str(traj_file), top=pdb_path, stride=frame_stride, chunk=chunk_size):
                if traj_chunk.n_frames == 0:
                    continue
                if align:
                    traj_chunk = traj_chunk.superpose(reference, atom_indices=align_indices)
                positions_list.append(traj_chunk.xyz.astype(np.float32))
                collected += traj_chunk.n_frames

                if use_torsion:
                    phi_angles = md.compute_phi(traj_chunk)[1]
                    psi_angles = md.compute_psi(traj_chunk)[1]
                    phi_list.append(phi_angles.astype(np.float32))
                    psi_list.append(psi_angles.astype(np.float32))
                if use_pairwise:
                    ca_traj = traj_chunk.atom_slice(traj_chunk.top.select("name CA"))
                    pair_dists, mask, partners = compute_pairwise_distances_per_residue(
                        ca_traj, step=pair_step, min_separation=pair_min_separation
                    )
                    pair_list.append(pair_dists.astype(np.float32))
                    if pair_mask is None:
                        pair_mask = mask
                        partner_residues = partners

                if max_frames is not None and collected >= max_frames:
                    stop_loading = True
                    break
        except Exception as exc:
            print(f"[prepare] Warning: failed to load {traj_file} ({exc}); skipping.")

    if not positions_list:
        raise RuntimeError("No trajectory frames were collected.")

    positions = np.concatenate(positions_list, axis=0)
    positions = positions * 10.0  # convert from nm (MDTraj default) to Å to match notebook
    positions = positions - positions.mean(axis=1, keepdims=True)
    total_frames = positions.shape[0]
    # Trim any excess if the last chunk overshot max_frames
    if max_frames is not None and total_frames > max_frames:
        keep = max_frames
        positions = positions[:keep]
        total_frames = positions.shape[0]

    extras = {}
    torsion_cond = None
    pair_cond = None
    phi_idx = psi_idx = phi_res_idx = psi_res_idx = None
    if use_torsion:
        phi_idx, _ = md.compute_phi(reference)
        psi_idx, _ = md.compute_psi(reference)
        # Map torsion definitions to residue indices for per-residue conditioning
        try:
            phi_res_idx = _torsion_residue_indices(reference, phi_idx)
            psi_res_idx = _torsion_residue_indices(reference, psi_idx)
        except Exception:
            phi_res_idx = None
            psi_res_idx = None
        n_phi = phi_idx.shape[0]
        n_psi = psi_idx.shape[0]
        phi_angles = np.concatenate(phi_list, axis=0)[:, :n_phi][:total_frames]
        psi_angles = np.concatenate(psi_list, axis=0)[:, :n_psi][:total_frames]
        phi_deg = np.degrees(phi_angles).astype(np.float32)
        psi_deg = np.degrees(psi_angles).astype(np.float32)
        phi_sin = np.sin(phi_angles, dtype=np.float32)
        phi_cos = np.cos(phi_angles, dtype=np.float32)
        psi_sin = np.sin(psi_angles, dtype=np.float32)
        psi_cos = np.cos(psi_angles, dtype=np.float32)
        torsion_cond = np.concatenate([phi_sin, phi_cos, psi_sin, psi_cos], axis=1).astype(np.float32)
        extras.update(
            dict(
                phi_angles=phi_angles,
                psi_angles=psi_angles,
                phi_angles_deg=phi_deg,
                psi_angles_deg=psi_deg,
                phi_sin=phi_sin,
                phi_cos=phi_cos,
                psi_sin=psi_sin,
                psi_cos=psi_cos,
                phi_res_idx=phi_res_idx if phi_res_idx is not None else np.array([], dtype=np.int32),
                psi_res_idx=psi_res_idx if psi_res_idx is not None else np.array([], dtype=np.int32),
            )
        )
    if use_pairwise:
        pair_distances = np.concatenate(pair_list, axis=0)[:total_frames]
        pair_cond = pair_distances.reshape(total_frames, -1).astype(np.float32)
        extras.update(
            dict(
                pair_distances=pair_distances,
                pair_mask=pair_mask,
                partner_residues=partner_residues,
                pair_step=np.array(pair_step, dtype=np.int32),
                pair_min_separation=np.array(pair_min_separation, dtype=np.int32),
            )
        )

    if conditioning_mode == "torsion":
        if cond_injection_mode == "per_residue":
            n_res = reference.n_residues
            per_res_feats = build_per_residue_torsion_features(
                phi_sin,
                phi_cos,
                psi_sin,
                psi_cos,
                phi_res_idx,
                psi_res_idx,
                n_res,
            )
            conditioning = per_res_feats.reshape(total_frames, -1)
            extras["per_res_torsion"] = np.array(True)
        else:
            conditioning = torsion_cond
    elif conditioning_mode == "pairwise":
        conditioning = pair_cond
    elif conditioning_mode == "hybrid":
        if torsion_cond is None or pair_cond is None:
            raise RuntimeError("Hybrid mode requires both torsion and pairwise features.")
        if cond_injection_mode == "per_residue":
            n_res = reference.n_residues
            per_res_feats = build_per_residue_torsion_features(
                phi_sin,
                phi_cos,
                psi_sin,
                psi_cos,
                phi_res_idx,
                psi_res_idx,
                n_res,
            )
            torsion_block = per_res_feats.reshape(total_frames, -1)
            extras["per_res_torsion"] = np.array(True)
        else:
            torsion_block = torsion_cond
        conditioning = np.concatenate([torsion_block, pair_cond], axis=1).astype(np.float32)
        extras["hybrid_torsion_dim"] = np.array(torsion_block.shape[1], dtype=np.int32)
        extras["hybrid_pair_dim"] = np.array(pair_cond.shape[1], dtype=np.int32)
    else:
        raise ValueError(f"Unknown conditioning mode '{conditioning_mode}'")

    # Optional PCA-based balanced subsampling in conditioning space
    if pca_balance and positions.shape[0] > 0 and conditioning.shape[1] >= 2:
        try:
            n_frames = conditioning.shape[0]
            # Fit PCA on a (possibly capped) subset
            fit_idx = np.arange(n_frames, dtype=int)
            if pca_balance_fit_cap and n_frames > pca_balance_fit_cap:
                fit_idx = np.linspace(0, n_frames - 1, num=pca_balance_fit_cap, dtype=int)
            cond_fit = conditioning[fit_idx]
            pca = PCA(n_components=2)
            pca.fit(cond_fit)
            # Project all frames
            proj_all = pca.transform(conditioning)  # (N, 2)
            # Build histogram for diagnostics and bin edges
            nbins = int(max(1, pca_balance_bins))
            counts, x_edges, y_edges = np.histogram2d(
                proj_all[:, 0], proj_all[:, 1], bins=nbins
            )
            # Assign each frame to a bin
            bx = np.clip(np.digitize(proj_all[:, 0], x_edges) - 1, 0, nbins - 1)
            by = np.clip(np.digitize(proj_all[:, 1], y_edges) - 1, 0, nbins - 1)
            bin_ids = bx + nbins * by
            # Balanced sampling per bin
            max_per = int(max(0, pca_balance_max_per_bin))
            if max_per <= 0:
                selected_idx = np.arange(n_frames, dtype=int)
            else:
                selected = []
                for b in range(nbins * nbins):
                    idx = np.where(bin_ids == b)[0]
                    if idx.size == 0:
                        continue
                    if idx.size <= max_per:
                        selected.extend(idx.tolist())
                    else:
                        # Randomly sample within bin for diversity
                        choice = np.random.choice(idx, size=max_per, replace=False)
                        selected.extend(choice.tolist())
                selected_idx = np.asarray(sorted(set(selected)), dtype=int)

            if selected_idx.size > 0 and selected_idx.size < n_frames:
                print(
                    f"[prepare] PCA-balanced subsampling: {selected_idx.size}/{n_frames} frames "
                    f"kept across {nbins}x{nbins} bins (max {max_per} per bin)."
                )
                positions = positions[selected_idx]
                conditioning = conditioning[selected_idx]
                # Subsample any frame-aligned extras (first dimension = frames)
                for key, val in list(extras.items()):
                    if isinstance(val, np.ndarray) and val.shape and val.shape[0] == n_frames:
                        extras[key] = val[selected_idx]
                total_frames = positions.shape[0]

            # Save a before/after density plot in PCA space
            try:
                plot_path = Path(pca_balance_plot) if pca_balance_plot else Path(output_path).with_name(
                    Path(output_path).stem + "_pca_balance.png"
                )
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
                hb1 = axes[0].hexbin(
                    proj_all[:, 0],
                    proj_all[:, 1],
                    gridsize=nbins,
                    cmap="magma",
                    norm=LogNorm(),
                )
                axes[0].set_title(f"Before (N={n_frames})")
                fig.colorbar(hb1, ax=axes[0], label="Counts")
                proj_sub = proj_all[selected_idx] if 'selected_idx' in locals() else proj_all
                hb2 = axes[1].hexbin(
                    proj_sub[:, 0],
                    proj_sub[:, 1],
                    gridsize=nbins,
                    cmap="magma",
                    norm=LogNorm(),
                )
                axes[1].set_title(f"After (N={proj_sub.shape[0]})")
                fig.colorbar(hb2, ax=axes[1], label="Counts")
                for ax in axes:
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel("PC 2")
                fig.suptitle("PCA Space Density Before/After Balancing", fontsize=14)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)
                print(f"[prepare] Saved PCA balance summary plot to {plot_path}")
            except Exception as e:
                print(f"[prepare] Warning: failed to save PCA balance plot ({e})")
        except Exception as e:
            print(f"[prepare] Warning: PCA-balance step failed ({e}); using unbalanced dataset.")
    # Optional TICA-based balanced subsampling to avoid bias
    if tica_balance and positions.shape[0] > 0 and conditioning.shape[1] >= 2:
        try:
            # Fit or load TICA
            proj_all = None
            model_path = tica_balance_model or ""
            if model_path and os.path.isfile(model_path):
                import pickle
                with open(model_path, "rb") as f:
                    tica_model = pickle.load(f)
                proj_all = tica_model.transform(conditioning)
            else:
                try:
                    from deeptime.decomposition import TICA as _TICA
                    tica_model = _TICA(lagtime=max(1, int(tica_balance_lagtime)), dim=2)
                    tica_model.fit([conditioning])
                    proj_all = tica_model.transform(conditioning)
                except Exception as exc:
                    print(f"[prepare] TICA balance skipped: unable to fit TICA model ({exc}).")
            if proj_all is not None:
                if proj_all.shape[1] > 2:
                    proj_all = proj_all[:, :2]
                n_frames = proj_all.shape[0]
                nbins = int(max(1, tica_balance_bins))
                counts, x_edges, y_edges = np.histogram2d(
                    proj_all[:, 0], proj_all[:, 1], bins=nbins
                )
                bx = np.clip(np.digitize(proj_all[:, 0], x_edges) - 1, 0, nbins - 1)
                by = np.clip(np.digitize(proj_all[:, 1], y_edges) - 1, 0, nbins - 1)
                bin_ids = bx + nbins * by
                max_per = int(max(0, tica_balance_max_per_bin))
                if max_per <= 0:
                    selected_idx = np.arange(n_frames, dtype=int)
                else:
                    selected = []
                    for b in range(nbins * nbins):
                        idx = np.where(bin_ids == b)[0]
                        if idx.size == 0:
                            continue
                        if idx.size <= max_per:
                            selected.extend(idx.tolist())
                        else:
                            choice = np.random.choice(idx, size=max_per, replace=False)
                            selected.extend(choice.tolist())
                    selected_idx = np.asarray(sorted(set(selected)), dtype=int)
                if selected_idx.size > 0 and selected_idx.size < n_frames:
                    print(
                        f"[prepare] TICA-balanced subsampling: {selected_idx.size}/{n_frames} frames "
                        f"kept across {nbins}x{nbins} bins (max {max_per} per bin)."
                    )
                    positions = positions[selected_idx]
                    conditioning = conditioning[selected_idx]
                    for key, val in list(extras.items()):
                        if isinstance(val, np.ndarray) and val.shape and val.shape[0] == n_frames:
                            extras[key] = val[selected_idx]
                    total_frames = positions.shape[0]
                if selected_idx.size > 0:
                    try:
                        plot_path = Path(tica_balance_plot) if tica_balance_plot else Path(output_path).with_name(
                            Path(output_path).stem + "_tica_balance.png"
                        )
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
                        hb1 = axes[0].hexbin(
                            proj_all[:, 0],
                            proj_all[:, 1],
                            gridsize=nbins,
                            cmap="magma",
                            norm=LogNorm(),
                        )
                        axes[0].set_title(f"Before (N={n_frames})")
                        fig.colorbar(hb1, ax=axes[0], label="Counts")
                        proj_sub = proj_all[selected_idx] if 'selected_idx' in locals() else proj_all
                        hb2 = axes[1].hexbin(
                            proj_sub[:, 0],
                            proj_sub[:, 1],
                            gridsize=nbins,
                            cmap="magma",
                            norm=LogNorm(),
                        )
                        axes[1].set_title(f"After (N={proj_sub.shape[0]})")
                        fig.colorbar(hb2, ax=axes[1], label="Counts")
                        for ax in axes:
                            ax.set_xlabel("TIC 1")
                            ax.set_ylabel("TIC 2")
                        fig.suptitle("TICA Space Density Before/After Balancing", fontsize=14)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])
                        plot_path.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(plot_path, dpi=200)
                        plt.close(fig)
                        print(f"[prepare] Saved TICA balance summary plot to {plot_path}")
                    except Exception as e:
                        print(f"[prepare] Warning: failed to save TICA balance plot ({e})")
        except Exception as e:
            print(f"[prepare] Warning: TICA-balance step failed ({e}); using unbalanced dataset.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extras.setdefault("conditioning", conditioning)
    extras.setdefault("conditioning_mode", np.array(conditioning_mode))
    extras.setdefault("cond_injection_mode", np.array(cond_injection_mode))
    np.savez_compressed(
        output_path,
        positions=positions.astype(np.float32),
        atomic_numbers=atomic_numbers,
        segment_counts=segment_counts,
        **extras,
    )
    print(f"[prepare] Saved dataset with {total_frames} frames to {output_path}")
    return output_path


class DiffusionDataset(Dataset):
    def __init__(self, positions: np.ndarray, cond: np.ndarray, cond_mean: np.ndarray, cond_std: np.ndarray):
        self.positions = torch.from_numpy(positions.astype(np.float32))
        norm_cond = (cond - cond_mean) / cond_std
        self.cond = torch.from_numpy(norm_cond.astype(np.float32))

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, idx: int):
        frame = self.positions[idx].transpose(0, 1).contiguous().unsqueeze(-1)
        return frame, self.cond[idx]


def _build_model(
    pdb_path: str,
    atomic_numbers: np.ndarray,
    heavy_mask: np.ndarray,
    model_cfg: dict,
    cond_dim: int,
) -> Tuple[ConditionalHierarchicalCartesianDiffusion, dict]:
    traj = md.load(pdb_path)
    res_idx_pad, res_mask, heavy_atom_indices = build_residue_maps(traj)
    atom_types = torch.from_numpy(atomic_numbers[heavy_mask]).long()
    defaults = {
        "d_model": 128,
        "latent_dim": 16,
        "nhead": 8,
        "num_layers_intra": 4,
        "num_layers_inter": 4,
        "dropout": 0.0,
        "atom_mixer": "transformer",
        "latent_level": "atom",
        "ungroup_mode": "split",
        "learn_sigma": True,
        "num_residue_sinks": 3,
    }
    config = defaults.copy()
    extra_cfg = dict(model_cfg or {})
    cond_injection_mode = extra_cfg.pop("cond_injection_mode", "global")
    config.update({k: extra_cfg.pop(k) for k in list(extra_cfg.keys()) if k in defaults})
    model = ConditionalHierarchicalCartesianDiffusion(
        n_atoms=heavy_mask.sum(),
        n_atom_types=int(atom_types.max().item()),
        R=res_idx_pad.shape[0],
        max_atoms_res=res_idx_pad.shape[1],
        res_idx_pad=res_idx_pad.clone(),
        res_mask=res_mask.clone(),
        atom_types=atom_types,
        cond_dim=cond_dim,
        cond_injection_mode=cond_injection_mode,
        **config,
    )
    config.update(
        {
            "n_atoms": int(heavy_mask.sum()),
            "n_atom_types": int(atom_types.max().item()),
            "R": int(res_idx_pad.shape[0]),
            "max_atoms_res": int(res_idx_pad.shape[1]),
            "cond_dim": cond_dim,
            "cond_injection_mode": cond_injection_mode,
        }
    )
    config.update(extra_cfg)
    return model, config

def _ensure_single_process_dist(device: str) -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for Muon. Please install PyTorch with distributed support.")
    if dist.is_initialized():
        return
    backend = "nccl" if device.startswith("cuda") and torch.cuda.is_available() else "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def _ensure_parameter_grads(params: Iterable[torch.nn.Parameter]) -> None:
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)


def _build_optimizers(model: torch.nn.Module, lr: float, use_muon: bool, device: str):
    params = [p for p in model.parameters() if p.requires_grad]
    if not use_muon:
        return [torch.optim.AdamW(params, lr=lr)]
    try:
        from flash_muon import Muon
    except ImportError as exc:
        raise RuntimeError("flash_muon is required when --use-muon is enabled. Please install flash-muon.") from exc
    _ensure_single_process_dist(device)
    muon_params = [p for p in params if p.ndim == 2 and p.grad is not None]
    adamw_params = [p for p in params if p.ndim != 2 or p.grad is None]
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=0.02, momentum=0.95, rank=0, world_size=1))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=lr, betas=(0.90, 0.95), weight_decay=0.01))
    if not optimizers:
        optimizers.append(torch.optim.AdamW(params, lr=lr))
    return optimizers


def train_diffusion(
    data_path: str,
    pdb_path: str,
    checkpoint_path: str,
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    diffusion_steps: int,
    model_cfg: dict,
    use_muon: bool = False,
    init_checkpoint: str | None = None,
) -> None:
    data = np.load(data_path)
    atomic_numbers = data["atomic_numbers"]
    heavy_mask = atomic_numbers != 1
    positions = data["positions"][:, heavy_mask, :]
    conditioning_mode = data.get("conditioning_mode", np.array("torsion")).item()
    if "conditioning" in data.files:
        cond = data["conditioning"]
    else:
        phi_sin = data["phi_sin"]
        phi_cos = data["phi_cos"]
        psi_sin = data["psi_sin"]
        psi_cos = data["psi_cos"]
        cond = np.concatenate([phi_sin, phi_cos, psi_sin, psi_cos], axis=1)
    cond_mean = cond.mean(axis=0, keepdims=True)
    cond_std = np.clip(cond.std(axis=0, keepdims=True), 1e-6, None)
    hybrid_torsion_dim = int(data.get("hybrid_torsion_dim", np.array(0)))
    hybrid_pair_dim = int(data.get("hybrid_pair_dim", np.array(0)))

    dataset = DiffusionDataset(positions, cond, cond_mean, cond_std)
    val_size = max(1, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model_cfg = dict(model_cfg)
    if conditioning_mode == "hybrid":
        if hybrid_torsion_dim <= 0:
            hybrid_torsion_dim = cond.shape[1] // 2
        model_cfg["hybrid_torsion_dim"] = hybrid_torsion_dim

    model, config = _build_model(
        pdb_path,
        atomic_numbers,
        heavy_mask,
        model_cfg,
        cond_dim=cond.shape[1],
    )
    if conditioning_mode == "hybrid":
        config["hybrid_torsion_dim"] = hybrid_torsion_dim
        config["hybrid_pair_dim"] = hybrid_pair_dim
    config["conditioning_mode"] = conditioning_mode
    model = model.to(device)
    print("=== Diffusion Training Configuration ===")
    print(f"Data: {data_path} | PDB: {pdb_path}")
    print(f"Device: {device} | Batch size: {batch_size} | Epochs: {epochs} | Steps: {diffusion_steps}")
    print(f"Learning rate: {lr} | Optimizer: {'Muon+AdamW' if use_muon else 'AdamW'}")
    print("Model config:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("===============================\n")
    print(model)
    print("\n===============================\n")
    # Optionally initialize from an existing checkpoint
    if init_checkpoint:
        try:
            init_path = Path(init_checkpoint)
            if init_path.exists():
                ckpt = torch.load(init_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"[train] Loaded init checkpoint from {init_path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            else:
                print(f"[train] Init checkpoint not found at {init_path}; continuing from random init.")
        except Exception as e:
            print(f"[train] Failed to load init checkpoint ({e}); continuing from random init.")

    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=diffusion_steps)
    optimizers = _build_optimizers(model, lr=lr, use_muon=use_muon, device=str(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    best_loss = float("inf")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - train", leave=False)
        for step, (xb, condb) in enumerate(train_bar, start=1):
            xb = xb.to(device)
            condb = condb.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (xb.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, xb, t, model_kwargs={"cond": condb})
            loss = loss_dict["loss"].mean()
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            _ensure_parameter_grads(trainable_params)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            for opt in optimizers:
                opt.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / step)
        avg_train = running_loss / max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - val", leave=False)
            for step, (xb, condb) in enumerate(val_bar, start=1):
                xb = xb.to(device)
                condb = condb.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (xb.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, xb, t, model_kwargs={"cond": condb})
                val_loss += loss_dict["loss"].mean().item()
                val_bar.set_postfix(loss=val_loss / step)
            val_loss /= max(1, len(val_loader))

        print(f"[train] Epoch {epoch:03d} | train {avg_train:.4f} | val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cond_mean": torch.from_numpy(cond_mean.squeeze().astype(np.float32)),
                    "cond_std": torch.from_numpy(cond_std.squeeze().astype(np.float32)),
                    "config": {
                        **config,
                        "diffusion_steps": diffusion_steps,
                        "conditioning_mode": conditioning_mode,
                    },
                },
                checkpoint_path,
            )
            print(f"[train] Saved checkpoint to {checkpoint_path} (val {val_loss:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare torsion-conditioned datasets and train diffusion checkpoints.")
    parser.add_argument("--pdb", default="", help="Reference PDB file.")
    parser.add_argument("--traj-dir", default="", help="Directory containing trajectory files.")
    parser.add_argument("--traj-pattern", default="*.dcd", help="Glob pattern to select trajectory files.")
    parser.add_argument("--work-dir", default="diffusion_runs", help="Directory to write datasets/checkpoints.")
    parser.add_argument("--data-path", default="", help="Optional precomputed dataset NPZ path.")
    parser.add_argument("--checkpoint-path", default="", help="Where to write the trained checkpoint.")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint to initialize model weights (resume/fine-tune).")
    parser.add_argument("--no-align", action="store_true", help="Disable alignment to the reference PDB before aggregation.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame from each trajectory.")
    parser.add_argument("--traj-chunk-frames", type=int, default=500, help="Frames per chunk when streaming trajectories (prevents large in-memory loads).")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on total frames (0 = all).")
    parser.add_argument("--conditioning-mode", choices=["torsion", "pairwise", "hybrid"], default="torsion", help="Conditioning features for diffusion (torsion sin/cos, sparse pairwise distances, or both).")
    parser.add_argument("--pair-step", type=int, default=10, help="Residue spacing for pairwise conditioning mode.")
    parser.add_argument("--pair-min-sep", type=int, default=10, help="Minimum residue separation for pairwise conditioning mode.")
    parser.add_argument("--device", default="cuda", help="Torch device for training (e.g., cuda or cpu).")
    parser.add_argument("--batch-size", type=int, default=64, help="Diffusion training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Diffusion training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Diffusion training learning rate.")
    parser.add_argument("--diffusion-steps", type=int, default=1000, help="Number of diffusion timesteps.")
    parser.add_argument("--use-muon", action="store_true", help="Use Flash-Attn Muon optimizer for matrix parameters (rest via AdamW).")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer width for diffusion model.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension for diffusion model.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num-layers-intra", type=int, default=4, help="Number of intra-residue transformer layers.")
    parser.add_argument("--num-layers-inter", type=int, default=4, help="Number of inter-residue transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout applied throughout the diffusion model.")
    parser.add_argument("--atom-mixer", choices=["transformer", "gated_deltanet"], default="transformer", help="Atom-level token mixer for intra-residue encoding/decoding.")
    parser.add_argument("--num-residue-sinks", type=int, default=3, help="Number of attention sink tokens to prepend at residue level (0 disables).")
    parser.add_argument("--cond-injection-mode", choices=["global", "per_residue"], default="global", help="How to inject conditioning information inside the diffusion network.")
    # Optional PCA-balanced subsampling of conditioning space
    parser.add_argument("--pca-balance", action="store_true", help="Enable PCA-based balanced subsampling of frames before saving dataset.")
    parser.add_argument("--pca-balance-bins", type=int, default=50, help="Number of bins per axis in PCA histogram (default 50).")
    parser.add_argument("--pca-balance-max-per-bin", type=int, default=0, help="Max frames per PCA bin (0 disables balancing).")
    parser.add_argument("--pca-balance-fit-cap", type=int, default=50000, help="Cap frames used to fit PCA (0 = all).")
    parser.add_argument("--pca-balance-plot", default="", help="Optional path for PCA balance before/after plot (PNG).")
    parser.add_argument("--tica-balance", action="store_true", help="Enable TICA-based balanced subsampling using a provided TICA model.")
    parser.add_argument("--tica-balance-bins", type=int, default=50, help="Number of bins per axis in TICA histogram (default 50).")
    parser.add_argument("--tica-balance-max-per-bin", type=int, default=0, help="Max frames per TICA bin (0 disables balancing).")
    parser.add_argument("--tica-balance-plot", default="", help="Optional path for TICA balance before/after plot (PNG).")
    parser.add_argument("--tica-balance-model", default="", help="Path to TICA model (pickle) for TICA-based balancing.")
    parser.add_argument("--tica-balance-lagtime", type=int, default=10, help="Lagtime for on-the-fly TICA fitting during balance.")
    parser.add_argument("--config", default="", help="YAML file to override CLI defaults for data/model/training.")
    parser.add_argument("--data-only", action="store_true", help="Only prepare the dataset; skip training.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip preparation and reuse an existing dataset.")
    return parser.parse_args()


def _merge_section(args: argparse.Namespace, section: dict, section_name: str):
    if not section:
        return
    for key, value in section.items():
        if section_name == "data" and key == "align":
            args.no_align = not bool(value)
            continue
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"[config] Warning: unknown option '{key}' in section '{section_name}'")


def apply_config(args: argparse.Namespace, config_path: str) -> None:
    if not config_path:
        return
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config support. Please install pyyaml.")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a mapping at the top level.")
    _merge_section(args, cfg.get("data"), "data")
    _merge_section(args, cfg.get("model"), "model")
    _merge_section(args, cfg.get("training"), "training")


def main():
    args = parse_args()
    apply_config(args, args.config)
    if not args.pdb:
        raise ValueError("--pdb is required (provide via CLI or config).")
    if not args.traj_dir and not args.skip_prepare:
        raise ValueError("--traj-dir is required unless --skip-prepare is set.")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path) if args.data_path else work_dir / "prepared_dataset.npz"
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else work_dir / "diffusion_checkpoint.pth"

    if not args.skip_prepare or not data_path.exists():
        max_frames = None if args.max_frames <= 0 else args.max_frames
        prepare_dataset(
            args.pdb,
            args.traj_dir,
            args.traj_pattern,
            str(data_path),
            align=not args.no_align,
            frame_stride=max(1, args.frame_stride),
            traj_chunk_frames=max(1, args.traj_chunk_frames),
            max_frames=max_frames,
            conditioning_mode=args.conditioning_mode,
            pair_step=args.pair_step,
            pair_min_separation=args.pair_min_sep,
            cond_injection_mode=args.cond_injection_mode,
            pca_balance=bool(getattr(args, "pca_balance", False)),
            pca_balance_bins=int(getattr(args, "pca_balance_bins", 50)),
            pca_balance_max_per_bin=int(getattr(args, "pca_balance_max_per_bin", 0)),
            pca_balance_fit_cap=int(getattr(args, "pca_balance_fit_cap", 50000)),
            pca_balance_plot=(args.pca_balance_plot or None),
            tica_balance=bool(getattr(args, "tica_balance", False)),
            tica_balance_bins=int(getattr(args, "tica_balance_bins", 50)),
            tica_balance_max_per_bin=int(getattr(args, "tica_balance_max_per_bin", 0)),
            tica_balance_plot=(args.tica_balance_plot or None),
            tica_balance_model=(args.tica_balance_model or getattr(args, "tica_model", "") or None),
        )
    else:
        print(f"[prepare] Skipping dataset prep and reusing {data_path}")

    if args.data_only:
        print("[train] Skipping diffusion training (data-only mode).")
        return

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = {
        "d_model": args.d_model,
        "latent_dim": args.latent_dim,
        "nhead": args.nhead,
        "num_layers_intra": args.num_layers_intra,
        "num_layers_inter": args.num_layers_inter,
        "dropout": args.dropout,
        "atom_mixer": args.atom_mixer,
        "cond_injection_mode": args.cond_injection_mode,
        "num_residue_sinks": max(0, int(args.num_residue_sinks)),
    }
    train_diffusion(
        str(data_path),
        args.pdb,
        str(checkpoint_path),
        device=str(device),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        diffusion_steps=args.diffusion_steps,
        model_cfg=model_cfg,
        use_muon=bool(getattr(args, "use_muon", False)),
        init_checkpoint=(args.init_checkpoint or None),
    )


if __name__ == "__main__":
    main()
