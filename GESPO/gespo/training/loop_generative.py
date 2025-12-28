"""
GESPO torsion-autoregressive generative loop with a geom2vec+VAMP backbone.

Key differences vs the standard discrete-start loop:
  - Start policy: `TorsionARPolicy` samples a full torsion sequence Θ.
  - Optional retrieval: Θ is mapped to a replay start frame from history by
    nearest-neighbor search in a periodic φ/ψ embedding space.
  - Rewards: per-trajectory ΔLOO-VAMP2 computed from geom2vec VAMP state
    probabilities (no ensemble/UCB), with an optional 2D bin-count bonus in the
    first two (linear) VAMP coordinates.
"""

from __future__ import annotations
from gespo.utils.logging import log_print

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import openmm
import torch
from openmm import unit
from openmm.app import Modeller, Simulation
from scipy.optimize import linear_sum_assignment

from ..config import TrainingConfig
from ..sampling import PeptideSampler
from ..utils.adk_metrics import ADK_GLOBAL_TOKEN_NAMES, build_adk_token_indices, compute_adk_global_values_from_frames
from ..utils.geometry import _dihedral, calculate_rmsd, find_all_backbone_dihedrals
from ..utils.openmm_utils import resolve_device
from ..utils.plotting import plot_analysis_summary, plot_ramachandran_by_residue
from .checkpointing import CheckpointManager
from .features import FeatureBuilder
from .geom2vec_backbone import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector, infer_and_cache
from .optimization import GRPOOptimizer
from .rewards import RewardCalculator
from .rl import StateLogitPolicy, TorsionARPolicy
from .sampler import Sampler
from .setup import initialize_projection_scorer
from .torsion_retrieval import (
    balanced_candidate_indices,
    compute_support_threshold,
    embed_phi_psi_from_cv_batch,
    embed_phi_psi_from_torsions,
    pairwise_sq_l2,
    sample_from_topk,
)


class SufficientStatistics:
    """
    Track sufficient statistics for fast leave-one-out VAMP-2 calculation.

    Maintains running covariance matrices C00, C11, C01 so that removing
    one trajectory requires only O(T) updates rather than full recomputation.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n_samples = 0
        self.sum_x = np.zeros(dim, dtype=np.float64)
        self.sum_y = np.zeros(dim, dtype=np.float64)
        self.C00 = np.zeros((dim, dim), dtype=np.float64)
        self.C11 = np.zeros((dim, dim), dtype=np.float64)
        self.C01 = np.zeros((dim, dim), dtype=np.float64)

    def add_trajectory(self, x: np.ndarray, y: np.ndarray) -> None:
        assert x.shape == y.shape
        n = int(x.shape[0])
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.C00 += x.T @ x
        self.C11 += y.T @ y
        self.C01 += x.T @ y
        self.n_samples += n

    def remove_trajectory(self, x: np.ndarray, y: np.ndarray) -> None:
        assert x.shape == y.shape
        n = int(x.shape[0])
        self.sum_x -= x.sum(axis=0)
        self.sum_y -= y.sum(axis=0)
        self.C00 -= x.T @ x
        self.C11 -= y.T @ y
        self.C01 -= x.T @ y
        self.n_samples -= n

    def compute_vamp2_score(self, epsilon: float = 1e-6) -> float:
        if self.n_samples < 2:
            return 0.0

        mu_x = self.sum_x / self.n_samples
        mu_y = self.sum_y / self.n_samples

        C00_centered = self.C00 / self.n_samples - np.outer(mu_x, mu_x)
        C11_centered = self.C11 / self.n_samples - np.outer(mu_y, mu_y)
        C01_centered = self.C01 / self.n_samples - np.outer(mu_x, mu_y)

        def _inv_sqrt(mat: np.ndarray) -> np.ndarray:
            vals, vecs = np.linalg.eigh(mat)
            vals = np.maximum(vals, float(epsilon))
            return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        try:
            C00_inv = _inv_sqrt(C00_centered)
            C11_inv = _inv_sqrt(C11_centered)
            K = C00_inv @ C01_centered @ C11_inv
            singular_values = np.linalg.svd(K, compute_uv=False)
            return float(np.sum(singular_values**2))
        except np.linalg.LinAlgError:
            return 0.0


def _align_states_by_overlap(
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Compute a permutation that best aligns new states to previous ones
    based on a confusion matrix of frame-wise assignments.
    """

    old_labels = np.asarray(old_labels, dtype=int)
    new_labels = np.asarray(new_labels, dtype=int)
    if old_labels.shape[0] != new_labels.shape[0]:
        raise ValueError(
            f"old_labels and new_labels must have same length, got {old_labels.shape[0]} vs {new_labels.shape[0]}"
        )

    old_populated = np.unique(old_labels)
    new_populated = np.unique(new_labels)
    old_populated = old_populated[(old_populated >= 0) & (old_populated < n_states)]
    new_populated = new_populated[(new_populated >= 0) & (new_populated < n_states)]

    if len(old_populated) == 0 or len(new_populated) == 0:
        return np.arange(n_states, dtype=int)

    overlap = np.zeros((len(old_populated), len(new_populated)), dtype=np.int64)
    old_to_idx = {state: idx for idx, state in enumerate(old_populated)}
    new_to_idx = {state: idx for idx, state in enumerate(new_populated)}

    for o, n in zip(old_labels, new_labels):
        if o in old_to_idx and n in new_to_idx:
            overlap[old_to_idx[o], new_to_idx[n]] += 1

    if overlap.sum() == 0:
        return np.arange(n_states, dtype=int)

    cost_matrix = -overlap.astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    permutation = np.arange(n_states, dtype=int)
    for old_idx, new_idx in zip(row_ind, col_ind):
        old_state = int(old_populated[int(old_idx)])
        new_state = int(new_populated[int(new_idx)])
        permutation[old_state] = new_state

    used_new = set(permutation[old_populated])
    all_new = set(range(n_states))
    unused_new = [s for s in sorted(all_new) if s not in used_new]

    for old_state in range(n_states):
        if old_state in old_populated:
            continue
        if old_state not in used_new:
            permutation[old_state] = old_state
            used_new.add(old_state)
        elif unused_new:
            new_state = unused_new.pop(0)
            permutation[old_state] = new_state
            used_new.add(new_state)

    log_print(
        f"[State Alignment] Matched {len(row_ind)} populated states: {len(old_populated)} old → {len(new_populated)} new"
    )
    return permutation


def train(args: argparse.Namespace) -> None:
    config = TrainingConfig.from_args(args)
    proj_mode = str(getattr(config, "projection", "geom2vec_vamp")).lower()
    if proj_mode != "geom2vec_vamp":
        raise ValueError(
            "The generative loop requires --projection=geom2vec_vamp. "
            f"Got --projection={proj_mode!r}."
        )
    if not str(getattr(config, "geom2vec_checkpoint", "") or "").strip():
        raise ValueError("--geom2vec-ckpt is required when --projection=geom2vec_vamp.")

    # Generative loop uses continuous torsion-based starts, not discrete state starts.
    config.discrete_state_starts = False
    # The generative loop uses torsion starts, but we still maintain a softmax
    # VAMP lobe (for LOO scoring / state labels) alongside a linear lobe (for
    # continuous diagnostics / binning).

    if not hasattr(config, "max_pca_train") or config.max_pca_train < 100000:
        config.max_pca_train = 100000
    if not hasattr(config, "history_reward_weight"):
        config.history_reward_weight = 5.0

    torch_device = "cuda" if torch.cuda.is_available() and config.platform != "CPU" else "cpu"
    gen_device = resolve_device(config.gen_device)

    log_print("GESPO Training (generative loop) with geom2vec VAMP + LOO rewards")
    log_print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    log_print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")

    # Initialize Environments
    log_print("Initializing environments...")
    envs = [PeptideSampler(args, generator_device=gen_device) for _ in range(config.grp_size)]
    env_ref = envs[0]

    # ------------------------------------------------------------------
    # TorsionAR metadata: residue types and angle mask
    # ------------------------------------------------------------------
    three_to_one = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    fallback_idx = len(aa_to_idx)
    try:
        ref_top = md.load(config.pdb).topology
        res_codes = [three_to_one.get(res.name.upper(), "X") for res in ref_top.residues]
        residue_types_1d = np.array([aa_to_idx.get(code, fallback_idx) for code in res_codes], dtype=np.int64)
    except Exception as exc_rt:
        log_print(f"[TorsionAR] Warning: failed to build residue type map from PDB: {exc_rt}")
        residue_types_1d = np.zeros((env_ref.n_residues,), dtype=np.int64)

    if residue_types_1d.shape[0] != env_ref.n_residues:
        log_print(
            f"[TorsionAR] Warning: residue type length {residue_types_1d.shape[0]} != env.n_residues {env_ref.n_residues}; "
            "truncating/padding."
        )
        if residue_types_1d.shape[0] > env_ref.n_residues:
            residue_types_1d = residue_types_1d[: env_ref.n_residues]
        else:
            pad = np.full((env_ref.n_residues - residue_types_1d.shape[0],), fallback_idx, dtype=np.int64)
            residue_types_1d = np.concatenate([residue_types_1d, pad], axis=0)

    residue_types_tensor = torch.as_tensor(residue_types_1d, device=torch_device, dtype=torch.long)
    residue_types_batch = residue_types_tensor.unsqueeze(0).expand(config.grp_size, -1)

    phi_exists = torch.from_numpy(env_ref._phi_exists.astype(bool)).to(torch_device)
    psi_exists = torch.from_numpy(env_ref._psi_exists.astype(bool)).to(torch_device)
    omega_exists = phi_exists & psi_exists
    angle_mask_1d = torch.stack([phi_exists, psi_exists, omega_exists], dim=-1)  # [L,3]
    angle_mask_batch = angle_mask_1d.unsqueeze(0).expand(config.grp_size, -1, -1)

    # Initialize TorsionAR start policy
    log_print("Initializing torsion autoregressive policy...")
    torsion_ckpt = getattr(args, "torsion_ar_checkpoint", "")
    if not torsion_ckpt:
        torsion_ckpt = getattr(args, "torsion_ar_out", "")
    if torsion_ckpt and not os.path.isfile(torsion_ckpt):
        log_print(f"[TorsionAR] Checkpoint '{torsion_ckpt}' not found; starting from scratch.")
        torsion_ckpt = ""

    ar_cfg: Dict[str, Any] = {}
    global_token_names: List[str] = []
    if torsion_ckpt:
        payload = torch.load(torsion_ckpt, map_location=torch_device)
        ar_cfg = payload.get("config", {}) or {}
        num_residues = int(ar_cfg.get("num_residues", env_ref.n_residues))
        residue_vocab_size = int(ar_cfg.get("residue_vocab_size", int(residue_types_1d.max()) + 1))
        d_model = int(ar_cfg.get("d_model", 256))
        n_heads = int(ar_cfg.get("n_heads", 8))
        n_layers = int(ar_cfg.get("n_layers", 4))
        kmix = int(ar_cfg.get("kmix", 8))
        n_global_tokens = int(ar_cfg.get("n_global_tokens", 0) or 0)
        try:
            names_cfg = ar_cfg.get("global_token_names", None)
        except Exception:
            names_cfg = None
        if isinstance(names_cfg, (list, tuple)) and len(names_cfg) == n_global_tokens:
            global_token_names = [str(x) for x in names_cfg]
        else:
            if n_global_tokens == 2:
                global_token_names = ["rmsd_nm", "e2e_nm"]
            elif n_global_tokens == len(ADK_GLOBAL_TOKEN_NAMES):
                global_token_names = list(ADK_GLOBAL_TOKEN_NAMES)
            else:
                global_token_names = [f"global_{i}" for i in range(max(0, n_global_tokens))]
        if num_residues != env_ref.n_residues:
            raise RuntimeError(
                f"TorsionAR checkpoint expects {num_residues} residues but env has {env_ref.n_residues}."
            )
        agent = TorsionARPolicy(
            num_residues=num_residues,
            residue_vocab_size=residue_vocab_size,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            dropout=0.1,
            kmix=kmix,
            n_global_tokens=n_global_tokens,
        ).to(torch_device)
        agent.load_state_dict(payload["state_dict"], strict=True)
        agent_old = TorsionARPolicy(
            num_residues=num_residues,
            residue_vocab_size=residue_vocab_size,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers,
            dropout=0.1,
            kmix=kmix,
            n_global_tokens=n_global_tokens,
        ).to(torch_device)
        agent_old.load_state_dict(payload["state_dict"], strict=True)
        agent_old.eval()
        log_print(f"[TorsionAR] Loaded pretrained policy from {torsion_ckpt}")
    else:
        num_residues = env_ref.n_residues
        residue_vocab_size = int(residue_types_1d.max()) + 1
        agent = TorsionARPolicy(num_residues=num_residues, residue_vocab_size=residue_vocab_size).to(torch_device)
        agent_old = TorsionARPolicy(num_residues=num_residues, residue_vocab_size=residue_vocab_size).to(torch_device)
        agent_old.load_state_dict(agent.state_dict())
        agent_old.eval()
        global_token_names = []
        log_print("[TorsionAR] Initialized new policy (no pretrained checkpoint provided)")

    policy_opts = [torch.optim.Adam(agent.parameters(), lr=config.lr)]
    optimizer = GRPOOptimizer(config, agent, agent_old, policy_opts)

    def _save_torsion_ar_policy(ep_idx_1based: int) -> None:
        ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
        if ckpt_dir is None:
            return
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        cfg = {}
        try:
            if torsion_ckpt:
                cfg = dict(ar_cfg) if isinstance(ar_cfg, dict) else {}
        except Exception:
            cfg = {}
        if not cfg:
            cfg = {
                "num_residues": int(getattr(agent, "num_residues", env_ref.n_residues)),
                "residue_vocab_size": int(getattr(agent, "residue_vocab_size", int(residue_types_1d.max()) + 1)),
                "d_model": int(getattr(agent, "d_model", 256)),
                "n_layers": int(getattr(agent, "num_layers", 4)),
                "n_heads": int(getattr(agent, "nhead", 8)),
                "kmix": int(getattr(agent, "kmix", 8)),
                "n_global_tokens": int(getattr(agent, "n_global_tokens", 0) or 0),
            }
        try:
            if global_token_names:
                cfg["global_token_names"] = list(global_token_names)
        except Exception:
            pass

        agent_path = ckpt_dir / f"torsion_ar_ep_{ep_idx_1based:04d}.pt"
        agent_old_path = ckpt_dir / f"torsion_ar_old_ep_{ep_idx_1based:04d}.pt"
        torch.save({"state_dict": agent.state_dict(), "config": cfg, "epoch": ep_idx_1based}, agent_path)
        torch.save({"state_dict": agent_old.state_dict(), "config": cfg, "epoch": ep_idx_1based}, agent_old_path)
        log_print(f"[TorsionAR] Saved policy checkpoints: {agent_path.name}, {agent_old_path.name}")

    # Checkpointing & History
    ckpt_manager = CheckpointManager(config)
    all_epoch_data: List[List[Any]] = []
    all_trajectory_lengths: List[List[int]] = []
    all_epoch_traj_dcd_paths: List[List[Optional[str]]] = []
    # Optional: for epochs/trajectories backed by a DCD on disk, store the
    # original frame indices used for each imported frame (e.g., after applying
    # history_dcd_stride or RMSD filtering). For trajectories produced by the
    # current run, this is typically None (implies 0..L-1).
    all_epoch_traj_dcd_frame_indices: List[List[Optional[np.ndarray]]] = []
    ep_start = 0
    pca_state = None

    # Retrieval cache for torsion-query replay starts.
    retrieval_hist_cv: List[np.ndarray] = []
    retrieval_hist_coords: List[np.ndarray] = []
    # Metadata for replay frames backed by on-disk trajectories.
    retrieval_hist_dcd_paths: List[Optional[str]] = []
    # Frame index in the source DCD (0-based); separate from `retrieval_hist_frame_idx`,
    # which is used for embedding/label mapping and is typically a dense 0..L-1 index.
    retrieval_hist_dcd_frame_idx: List[int] = []
    # Per-frame global metrics aligned with retrieval_hist_cv/coords (shape [G]):
    # typically matches the TorsionARPolicy global-token ordering.
    retrieval_hist_globals: List[np.ndarray] = []
    retrieval_hist_keys: List[str] = []
    retrieval_hist_frame_idx: List[int] = []
    retrieval_hist_labels: Optional[np.ndarray] = None
    retrieval_rng = np.random.default_rng(0)

    global_dim = int(getattr(agent_old, "n_global_tokens", 0) or 0)
    if global_dim > 0 and (not global_token_names or len(global_token_names) != global_dim):
        global_token_names = [f"global_{i}" for i in range(global_dim)]

    # Precompute CA mapping (and optional ADK indices) for fast global-metric extraction
    # from protein coords.
    retrieval_ca_local: np.ndarray = np.zeros((0,), dtype=np.int64)
    retrieval_ref_ca_nm: np.ndarray = np.zeros((0, 3), dtype=np.float64)
    retrieval_protein_indices: np.ndarray = np.zeros((0,), dtype=np.int64)
    adk_token_idx = None
    try:
        protein_indices = env_ref.md_topology.select("protein")
        if protein_indices is None or len(protein_indices) == 0:
            protein_indices = np.arange(env_ref.md_topology.n_atoms, dtype=np.int64)
        retrieval_protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
        idx_to_local = {int(a): int(i) for i, a in enumerate(np.asarray(protein_indices).reshape(-1))}
        ca_full = np.asarray(env_ref.ca_atom_indices, dtype=np.int64).reshape(-1)
        ca_local_by_slot = np.array([idx_to_local.get(int(a), -1) if int(a) >= 0 else -1 for a in ca_full], dtype=np.int64)
        valid_slots = (ca_local_by_slot >= 0) & np.asarray(getattr(env_ref, "_valid_ca_mask", ca_local_by_slot >= 0), dtype=bool)
        retrieval_ca_local = ca_local_by_slot[valid_slots].astype(np.int64, copy=False)
        retrieval_ref_ca_nm = np.asarray(getattr(env_ref, "_ref_ca_nm", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)[
            valid_slots
        ]
        if global_dim == len(ADK_GLOBAL_TOKEN_NAMES) or any(n in set(ADK_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                adk_token_idx = build_adk_token_indices(env_ref.md_topology, protein_indices=retrieval_protein_indices)
            except Exception:
                adk_token_idx = None
    except Exception:
        retrieval_ca_local = np.zeros((0,), dtype=np.int64)
        retrieval_ref_ca_nm = np.zeros((0, 3), dtype=np.float64)
        retrieval_protein_indices = np.zeros((0,), dtype=np.int64)
        adk_token_idx = None

    def _compute_globals_from_coords(coords_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute per-frame global metrics for a list of protein coordinate frames (nm).
        Returns [N,G] float32 aligned to `global_token_names`; falls back to zeros on failure.
        """
        if not coords_list:
            return np.zeros((0, max(0, global_dim)), dtype=np.float32)
        if global_dim <= 0:
            return np.zeros((len(coords_list), 0), dtype=np.float32)

        out = np.zeros((len(coords_list), global_dim), dtype=np.float32)

        # Default metrics (if requested by name): CA RMSD and end-to-end CA distance.
        need_rmsd = any(str(n).lower().startswith("rmsd") for n in global_token_names)
        need_e2e = any(str(n).lower().startswith("e2e") for n in global_token_names)
        rmsd_nm = None
        e2e_nm = None
        if (need_rmsd or need_e2e) and retrieval_ca_local.size >= 2 and retrieval_ref_ca_nm.shape[0] == retrieval_ca_local.size:
            try:
                coords_traj_ca = np.asarray(
                    [np.asarray(c, dtype=np.float64)[retrieval_ca_local] for c in coords_list], dtype=np.float64
                )
                if need_rmsd:
                    rmsd_nm = calculate_rmsd(coords_traj_ca, retrieval_ref_ca_nm, atom_indices=None).astype(
                        np.float32, copy=False
                    )
                if need_e2e:
                    e2e_nm = np.linalg.norm(coords_traj_ca[:, 0, :] - coords_traj_ca[:, -1, :], axis=1).astype(
                        np.float32, copy=False
                    )
            except Exception:
                rmsd_nm = None
                e2e_nm = None

        # ADK hinge/distance metrics (if requested).
        adk_vals = None
        if adk_token_idx is not None and any(n in set(ADK_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                adk_vals = compute_adk_global_values_from_frames(
                    coords_list,
                    ca_local=retrieval_ca_local,
                    ref_ca_nm=retrieval_ref_ca_nm,
                    idx=adk_token_idx,
                )
            except Exception:
                adk_vals = None

        try:
            adk_name_to_col = {name: i for i, name in enumerate(ADK_GLOBAL_TOKEN_NAMES)}
            for j, name in enumerate(global_token_names):
                key = str(name)
                key_l = key.lower()
                if key_l.startswith("rmsd"):
                    if adk_vals is not None:
                        out[:, j] = adk_vals[:, 0]
                    elif rmsd_nm is not None:
                        out[:, j] = rmsd_nm
                elif key_l.startswith("e2e"):
                    if e2e_nm is not None:
                        out[:, j] = e2e_nm
                elif adk_vals is not None and key in adk_name_to_col:
                    out[:, j] = adk_vals[:, adk_name_to_col[key]]
        except Exception:
            pass

        out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
        out = np.clip(out, 0.0, None)
        return out

    # Global (history-aware) LOO-VAMP sufficient statistics so trajectory rewards reflect
    # the marginal improvement to VAMP-2 over the accumulated dataset.
    loo_vamp_stats: Optional[SufficientStatistics] = None
    loo_vamp_stats_keys: set[str] = set()
    loo_vamp_stats_signature: Optional[Tuple[int, int]] = None  # (n_states, lag_embed)

    # Optional 2D bin-count statistics in the first two VAMP coordinates.
    # Used as an exploration bonus in the generative loop.
    vamp_bin_mu: Optional[np.ndarray] = None  # shape (2,)
    vamp_bin_std: Optional[np.ndarray] = None  # shape (2,)
    vamp_bin_counts: Optional[np.ndarray] = None  # shape (bins, bins)
    vamp_bin_total: int = 0
    vamp_bin_counted_keys: set[str] = set()
    vamp_bin_signature: Optional[Tuple[int, float, int, int]] = None  # (bins, clip, max_samples, n_states)
    vamp_bin_norm_ema_m1: Optional[float] = None
    vamp_bin_norm_ema_m2: Optional[float] = None
    vamp_bin_norm_ema_scale: Optional[float] = None

    def _invalidate_loo_vamp_stats(tag: str) -> None:
        nonlocal loo_vamp_stats, loo_vamp_stats_keys, loo_vamp_stats_signature
        loo_vamp_stats = None
        loo_vamp_stats_keys = set()
        loo_vamp_stats_signature = None
        log_print(f"[LOO VAMP] Invalidated global stats ({tag}).")

    def _invalidate_vamp_bin_stats(tag: str) -> None:
        nonlocal vamp_bin_mu, vamp_bin_std, vamp_bin_counts, vamp_bin_total, vamp_bin_counted_keys, vamp_bin_signature
        nonlocal vamp_bin_norm_ema_m1, vamp_bin_norm_ema_m2, vamp_bin_norm_ema_scale
        vamp_bin_mu = None
        vamp_bin_std = None
        vamp_bin_counts = None
        vamp_bin_total = 0
        vamp_bin_counted_keys = set()
        vamp_bin_signature = None
        vamp_bin_norm_ema_m1 = None
        vamp_bin_norm_ema_m2 = None
        vamp_bin_norm_ema_scale = None
        log_print(f"[VAMP Bins] Invalidated stats ({tag}).")

    def _canon_vamp_output(z_raw: np.ndarray, *, n_states: int, perm: Optional[np.ndarray]) -> np.ndarray:
        """Return canonical VAMP output array shaped [T, n_states], applying optional permutation."""
        z = np.asarray(z_raw, dtype=np.float32)
        if z.ndim != 2:
            z = z.reshape(z.shape[0], -1)
        if z.shape[1] != int(n_states):
            if z.shape[1] > int(n_states):
                z = z[:, : int(n_states)]
            else:
                z = np.pad(z, ((0, 0), (0, int(n_states) - z.shape[1])), mode="constant")
        if perm is not None:
            try:
                perm = np.asarray(perm, dtype=int).reshape(-1)
                if perm.size == int(n_states) and int(perm.max(initial=0)) < int(n_states):
                    z = z[:, perm]
            except Exception:
                pass
        return z

    def _vamp_bin_indices(z2: np.ndarray, *, bins: int, clip: float) -> Tuple[np.ndarray, np.ndarray]:
        """Map standardized 2D coords to integer bin indices (ix, iy) in [0, bins-1]."""
        z2 = np.asarray(z2, dtype=np.float32)
        if z2.ndim != 2 or z2.shape[1] != 2:
            raise ValueError("Expected z2 with shape [T, 2].")
        bins_i = int(bins)
        clip_f = float(clip)
        if bins_i <= 1 or clip_f <= 0:
            raise ValueError("Invalid bin parameters.")
        z2 = np.clip(z2, -clip_f, clip_f)
        scale = float(bins_i) / (2.0 * clip_f)
        ix = np.floor((z2[:, 0] + clip_f) * scale).astype(np.int64, copy=False)
        iy = np.floor((z2[:, 1] + clip_f) * scale).astype(np.int64, copy=False)
        ix = np.clip(ix, 0, bins_i - 1)
        iy = np.clip(iy, 0, bins_i - 1)
        return ix, iy

    def _iter_history_trajs(upto_epoch_exclusive: int):
        """Yield (epoch_idx, traj_idx, L_i, dcd_path, coords_nm_or_none)."""
        upto = max(0, min(int(upto_epoch_exclusive), len(all_epoch_data)))
        for epoch_idx in range(upto):
            epoch_data = all_epoch_data[epoch_idx]
            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            if not traj_lens:
                continue
            cursor = 0
            for traj_idx, L in enumerate(traj_lens):
                L_i = int(L)
                if L_i <= 0:
                    continue
                cursor_next = cursor + L_i
                dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                coords_nm = None
                if not dcd_path:
                    try:
                        coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor:cursor_next]]
                        coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                    except Exception:
                        coords_nm = None
                yield epoch_idx, traj_idx, L_i, dcd_path, coords_nm
                cursor = cursor_next

    def _ensure_vamp_bin_history_counts(*, upto_epoch_exclusive: int, n_states: int, tag: str) -> None:
        """Populate/extend the 2D VAMP-bin histogram from cached history trajectories."""
        nonlocal vamp_bin_mu, vamp_bin_std, vamp_bin_counts, vamp_bin_total, vamp_bin_counted_keys, vamp_bin_signature

        if (
            geom2vec_cache is None
            or geom2vec_projector_linear is None
            or getattr(geom2vec_projector_linear, "trained", None) is None
        ):
            return

        bins = int(getattr(config, "vamp_bin_bins", 50) or 50)
        clip = float(getattr(config, "vamp_bin_clip", 3.0) or 3.0)
        max_samples = int(getattr(config, "vamp_bin_max_samples", 20000) or 20000)
        n_states_i = int(n_states)
        if bins <= 1 or clip <= 0 or n_states_i < 2:
            return

        sig = (int(bins), float(clip), int(max_samples), int(n_states_i))
        if vamp_bin_counts is None or vamp_bin_signature != sig:
            vamp_bin_mu = None
            vamp_bin_std = None
            vamp_bin_counts = np.zeros((bins, bins), dtype=np.int64)
            vamp_bin_total = 0
            vamp_bin_counted_keys = set()
            vamp_bin_signature = sig

        if vamp_bin_counts is None:
            return

        perm = np.asarray(geom2vec_state_permutation_map, dtype=int)

        # Estimate mean/std (once) from a capped sample of history frames.
        if vamp_bin_mu is None or vamp_bin_std is None:
            samples: List[np.ndarray] = []
            n_collected = 0
            for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in _iter_history_trajs(upto_epoch_exclusive):
                if n_collected >= max_samples:
                    break
                key = _geom2vec_cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
                cached_key = _geom2vec_ensure_cached(
                    key=key,
                    dcd_path=dcd_path,
                    coords_nm=coords_nm,
                    overwrite=False,
                    meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                )
                if cached_key is None:
                    continue
                try:
                    graph, ca = geom2vec_cache.load(cached_key)
                except Exception:
                    continue
                try:
                    z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                    z = _canon_vamp_output(z_raw, n_states=n_states_i, perm=perm)
                    z2 = z[:, :2]
                except Exception:
                    continue
                if z2.size == 0:
                    continue
                remaining = max_samples - n_collected
                if int(z2.shape[0]) > int(remaining) and remaining > 0:
                    try:
                        idx = np.linspace(0, int(z2.shape[0]) - 1, num=int(remaining), dtype=np.int64)
                        z2 = z2[idx]
                    except Exception:
                        z2 = z2[: int(remaining)]
                samples.append(np.asarray(z2, dtype=np.float64))
                n_collected += int(z2.shape[0])
            if samples:
                arr = np.concatenate(samples, axis=0)
                mu = arr.mean(axis=0)
                std = arr.std(axis=0)
                std = np.where(std > 1e-6, std, 1.0)
                vamp_bin_mu = mu.astype(np.float32, copy=False)
                vamp_bin_std = std.astype(np.float32, copy=False)
            else:
                vamp_bin_mu = np.zeros((2,), dtype=np.float32)
                vamp_bin_std = np.ones((2,), dtype=np.float32)

        # Add any missing history keys into the histogram counts.
        added = 0
        skipped_missing = 0
        for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in _iter_history_trajs(upto_epoch_exclusive):
            key = _geom2vec_cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
            key_s = str(key)
            if key_s in vamp_bin_counted_keys:
                continue

            cached_key = key if _geom2vec_cache_has_key(key) else _geom2vec_ensure_cached(
                key=key,
                dcd_path=dcd_path,
                coords_nm=coords_nm,
                overwrite=False,
                meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
            )
            if cached_key is None:
                skipped_missing += 1
                continue
            cached_key_s = str(cached_key)
            if cached_key_s in vamp_bin_counted_keys:
                vamp_bin_counted_keys.add(key_s)
                continue

            try:
                graph, ca = geom2vec_cache.load(cached_key)
            except Exception:
                skipped_missing += 1
                continue

            try:
                z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                z = _canon_vamp_output(z_raw, n_states=n_states_i, perm=perm)
                z2 = z[:, :2]
                z2 = (z2 - vamp_bin_mu[None, :]) / vamp_bin_std[None, :]
                ix, iy = _vamp_bin_indices(z2, bins=bins, clip=clip)
                np.add.at(vamp_bin_counts, (ix, iy), 1)
                vamp_bin_total += int(ix.shape[0])
                vamp_bin_counted_keys.add(key_s)
                vamp_bin_counted_keys.add(cached_key_s)
                added += 1
            except Exception:
                skipped_missing += 1
                continue

        if added:
            try:
                visited = int(np.count_nonzero(vamp_bin_counts))
                nbins = int(bins) * int(bins)
                total = int(vamp_bin_total)
                log_print(
                    f"[VAMP Bins] Seeded history ({tag}): added_trajs={added}, skipped_missing={skipped_missing}, "
                    f"visited={visited}/{nbins}, total_frames={total}"
                )
            except Exception:
                log_print(f"[VAMP Bins] Seeded history ({tag}): added_trajs={added}, skipped_missing={skipped_missing}")

    def _ensure_loo_vamp_history_stats(
        *,
        upto_epoch_exclusive: int,
        n_states: int,
        lag_embed: int,
        tag: str,
    ) -> None:
        """
        Ensure `loo_vamp_stats` contains sufficient statistics for all cached
        trajectories in `all_epoch_data[:upto_epoch_exclusive]` under the *current* VAMP lobe.
        """
        nonlocal loo_vamp_stats, loo_vamp_stats_keys, loo_vamp_stats_signature

        if geom2vec_cache is None or geom2vec_projector is None or not _geom2vec_is_trained():
            return

        sig = (int(n_states), int(lag_embed))
        if loo_vamp_stats is None or loo_vamp_stats_signature != sig:
            loo_vamp_stats = SufficientStatistics(int(n_states))
            loo_vamp_stats_keys = set()
            loo_vamp_stats_signature = sig

        upto = max(0, min(int(upto_epoch_exclusive), len(all_epoch_data)))
        if upto == 0:
            return

        added = 0
        skipped_short = 0
        skipped_missing = 0
        perm = np.asarray(geom2vec_state_permutation_map, dtype=int)

        for epoch_idx in range(upto):
            epoch_data = all_epoch_data[epoch_idx]
            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            if not traj_lens:
                continue

            cursor = 0
            for traj_idx, L in enumerate(traj_lens):
                L_i = int(L)
                if L_i <= 0:
                    continue
                dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                key = _geom2vec_cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)
                cursor_next = cursor + L_i

                if key in loo_vamp_stats_keys:
                    cursor = cursor_next
                    continue

                cached_key = None
                if _geom2vec_cache_has_key(key):
                    cached_key = key
                else:
                    coords_nm = None
                    if not dcd_path:
                        try:
                            coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor:cursor_next]]
                            coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                        except Exception:
                            coords_nm = None
                    cached_key = _geom2vec_ensure_cached(
                        key=key,
                        dcd_path=dcd_path,
                        coords_nm=coords_nm,
                        overwrite=False,
                        meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                    )

                cursor = cursor_next

                if cached_key is None:
                    skipped_missing += 1
                    continue
                if cached_key in loo_vamp_stats_keys:
                    continue

                try:
                    rec = geom2vec_cache.get(cached_key)
                    if int(rec.frames) <= int(lag_embed):
                        loo_vamp_stats_keys.add(str(cached_key))
                        skipped_short += 1
                        continue
                except Exception:
                    pass

                try:
                    graph, ca = geom2vec_cache.load(cached_key)
                except Exception:
                    skipped_missing += 1
                    continue

                try:
                    probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)
                    probs = _canon_vamp_output(probs_raw, n_states=n_states, perm=perm)
                    T = int(probs.shape[0])
                    if T > lag_embed:
                        loo_vamp_stats.add_trajectory(probs[:-lag_embed], probs[lag_embed:])  # type: ignore[union-attr]
                except Exception:
                    skipped_missing += 1
                    continue

                loo_vamp_stats_keys.add(str(cached_key))
                added += 1

        if added:
            log_print(
                f"[LOO VAMP] Seeded global stats ({tag}): added_trajs={added}, "
                f"skipped_short={skipped_short}, skipped_missing={skipped_missing}"
            )

    def _append_retrieval_cache(
        epoch_data: List[Any],
        *,
        tag: str,
        epoch_idx: Optional[int] = None,
        traj_lengths: Optional[Sequence[int]] = None,
        traj_dcd_paths: Optional[Sequence[Optional[str]]] = None,
        traj_dcd_frame_indices: Optional[Sequence[Optional[Sequence[int]]]] = None,
        log: bool = True,
    ) -> None:
        nonlocal retrieval_hist_cv
        nonlocal retrieval_hist_coords
        nonlocal retrieval_hist_dcd_paths
        nonlocal retrieval_hist_dcd_frame_idx
        nonlocal retrieval_hist_globals
        nonlocal retrieval_hist_labels
        nonlocal retrieval_hist_keys
        nonlocal retrieval_hist_frame_idx
        if not epoch_data:
            return
        old_len = len(retrieval_hist_cv)
        new_cv = [np.asarray(item[0], dtype=np.float32) for item in epoch_data]
        new_coords = [np.asarray(item[1], dtype=np.float32) for item in epoch_data]
        retrieval_hist_cv.extend(new_cv)
        retrieval_hist_coords.extend(new_coords)
        try:
            glob_arr = _compute_globals_from_coords(new_coords)
            if glob_arr.shape[0] != len(new_coords):
                raise ValueError("global metric length mismatch")
            retrieval_hist_globals.extend(list(glob_arr))
        except Exception:
            retrieval_hist_globals.extend([np.zeros((max(0, global_dim),), dtype=np.float32) for _ in range(len(new_coords))])

        new_keys: List[str] = []
        new_frame_idx: List[int] = []
        new_dcd_paths: List[Optional[str]] = []
        new_dcd_frame_idx: List[int] = []
        if epoch_idx is not None and traj_lengths is not None:
            cursor = 0
            for traj_i, L in enumerate(traj_lengths):
                L_i = int(L)
                if L_i <= 0:
                    continue
                dcd_path = traj_dcd_paths[traj_i] if traj_dcd_paths is not None and traj_i < len(traj_dcd_paths) else None
                key = _geom2vec_cache_key_for_history_traj(int(epoch_idx), int(traj_i), dcd_path)
                frame_idx_seq: Optional[Sequence[int]] = None
                if traj_dcd_frame_indices is not None and traj_i < len(traj_dcd_frame_indices):
                    try:
                        frame_idx_seq = traj_dcd_frame_indices[traj_i]
                    except Exception:
                        frame_idx_seq = None
                use_frame_seq = frame_idx_seq is not None and len(frame_idx_seq) == int(L_i)
                for local_i in range(L_i):
                    new_keys.append(key)
                    new_frame_idx.append(int(local_i))
                    new_dcd_paths.append(str(dcd_path) if dcd_path else None)
                    if use_frame_seq:
                        try:
                            new_dcd_frame_idx.append(int(frame_idx_seq[local_i]))  # type: ignore[index]
                        except Exception:
                            new_dcd_frame_idx.append(int(local_i))
                    else:
                        new_dcd_frame_idx.append(int(local_i))
                cursor += L_i
            if len(new_keys) != len(new_cv):
                # Fall back to empty keys if trajectory lengths don't match the flattened epoch data.
                new_keys = [""] * len(new_cv)
                new_frame_idx = [0] * len(new_cv)
                new_dcd_paths = [None] * len(new_cv)
                new_dcd_frame_idx = [0] * len(new_cv)
        else:
            new_keys = [""] * len(new_cv)
            new_frame_idx = [0] * len(new_cv)
            new_dcd_paths = [None] * len(new_cv)
            new_dcd_frame_idx = [0] * len(new_cv)

        retrieval_hist_keys.extend(new_keys)
        retrieval_hist_frame_idx.extend(new_frame_idx)
        retrieval_hist_dcd_paths.extend(new_dcd_paths)
        retrieval_hist_dcd_frame_idx.extend(new_dcd_frame_idx)

        # Keep labels for existing frames, but mark newly appended frames as
        # "unlabeled" (-1). We refresh these lazily before the next retrieval
        # query, and fully recompute on projector refresh.
        try:
            if retrieval_hist_labels is not None and int(retrieval_hist_labels.shape[0]) == int(old_len):
                pad = -np.ones((len(new_cv),), dtype=np.int64)
                retrieval_hist_labels = np.concatenate([retrieval_hist_labels.astype(np.int64, copy=False), pad], axis=0)
            else:
                retrieval_hist_labels = None
        except Exception:
            retrieval_hist_labels = None

        if log:
            log_print(f"[Retrieval Cache] Appended {len(new_cv)} frames ({tag}); cache_frames={len(retrieval_hist_cv)}")

    def _retrieval_cache_cap() -> int:
        cap = int(getattr(config, "retrieval_cache_max_frames", 0) or 0)
        if cap <= 0:
            cap = int(getattr(config, "max_pca_train", 0) or 0)
        return int(cap)

    def _key_source_bucket(key: str) -> str:
        if not key:
            return "missing"
        if key.startswith("external/"):
            return "external"
        if key.startswith("history_bootstrap/"):
            return "history_bootstrap"
        return "run"

    def _apply_retrieval_cache_keep_indices(keep_idx: np.ndarray, *, tag: str) -> None:
        """Apply an index mask to all retrieval-cache arrays consistently."""
        nonlocal retrieval_hist_cv
        nonlocal retrieval_hist_coords
        nonlocal retrieval_hist_dcd_paths
        nonlocal retrieval_hist_dcd_frame_idx
        nonlocal retrieval_hist_globals
        nonlocal retrieval_hist_labels
        nonlocal retrieval_hist_keys
        nonlocal retrieval_hist_frame_idx

        keep_idx = np.asarray(keep_idx, dtype=np.int64).reshape(-1)
        n_before = len(retrieval_hist_cv)
        labels_before = retrieval_hist_labels
        if keep_idx.size == 0:
            retrieval_hist_cv = []
            retrieval_hist_coords = []
            retrieval_hist_dcd_paths = []
            retrieval_hist_dcd_frame_idx = []
            retrieval_hist_globals = []
            retrieval_hist_keys = []
            retrieval_hist_frame_idx = []
            retrieval_hist_labels = None
            log_print(f"[Retrieval Cache] Trimmed to 0 frames ({tag}).")
            return

        retrieval_hist_cv = [retrieval_hist_cv[int(i)] for i in keep_idx]
        retrieval_hist_coords = [retrieval_hist_coords[int(i)] for i in keep_idx]
        retrieval_hist_dcd_paths = [retrieval_hist_dcd_paths[int(i)] for i in keep_idx] if retrieval_hist_dcd_paths else []
        retrieval_hist_dcd_frame_idx = (
            [retrieval_hist_dcd_frame_idx[int(i)] for i in keep_idx] if retrieval_hist_dcd_frame_idx else []
        )
        retrieval_hist_globals = [retrieval_hist_globals[int(i)] for i in keep_idx] if retrieval_hist_globals else []
        retrieval_hist_keys = [retrieval_hist_keys[int(i)] for i in keep_idx]
        retrieval_hist_frame_idx = [retrieval_hist_frame_idx[int(i)] for i in keep_idx]
        if labels_before is not None and isinstance(labels_before, np.ndarray) and int(labels_before.shape[0]) == int(n_before):
            try:
                retrieval_hist_labels = np.asarray(labels_before, dtype=np.int64).reshape(-1)[keep_idx]
            except Exception:
                retrieval_hist_labels = None
        else:
            retrieval_hist_labels = None

    def _rebalance_retrieval_cache(tag: str) -> None:
        """
        Enforce `retrieval_cache_max_frames` via a uniform downsample.

        Note: the generative loop intentionally does not do cluster-balanced cache
        reweighting (no VAMP-label balancing).
        """
        nonlocal retrieval_hist_cv, retrieval_hist_coords, retrieval_hist_labels, retrieval_hist_keys, retrieval_hist_frame_idx

        cap = _retrieval_cache_cap()
        if cap <= 0:
            return
        n = len(retrieval_hist_cv)
        if n <= cap:
            return

        src_counts_before: Dict[str, int] = {}
        try:
            for k in retrieval_hist_keys:
                b = _key_source_bucket(str(k))
                src_counts_before[b] = src_counts_before.get(b, 0) + 1
        except Exception:
            src_counts_before = {}

        keep = retrieval_rng.choice(int(n), size=int(cap), replace=False)
        keep = np.sort(np.asarray(keep, dtype=np.int64))
        _apply_retrieval_cache_keep_indices(keep, tag=tag)

        log_print(
            f"[Retrieval Cache] Capped uniformly ({tag}): frames={int(n)}→{len(retrieval_hist_cv)} "
            f"(cap={cap}, retrieval_cache_max_frames={getattr(config, 'retrieval_cache_max_frames', 0)}, "
            f"max_pca_train={getattr(config, 'max_pca_train', 0)})"
        )
        if src_counts_before:
            try:
                src_counts_after: Dict[str, int] = {}
                for k in retrieval_hist_keys:
                    b = _key_source_bucket(str(k))
                    src_counts_after[b] = src_counts_after.get(b, 0) + 1
                log_print(f"[Retrieval Cache] Sources ({tag}): before={src_counts_before} after={src_counts_after}")
            except Exception:
                pass

    def _rebuild_retrieval_cache_from_history(tag: str) -> None:
        """Rebuild the retrieval cache from *all* currently loaded history epochs."""
        nonlocal retrieval_hist_cv
        nonlocal retrieval_hist_coords
        nonlocal retrieval_hist_dcd_paths
        nonlocal retrieval_hist_dcd_frame_idx
        nonlocal retrieval_hist_globals
        nonlocal retrieval_hist_labels
        nonlocal retrieval_hist_keys
        nonlocal retrieval_hist_frame_idx

        retrieval_hist_cv = []
        retrieval_hist_coords = []
        retrieval_hist_dcd_paths = []
        retrieval_hist_dcd_frame_idx = []
        retrieval_hist_globals = []
        retrieval_hist_keys = []
        retrieval_hist_frame_idx = []
        retrieval_hist_labels = None

        if not all_epoch_data:
            return

        for epoch_idx, epoch_data in enumerate(all_epoch_data):
            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            traj_frame_idx = (
                all_epoch_traj_dcd_frame_indices[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_frame_indices) else []
            )
            if not traj_lens and epoch_data:
                traj_lens = [len(epoch_data)]
                traj_paths = [None]
                traj_frame_idx = [None]
            _append_retrieval_cache(
                epoch_data,
                tag=f"{tag}:epoch_{epoch_idx:04d}",
                epoch_idx=int(epoch_idx),
                traj_lengths=traj_lens,
                traj_dcd_paths=traj_paths,
                traj_dcd_frame_indices=traj_frame_idx,
                log=False,
            )

        cap = _retrieval_cache_cap()
        src_counts: Dict[str, int] = {}
        try:
            for k in retrieval_hist_keys:
                b = _key_source_bucket(str(k))
                src_counts[b] = src_counts.get(b, 0) + 1
        except Exception:
            src_counts = {}

        log_print(
            f"[Retrieval Cache] Rebuilt from history ({tag}): frames={len(retrieval_hist_cv)} "
            f"(cap={cap if cap > 0 else 'none'}, retrieval_cache_max_frames={getattr(config, 'retrieval_cache_max_frames', 0)}, "
            f"max_pca_train={getattr(config, 'max_pca_train', 0)})"
        )
        if src_counts:
            log_print(f"[Retrieval Cache] Sources ({tag}): {src_counts}")

    def _refresh_retrieval_cache_labels(tag: str, *, only_indices: Optional[np.ndarray] = None) -> None:
        """Assign VAMP state labels to the in-memory replay buffer for balanced sampling.

        When `only_indices` is provided, only updates those positions in-place
        (useful for newly appended frames marked with -1).
        """
        nonlocal retrieval_hist_labels

        if not retrieval_hist_cv:
            return
        if geom2vec_cache is None or geom2vec_projector is None or not _geom2vec_is_trained():
            return
        if len(retrieval_hist_keys) != len(retrieval_hist_cv) or len(retrieval_hist_frame_idx) != len(retrieval_hist_cv):
            retrieval_hist_labels = np.zeros((len(retrieval_hist_cv),), dtype=np.int64)
            log_print(f"[Retrieval Cache] Warning: missing replay key metadata; labels reset to zeros ({tag}).")
            return

        n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
        perm = np.asarray(geom2vec_state_permutation_map, dtype=int)
        if perm.ndim != 1 or int(perm.shape[0]) != int(n_states):
            perm = np.arange(n_states, dtype=int)

        groups: Dict[str, List[int]] = {}
        idx_arr: Optional[np.ndarray] = None
        if only_indices is not None:
            try:
                idx_arr = np.asarray(only_indices, dtype=np.int64).reshape(-1)
                idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(retrieval_hist_cv))]
            except Exception:
                idx_arr = None
        if idx_arr is None:
            for i, key in enumerate(retrieval_hist_keys):
                if key:
                    groups.setdefault(key, []).append(int(i))
        else:
            for pos in idx_arr.tolist():
                key = retrieval_hist_keys[int(pos)]
                if key:
                    groups.setdefault(key, []).append(int(pos))

        if not groups:
            if retrieval_hist_labels is None or retrieval_hist_labels.shape[0] != len(retrieval_hist_cv):
                retrieval_hist_labels = np.zeros((len(retrieval_hist_cv),), dtype=np.int64)
            return

        if retrieval_hist_labels is None or retrieval_hist_labels.shape[0] != len(retrieval_hist_cv) or idx_arr is None:
            labels_out = np.zeros((len(retrieval_hist_cv),), dtype=np.int64)
        else:
            labels_out = np.asarray(retrieval_hist_labels, dtype=np.int64).copy()
            # Clear missing labels to a safe default before filling.
            labels_out[idx_arr] = 0
        missing = 0
        for key, idxs in groups.items():
            try:
                rec = geom2vec_cache.get(key)
            except KeyError:
                missing += 1
                continue
            stride_eff = int(rec.meta.get("stride", geom2vec_embed_stride) or geom2vec_embed_stride)
            stride_eff = max(1, stride_eff)

            try:
                probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                probs = probs_raw[:, perm]
                if probs.shape[1] != n_states:
                    if probs.shape[1] > n_states:
                        probs = probs[:, :n_states]
                    else:
                        probs = np.pad(probs, ((0, 0), (0, n_states - probs.shape[1])), mode="constant")
                labels_embed = np.argmax(probs, axis=1).astype(np.int64, copy=False)
            except Exception as exc_probs:
                log_print(f"[Retrieval Cache] Warning: failed to compute VAMP labels for key={key} ({exc_probs})")
                continue

            T = int(labels_embed.shape[0])
            if T <= 0:
                continue
            for pos in idxs:
                raw_i = int(retrieval_hist_frame_idx[pos])
                embed_i = raw_i // stride_eff
                if embed_i < 0:
                    embed_i = 0
                elif embed_i >= T:
                    embed_i = T - 1
                labels_out[pos] = int(labels_embed[embed_i])

        retrieval_hist_labels = labels_out

        try:
            if idx_arr is not None:
                log_print(f"[Retrieval Cache] Updated VAMP labels for {int(idx_arr.size)}/{len(labels_out)} frames ({tag}).")
                return
            counts = np.bincount(labels_out.clip(0, n_states - 1), minlength=n_states)
            cap = _retrieval_cache_cap()
            cfg_cap = int(getattr(config, "retrieval_cache_max_frames", 0) or 0)
            cfg_max_train = int(getattr(config, "max_pca_train", 0) or 0)
            empty_keys = 0
            src_counts: Dict[str, int] = {}
            try:
                for k in retrieval_hist_keys:
                    if not k:
                        empty_keys += 1
                    b = _key_source_bucket(str(k))
                    src_counts[b] = src_counts.get(b, 0) + 1
            except Exception:
                src_counts = {}
            miss_msg = f", missing_keys={missing}" if missing else ""
            src_msg = f", sources={src_counts}" if src_counts else ""
            cap_msg = (
                f", cap={cap if cap > 0 else 'none'}(retrieval_cache_max_frames={cfg_cap},max_pca_train={cfg_max_train})"
            )
            log_print(
                f"[Retrieval Cache] Refreshed VAMP cluster labels ({tag}): "
                f"frames={len(labels_out)}, keys={len(groups)}, pop={counts.tolist()}, empty_keys={empty_keys}{miss_msg}{cap_msg}{src_msg}"
            )
        except Exception:
            pass

    # Geom2Vec VAMP backbone state
    geom2vec_cache: Optional[EmbeddingCache] = None
    # Softmax VAMP lobe: used for state labels and LOO-VAMP scoring.
    geom2vec_projector: Optional[Geom2VecVAMPProjector] = None
    # Linear VAMP lobe: used for continuous diagnostics / VAMP-bin exploration bonus.
    geom2vec_projector_linear: Optional[Geom2VecVAMPProjector] = None
    geom2vec_state_permutation_map = np.arange(int(getattr(config, "geom2vec_dim", 2) or 2), dtype=int)
    geom2vec_state_inverse_permutation_map = np.arange(int(getattr(config, "geom2vec_dim", 2) or 2), dtype=int)
    geom2vec_lagtime_embed = 1
    geom2vec_embed_stride = int(getattr(config, "geom2vec_embed_stride", 2) or 2)
    geom2vec_embed_batch_size = int(getattr(config, "geom2vec_embed_batch_size", 32) or 32)
    geom2vec_reduction = str(getattr(config, "geom2vec_reduction", "sum") or "sum")
    geom2vec_center_ca = bool(getattr(config, "geom2vec_center_ca", True))
    geom2vec_hidden_channels_opt: Optional[int] = None
    geom2vec_md_top_protein = None
    geom2vec_topology_cache: Optional[Geom2VecTopologyCache] = None
    geom2vec_cache_root_path: Optional[Path] = None
    geom2vec_device_str = ""

    def _geom2vec_is_trained() -> bool:
        return geom2vec_projector is not None and getattr(geom2vec_projector, "trained", None) is not None

    def _geom2vec_import_module():
        try:
            import geom2vec  # type: ignore

            if hasattr(geom2vec, "create_model_from_checkpoint"):
                return geom2vec
        except ImportError:
            pass

        import importlib
        import sys

        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "apps" / "geom2vec" / "src"
            if (candidate / "geom2vec").exists():
                sys.path.insert(0, str(candidate))
                # If a namespace package shadowed the real geom2vec, purge it before retrying.
                for name in list(sys.modules.keys()):
                    if name == "geom2vec" or name.startswith("geom2vec."):
                        sys.modules.pop(name, None)
                importlib.invalidate_caches()
                import geom2vec  # type: ignore

                if hasattr(geom2vec, "create_model_from_checkpoint"):
                    return geom2vec
                raise ImportError(
                    "Imported `geom2vec` but it does not expose `create_model_from_checkpoint` "
                    "(possible namespace-package shadowing)."
                )

        raise ImportError(
            "Could not import `geom2vec`. Install it (e.g. `pip install -e apps/geom2vec`) "
            "or ensure `apps/geom2vec/src` is on PYTHONPATH."
        )

    def _geom2vec_cache_key_for_history_traj(epoch_idx: int, traj_idx: int, dcd_path: Optional[str]) -> str:
        if dcd_path:
            p = Path(str(dcd_path))
            if geom2vec_cache_root_path is not None:
                try:
                    rel = p.relative_to(geom2vec_cache_root_path)
                    return rel.with_suffix("").as_posix()
                except Exception:
                    pass
            try:
                import hashlib

                abs_path = str(p.resolve())
                payload = abs_path.encode("utf-8")
                try:
                    h = hashlib.md5(payload, usedforsecurity=False)
                except TypeError:
                    h = hashlib.md5(payload)
                digest = h.hexdigest()[:10]
                return (Path("external") / f"{digest}_{p.stem}").as_posix()
            except Exception:
                return (Path("external") / p.name).with_suffix("").as_posix()
        return (Path("history_bootstrap") / f"epoch_{epoch_idx:04d}" / f"traj_{traj_idx:03d}").as_posix()

    def _geom2vec_cache_has_key(key: str) -> bool:
        if geom2vec_cache is None:
            return False
        try:
            rec = geom2vec_cache.get(key)
        except KeyError:
            return False
        try:
            return Path(rec.embedding_path).exists() and Path(rec.ca_path).exists()
        except Exception:
            return False

    def _geom2vec_ensure_cached(
        *,
        key: str,
        dcd_path: Optional[str],
        coords_nm: Optional[np.ndarray],
        overwrite: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if geom2vec_cache is None or geom2vec_projector is None or geom2vec_topology_cache is None:
            return None
        if not overwrite and _geom2vec_cache_has_key(key):
            return key

        if coords_nm is None:
            if not dcd_path:
                return None
            if geom2vec_md_top_protein is None:
                return None
            try:
                traj = md.load(str(dcd_path), top=geom2vec_md_top_protein, stride=int(geom2vec_embed_stride))
                coords_nm = np.asarray(traj.xyz, dtype=np.float32)
            except Exception:
                # In explicit-solvent full-coordinate replay mode, DCDs may contain the
                # full system. Fall back to loading with the full topology and slicing.
                try:
                    md_top_full = env_ref.md_topology
                    protein_indices = md_top_full.select("protein")
                    if protein_indices is None or len(protein_indices) == 0:
                        protein_indices = np.arange(md_top_full.n_atoms, dtype=int)
                    traj_full = md.load(str(dcd_path), top=md_top_full, stride=int(geom2vec_embed_stride))
                    traj = traj_full.atom_slice(protein_indices)
                    coords_nm = np.asarray(traj.xyz, dtype=np.float32)
                except Exception as exc2:
                    log_print(f"[Geom2Vec Cache] Failed to load DCD {dcd_path}: {exc2}")
                    return None
            effective_stride = int(geom2vec_embed_stride)
            infer_stride = 1
        else:
            effective_stride = int(geom2vec_embed_stride)
            infer_stride = int(geom2vec_embed_stride)

        try:
            rec = infer_and_cache(
                geom2vec_cache,
                key=key,
                model=geom2vec_projector.rep_model,
                topology=geom2vec_topology_cache,
                coords=coords_nm,
                coords_in_nm=True,
                stride=infer_stride,
                batch_size=int(geom2vec_embed_batch_size),
                reduction=str(geom2vec_reduction),
                hidden_channels=geom2vec_hidden_channels_opt,
                device=torch.device(str(geom2vec_device_str or torch_device)),
                center_ca=bool(geom2vec_center_ca),
                overwrite=overwrite,
                meta={
                    **(meta or {}),
                    "stride": int(effective_stride),
                    "source_dcd": str(dcd_path) if dcd_path else "",
                },
            )
            return rec.key
        except Exception as exc:
            log_print(f"[Geom2Vec Cache] Failed to compute/store embeddings for key={key}: {exc}")
            return None

    geom2vec_device_str = str(getattr(config, "geom2vec_device", "") or "").strip() or str(torch_device)
    geom2vec_hidden_channels = int(getattr(config, "geom2vec_hidden_channels", 0) or 0)
    geom2vec_hidden_channels_opt = int(geom2vec_hidden_channels) if geom2vec_hidden_channels > 0 else None

    cache_root = str(getattr(config, "geom2vec_cache_root", "") or "").strip() or str(
        getattr(config, "snapshot_dir", "") or "."
    )
    manifest_name = str(getattr(config, "geom2vec_manifest_name", "geom2vec_manifest.jsonl") or "geom2vec_manifest.jsonl")
    geom2vec_cache = EmbeddingCache(cache_root, manifest_name=manifest_name)
    geom2vec_cache_root_path = Path(cache_root).resolve()

    geom2vec = _geom2vec_import_module()
    rep_ckpt = str(getattr(config, "geom2vec_checkpoint", "") or "").strip()
    rep_model = geom2vec.create_model_from_checkpoint(rep_ckpt, device=geom2vec_device_str)

    # Build a protein-only topology cache aligned with the rollout DCDs.
    try:
        md_top_full = env_ref.md_topology
        protein_indices = md_top_full.select("protein")
        if protein_indices is None or len(protein_indices) == 0:
            protein_indices = np.arange(md_top_full.n_atoms, dtype=int)
        geom2vec_md_top_protein = md_top_full.subset(protein_indices)
    except Exception:
        geom2vec_md_top_protein = None

    if geom2vec_md_top_protein is None:
        raise RuntimeError("[Geom2Vec] Failed to build protein-only mdtraj topology for embedding inference.")

    geom2vec_topology_cache = Geom2VecTopologyCache.from_mdtraj(
        geom2vec_md_top_protein,
        exclude_hydrogens=bool(getattr(config, "geom2vec_exclude_hydrogens", True)),
        meta={"selection": "protein"},
    )

    geom2vec_projector = Geom2VecVAMPProjector(
        rep_model=rep_model,
        topology=geom2vec_topology_cache,
        cache=geom2vec_cache,
        device=torch.device(geom2vec_device_str),
        hidden_channels=geom2vec_hidden_channels_opt,
    )
    geom2vec_projector_linear = Geom2VecVAMPProjector(
        rep_model=rep_model,
        topology=geom2vec_topology_cache,
        cache=geom2vec_cache,
        device=torch.device(geom2vec_device_str),
        hidden_channels=geom2vec_hidden_channels_opt,
    )

    lag_raw = int(getattr(config, "geom2vec_lagtime", 1) or 1)
    geom2vec_lagtime_embed = max(1, int(round(lag_raw / max(1, int(geom2vec_embed_stride)))))

    def _refresh_geom2vec_vamp(reason: str) -> None:
        """Train geom2vec VAMP lobe on cached embeddings and keep stable state indexing."""
        nonlocal geom2vec_state_permutation_map
        nonlocal geom2vec_state_inverse_permutation_map
        nonlocal geom2vec_lagtime_embed

        if geom2vec_cache is None or geom2vec_projector is None:
            log_print(f"[Geom2Vec Refresh] Skipped ({reason}): projector/cache not initialized.")
            return
        if not all_epoch_data:
            log_print(f"[Geom2Vec Refresh] Skipped ({reason}): no history frames available.")
            return

        lag_raw_i = int(getattr(config, "geom2vec_lagtime", 1) or 1)
        geom2vec_lagtime_embed = max(1, int(round(lag_raw_i / max(1, int(geom2vec_embed_stride)))))
        lag_embed = int(geom2vec_lagtime_embed)

        def _natural_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
            import re

            parts = re.split(r"(\d+)", text)
            key: List[Tuple[int, object]] = []
            for part in parts:
                if not part:
                    continue
                if part.isdigit():
                    key.append((1, int(part)))
                else:
                    key.append((0, part.lower()))
            return tuple(key)

        all_keys: List[str] = []
        for epoch_idx, epoch_data in enumerate(all_epoch_data):
            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            if not traj_lens:
                continue

            cursor = 0
            for traj_idx, L in enumerate(traj_lens):
                L_i = int(L)
                if L_i <= 0:
                    continue
                dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                key = _geom2vec_cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)

                if _geom2vec_cache_has_key(key):
                    all_keys.append(str(key))
                    cursor += L_i
                    continue

                coords_nm = None
                try:
                    coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor : cursor + L_i]]
                    coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                except Exception:
                    coords_nm = None

                cached_key = _geom2vec_ensure_cached(
                    key=key,
                    dcd_path=dcd_path,
                    coords_nm=coords_nm,
                    overwrite=False,
                    meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                )
                if cached_key is not None:
                    all_keys.append(str(cached_key))

                cursor += L_i

        if not all_keys:
            log_print(f"[Geom2Vec Refresh] Skipped ({reason}): no trajectories could be cached.")
            return

        keys_sorted = sorted(set(all_keys), key=_natural_sort_key)

        max_train_raw = int(getattr(config, "max_pca_train", 0) or 0)
        max_train_embed = int(math.ceil(max_train_raw / max(1, int(geom2vec_embed_stride)))) if max_train_raw > 0 else 0

        train_keys: List[str] = []
        total_embed_frames = 0
        for key in reversed(keys_sorted):
            try:
                rec = geom2vec_cache.get(key)
            except KeyError:
                continue
            if int(rec.frames) <= lag_embed:
                continue
            if max_train_embed > 0 and total_embed_frames + int(rec.frames) > max_train_embed and train_keys:
                break
            train_keys.append(key)
            total_embed_frames += int(rec.frames)
        train_keys = list(reversed(train_keys))

        if not train_keys:
            log_print(f"[Geom2Vec Refresh] Skipped ({reason}): no cached trajectories with frames > lag ({lag_embed}).")
            return

        old_labels_by_key: Optional[Dict[str, np.ndarray]] = None
        if _geom2vec_is_trained():
            old_labels_by_key = {}
            for key in train_keys:
                try:
                    probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                    probs_canon = probs_raw[:, geom2vec_state_permutation_map]
                    old_labels_by_key[key] = np.argmax(probs_canon, axis=1).astype(np.int64, copy=False)
                except Exception:
                    continue

        n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
        pair_stride = int(getattr(config, "geom2vec_vamp_pair_stride", 1) or 1)
        train_fraction = float(getattr(config, "geom2vec_vamp_train_fraction", 0.8) or 0.8)
        n_epochs = int(getattr(config, "geom2vec_vamp_epochs", 10) or 10)
        batch_size = int(getattr(config, "geom2vec_vamp_batch_size", 256) or 256)
        learning_rate = float(getattr(config, "geom2vec_vamp_learning_rate", 5e-4) or 5e-4)
        weight_decay = float(getattr(config, "geom2vec_vamp_weight_decay", 0.0) or 0.0)
        optimizer_name = str(getattr(config, "geom2vec_vamp_optimizer", "Adam") or "Adam")

        log_print(
            f"[Geom2Vec Refresh] {reason}: trajs={len(train_keys)}, embed_frames~{total_embed_frames}, "
            f"lag_embed={lag_embed}, pair_stride={pair_stride}"
        )

        try:
            seed_base = int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42)
            epsilon = float(getattr(config, "geom2vec_vamp_epsilon", 1e-6) or 1e-6)
            inverse_mode = str(getattr(config, "geom2vec_vamp_inverse_mode", "regularize") or "regularize")
            # Train a softmax lobe for LOO-VAMP scoring + stable state labels.
            geom2vec_projector.fit_from_cache(
                keys=train_keys,
                n_states=n_states,
                lag_time=lag_embed,
                seed=int(seed_base),
                pair_stride=pair_stride,
                train_fraction=train_fraction,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer=optimizer_name,
                epsilon=epsilon,
                inverse_mode=inverse_mode,
                softmax_output=True,
            )
            # Optionally train a linear lobe for continuous diagnostics / binning.
            if geom2vec_projector_linear is not None:
                try:
                    geom2vec_projector_linear.fit_from_cache(
                        keys=train_keys,
                        n_states=n_states,
                        lag_time=lag_embed,
                        seed=int(seed_base) + 1000,
                        pair_stride=pair_stride,
                        train_fraction=train_fraction,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        optimizer=optimizer_name,
                        epsilon=epsilon,
                        inverse_mode=inverse_mode,
                        softmax_output=False,
                    )
                except Exception as exc_lin:
                    log_print(f"[Geom2Vec Refresh] Warning: failed to train linear VAMP lobe (keeping previous): {exc_lin}")
        except Exception as exc:
            log_print(f"[Geom2Vec Refresh] CRITICAL ERROR: Failed to train geom2vec softmax VAMP lobe: {exc}")
            try:
                outcomes = getattr(geom2vec_projector, "last_fit_outcomes", None)
                if outcomes:
                    log_print("[Geom2Vec Refresh] Fit outcomes:")
                    for rec in outcomes:
                        if rec.get("ok"):
                            continue
                        midx = rec.get("model_idx", "?")
                        err = rec.get("error", rec)
                        log_print(f"  - model {midx}: {err}")
            except Exception:
                pass
            import traceback

            traceback.print_exc()
            return

        try:
            outcomes_sm = getattr(geom2vec_projector, "last_fit_outcomes", None)
            if outcomes_sm:
                rec = outcomes_sm[0]
                log_print(
                    f"[Geom2Vec Refresh] Softmax fit: val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
                )
            if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "last_fit_outcomes", None):
                rec = geom2vec_projector_linear.last_fit_outcomes[0]
                log_print(
                    f"[Geom2Vec Refresh] Linear fit: val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
                )
        except Exception:
            pass

        if old_labels_by_key is not None:
            old_all: List[np.ndarray] = []
            new_all: List[np.ndarray] = []
            for key in train_keys:
                old_lbl = old_labels_by_key.get(key)
                if old_lbl is None:
                    continue
                try:
                    probs_new_raw = geom2vec_projector.state_probs_from_cache_key(key)
                except Exception:
                    continue
                new_lbl = np.argmax(probs_new_raw, axis=1).astype(np.int64, copy=False)
                if new_lbl.shape[0] != old_lbl.shape[0]:
                    continue
                old_all.append(old_lbl)
                new_all.append(new_lbl)

            if old_all and new_all:
                old_flat = np.concatenate(old_all, axis=0)
                new_flat = np.concatenate(new_all, axis=0)
                try:
                    perm = _align_states_by_overlap(old_labels=old_flat, new_labels=new_flat, n_states=n_states)
                    geom2vec_state_permutation_map = perm
                    inv_perm = np.zeros_like(perm)
                    inv_perm[perm] = np.arange(len(perm))
                    geom2vec_state_inverse_permutation_map = inv_perm
                    log_print(f"[Geom2Vec Refresh] Applied confusion-matrix alignment: {perm}")
                except Exception as exc_align:
                    log_print(f"[Geom2Vec Refresh] Warning: overlap alignment failed: {exc_align}; using identity mapping.")
                    identity = np.arange(n_states, dtype=int)
                    geom2vec_state_permutation_map = identity
                    geom2vec_state_inverse_permutation_map = identity
        else:
            identity = np.arange(n_states, dtype=int)
            geom2vec_state_permutation_map = identity
            geom2vec_state_inverse_permutation_map = identity

        if config.snapshot_dir:
            out_dir = Path(config.snapshot_dir) / "geom2vec_vamp"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                lobe_softmax = geom2vec_projector._require_lobe()
                # Keep the legacy filename pointing at the softmax lobe.
                torch.save(lobe_softmax.state_dict(), out_dir / "lobe_state_dict.pt")
                torch.save(lobe_softmax.state_dict(), out_dir / "lobe_state_dict_softmax.pt")
                if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "trained", None) is not None:
                    lobe_lin = geom2vec_projector_linear._require_lobe()
                    torch.save(lobe_lin.state_dict(), out_dir / "lobe_state_dict_linear.pt")
                torch.save(
                    {
                        "train_keys": train_keys,
                        "lag_embed": int(lag_embed),
                        "embed_stride": int(geom2vec_embed_stride),
                        "state_permutation_map": np.asarray(geom2vec_state_permutation_map, dtype=np.int64),
                        "state_inverse_permutation_map": np.asarray(geom2vec_state_inverse_permutation_map, dtype=np.int64),
                        "softmax_fit_outcomes": getattr(geom2vec_projector, "last_fit_outcomes", None),
                        "linear_fit_outcomes": (
                            getattr(geom2vec_projector_linear, "last_fit_outcomes", None) if geom2vec_projector_linear is not None else None
                        ),
                    },
                    out_dir / "lobe_meta.pt",
                )
                log_print(f"[Geom2Vec Refresh] Saved lobe to {out_dir}")
            except Exception as exc_save:
                log_print(f"[Geom2Vec Refresh] Warning: failed to save lobe weights/meta: {exc_save}")

        # The VAMP lobe weights (and potentially the state permutation) have changed;
        # invalidate any cached global LOO statistics so rewards stay consistent.
        _invalidate_loo_vamp_stats(tag=f"geom2vec refresh: {reason}")
        _invalidate_vamp_bin_stats(tag=f"geom2vec refresh: {reason}")
        try:
            _refresh_retrieval_cache_labels(tag=f"geom2vec refresh: {reason}")
        except Exception as exc_lbl:
            log_print(f"[Retrieval Cache] Warning: failed to refresh VAMP labels ({exc_lbl})")

        if getattr(config, "reset_policy_on_projector_refresh", False):
            try:
                if isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy):
                    with torch.no_grad():
                        if getattr(agent, "logits", None) is not None:
                            agent.logits.zero_()
                        if getattr(agent_old, "logits", None) is not None:
                            agent_old.logits.zero_()
                    for opt in policy_opts:
                        opt.state.clear()
                    log_print("[Geom2Vec Refresh] Reset StateLogitPolicy logits to uniform after refresh.")
            except Exception as exc_reset:
                log_print(f"[Geom2Vec Refresh] Warning: failed to reset policy after refresh: {exc_reset}")

    # Load Checkpoint
    latest_ckpt, ckpt_data = ckpt_manager.load_latest()
    if config.resume and ckpt_data:
        ep_start, _, pca_state = ckpt_manager.restore_states(agent, agent_old, policy_opts, ckpt_data, torch_device)
        imported_frames, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.snapshot_dir or "."
        )
        if imported_frames > 0:
            log_print(f"[Resume] Rebuilt history from {imported_frames} frames (DCD).")
            all_trajectory_lengths.append(imported_lens)
            traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
            if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
            else:
                all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
            traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
            if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                all_epoch_traj_dcd_frame_indices.append(
                    [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                )
            else:
                all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))
        else:
            restored, restored_lens = ckpt_manager.rebuild_history_from_npz(all_epoch_data, base_dir=config.snapshot_dir or ".")
            if restored > 0:
                log_print(f"[Resume] Rebuilt history from {restored} NPZ epochs.")
                all_trajectory_lengths.extend(restored_lens)
                for lens in restored_lens:
                    all_epoch_traj_dcd_paths.append([None] * len(lens))
                    all_epoch_traj_dcd_frame_indices.append([None] * len(lens))

    # Bootstrap history from external DCDs if provided.
    if getattr(config, "history_dcd_dir", ""):
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(all_epoch_data, env_ref, base_dir=config.history_dcd_dir)
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
            traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
            if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
            else:
                all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
            traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
            if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                all_epoch_traj_dcd_frame_indices.append(
                    [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                )
            else:
                all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))
            log_print(
                f"[History DCD] Imported {imported} frames; retrieval cache will be rebuilt from all loaded history before training."
            )

    # Discrete state dimension (number of VAMP states)
    n_state_components = int(getattr(config, "geom2vec_dim", 2) or 2)
    config.n_discrete_states = n_state_components
    n_discrete_states = n_state_components
    log_print("Initializing geom2vec VAMP backbone...")

    # Initialize PCA scorer (for feature building)
    pca_scorer = initialize_projection_scorer(config, envs[0].n_cvs, args)
    if pca_state:
        pca_scorer.set_state(pca_state)

    feature_builder = FeatureBuilder(config, generator=envs[0].generator, pca_scorer=pca_scorer, env=envs[0])

    # Core reward calculator (trajectory + launch rewards). Launch is disabled.
    reward_calculator = RewardCalculator(
        config,
        pca_scorer,
        feature_builder,
        None,
        None,
        None,
        None,
    )

    sampler = Sampler(config, envs, reward_calculator, torch_device)

    # Stats Log
    stats_path = Path(config.stats_log) if config.stats_log else None
    if stats_path and not stats_path.exists():
        header = (
            "epoch,inner,valid_count,pg_h,pg_s,ent_coef,ent_val,loss,"
            "adv_h_mean,adv_h_std,adv_s_mean,adv_s_std,"
            "dlogp_h_mean,dlogp_h_std,ratio_h_mean,ratio_h_std,"
            "dlogp_s_mean,dlogp_s_std,ratio_s_mean,ratio_s_std,"
            "kl_h,kl_s,grad_norm,pre_clip_norm,avg_launch,avg_traj,avg_novelty,avg_pca,avg_loo_vamp\n"
        )
        stats_path.write_text(header)

    def _run_anchoring_round() -> None:
        """Seed history with short PDB-grounded trajectories before training."""
        if not getattr(config, "anchoring_round", False):
            return

        anchor_dir = None
        if config.snapshot_dir:
            anchor_dir = Path(config.snapshot_dir) / "anchoring"
            anchor_dir.mkdir(parents=True, exist_ok=True)

            source_dir = getattr(config, "anchoring_source_dir", "") or str(anchor_dir)

            imported = 0
            imported_lens: List[int] = []
            try:
                imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(all_epoch_data, env_ref, base_dir=source_dir)
            except Exception:
                imported = 0
                imported_lens = []
            if imported > 0:
                all_trajectory_lengths.append(imported_lens)
                traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
                if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                    all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
                else:
                    all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
                traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
                if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                    all_epoch_traj_dcd_frame_indices.append(
                        [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                    )
                else:
                    all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))
                log_print(f"[Anchoring] Detected existing anchoring DCDs in {source_dir}; skipped rerun ({imported} frames loaded).")
                return

            restored = 0
            restored_lens: List[List[int]] = []
            try:
                restored, restored_lens = ckpt_manager.rebuild_history_from_npz(all_epoch_data, base_dir=source_dir)
            except Exception:
                restored = 0
                restored_lens = []
            if restored > 0:
                all_trajectory_lengths.extend(restored_lens)
                for lens in restored_lens:
                    all_epoch_traj_dcd_paths.append([None] * len(lens))
                    all_epoch_traj_dcd_frame_indices.append([None] * len(lens))
                log_print(f"[Anchoring] Detected existing anchoring data in {source_dir}; skipped rerun ({restored} NPZ epochs loaded).")
                return

        log_print("[Anchoring] Running initial anchoring trajectories...", flush=True)

        desired_traj_count = int(getattr(config, "anchoring_num_trajs", 0) or len(envs))
        if desired_traj_count <= 0:
            desired_traj_count = len(envs)

        epoch_data: List[Any] = []
        traj_lengths: List[int] = []
        traj_dcd_paths: List[Optional[str]] = []

        for traj_idx in range(desired_traj_count):
            env = envs[traj_idx % len(envs)]

            cv_targets: List[float]
            try:
                pdb_pos_nm = np.array(env.clean_modeller.getPositions().value_in_unit(unit.nanometer), dtype=np.float32)
                cv_targets = [float(_dihedral(pdb_pos_nm, atoms)) for _, atoms in env.cv_definitions]
            except Exception:
                n_cvs = int(getattr(env, "n_cvs", 0) or 0)
                cv_targets = [0.0] * n_cvs

            pdb_only_launch = bool(getattr(config, "anchoring_pdb_only", False)) or ((traj_idx % 4) == 0)
            if pdb_only_launch:
                env.reset(cv_targets, force_pdb_coords=True)
            else:
                reset_cvs = env.reset(cv_targets, force_pdb_coords=False)
                if getattr(env, "reset_failed", False) or reset_cvs is None:
                    log_print("[Anchoring] Generator reset failed; falling back to PDB replay.", flush=True)
                    env.reset(cv_targets, force_pdb_coords=True)

            h_scales = [0.0] * config.num_mixtures
            sigmas = [[max(config.sigma_min, 1e-3)] * env.n_cvs for _ in range(config.num_mixtures)]

            original_steps = env.n_md_steps
            try:
                anchor_steps = int(getattr(config, "anchoring_steps", 0) or env.n_md_steps)
                env.n_md_steps = anchor_steps
                traj_data, is_stable, dcd_path = env.step(
                    h_scales,
                    sigmas,
                    record_frequency=config.record_stride,
                    output_dir=anchor_dir,
                    epoch_index=traj_idx + 1,
                    walker_idx=traj_idx % len(envs),
                    equilibration_steps=config.equilibration_steps,
                )
            finally:
                env.n_md_steps = original_steps

            if not (traj_data and is_stable):
                log_print(f"[Anchoring] Trajectory {traj_idx+1}/{desired_traj_count} unstable or empty; skipping.")
                continue

            epoch_data.extend(traj_data)
            traj_lengths.append(len(traj_data))
            traj_dcd_paths.append(str(dcd_path) if dcd_path else None)
            log_print(f"[Anchoring] Completed trajectory {traj_idx+1}/{desired_traj_count}.")

        if epoch_data:
            all_epoch_data.append(epoch_data)
            all_trajectory_lengths.append(traj_lengths)
            all_epoch_traj_dcd_paths.append(traj_dcd_paths)
            all_epoch_traj_dcd_frame_indices.append([None] * len(traj_dcd_paths))

            try:
                cv_arr = np.asarray([x[0] for x in epoch_data], dtype=np.float32)
                feat_arr = np.asarray(
                    [
                        np.asarray(
                            feature_builder.build_projection_features(
                                np.asarray(x[0], dtype=np.float32),
                                np.asarray(x[1], dtype=np.float32),
                            ),
                            dtype=np.float32,
                        )
                        for x in epoch_data
                    ],
                    dtype=np.float32,
                )
                if feat_arr.ndim > 2:
                    feat_arr = feat_arr.reshape(feat_arr.shape[0], -1)
                pca_scorer.update_model(cv_arr, current_ep=0, features=feat_arr, trajectory_lengths=traj_lengths)
                log_print("[Anchoring] Updated PCA scorer with anchoring data.")
            except Exception as exc:
                log_print(f"[Anchoring] PCA scorer update failed: {exc}")

        log_print("[Anchoring] Done.")

    # Optional anchoring to seed history
    _run_anchoring_round()

    # Build/refresh the in-memory retrieval cache from *all* loaded history sources
    # (resume + external history DCDs + anchoring). This keeps replay starts stable
    # and ensures older history frames are not silently dropped by load order.
    if all_epoch_data:
        try:
            _rebuild_retrieval_cache_from_history(tag="pre-train")
        except Exception as exc_rebuild:
            log_print(f"[Retrieval Cache] Warning: failed to rebuild from history ({exc_rebuild})")

    # Warm-start projector immediately after anchoring so LOO rewards are ready for epoch 1
    if all_epoch_data and not _geom2vec_is_trained():
        _refresh_geom2vec_vamp(reason="post-anchoring warm-start")

    # Plotting for "this round": exclude any bootstrapped history/anchoring epochs.
    round_epoch_data_start = len(all_epoch_data)

    log_print(f"Starting training from epoch {ep_start+1}...")

    for ep in range(ep_start, config.episodes):
        log_print(f"\n{'='*60}")
        log_print(f"Starting Epoch {ep+1}/{config.episodes}")
        log_print(f"{'='*60}")
        epoch_start_time = time.time()

        # A. Refresh projector (every N epochs)
        refresh_interval = int(getattr(config, "pca_refresh_interval", 1) or 1)
        model_ready = _geom2vec_is_trained()
        if refresh_interval <= 0:
            should_update = not model_ready
        else:
            should_update = ((ep + 1) % refresh_interval == 0) or (not model_ready)

        if should_update and all_epoch_data:
            log_print(f"[Epoch {ep+1}] Refreshing Geom2Vec VAMP lobe...")
            _refresh_geom2vec_vamp(reason=f"epoch {ep+1}")
            model_ready = _geom2vec_is_trained()

        # Ensure history-aware LOO stats match the current VAMP lobe before sampling.
        if model_ready:
            try:
                n_states_hist = int(getattr(config, "geom2vec_dim", 2) or 2)
                lag_hist = int(geom2vec_lagtime_embed)
                _ensure_loo_vamp_history_stats(
                    upto_epoch_exclusive=len(all_epoch_data),
                    n_states=n_states_hist,
                    lag_embed=lag_hist,
                    tag=f"epoch {ep+1} pre-sample",
                )
            except Exception as exc_loo_seed:
                log_print(f"[LOO VAMP] Warning: failed to seed global stats ({exc_loo_seed})")

        # Enforce retrieval cache cap (uniform downsample; no VAMP-label balancing).
        if retrieval_hist_cv and retrieval_hist_coords:
            try:
                _rebalance_retrieval_cache(tag=f"epoch {ep+1}")
            except Exception as exc_reb:
                log_print(f"[Retrieval Cache] Warning: cache cap failed ({exc_reb})")

        # B. Sample torsion starts from autoregressive policy
        log_print(f"[Epoch {ep+1}] Phase B: Sampling torsion starts from TorsionAR policy...")

        rand_mask = torch.zeros(config.grp_size, dtype=torch.bool, device=torch_device)

        start_globals: Optional[torch.Tensor] = None
        global_ranges = None
        with torch.no_grad():
            if int(getattr(agent_old, "n_global_tokens", 0) or 0) > 0:
                # Optional: set target ranges for (named) global tokens.
                # Format: [batch, n_tokens, 2] where [:, :, 0]=min, [:, :, 1]=max
                # Use inf/-inf to disable constraints per token.
                has_rmsd_range = hasattr(config, "global_rmsd_range") and config.global_rmsd_range is not None
                has_e2e_range = hasattr(config, "global_e2e_range") and config.global_e2e_range is not None
                
                if has_rmsd_range or has_e2e_range:
                    try:
                        n_gtok = int(getattr(agent_old, "n_global_tokens", 0) or 0)
                        if n_gtok <= 0:
                            global_ranges = None
                        else:
                            # Initialize with inf (no constraint by default)
                            global_ranges = torch.full((config.grp_size, n_gtok, 2), float("inf"), device=torch_device)
                            global_ranges[:, :, 0] = float("-inf")  # Set all mins to -inf initially

                        idx_rmsd = None
                        idx_e2e = None
                        try:
                            for i, name in enumerate(global_token_names):
                                key = str(name).lower()
                                if idx_rmsd is None and key.startswith("rmsd"):
                                    idx_rmsd = int(i)
                                if idx_e2e is None and key.startswith("e2e"):
                                    idx_e2e = int(i)
                        except Exception:
                            idx_rmsd = None
                            idx_e2e = None

                        range_msgs = []
                        if has_rmsd_range:
                            rmsd_min, rmsd_max = config.global_rmsd_range
                            if global_ranges is not None:
                                j = int(idx_rmsd) if idx_rmsd is not None else 0
                                if 0 <= j < global_ranges.shape[1]:
                                    global_ranges[:, j, 0] = float(rmsd_min)
                                    global_ranges[:, j, 1] = float(rmsd_max)
                                    label = global_token_names[j] if j < len(global_token_names) else "rmsd"
                                    range_msgs.append(f"{label}=[{rmsd_min:.3f}, {rmsd_max:.3f}]")
                                else:
                                    log_print(
                                        f"[TorsionAR][Epoch {ep+1}] Warning: RMSD range set but no RMSD token in globals; ignoring.",
                                        flush=True,
                                    )
                        
                        if has_e2e_range:
                            e2e_min, e2e_max = config.global_e2e_range
                            if global_ranges is not None:
                                j = int(idx_e2e) if idx_e2e is not None else (1 if global_ranges.shape[1] >= 2 else -1)
                                if 0 <= j < global_ranges.shape[1]:
                                    global_ranges[:, j, 0] = float(e2e_min)
                                    global_ranges[:, j, 1] = float(e2e_max)
                                    label = global_token_names[j] if j < len(global_token_names) else "e2e"
                                    range_msgs.append(f"{label}=[{e2e_min:.3f}, {e2e_max:.3f}]")
                                else:
                                    log_print(
                                        f"[TorsionAR][Epoch {ep+1}] Warning: E2E range set but no E2E token in globals; ignoring.",
                                        flush=True,
                                    )
                        
                        if range_msgs:
                            log_print(
                                f"[TorsionAR][Epoch {ep+1}] Using global ranges: {', '.join(range_msgs)}",
                                flush=True,
                            )
                    except Exception as exc_ranges:
                        log_print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to parse global ranges ({exc_ranges})")
                        global_ranges = None
                
                start_torsions, start_globals = agent_old.sample_with_globals(
                    batch_size=config.grp_size,
                    residue_types=residue_types_batch,
                    angle_mask=angle_mask_batch,
                    global_ranges=global_ranges,
                )
                start_torsions = start_torsions.detach()
            else:
                start_torsions = agent_old.sample(
                    batch_size=config.grp_size,
                    residue_types=residue_types_batch,
                    angle_mask=angle_mask_batch,
                ).detach()

        try:
            phi_prop = start_torsions[..., 0][angle_mask_batch[..., 0]]
            psi_prop = start_torsions[..., 1][angle_mask_batch[..., 1]]
            if phi_prop.numel() > 0 and psi_prop.numel() > 0:
                log_print(
                    f"[TorsionAR][Epoch {ep+1}] Proposed φ (rad): mean={phi_prop.mean().item():+.3f}, "
                    f"std={phi_prop.std().item():.3f}, min={phi_prop.min().item():+.3f}, max={phi_prop.max().item():+.3f}"
                )
                log_print(
                    f"[TorsionAR][Epoch {ep+1}] Proposed ψ (rad): mean={psi_prop.mean().item():+.3f}, "
                    f"std={psi_prop.std().item():.3f}, min={psi_prop.min().item():+.3f}, max={psi_prop.max().item():+.3f}"
                )
        except Exception as exc_prop:
            log_print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute proposed torsion stats ({exc_prop})")

        if start_globals is not None:
            try:
                g = start_globals.detach()
                ncols = int(g.shape[1]) if g.ndim == 2 else 0
                for j in range(ncols):
                    name = global_token_names[j] if j < len(global_token_names) else f"global_{j}"
                    key = str(name).lower()
                    unit = "nm" if key.endswith("_nm") else ("rad" if key.endswith("_rad") else "")
                    v = g[:, j]
                    if unit:
                        log_print(
                            f"[TorsionAR][Epoch {ep+1}] Proposed {name} ({unit}): mean={v.mean().item():.3f}, "
                            f"std={v.std().item():.3f}, min={v.min().item():.3f}, max={v.max().item():.3f}"
                        )
                    else:
                        log_print(
                            f"[TorsionAR][Epoch {ep+1}] Proposed {name}: mean={v.mean().item():.3f}, "
                            f"std={v.std().item():.3f}, min={v.min().item():.3f}, max={v.max().item():.3f}"
                        )
            except Exception as exc_g:
                log_print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute global-token stats ({exc_g})")

        use_retrieval = bool(getattr(config, "use_replay_resets", True)) and bool(retrieval_hist_cv) and bool(retrieval_hist_coords)
        replay_coords = None
        starts: torch.Tensor
        start_cluster_entropy: Optional[float] = None
        start_cluster_bonus: Optional[np.ndarray] = None

        if use_retrieval:
            try:
                labels = retrieval_hist_labels
                n_clusters = int(max(1, n_discrete_states))
                if labels is None or labels.shape[0] != len(retrieval_hist_cv):
                    _refresh_retrieval_cache_labels(tag=f"epoch {ep+1} pre-retrieval")
                    labels = retrieval_hist_labels
                if labels is None or labels.shape[0] != len(retrieval_hist_cv):
                    labels = np.zeros((len(retrieval_hist_cv),), dtype=np.int64)
                else:
                    try:
                        missing_idx = np.nonzero(labels < 0)[0]
                    except Exception:
                        missing_idx = np.zeros((0,), dtype=np.int64)
                    if missing_idx.size:
                        _refresh_retrieval_cache_labels(tag=f"epoch {ep+1} fill-missing", only_indices=missing_idx)
                        labels = retrieval_hist_labels
                        if labels is None or labels.shape[0] != len(retrieval_hist_cv):
                            labels = np.zeros((len(retrieval_hist_cv),), dtype=np.int64)

                m_candidates = int(getattr(config, "retrieval_candidates", 2048) or 2048)
                cand_idx = balanced_candidate_indices(labels=labels, n_clusters=n_clusters, m=m_candidates, rng=retrieval_rng)
                if cand_idx.size == 0:
                    raise RuntimeError("No retrieval candidates available.")

                cand_cv = np.asarray([retrieval_hist_cv[int(i)] for i in cand_idx], dtype=np.float32)
                cand_emb = embed_phi_psi_from_cv_batch(
                    cand_cv,
                    cv_residue_slots=np.asarray(env_ref.cv_residue_slots, dtype=np.int64),
                    cv_is_phi=np.asarray(env_ref.cv_is_phi, dtype=bool),
                    phi_exists=np.asarray(env_ref._phi_exists, dtype=bool),
                    psi_exists=np.asarray(env_ref._psi_exists, dtype=bool),
                )

                # Optional: incorporate global metrics (global tokens) into retrieval.
                #
                # Behavior (hierarchical):
                #   1) Rank candidates by global-token distance.
                #   2) Within the closest global pool, rank/sample by torsion (φ/ψ) distance.
                #
                # This ensures global objectives dominate retrieval, with torsions as a secondary key.
                use_global_retrieval = False
                global_norm_mu = None
                global_norm_std = None
                cand_glob_norm = None
                global_w_raw = getattr(config, "retrieval_global_weight", 1.0)
                global_w = float(1.0 if global_w_raw is None else global_w_raw)
                if (
                    bool(getattr(config, "retrieval_use_global_metrics", True))
                    and global_w != 0.0
                    and start_globals is not None
                    and retrieval_hist_globals
                    and len(retrieval_hist_globals) == len(retrieval_hist_cv)
                    and int(getattr(agent_old, "n_global_tokens", 0) or 0) > 0
                ):
                    cand_glob = np.asarray([retrieval_hist_globals[int(i)] for i in cand_idx], dtype=np.float32)  # [M,2]
                    cand_y = np.log1p(np.clip(cand_glob, 0.0, None)).astype(np.float32, copy=False)
                    mu = cand_y.mean(axis=0).astype(np.float32, copy=False)
                    std = cand_y.std(axis=0).astype(np.float32, copy=False)
                    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
                    cand_glob_norm = ((cand_y - mu[None, :]) / std[None, :]) * float(global_w)
                    use_global_retrieval = True
                    global_norm_mu = mu
                    global_norm_std = std

                # Support thresholds for the "out-of-support" resample guard.
                support_thr_tors = compute_support_threshold(cand_emb, rng=retrieval_rng, sample_size=256, quantile=0.99)
                support_thr_glob = float("inf")
                if use_global_retrieval and cand_glob_norm is not None:
                    support_thr_glob = compute_support_threshold(
                        cand_glob_norm.astype(np.float32, copy=False),
                        rng=retrieval_rng,
                        sample_size=256,
                        quantile=0.99,
                    )
                max_resamples = int(getattr(config, "retrieval_max_resamples", 3) or 3)
                topk = int(getattr(config, "retrieval_topk", 32) or 32)

                chosen_cv = np.zeros((config.grp_size, env_ref.n_cvs), dtype=np.float32)
                chosen_coords: List[Any] = []
                chosen_hist_idx = np.full((config.grp_size,), -1, dtype=np.int64)
                use_full_replay_coords = bool(getattr(config, "explicit_solvent_replay_full_coords", False)) and bool(
                    getattr(env_ref, "explicit_solvent", False)
                )

                def _make_replay_payload(hist_i: int) -> Any:
                    coords_nm = np.asarray(retrieval_hist_coords[hist_i], dtype=np.float32)
                    if not use_full_replay_coords:
                        return coords_nm
                    dcd_path = None
                    try:
                        if retrieval_hist_dcd_paths and hist_i < len(retrieval_hist_dcd_paths):
                            dcd_path = retrieval_hist_dcd_paths[hist_i]
                    except Exception:
                        dcd_path = None
                    dcd_frame = None
                    try:
                        if retrieval_hist_dcd_frame_idx and hist_i < len(retrieval_hist_dcd_frame_idx):
                            dcd_frame = int(retrieval_hist_dcd_frame_idx[hist_i])
                        elif retrieval_hist_frame_idx and hist_i < len(retrieval_hist_frame_idx):
                            dcd_frame = int(retrieval_hist_frame_idx[hist_i])
                    except Exception:
                        dcd_frame = None
                    return {
                        "protein_coords_nm": coords_nm,
                        "dcd_path": str(dcd_path) if dcd_path else None,
                        "dcd_frame_idx": int(dcd_frame) if dcd_frame is not None else None,
                    }

                for w in range(config.grp_size):
                    q_tors = start_torsions[w : w + 1]
                    accepted = False
                    for attempt in range(max_resamples + 1):
                        q_tors_emb = embed_phi_psi_from_torsions(
                            q_tors,
                            phi_exists=phi_exists,
                            psi_exists=psi_exists,
                        )  # [1, D]

                        # Stage 1: global-token sort to build a small pool.
                        # Stage 2: torsion distance within that pool.
                        if use_global_retrieval and cand_glob_norm is not None and start_globals is not None and global_norm_mu is not None and global_norm_std is not None:
                            try:
                                q_glob = start_globals[w : w + 1].detach().cpu().numpy().astype(np.float32, copy=False)  # [1,2]
                                q_y = np.log1p(np.clip(q_glob, 0.0, None)).astype(np.float32, copy=False)
                                q_norm = ((q_y - global_norm_mu[None, :]) / global_norm_std[None, :]) * float(global_w)
                                d2_glob = pairwise_sq_l2(q_norm.astype(np.float32, copy=False), cand_glob_norm)[0]  # [M]
                            except Exception:
                                d2_glob = np.zeros((cand_emb.shape[0],), dtype=np.float32)

                            if d2_glob.size == 0:
                                break

                            glob_order = np.argsort(d2_glob)
                            global_pool = min(int(d2_glob.size), max(int(topk) * 8, 256))
                            stage1_pos = glob_order[:global_pool]
                            d2_tors = pairwise_sq_l2(q_tors_emb, cand_emb[stage1_pos])[0]  # [global_pool]
                            d_min_glob = float(d2_glob[glob_order[0]]) if glob_order.size else float("inf")
                            d_min_tors = float(np.min(d2_tors)) if d2_tors.size else float("inf")

                            out_of_support = (
                                (np.isfinite(support_thr_glob) and d_min_glob > float(support_thr_glob))
                                or (np.isfinite(support_thr_tors) and d_min_tors > float(support_thr_tors))
                            )
                            if out_of_support and attempt < max_resamples:
                                with torch.no_grad():
                                    if int(getattr(agent_old, "n_global_tokens", 0) or 0) > 0:
                                        # Resample with same constraints if they were used initially
                                        resample_ranges = global_ranges[w:w+1] if global_ranges is not None else None
                                        q_tors_new, q_glob_new = agent_old.sample_with_globals(
                                            batch_size=1,
                                            residue_types=residue_types_batch[w : w + 1],
                                            angle_mask=angle_mask_batch[w : w + 1],
                                            global_ranges=resample_ranges,
                                        )
                                        q_tors = q_tors_new.detach()
                                        if start_globals is not None and q_glob_new is not None:
                                            start_globals[w : w + 1] = q_glob_new
                                    else:
                                        q_tors = agent_old.sample(
                                            batch_size=1,
                                            residue_types=residue_types_batch[w : w + 1],
                                            angle_mask=angle_mask_batch[w : w + 1],
                                        ).detach()
                                # Update the action actually taken for policy update.
                                start_torsions[w : w + 1] = q_tors
                                continue

                            k = min(int(topk), int(d2_tors.size))
                            topk_idx_local = np.argpartition(d2_tors, k - 1)[:k]
                            topk_d2 = d2_tors[topk_idx_local]
                            tau = float(np.median(topk_d2) + 1e-6)
                            pick_local = sample_from_topk(topk_idx_local, topk_d2, rng=retrieval_rng, temperature=tau)
                            pos = int(stage1_pos[int(pick_local)])
                            hist_i = int(cand_idx[pos])
                            chosen_hist_idx[w] = hist_i
                            chosen_cv[w] = np.asarray(retrieval_hist_cv[hist_i], dtype=np.float32)
                            chosen_coords.append(_make_replay_payload(hist_i))
                            accepted = True
                            break

                        # No globals: torsion-only retrieval as before.
                        d2 = pairwise_sq_l2(q_tors_emb, cand_emb)[0]  # [M]
                        d_min = float(np.min(d2)) if d2.size else float("inf")

                        if d2.size == 0:
                            break

                        if np.isfinite(support_thr_tors) and d_min > float(support_thr_tors) and attempt < max_resamples:
                            with torch.no_grad():
                                if int(getattr(agent_old, "n_global_tokens", 0) or 0) > 0:
                                    # Resample with same constraints if they were used initially
                                    resample_ranges = global_ranges[w:w+1] if global_ranges is not None else None
                                    q_tors_new, q_glob_new = agent_old.sample_with_globals(
                                        batch_size=1,
                                        residue_types=residue_types_batch[w : w + 1],
                                        angle_mask=angle_mask_batch[w : w + 1],
                                        global_ranges=resample_ranges,
                                    )
                                    q_tors = q_tors_new.detach()
                                    if start_globals is not None and q_glob_new is not None:
                                        start_globals[w : w + 1] = q_glob_new
                                else:
                                    q_tors = agent_old.sample(
                                        batch_size=1,
                                        residue_types=residue_types_batch[w : w + 1],
                                        angle_mask=angle_mask_batch[w : w + 1],
                                    ).detach()
                            # Update the action actually taken for policy update.
                            start_torsions[w : w + 1] = q_tors
                            continue

                        k = min(int(topk), int(d2.size))
                        topk_idx = np.argpartition(d2, k - 1)[:k]
                        topk_d2 = d2[topk_idx]
                        tau = float(np.median(topk_d2) + 1e-6)
                        pick = sample_from_topk(topk_idx, topk_d2, rng=retrieval_rng, temperature=tau)
                        hist_i = int(cand_idx[int(pick)])
                        chosen_hist_idx[w] = hist_i
                        chosen_cv[w] = np.asarray(retrieval_hist_cv[hist_i], dtype=np.float32)
                        chosen_coords.append(_make_replay_payload(hist_i))
                        accepted = True
                        break

                    if not accepted:
                        if use_global_retrieval and cand_glob_norm is not None:
                            try:
                                q_glob = start_globals[w : w + 1].detach().cpu().numpy().astype(np.float32, copy=False)  # [1,2]
                                q_y = np.log1p(np.clip(q_glob, 0.0, None)).astype(np.float32, copy=False)
                                q_norm = ((q_y - global_norm_mu[None, :]) / global_norm_std[None, :]) * float(global_w)
                                d2_glob = pairwise_sq_l2(q_norm.astype(np.float32, copy=False), cand_glob_norm)[0]
                                glob_order = np.argsort(d2_glob)
                                pos = int(glob_order[0]) if glob_order.size else 0
                                hist_i = int(cand_idx[pos])
                            except Exception:
                                hist_i = int(cand_idx[0])
                        else:
                            hist_i = int(cand_idx[0])
                        chosen_hist_idx[w] = hist_i
                        chosen_cv[w] = np.asarray(retrieval_hist_cv[hist_i], dtype=np.float32)
                        chosen_coords.append(_make_replay_payload(hist_i))

                # Use AR-proposed torsion targets for the CV conditioning; replay coordinates
                # provide the retrieved frame, and (optionally) targeted MD will steer toward
                # these CVs/globals.
                phi_psi = start_torsions[..., :2]
                phi_psi_np = phi_psi.detach().cpu().numpy()
                try:
                    cv_batch = env_ref.residue_batch_to_cv(phi_psi_np)
                except Exception as exc_cv:
                    log_print(f"[TorsionAR] Warning: residue→CV mapping failed ({exc_cv}); using replay CVs.")
                    cv_batch = chosen_cv
                starts = torch.as_tensor(cv_batch, device=torch_device, dtype=torch.float32)
                replay_coords = chosen_coords

                # Optional start-cluster entropy bonus (requires labels)
                try:
                    if labels is not None and chosen_hist_idx.size == config.grp_size:
                        start_cluster_ids = labels[chosen_hist_idx.clip(0, labels.shape[0] - 1)]
                        counts = np.bincount(start_cluster_ids.clip(0, n_clusters - 1), minlength=n_clusters)
                        total = int(counts.sum())
                        if total > 0:
                            p = counts.astype(np.float64) / float(total)
                            p_nz = p[p > 0]
                            start_cluster_entropy = float(-(p_nz * np.log(p_nz)).sum())
                            p_per = p[start_cluster_ids.clip(0, n_clusters - 1)]
                            bonus = (-np.log(p_per + 1e-12)).astype(np.float32)
                            start_cluster_bonus = bonus
                            log_print(f"[Retrieval][Epoch {ep+1}] Start clusters: H={start_cluster_entropy:.3f} nats, counts={counts.tolist()}")
                except Exception as exc_sc:
                    log_print(f"[Retrieval][Epoch {ep+1}] Warning: failed to compute start-cluster stats ({exc_sc})")
            except Exception as exc_retrieval:
                log_print(f"[Retrieval] Warning: retrieval starts failed ({exc_retrieval}); falling back to generator starts.")
                use_retrieval = False

        if not use_retrieval:
            phi_psi = start_torsions[..., :2]
            phi_psi_np = phi_psi.detach().cpu().numpy()
            try:
                cv_batch = env_ref.residue_batch_to_cv(phi_psi_np)
            except Exception as exc_cv:
                log_print(f"[TorsionAR] Warning: residue→CV mapping failed ({exc_cv}); falling back to random CV starts.")
                cv_batch = np.random.uniform(low=-math.pi, high=math.pi, size=(config.grp_size, env_ref.n_cvs)).astype(np.float32)
            starts = torch.as_tensor(cv_batch, device=torch_device, dtype=torch.float32)
            replay_coords = None

        rollout_token_entropy: Optional[np.ndarray] = None
        try:
            with torch.no_grad():
                ent_seq = agent_old.mean_token_entropy(
                    start_torsions.to(torch_device),
                    residue_types=residue_types_batch,
                    angle_mask=angle_mask_batch,
                    include_omega=False,
                    global_values=start_globals.to(torch_device) if start_globals is not None else None,
                )
                rollout_token_entropy = ent_seq.detach().cpu().numpy().astype(np.float32)
        except Exception as exc_ent:
            log_print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute token entropy ({exc_ent})")
            rollout_token_entropy = None

        try:
            for w_idx in range(config.grp_size):
                envs[w_idx]._launch_state_id = -1
        except Exception:
            pass

        a_h_scales = torch.zeros(config.grp_size, config.num_mixtures, device=torch_device)
        a_sigmas = torch.zeros(config.grp_size, config.num_mixtures, envs[0].n_cvs, device=torch_device)

        epoch_dir = None
        if config.save_epoch_coords and config.snapshot_dir:
            epoch_dir = Path(config.snapshot_dir) / f"epoch_{ep+1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

        current_epoch_data, reward_tensors, metrics, active_mask_list = sampler.sample_epoch(
            ep,
            starts,
            None,
            rand_mask,
            a_h_scales,
            a_sigmas,
            epoch_dir,
            replay_coords=replay_coords,
            replay_global_targets=(start_globals if start_globals is not None else None),
            replay_global_token_names=(list(global_token_names) if start_globals is not None else None),
        )

        launch_rewards = reward_tensors.get("launch", torch.zeros(config.grp_size, device=torch_device))
        if launch_rewards.numel() == 0:
            launch_rewards = torch.zeros(config.grp_size, device=torch_device)
        metrics["avg_launch"] = float(launch_rewards.detach().mean().item())

        all_epoch_data.append(current_epoch_data)
        if "trajectory_lengths" in metrics:
            all_trajectory_lengths.append(metrics["trajectory_lengths"])
        else:
            all_trajectory_lengths.append([])

        traj_lens = all_trajectory_lengths[-1]
        traj_dcd_paths = metrics.get("trajectory_dcd_paths", None)
        if traj_dcd_paths is None or len(traj_dcd_paths) != len(traj_lens):
            traj_dcd_paths = [None] * len(traj_lens)
        all_epoch_traj_dcd_paths.append(traj_dcd_paths)
        traj_dcd_frame_idx = [None] * len(traj_dcd_paths)
        all_epoch_traj_dcd_frame_indices.append(traj_dcd_frame_idx)

        try:
            epoch_idx_cur = len(all_epoch_data) - 1
            _append_retrieval_cache(
                current_epoch_data,
                tag=f"epoch {ep+1}",
                epoch_idx=epoch_idx_cur,
                traj_lengths=traj_lens,
                traj_dcd_paths=traj_dcd_paths,
                traj_dcd_frame_indices=traj_dcd_frame_idx,
            )
        except Exception as exc_append:
            log_print(f"[Retrieval Cache] Warning: append failed: {exc_append}")

        # D. Compute LOO VAMP rewards from the softmax VAMP lobe outputs (probabilities).
        loo_vamp_rewards = np.zeros(config.grp_size, dtype=np.float32)
        loo_vamp_contributions = None
        vamp_bin_bonus = None
        if len(current_epoch_data) > 0 and "trajectory_lengths" in metrics:
            if model_ready and geom2vec_cache is not None and geom2vec_projector is not None:
                try:
                    lag_embed = int(geom2vec_lagtime_embed)
                    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)

                    vamp_bin_ent_coef = float(getattr(config, "vamp_bin_entropy_coef", -1.0) or -1.0)
                    if vamp_bin_ent_coef < 0:
                        vamp_bin_ent_coef = float(getattr(config, "ent_coef", 0.0) or 0.0)
                    use_vamp_bin_bonus = vamp_bin_ent_coef > 0

                    traj_lengths_raw = [int(x) for x in metrics.get("trajectory_lengths", [])]
                    cursor = 0
                    cached_keys_epoch: List[Optional[str]] = []

                    for walker_idx in range(config.grp_size):
                        L_i = traj_lengths_raw[walker_idx] if walker_idx < len(traj_lengths_raw) else 0
                        dcd_path = traj_dcd_paths[walker_idx] if walker_idx < len(traj_dcd_paths) else None

                        coords_nm = None
                        if L_i > 0:
                            seg = current_epoch_data[cursor : cursor + L_i]
                            try:
                                coords_list = [
                                    np.asarray(fr[1], dtype=np.float32) for fr in seg if fr and fr[1] is not None
                                ]
                                if len(coords_list) == L_i:
                                    coords_nm = np.asarray(coords_list, dtype=np.float32)
                            except Exception:
                                coords_nm = None
                        cursor += max(0, L_i)

                        key = _geom2vec_cache_key_for_history_traj(len(all_epoch_data) - 1, walker_idx, dcd_path)
                        cached_key = _geom2vec_ensure_cached(
                            key=key,
                            dcd_path=dcd_path,
                            coords_nm=coords_nm,
                            overwrite=False,
                            meta={
                                "epoch_idx": int(len(all_epoch_data) - 1),
                                "traj_idx": int(walker_idx),
                                "raw_frames": int(L_i),
                            },
                        )
                        cached_keys_epoch.append(cached_key)

                    perm = np.asarray(geom2vec_state_permutation_map, dtype=int)

                    # Seed global (history) sufficient statistics, excluding the just-appended epoch.
                    _ensure_loo_vamp_history_stats(
                        upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                        n_states=n_states,
                        lag_embed=lag_embed,
                        tag=f"epoch {ep+1} pre-reward",
                    )
                    if loo_vamp_stats is None or loo_vamp_stats_signature != (n_states, lag_embed):
                        raise RuntimeError("Global LOO VAMP stats not initialized.")

                    if use_vamp_bin_bonus:
                        _ensure_vamp_bin_history_counts(
                            upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                            n_states=n_states,
                            tag=f"epoch {ep+1} pre-reward",
                        )

                    pair_counts = np.zeros(config.grp_size, dtype=np.int64)
                    traj_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
                        None for _ in range(len(cached_keys_epoch))
                    ]

                    vamp_bin_ready = (
                        bool(use_vamp_bin_bonus)
                        and vamp_bin_counts is not None
                        and vamp_bin_mu is not None
                        and vamp_bin_std is not None
                        and geom2vec_projector_linear is not None
                        and getattr(geom2vec_projector_linear, "trained", None) is not None
                    )
                    traj_z2_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))] if vamp_bin_ready else None
                    )

                    for t_i, cached_key in enumerate(cached_keys_epoch):
                        if cached_key is None:
                            continue
                        try:
                            graph, ca = geom2vec_cache.load(cached_key)
                        except Exception:
                            continue
                        try:
                            probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)
                            probs = _canon_vamp_output(probs_raw, n_states=n_states, perm=perm)
                            T = int(probs.shape[0])
                            if T > lag_embed:
                                traj_pairs[t_i] = (probs[:-lag_embed], probs[lag_embed:])
                                pair_counts[t_i] = max(1, int(traj_pairs[t_i][0].shape[0]))  # type: ignore[index]
                        except Exception:
                            traj_pairs[t_i] = None

                        if traj_z2_best is None or geom2vec_projector_linear is None:
                            continue
                        try:
                            z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                            z = _canon_vamp_output(z_raw, n_states=n_states, perm=perm)
                            if z.shape[1] >= 2:
                                traj_z2_best[t_i] = np.asarray(z[:, :2], dtype=np.float32)
                        except Exception:
                            traj_z2_best[t_i] = None

                    # VAMP-bin count exploration bonus (first two VAMP coords).
                    if (
                        traj_z2_best is not None
                        and vamp_bin_counts is not None
                        and vamp_bin_mu is not None
                        and vamp_bin_std is not None
                    ):
                        bins_v = int(getattr(config, "vamp_bin_bins", 50) or 50)
                        clip_v = float(getattr(config, "vamp_bin_clip", 3.0) or 3.0)
                        nbins = int(bins_v) * int(bins_v)
                        bonuses = np.zeros(config.grp_size, dtype=np.float32)
                        frame_counts = np.zeros(config.grp_size, dtype=np.int64)

                        for i, z2 in enumerate(traj_z2_best):
                            if z2 is None or z2.size == 0:
                                continue
                            z2s = (z2 - vamp_bin_mu[None, :]) / vamp_bin_std[None, :]
                            ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                            c = vamp_bin_counts[ix, iy].astype(np.float64, copy=False)
                            bonuses[i] = float((1.0 / np.sqrt(c + 1.0)).mean())
                            frame_counts[i] = int(ix.shape[0])

                        vamp_bin_bonus = bonuses

                        # Update histogram with current epoch trajectories (for future epochs).
                        for i, z2 in enumerate(traj_z2_best):
                            cached_key = cached_keys_epoch[i] if i < len(cached_keys_epoch) else None
                            if cached_key is None or z2 is None or z2.size == 0:
                                continue
                            key_s = str(cached_key)
                            if key_s in vamp_bin_counted_keys:
                                continue
                            z2s = (z2 - vamp_bin_mu[None, :]) / vamp_bin_std[None, :]
                            ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                            np.add.at(vamp_bin_counts, (ix, iy), 1)
                            vamp_bin_total += int(ix.shape[0])
                            vamp_bin_counted_keys.add(key_s)

                        try:
                            visited = int(np.count_nonzero(vamp_bin_counts))
                            log_print(
                                f"[VAMP Bins - Epoch {ep+1}] bonus_mean={float(bonuses.mean()):.4f} "
                                f"visited={visited}/{nbins} total_frames={int(vamp_bin_total)}"
                            )
                            log_print("  Per-trajectory bin bonuses (mean 1/sqrt(count+1)):")
                            for i in range(config.grp_size):
                                if frame_counts[i] <= 0:
                                    continue
                                log_print(
                                    f"    Traj {i:2d}: bonus={float(bonuses[i]):+.4f} frames={int(frame_counts[i])}"
                                )
                        except Exception:
                            pass

                    # Add current epoch trajectories to global stats (once per key).
                    for t_i, cached_key in enumerate(cached_keys_epoch):
                        if cached_key is None:
                            continue
                        key_s = str(cached_key)
                        if key_s in loo_vamp_stats_keys:
                            continue
                        pair = traj_pairs[t_i] if t_i < len(traj_pairs) else None
                        if pair is not None and loo_vamp_stats is not None:
                            loo_vamp_stats.add_trajectory(pair[0], pair[1])
                        loo_vamp_stats_keys.add(key_s)

                    if loo_vamp_stats is None:
                        raise RuntimeError("Global LOO VAMP stats not initialized.")

                    vamp_baseline = float(loo_vamp_stats.compute_vamp2_score())
                    contribs = np.zeros(config.grp_size, dtype=np.float32)

                    for i, pair in enumerate(traj_pairs):
                        if pair is None:
                            continue
                        loo_vamp_stats.remove_trajectory(pair[0], pair[1])
                        vamp_wo = float(loo_vamp_stats.compute_vamp2_score())
                        contribs[i] = float(vamp_baseline - vamp_wo)
                        loo_vamp_stats.add_trajectory(pair[0], pair[1])

                    loo_vamp_rewards = contribs.astype(np.float32, copy=False)
                    loo_vamp_contributions = contribs.astype(np.float32, copy=False)

                    log_print(f"\n[LOO VAMP Statistics - Epoch {ep+1}]")
                    log_print(f"  Baseline VAMP-2 (history+epoch trajectories): {vamp_baseline:.4f}")

                    total_pairs = int(getattr(loo_vamp_stats, "n_samples", 0) or 0)
                    if total_pairs > 0:
                        try:
                            stride_eff = int(geom2vec_embed_stride)
                        except Exception:
                            stride_eff = 0
                        log_print(
                            f"  Total transition pairs (T-lag): {total_pairs} "
                            f"(lag_embed={int(lag_embed)}, embed_stride={stride_eff})"
                        )

                    log_print("  Per-trajectory contributions (delta_i = S_all - S_minus_i):")
                    for i in range(config.grp_size):
                        if pair_counts[i] <= 0:
                            continue
                        frac = float(pair_counts[i]) / float(total_pairs) if total_pairs > 0 else 0.0
                        raw_per_pair = float(contribs[i]) / float(pair_counts[i]) if pair_counts[i] > 0 else 0.0
                        log_print(
                            f"    Traj {i:2d}: raw={float(contribs[i]):+.4f}  "
                            f"pairs={int(pair_counts[i])} ({frac:.2%})  raw/pair={raw_per_pair:+.2e}  "
                            f"score={float(loo_vamp_rewards[i]):+.4f}"
                        )
                except Exception as exc:
                    log_print(f"[LOO VAMP] Warning: Failed to compute geom2vec LOO rewards: {exc}")
                    import traceback

                    traceback.print_exc()
                    loo_vamp_rewards = np.zeros(config.grp_size, dtype=np.float32)
        # E. Policy Update (GRPO) for torsion AR starts
        if len(current_epoch_data) > 0:
            hill_data = None

            launch_rewards = reward_tensors.get("launch", torch.zeros(config.grp_size, device=torch_device))
            if launch_rewards.device != torch_device:
                launch_rewards = launch_rewards.to(torch_device)
                reward_tensors["launch"] = launch_rewards

            loo_weight = float(getattr(config, "loo_vamp_reward_weight", 1.0) or 1.0)
            start_cluster_ent_coef = float(getattr(config, "start_cluster_entropy_coef", -1.0) or -1.0)
            if start_cluster_ent_coef < 0:
                start_cluster_ent_coef = float(getattr(config, "ent_coef", 0.0) or 0.0)
            vamp_bin_ent_coef = float(getattr(config, "vamp_bin_entropy_coef", -1.0) or -1.0)
            if vamp_bin_ent_coef < 0:
                vamp_bin_ent_coef = float(getattr(config, "ent_coef", 0.0) or 0.0)

            if loo_vamp_rewards is not None and np.any(np.abs(loo_vamp_rewards) > 1e-8):
                loo_vamp_tensor = torch.tensor(loo_vamp_rewards, device=torch_device, dtype=torch.float32)
                combined_rewards = loo_weight * loo_vamp_tensor
                log_print(f"[Rewards] Using LOO VAMP only (weight={loo_weight:.2f}): loo_mean={loo_vamp_tensor.mean():.4f}")
            else:
                combined_rewards = torch.zeros_like(launch_rewards)

            if start_cluster_ent_coef > 0 and start_cluster_bonus is not None:
                bonus_tensor = torch.as_tensor(start_cluster_bonus, device=torch_device, dtype=torch.float32)
                combined_rewards = combined_rewards + start_cluster_ent_coef * bonus_tensor
                try:
                    h_val = float(start_cluster_entropy) if start_cluster_entropy is not None else float("nan")
                    log_print(
                        f"[Rewards] Added start-cluster entropy bonus: coef={start_cluster_ent_coef:.2e}, "
                        f"H={h_val:.3f}, bonus_mean={bonus_tensor.mean().item():.4f}"
                    )
                except Exception:
                    log_print(f"[Rewards] Added start-cluster entropy bonus (coef={start_cluster_ent_coef:.2e}).")

            if vamp_bin_ent_coef > 0 and vamp_bin_bonus is not None:
                bonus_raw = np.asarray(vamp_bin_bonus, dtype=np.float32)
                bonus_raw_mean = float(np.mean(bonus_raw))
                bonus_raw_std = float(np.std(bonus_raw))
                try:
                    metrics["avg_vamp_bin_bonus_raw"] = bonus_raw_mean
                    metrics["std_vamp_bin_bonus_raw"] = bonus_raw_std
                except Exception:
                    pass

                # Normalize the bin bonus to be comparable in scale to LOO-VAMP rewards.
                eps = 1e-8
                ema_beta = float(getattr(config, "vamp_bin_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = bonus_raw_mean
                cur_m2 = float(np.mean(bonus_raw * bonus_raw))
                if not np.isfinite(cur_m2):
                    cur_m2 = float(cur_m1 * cur_m1 + bonus_raw_std * bonus_raw_std)

                if use_ema:
                    if vamp_bin_norm_ema_m1 is None or vamp_bin_norm_ema_m2 is None:
                        vamp_bin_norm_ema_m1 = float(cur_m1)
                        vamp_bin_norm_ema_m2 = float(cur_m2)
                    else:
                        vamp_bin_norm_ema_m1 = float(ema_beta) * float(vamp_bin_norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        vamp_bin_norm_ema_m2 = float(ema_beta) * float(vamp_bin_norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)

                    mean_use = float(vamp_bin_norm_ema_m1)
                    var_use = float(max(float(vamp_bin_norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(bonus_raw_std)

                if np.isfinite(std_use) and std_use > eps:
                    bonus_z = (bonus_raw - mean_use) / std_use
                else:
                    bonus_z = bonus_raw * 0.0

                loo_std = 0.0
                try:
                    loo_std = float(np.std(np.asarray(loo_vamp_rewards, dtype=np.float32)))
                except Exception:
                    loo_std = 0.0

                target_scale_cur = float(loo_weight) * loo_std if (np.isfinite(loo_std) and loo_std > eps) else 1.0
                if use_ema:
                    if vamp_bin_norm_ema_scale is None:
                        vamp_bin_norm_ema_scale = float(target_scale_cur)
                    else:
                        vamp_bin_norm_ema_scale = float(ema_beta) * float(vamp_bin_norm_ema_scale) + float(1.0 - ema_beta) * float(target_scale_cur)
                    target_scale = float(vamp_bin_norm_ema_scale)
                else:
                    target_scale = float(target_scale_cur)

                bonus_norm = bonus_z * float(target_scale)

                bonus_tensor = torch.as_tensor(bonus_norm, device=torch_device, dtype=torch.float32)
                combined_rewards = combined_rewards + vamp_bin_ent_coef * bonus_tensor
                try:
                    metrics["avg_vamp_bin_bonus"] = float(bonus_tensor.mean().item())
                    metrics["std_vamp_bin_bonus"] = float(bonus_tensor.std(unbiased=False).item())
                    metrics["avg_vamp_bin_bonus_norm_mean"] = float(mean_use)
                    metrics["avg_vamp_bin_bonus_norm_std"] = float(std_use)
                    metrics["avg_vamp_bin_bonus_norm_scale"] = float(target_scale)
                except Exception:
                    pass
                try:
                    log_print(
                        f"[Rewards] Added VAMP-bin bonus: coef={vamp_bin_ent_coef:.2e}, "
                        f"raw_mean={bonus_raw_mean:.4f} raw_std={bonus_raw_std:.4f} "
                        f"ema_beta={ema_beta:.2f} norm_std={std_use:.4f} "
                        f"scale={target_scale:.3e} norm_mean={bonus_tensor.mean().item():+.4f}"
                    )
                except Exception:
                    log_print(f"[Rewards] Added VAMP-bin bonus (coef={vamp_bin_ent_coef:.2e}).")

            reward_tensors["launch"] = combined_rewards
            try:
                metrics["avg_hill"] = float(combined_rewards.mean().item())
            except Exception:
                pass

            valid_mask = metrics.get("valid_mask", torch.ones(config.grp_size, dtype=torch.bool, device=torch_device))
            if valid_mask.device != combined_rewards.device:
                valid_mask = valid_mask.to(combined_rewards.device)

            if rand_mask is not None:
                if rand_mask.device != valid_mask.device:
                    rand_mask = rand_mask.to(valid_mask.device)
                valid_mask = valid_mask & (~rand_mask)

            start_data = {
                "mode": "torsion_ar",
                "torsions": start_torsions.to(torch_device),
                "residue_types": residue_types_batch,
                "angle_mask": angle_mask_batch,
                "global_values": start_globals.to(torch_device) if start_globals is not None else None,
                "rewards": combined_rewards,
                "valid_mask": valid_mask,
            }

            update_stats = optimizer.update(start_data, hill_data)

            if config.sync_interval > 0 and (ep + 1) % config.sync_interval == 0:
                agent_old.load_state_dict(agent.state_dict())
                log_print(f"[Policy] Synced old policy at epoch {ep+1}")
        else:
            update_stats = {}

        log_print(f"[Epoch {ep+1}] Phase E: Logging & checkpointing...")
        log_print(f"Epoch {ep+1} Complete. Avg Reward: {metrics.get('avg_hill', 0.0):.4f}")

        if rollout_token_entropy is not None and rollout_token_entropy.size > 0:
            try:
                ent_mean = float(np.mean(rollout_token_entropy))
                ent_std = float(np.std(rollout_token_entropy))
                log_print(f"[TorsionAR][Epoch {ep+1}] Token entropy (φ/ψ, nats/token): mean={ent_mean:.3f}, std={ent_std:.3f}")
            except Exception:
                pass

        if stats_path:
            avg_loo_vamp = loo_vamp_rewards.mean() if loo_vamp_rewards is not None else 0.0
            pg_s_val = update_stats.get("pg_s", 0.0) if update_stats else 0.0
            ent_val = update_stats.get("entropy", 0.0) if update_stats else 0.0
            loss_val = update_stats.get("loss", 0.0) if update_stats else 0.0
            adv_s_mean = update_stats.get("adv_s_mean", 0.0) if update_stats else 0.0
            adv_s_std = update_stats.get("adv_s_std", 0.0) if update_stats else 0.0
            kl_s_val = update_stats.get("kl_s", 0.0) if update_stats else 0.0
            avg_launch = metrics.get("avg_launch", 0.0)

            row = (
                f"{ep+1},0,{len(current_epoch_data)},0.0,{pg_s_val:.6f},"
                f"0.0,{ent_val:.6f},{loss_val:.6f},0.0,0.0,"
                f"{adv_s_mean:.6f},{adv_s_std:.6f},"
                f"0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
                f"0.0,{kl_s_val:.6f},0.0,0.0,"
                f"{avg_launch:.6f},{metrics.get('avg_hill', 0.0):.6f},"
                f"0.000000,0.000000,{avg_loo_vamp:.6f}\n"
            )
            with stats_path.open("a") as f:
                f.write(row)

        if config.checkpoint_interval > 0 and (ep + 1) % config.checkpoint_interval == 0:
            ckpt_manager.save(ep, agent, agent_old, policy_opts, {}, pca_scorer, vars(args))
            try:
                _save_torsion_ar_policy(ep + 1)
            except Exception as exc_ar_save:
                log_print(f"[TorsionAR] Warning: failed to save torsion AR checkpoint: {exc_ar_save}")

        if config.plot_freq > 0 and (ep + 1) % config.plot_freq == 0:
            all_cv_samples = [item[0] for epoch_data in all_epoch_data for item in epoch_data]
            all_cv_array = np.asarray(all_cv_samples, dtype=np.float32)
            plot_ramachandran_by_residue(all_cv_array, envs[0].cv_names, f"ramachandran_ep{ep+1:03d}.png")

            try:
                plot_analysis_summary(
                    all_epoch_data[round_epoch_data_start:],
                    pca_scorer,
                    envs[0].cv_names,
                    config.pdb,
                    f"analysis_summary_ep{ep+1:03d}.png",
                    feature_builder=feature_builder,
                    vamp_bin_counts=vamp_bin_counts,
                    vamp_bin_clip=float(getattr(config, "vamp_bin_clip", 3.0) or 3.0),
                    vamp_bin_total=int(vamp_bin_total),
                )
            except Exception as exc_plot:
                log_print(f"[Plot] Warning: summary plot failed: {exc_plot}")

        elapsed = time.time() - epoch_start_time
        log_print(f"[Epoch {ep+1}] Complete in {elapsed:.1f}s.")

        gc.collect()
