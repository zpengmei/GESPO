"""
GESPO torsion-autoregressive generative *path* loop with a geom2vec + VAMP backbone.

Key differences vs the standard discrete-start loop:
  - Start policy: `TorsionARPolicy` samples a full torsion sequence Θ.
  - Optional retrieval: Θ is mapped to a replay start frame from history by
    nearest-neighbor search in a periodic φ/ψ embedding space.
  - Boundaries: endpoint (A/B) boundary masks derived from the two endpoint PDBs
    are used for committor estimation and committor-progress rewards in path mode.
  - Rewards: ΔLOO-(Stop)VAMP2 + optional VAMP-bin + optional MSM-IG, plus an
    additional committor-progress term Q_tilde.
"""

from __future__ import annotations
from dataclasses import dataclass

from gespo.training.components.geom2vec_manager import Geom2VecManager, align_states_by_overlap
from gespo.training.components.retrieval_manager import RetrievalManager
from gespo.training.components.policy_io import save_torsion_ar_policy
from gespo.training.components.global_metrics import compute_globals_from_coords as _compute_globals_impl
from gespo.training.components.vamp_bin_bonus import VampBinBonus
from gespo.training.components.loo_vamp_seeder import ensure_loo_vamp_history_stats as _ensure_loo_vamp_impl
from gespo.training.components.retrieval_labels import refresh_retrieval_cache_labels as _refresh_labels_impl
from gespo.training.components.msm_ig_bonus import MSMIGBonus
from gespo.training.components import graph_algorithms as _graph
from gespo.training.components.endpoint_loader import load_endpoint_ca_ref, load_boundary_plugin
from gespo.training.components.vamp_refresh import refresh_geom2vec_vamp as _refresh_geom2vec_vamp_impl
from gespo.training.components import committor_utils as _cu
from gespo.training.components.committor_state import CommittorState
from gespo.training.dga_committor import (
    DGAKMeansCommittorModel,
    VampDiscreteCommittorModel,
    fit_dga_committor_kmeans,
    fit_vamp_discrete_committor,
)
from gespo.training.geom2vec_backbone.vamp_trainer import train_vamp_lobe_from_cache

import argparse
import gc
import json
import math
import copy
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sys

# Suppress MDTraj DCD plugin messages (dcdplugin stdout/stderr noise)
try:
    _stderr_fd = sys.stderr.fileno()
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _old_stderr = os.dup(_stderr_fd)
    os.dup2(_devnull, _stderr_fd)
    import mdtraj as md
    os.dup2(_old_stderr, _stderr_fd)
    os.close(_devnull)
    os.close(_old_stderr)
except (AttributeError, OSError):
    # stderr doesn't support fileno() (e.g., redirected/tee'd), just import normally
    import mdtraj as md
import numpy as np
import openmm
import torch
from openmm import unit
from openmm.app import Modeller, Simulation
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma, gammaln

from gespo.config import TrainingConfig
from gespo.sampling import PeptideSampler
from gespo.utils.adk_metrics import ADK_GLOBAL_TOKEN_NAMES, build_adk_token_indices, compute_adk_global_values_from_frames
from gespo.utils.ppi_metrics import PPI_GLOBAL_TOKEN_NAMES, PPITokenIndices, build_ppi_token_indices, compute_ppi_globals_from_frames, set_native_contacts
from gespo.utils.geometry import _dihedral, calculate_rmsd, find_all_backbone_dihedrals
from gespo.utils.openmm_utils import resolve_device
from gespo.utils.plotting import plot_analysis_summary, plot_ramachandran_by_residue
from gespo.utils.selection import infer_protein_atom_indices
from gespo.training.checkpointing import CheckpointManager
from gespo.training.features import FeatureBuilder
from gespo.training.geom2vec_backbone import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector, infer_and_cache
from gespo.training.geom2vec_backbone.inference import infer_geom2vec_embeddings
from gespo.training.optimization import GRPOOptimizer
from gespo.training.rl import StateLogitPolicy, TorsionARPolicy
from gespo.training.sampler import Sampler
from gespo.training.setup import initialize_policy
from gespo.training.torsion_retrieval import (
    balanced_candidate_indices,
    build_residue_torsions_from_cv_and_coords,
    compute_support_threshold,
    embed_sincos_from_torsions_np,
    pairwise_sq_l2,
    sample_from_topk,
)
from gespo.training.vamp_stats import SufficientStatistics as VAMPSufficientStatistics
from gespo.training.components.lobe_projector import (
    state_probs_from_output_np as _state_probs_from_output_np,
    TrainedLobe as _TrainedLobe,
    VampPairRecord as _VampPairRecord,
    LobeProjector as _LobeProjector,
)



def train(args: argparse.Namespace) -> None:
    config = TrainingConfig.from_args(args)
    pdb_b = str(getattr(config, "pdb_b", "") or "").strip()
    if bool(getattr(config, "path_sampling", False)) and not pdb_b:
        raise ValueError("--path-sampling requires --pdb-b.")
    if pdb_b:
        from ..utils.endpoints import validate_endpoint_pdbs

        validate_endpoint_pdbs(str(getattr(config, "pdb_a", config.pdb) or config.pdb), pdb_b)
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

    print("GESPO Training (generative loop) with geom2vec VAMP + LOO rewards")
    print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")

    # Initialize Environments
    print("Initializing environments...")
    n_env_gpus = 0
    openmm_device_ids: Optional[List[int]] = None
    geom2vec_reserved_device: Optional[int] = None
    geom2vec_reserved_devices: set[int] = set()
    try:
        geom2vec_dev = str(getattr(config, "geom2vec_device", "") or "").strip()
        if geom2vec_dev.startswith("cuda:"):
            geom2vec_reserved_device = int(geom2vec_dev.split(":", 1)[1])
    except Exception:
        geom2vec_reserved_device = None
    try:
        if bool(getattr(config, "enable_multi_gpu", False)) and torch.cuda.is_available() and config.platform != "CPU":
            n_env_gpus = int(torch.cuda.device_count())
    except Exception:
        n_env_gpus = 0
    if n_env_gpus > 1:
        openmm_device_ids = list(range(int(n_env_gpus)))
        # Reserve devices for geom2vec (torch) inference/training when requested.
        reserve_geom2vec = bool(getattr(config, "geom2vec_reserve_gpus", False))
        dp_enabled = bool(getattr(config, "geom2vec_data_parallel", False))
        dp_devices_s = str(getattr(config, "geom2vec_data_parallel_devices", "") or "").strip()
        if dp_enabled:
            dp_ids: List[int] = []
            for part in dp_devices_s.split(","):
                p = part.strip()
                if not p:
                    continue
                try:
                    di = int(p)
                except Exception:
                    continue
                if 0 <= di < int(n_env_gpus) and di not in dp_ids:
                    dp_ids.append(di)
            if dp_ids:
                geom2vec_reserved_devices = set(int(x) for x in dp_ids)
            else:
                # Empty => "all visible" for DP; warn because this will contend with OpenMM by default.
                geom2vec_reserved_devices = set(int(i) for i in range(int(n_env_gpus)))
                print(
                    "[Multi-GPU] Warning: --geom2vec-data-parallel enabled with no explicit device list; "
                    "geom2vec will use all visible GPUs and share with OpenMM walkers (default), which may contend.",
                    flush=True,
                )
        elif geom2vec_reserved_device is not None and 0 <= int(geom2vec_reserved_device) < int(n_env_gpus):
            geom2vec_reserved_devices = {int(geom2vec_reserved_device)}

        if geom2vec_reserved_devices and int(n_env_gpus) > 1:
            if reserve_geom2vec:
                openmm_device_ids = [i for i in openmm_device_ids if int(i) not in geom2vec_reserved_devices]
                if openmm_device_ids:
                    print(
                        f"[Multi-GPU] Reserving CUDA devices {sorted(geom2vec_reserved_devices)} for geom2vec; "
                        f"assigning {config.grp_size} walkers across OpenMM devices {openmm_device_ids} (round-robin).",
                        flush=True,
                    )
                else:
                    # Fall back: nothing left for OpenMM; share devices.
                    openmm_device_ids = list(range(int(n_env_gpus)))
                    print(
                        f"[Multi-GPU] Warning: geom2vec reserved devices {sorted(geom2vec_reserved_devices)} leave no GPUs for OpenMM; "
                        "sharing devices.",
                        flush=True,
                    )
            else:
                print(
                    f"[Multi-GPU] geom2vec will use CUDA devices {sorted(geom2vec_reserved_devices)} and share with OpenMM walkers "
                    "(default). Set --geom2vec-reserve-gpus to dedicate GPUs.",
                    flush=True,
                )
        if openmm_device_ids:
            print(f"[Multi-GPU] OpenMM device pool: {openmm_device_ids}")
    envs = []
    for env_idx in range(int(config.grp_size)):
        env_args = args
        if n_env_gpus > 1 and openmm_device_ids:
            env_args = copy.copy(args)
            env_args.device = int(openmm_device_ids[int(env_idx % len(openmm_device_ids))])
        envs.append(PeptideSampler(env_args, generator_device=gen_device))
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
        print(f"[TorsionAR] Warning: failed to build residue type map from PDB: {exc_rt}")
        residue_types_1d = np.zeros((env_ref.n_residues,), dtype=np.int64)

    if residue_types_1d.shape[0] != env_ref.n_residues:
        print(
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
    angle_mask_1d = None
    angle_mask_batch = None
    angle_exists_np = None
    chi_torsion_atoms = np.zeros((0, 4), dtype=np.int64)
    chi_residue_idx = np.zeros((0,), dtype=np.int64)
    chi_order_idx = np.zeros((0,), dtype=np.int64)
    torsion_embed_indices: List[int] = []

    # Initialize TorsionAR start policy
    print("Initializing torsion autoregressive policy...")
    torsion_ckpt = getattr(args, "torsion_ar_checkpoint", "")
    if not torsion_ckpt:
        torsion_ckpt = getattr(args, "torsion_ar_out", "")
    if torsion_ckpt and not os.path.isfile(torsion_ckpt):
        print(f"[TorsionAR] Checkpoint '{torsion_ckpt}' not found; starting from scratch.")
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
        n_angles = int(ar_cfg.get("n_angles", 3) or 3)
        angle_names = ar_cfg.get("angle_names", None)
        if not isinstance(angle_names, (list, tuple)) or len(angle_names) != n_angles:
            angle_names = None
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
            elif n_global_tokens == 3:
                global_token_names = ["rmsd_nm", "e2e_nm", "rg_nm"]
            elif n_global_tokens == len(ADK_GLOBAL_TOKEN_NAMES) - 1:
                # ADK domain-only globals (theta/dist, no RMSD).
                global_token_names = list(ADK_GLOBAL_TOKEN_NAMES[1:])
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
            n_angles=n_angles,
            angle_names=angle_names,
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
            n_angles=n_angles,
            angle_names=angle_names,
            n_global_tokens=n_global_tokens,
        ).to(torch_device)
        agent_old.load_state_dict(payload["state_dict"], strict=True)
        agent_old.eval()
        print(f"[TorsionAR] Loaded pretrained policy from {torsion_ckpt}")
    else:
        num_residues = env_ref.n_residues
        residue_vocab_size = int(residue_types_1d.max()) + 1
        torsion_mode = str(getattr(args, "torsion_ar_mode", "backbone") or "backbone").strip().lower()
        if torsion_mode == "backbone":
            n_angles = 3
        elif torsion_mode in {"backbone+sidechain", "backbone_sidechain"}:
            n_angles = 7
        else:
            print(f"[TorsionAR] Warning: unknown torsion_ar_mode='{torsion_mode}', defaulting to 'backbone'.")
            n_angles = 3
        agent = TorsionARPolicy(
            num_residues=num_residues,
            residue_vocab_size=residue_vocab_size,
            n_angles=n_angles,
        ).to(torch_device)
        agent_old = TorsionARPolicy(
            num_residues=num_residues,
            residue_vocab_size=residue_vocab_size,
            n_angles=n_angles,
        ).to(torch_device)
        agent_old.load_state_dict(agent.state_dict())
        agent_old.eval()
        global_token_names = []
        print("[TorsionAR] Initialized new policy (no pretrained checkpoint provided)")

    # ------------------------------------------------------------------
    # Angle masks and χ torsion map (policy sampling, retrieval, steering).
    # ------------------------------------------------------------------
    n_angles = int(getattr(agent_old, "n_angles", 3) or 3)
    n_chi = max(0, int(n_angles) - 3)

    chi_exists_np = np.zeros((env_ref.n_residues, n_chi), dtype=bool)
    chi_torsion_atoms = np.zeros((0, 4), dtype=np.int64)
    chi_residue_idx = np.zeros((0,), dtype=np.int64)
    chi_order_idx = np.zeros((0,), dtype=np.int64)

    if n_chi > 0:
        try:
            prot_idx = None
            try:
                prot_idx = env_ref.md_topology.select("protein")
            except Exception:
                prot_idx = None
            if prot_idx is None or len(prot_idx) == 0:
                prot_idx = infer_protein_atom_indices(env_ref.md_topology, include_caps=True, allow_empty=True)
            prot_top = env_ref.md_topology.subset(np.asarray(prot_idx, dtype=np.int64))
            if int(prot_top.n_residues) != int(env_ref.n_residues):
                raise RuntimeError(
                    f"protein topology residue count {int(prot_top.n_residues)} != env_ref.n_residues {int(env_ref.n_residues)}"
                )
            rng_chi = np.random.default_rng(0)
            dummy_xyz = rng_chi.normal(size=(1, int(prot_top.n_atoms), 3)).astype(np.float32)
            tr_dummy = md.Trajectory(dummy_xyz, prot_top)

            tors_atoms_list: List[List[int]] = []
            res_list: List[int] = []
            order_list: List[int] = []
            for order in range(1, n_chi + 1):
                fn = getattr(md, f"compute_chi{order}", None)
                if fn is None:
                    continue
                idx, _ = fn(tr_dummy)
                idx = np.asarray(idx, dtype=np.int64).reshape(-1, 4)
                for tors in idx:
                    try:
                        res = int(prot_top.atom(int(tors[1])).residue.index)
                    except Exception:
                        continue
                    if res < 0 or res >= int(env_ref.n_residues):
                        continue
                    chi_exists_np[res, order - 1] = True
                    tors_atoms_list.append([int(x) for x in tors.tolist()])
                    res_list.append(res)
                    order_list.append(int(order - 1))

            if tors_atoms_list:
                chi_torsion_atoms = np.asarray(tors_atoms_list, dtype=np.int64)
                chi_residue_idx = np.asarray(res_list, dtype=np.int64)
                chi_order_idx = np.asarray(order_list, dtype=np.int64)
        except Exception as exc_chi:
            print(f"[TorsionAR] Warning: failed to build χ torsion map ({exc_chi}); disabling χ in retrieval/steering.")
            chi_exists_np = np.zeros((env_ref.n_residues, n_chi), dtype=bool)
            chi_torsion_atoms = np.zeros((0, 4), dtype=np.int64)
            chi_residue_idx = np.zeros((0,), dtype=np.int64)
            chi_order_idx = np.zeros((0,), dtype=np.int64)

    angle_exists_np = np.zeros((env_ref.n_residues, n_angles), dtype=bool)
    angle_exists_np[:, 0] = np.asarray(env_ref._phi_exists, dtype=bool)
    angle_exists_np[:, 1] = np.asarray(env_ref._psi_exists, dtype=bool)
    if n_angles >= 3:
        angle_exists_np[:, 2] = angle_exists_np[:, 0] & angle_exists_np[:, 1]
    if n_chi > 0 and n_angles > 3:
        angle_exists_np[:, 3 : 3 + n_chi] = chi_exists_np[:, :n_chi]

    angle_mask_1d = torch.from_numpy(angle_exists_np.astype(bool)).to(torch_device)
    angle_mask_batch = angle_mask_1d.unsqueeze(0).expand(config.grp_size, -1, -1)
    torsion_embed_indices = [0, 1] + ([i for i in range(3, int(n_angles))] if int(n_angles) > 3 else [])

    policy_opts = [torch.optim.Adam(agent.parameters(), lr=config.lr)]
    optimizer = GRPOOptimizer(config, agent, agent_old, policy_opts)

    def _save_torsion_ar_policy(ep_idx_1based: int) -> None:
        save_torsion_ar_policy(
            ep_idx_1based,
            checkpoint_dir=config.checkpoint_dir,
            agent=agent,
            agent_old=agent_old,
            ar_cfg=ar_cfg,
            torsion_ckpt=torsion_ckpt,
            env_ref=env_ref,
            residue_types_1d=residue_types_1d,
            global_token_names=global_token_names,
        )

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
    # Per-epoch flag: True = include in replay pool for discrete-state starts
    # but EXCLUDE from VAMP training / LOO-VAMP reward scoring.
    # Use case: pretrain trajectories with different time spacing.
    all_epoch_vamp_exclude: List[bool] = []
    ep_start = 0
    pca_state = None


    # Retrieval cache manager for torsion-query replay starts
    # Retrieval cache cap is intentionally decoupled from PCA training caps:
    # - `retrieval_cache_max_frames` controls only the in-memory replay/retrieval cache.
    # - `max_pca_train` controls PCA/VAMP fitting subsampling.
    retrieval_cache_max = int(getattr(config, "retrieval_cache_max_frames", 0) or 0)
    
    global_dim = int(getattr(agent_old, "n_global_tokens", 0) or 0)
    retrieval_mgr = RetrievalManager(
        max_cache_size=retrieval_cache_max,
        global_dim=global_dim,
        seed=0
    )
    
    # Keep backward-compatible references (will access through manager)
    retrieval_mgr.cv = []  # Managed by retrieval_mgr
    retrieval_mgr.coords = []
    retrieval_mgr.dcd_paths = []
    retrieval_mgr.dcd_frame_idx = []
    retrieval_mgr.globals = []
    retrieval_mgr.keys = []
    retrieval_mgr.frame_idx = []
    retrieval_mgr.labels = None

    if global_dim > 0 and (not global_token_names or len(global_token_names) != global_dim):
        global_token_names = [f"global_{i}" for i in range(global_dim)]

    # Precompute CA mapping (and optional ADK indices) for fast global-metric extraction
    # from protein coords.
    retrieval_ca_local: np.ndarray = np.zeros((0,), dtype=np.int64)
    retrieval_ref_ca_nm: np.ndarray = np.zeros((0, 3), dtype=np.float64)
    global_ref_ca_a_nm: Optional[np.ndarray] = None
    global_ref_ca_b_nm: Optional[np.ndarray] = None
    retrieval_protein_indices: np.ndarray = np.zeros((0,), dtype=np.int64)
    adk_token_idx = None
    ppi_token_idx: Optional[PPITokenIndices] = None
    try:
        protein_indices = env_ref.md_topology.select("protein")
        if protein_indices is None or len(protein_indices) == 0:
            from ..utils.selection import infer_protein_atom_indices

            protein_indices = infer_protein_atom_indices(env_ref.md_topology, include_caps=True, allow_empty=True)
        retrieval_protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
        idx_to_local = {int(a): int(i) for i, a in enumerate(np.asarray(protein_indices).reshape(-1))}
        ca_full = np.asarray(env_ref.ca_atom_indices, dtype=np.int64).reshape(-1)
        ca_local_by_slot = np.array([idx_to_local.get(int(a), -1) if int(a) >= 0 else -1 for a in ca_full], dtype=np.int64)
        valid_slots = (ca_local_by_slot >= 0) & np.asarray(getattr(env_ref, "_valid_ca_mask", ca_local_by_slot >= 0), dtype=bool)
        retrieval_ca_local = ca_local_by_slot[valid_slots].astype(np.int64, copy=False)
        retrieval_ref_ca_nm = np.asarray(getattr(env_ref, "_ref_ca_nm", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)[
            valid_slots
        ]

        def _load_ca_ref_for_env_slots(pdb_path: str) -> Optional[np.ndarray]:
            pdb_path = str(pdb_path or "").strip()
            if not pdb_path:
                return None
            try:
                tr_end = md.load(pdb_path)
                if int(tr_end.n_frames) <= 0:
                    return None
                top_end = tr_end.topology
                xyz_end = np.asarray(tr_end.xyz[0], dtype=np.float32)  # nm

                # residue_index -> CA position (nm)
                ca_by_res: Dict[int, np.ndarray] = {}
                for res in top_end.residues:
                    ai = None
                    for atom in res.atoms:
                        if str(atom.name) == "CA":
                            ai = int(atom.index)
                            break
                    if ai is None:
                        continue
                    if 0 <= ai < int(xyz_end.shape[0]):
                        ca_by_res[int(res.index)] = xyz_end[ai]

                ca_full = np.asarray(env_ref.ca_atom_indices, dtype=np.int64).reshape(-1)
                out_full = np.zeros((int(ca_full.shape[0]), 3), dtype=np.float64)
                ok_full = np.zeros((int(ca_full.shape[0]),), dtype=bool)
                env_residue_indices = [int(r.index) for r in env_ref.md_topology.residues]
                if len(env_residue_indices) != int(ca_full.shape[0]):
                    return None
                for i in range(int(ca_full.shape[0])):
                    pos = ca_by_res.get(env_residue_indices[int(i)], None)
                    if pos is None:
                        continue
                    out_full[i] = np.asarray(pos, dtype=np.float64)
                    ok_full[i] = True
                if int(ok_full[valid_slots].sum()) < 2:
                    return None
                return out_full[valid_slots]
            except Exception:
                return None

        # If the AR checkpoint uses RMSD-to-A/B globals, load endpoint CA references
        # aligned to env_ref CA slots (not geom2vec token order).
        try:
            need_rmsd_a = any(str(n).lower().startswith(("rmsd_a", "rmsda")) for n in global_token_names)
            need_rmsd_b = any(str(n).lower().startswith(("rmsd_b", "rmsdb")) for n in global_token_names)
            if need_rmsd_a:
                pdb_a_path = str(getattr(config, "pdb_a", config.pdb) or config.pdb)
                global_ref_ca_a_nm = _load_ca_ref_for_env_slots(pdb_a_path)
            if need_rmsd_b:
                pdb_b_path = str(getattr(config, "pdb_b", "") or "")
                global_ref_ca_b_nm = _load_ca_ref_for_env_slots(pdb_b_path)
        except Exception:
            global_ref_ca_a_nm = None
            global_ref_ca_b_nm = None

        if global_dim == len(ADK_GLOBAL_TOKEN_NAMES) or any(n in set(ADK_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                adk_token_idx = build_adk_token_indices(env_ref.md_topology, protein_indices=retrieval_protein_indices)
            except Exception:
                adk_token_idx = None
        if any(n in set(PPI_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                ppi_token_idx = build_ppi_token_indices(env_ref.md_topology, protein_indices=retrieval_protein_indices)
                if retrieval_ref_ca_nm.shape[0] > 0:
                    ppi_token_idx = set_native_contacts(ppi_token_idx, retrieval_ref_ca_nm)
            except Exception:
                ppi_token_idx = None
    except Exception:
        retrieval_ca_local = np.zeros((0,), dtype=np.int64)
        retrieval_ref_ca_nm = np.zeros((0, 3), dtype=np.float64)
        retrieval_protein_indices = np.zeros((0,), dtype=np.int64)
        adk_token_idx = None
        ppi_token_idx = None

    def _compute_globals_from_coords(coords_list: List[np.ndarray]) -> np.ndarray:
        return _compute_globals_impl(
            coords_list,
            global_dim=global_dim,
            global_token_names=global_token_names,
            ca_local=retrieval_ca_local,
            ref_ca_nm=retrieval_ref_ca_nm,
            ref_ca_a_nm=global_ref_ca_a_nm,
            ref_ca_b_nm=global_ref_ca_b_nm,
            adk_token_indices=adk_token_idx,
            ppi_token_indices=ppi_token_idx,
        )

    # LOO rewards in standard VAMP space (no stop/stopping-time modifications).
    n_models = 1  # Keep single-model semantics (no UCB/ensemble) for this loop.
    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
    loo_vamp_stats: Optional[VAMPSufficientStatistics] = None
    loo_vamp_stats_keys: set[str] = set()
    loo_vamp_stats_signature: Optional[Tuple[int, int, bool, str]] = None  # (K, lag_embed, symmetrized, feature_mode)

    # Committor state (path-sampling mode).
    committor_state = CommittorState()

    # VAMP-bin exploration bonus manager (2D bin-count statistics).
    vamp_bin_mgr = VampBinBonus()

    # MSM information-gain exploration bonus manager.
    msm_ig_mgr = MSMIGBonus()


    def _invalidate_loo_vamp_stats(tag: str) -> None:
        """Invalidate LOO-VAMP sufficient statistics (projector-dependent)."""
        nonlocal loo_vamp_stats, loo_vamp_stats_keys, loo_vamp_stats_signature
        loo_vamp_stats = None
        loo_vamp_stats_keys = set()
        loo_vamp_stats_signature = None
        print(f"[LOO VAMP] Invalidated stats ({tag}).")

    def _invalidate_vamp_bin_stats(tag: str) -> None:
        vamp_bin_mgr.invalidate(tag)
        print(f"[VAMP Bin] Invalidated bin stats ({tag}).")

    def _invalidate_msm_ig_stats(tag: str) -> None:
        msm_ig_mgr.invalidate(tag)

    def _invalidate_committor_state(tag: str) -> None:
        committor_state.invalidate_model(tag)

    def _invalidate_committor_tube_stats(tag: str) -> None:
        """Invalidate tube-diversity stats derived from the committor model."""
        committor_state.tube_counts = None
        committor_state.tube_total_weight = 0.0
        committor_state.tube_counted_keys = set()
        committor_state.tube_signature = None
        committor_state.tube_norm_ema_m1 = None
        committor_state.tube_norm_ema_m2 = None
        committor_state.edge_counts = None
        committor_state.edge_total_count = 0.0
        committor_state.edge_counted_keys = set()
        committor_state.edge_signature = None
        committor_state.edge_norm_ema_m1 = None
        committor_state.edge_norm_ema_m2 = None
        committor_state.conn_adj = None
        committor_state.conn_edges = set()
        committor_state.conn_a_cells = set()
        committor_state.conn_b_cells = set()
        committor_state.conn_counted_keys = set()
        committor_state.conn_signature = None
        committor_state.conn_norm_ema_m1 = None
        committor_state.conn_norm_ema_m2 = None
        print(f"[Committor Tube] Invalidated tube stats ({tag}).")

    def _invalidate_boundary_vac_kmeans(tag: str) -> None:
        """Invalidate VAC-kmeans boundary model (projector-dependent)."""
        committor_state.boundary_vac_kmeans = None
        committor_state.boundary_vac_mu = None
        committor_state.boundary_vac_std = None
        committor_state.boundary_vac_a_clusters = None
        committor_state.boundary_vac_b_clusters = None
        committor_state.boundary_vac_signature = None
        print(f"[Path Boundaries] Invalidated VAC-kmeans boundary model ({tag}).")

    def _iter_history_trajs(upto_epoch_exclusive: int):
        """Yield history trajectories (epoch_idx, traj_idx, L_i, dcd_path, coords_nm).

        This is used by multiple reward helpers (MSM-IG, VAMP-bin) to populate
        history statistics without duplicating traversal logic.
        """

        upto = max(0, min(int(upto_epoch_exclusive), len(all_epoch_data)))
        for epoch_idx in range(upto):
            # Skip epochs quarantined from VAMP scoring (e.g. pretrain trajs
            # with incompatible time spacing).
            if epoch_idx < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[epoch_idx]:
                continue

            epoch_data = all_epoch_data[epoch_idx]
            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            if not traj_lens:
                continue
            cursor = 0
            for traj_idx, L in enumerate(traj_lens):
                L_i = int(L)
                cursor_next = cursor + max(0, L_i)
                if L_i <= 0:
                    cursor = cursor_next
                    continue
                dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                coords_nm = None
                if not dcd_path:
                    try:
                        coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor:cursor_next]]
                        if coords_list and len(coords_list) == L_i:
                            coords_nm = np.asarray(coords_list, dtype=np.float32)
                    except Exception:
                        coords_nm = None
                cursor = cursor_next
                yield int(epoch_idx), int(traj_idx), int(L_i), dcd_path, coords_nm

    _vamp_bin_indices = VampBinBonus.bin_indices

    def _ensure_vamp_bin_history_counts(*, upto_epoch_exclusive: int, n_states: int, tag: str) -> None:
        vamp_bin_mgr.ensure_history_counts(
            upto_epoch_exclusive=upto_epoch_exclusive,
            n_states=n_states,
            tag=tag,
            config=config,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_mgr=geom2vec_mgr,
            iter_history_trajs=_iter_history_trajs,
        )

    _msm_ig_reward_from_counts = MSMIGBonus.reward_from_counts
    _msm_transition_counts = MSMIGBonus.transition_counts

    def _ensure_msm_ig_history_counts(*, upto_epoch_exclusive: int, n_states: int, lag_embed: int, tag: str) -> None:
        msm_ig_mgr.ensure_history_counts(
            upto_epoch_exclusive=upto_epoch_exclusive,
            n_states=n_states,
            lag_embed=lag_embed,
            tag=tag,
            config=config,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector=geom2vec_projector,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_mgr=geom2vec_mgr,
            geom2vec_state_permutation_map=geom2vec_state_permutation_map,
            iter_history_trajs=_iter_history_trajs,
        )

    def _ensure_loo_vamp_history_stats(
        *,
        upto_epoch_exclusive: int,
        n_states: int,
        lag_embed: int,
        tag: str,
    ) -> None:
        nonlocal loo_vamp_stats, loo_vamp_stats_keys, loo_vamp_stats_signature
        _stats_ref = [loo_vamp_stats]
        _keys_ref = [loo_vamp_stats_keys]
        _sig_ref = [loo_vamp_stats_signature]
        _ensure_loo_vamp_impl(
            loo_vamp_stats_ref=_stats_ref,
            loo_vamp_stats_keys_ref=_keys_ref,
            loo_vamp_stats_signature_ref=_sig_ref,
            upto_epoch_exclusive=upto_epoch_exclusive,
            n_states=n_states,
            lag_embed=lag_embed,
            tag=tag,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector=geom2vec_projector,
            geom2vec_mgr=geom2vec_mgr,
            all_epoch_data=all_epoch_data,
            all_trajectory_lengths=all_trajectory_lengths,
            all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
            all_epoch_vamp_exclude=all_epoch_vamp_exclude,
        )
        loo_vamp_stats = _stats_ref[0]
        loo_vamp_stats_keys = _keys_ref[0]
        loo_vamp_stats_signature = _sig_ref[0]


    def _ensure_committor_tube_history_counts(*, upto_epoch_exclusive: int, n_bins: int, tag: str) -> None:
        from .components.committor_tube import ensure_committor_tube_history_counts as _tube_impl
        _tube_impl(
            upto_epoch_exclusive=upto_epoch_exclusive,
            n_bins=n_bins,
            tag=tag,
            config=config,
            committor_state=committor_state,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector=geom2vec_projector,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_mgr=geom2vec_mgr,
            geom2vec_state_permutation_map=geom2vec_state_permutation_map,
            geom2vec_embed_stride=geom2vec_embed_stride,
            n_states=n_states,
            iter_history_trajs=_iter_history_trajs,
            rmsd_to_endpoints_fn=_rmsd_to_endpoints,
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
        if not epoch_data:
            return
        old_len = len(retrieval_mgr.cv)
        new_cv = [np.asarray(item[0], dtype=np.float32) for item in epoch_data]
        new_coords = [np.asarray(item[1], dtype=np.float32) for item in epoch_data]
        retrieval_mgr.cv.extend(new_cv)
        retrieval_mgr.coords.extend(new_coords)
        try:
            glob_arr = _compute_globals_from_coords(new_coords)
            if glob_arr.shape[0] != len(new_coords):
                raise ValueError("global metric length mismatch")
            retrieval_mgr.globals.extend(list(glob_arr))
        except Exception:
            retrieval_mgr.globals.extend([np.zeros((max(0, global_dim),), dtype=np.float32) for _ in range(len(new_coords))])

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
                key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_i), dcd_path)
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

        retrieval_mgr.keys.extend(new_keys)
        retrieval_mgr.frame_idx.extend(new_frame_idx)
        retrieval_mgr.dcd_paths.extend(new_dcd_paths)
        retrieval_mgr.dcd_frame_idx.extend(new_dcd_frame_idx)

        # Keep labels for existing frames, but mark newly appended frames as
        # "unlabeled" (-1). We refresh these lazily before the next retrieval
        # query, and fully recompute on projector refresh.
        try:
            if retrieval_mgr.labels is not None and int(retrieval_mgr.labels.shape[0]) == int(old_len):
                pad = -np.ones((len(new_cv),), dtype=np.int64)
                retrieval_mgr.labels = np.concatenate([retrieval_mgr.labels.astype(np.int64, copy=False), pad], axis=0)
            else:
                retrieval_mgr.labels = None
        except Exception:
            retrieval_mgr.labels = None

        if log:
            print(f"[Retrieval Cache] Appended {len(new_cv)} frames ({tag}); cache_frames={len(retrieval_mgr.cv)}")

    def _retrieval_cache_cap() -> int:
        return int(getattr(config, "retrieval_cache_max_frames", 0) or 0)

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

        keep_idx = np.asarray(keep_idx, dtype=np.int64).reshape(-1)
        n_before = len(retrieval_mgr.cv)
        labels_before = retrieval_mgr.labels
        if keep_idx.size == 0:
            retrieval_mgr.cv = []
            retrieval_mgr.coords = []
            retrieval_mgr.dcd_paths = []
            retrieval_mgr.dcd_frame_idx = []
            retrieval_mgr.globals = []
            retrieval_mgr.keys = []
            retrieval_mgr.frame_idx = []
            retrieval_mgr.labels = None
            print(f"[Retrieval Cache] Trimmed to 0 frames ({tag}).")
            return

        retrieval_mgr.cv = [retrieval_mgr.cv[int(i)] for i in keep_idx]
        retrieval_mgr.coords = [retrieval_mgr.coords[int(i)] for i in keep_idx]
        retrieval_mgr.dcd_paths = [retrieval_mgr.dcd_paths[int(i)] for i in keep_idx] if retrieval_mgr.dcd_paths else []
        retrieval_mgr.dcd_frame_idx = (
            [retrieval_mgr.dcd_frame_idx[int(i)] for i in keep_idx] if retrieval_mgr.dcd_frame_idx else []
        )
        retrieval_mgr.globals = [retrieval_mgr.globals[int(i)] for i in keep_idx] if retrieval_mgr.globals else []
        retrieval_mgr.keys = [retrieval_mgr.keys[int(i)] for i in keep_idx]
        retrieval_mgr.frame_idx = [retrieval_mgr.frame_idx[int(i)] for i in keep_idx]
        if labels_before is not None and isinstance(labels_before, np.ndarray) and int(labels_before.shape[0]) == int(n_before):
            try:
                retrieval_mgr.labels = np.asarray(labels_before, dtype=np.int64).reshape(-1)[keep_idx]
            except Exception:
                retrieval_mgr.labels = None
        else:
            retrieval_mgr.labels = None

    def _rebalance_retrieval_cache(tag: str) -> None:
        """
        Enforce `retrieval_cache_max_frames` via a uniform downsample.

        Note: the generative loop intentionally does not do cluster-balanced cache
        reweighting (no VAMP-label balancing).
        """

        cap = _retrieval_cache_cap()
        if cap <= 0:
            return
        n = len(retrieval_mgr.cv)
        if n <= cap:
            return

        src_counts_before: Dict[str, int] = {}
        try:
            for k in retrieval_mgr.keys:
                b = _key_source_bucket(str(k))
                src_counts_before[b] = src_counts_before.get(b, 0) + 1
        except Exception:
            src_counts_before = {}

        keep = retrieval_mgr.rng.choice(int(n), size=int(cap), replace=False)
        keep = np.sort(np.asarray(keep, dtype=np.int64))
        _apply_retrieval_cache_keep_indices(keep, tag=tag)

        print(
            f"[Retrieval Cache] Capped uniformly ({tag}): frames={int(n)}→{len(retrieval_mgr.cv)} "
            f"(cap={cap}, retrieval_cache_max_frames={getattr(config, 'retrieval_cache_max_frames', 0)})"
        )
        if src_counts_before:
            try:
                src_counts_after: Dict[str, int] = {}
                for k in retrieval_mgr.keys:
                    b = _key_source_bucket(str(k))
                    src_counts_after[b] = src_counts_after.get(b, 0) + 1
                print(f"[Retrieval Cache] Sources ({tag}): before={src_counts_before} after={src_counts_after}")
            except Exception:
                pass

    def _rebuild_retrieval_cache_from_history(tag: str) -> None:
        """Rebuild the retrieval cache from *all* currently loaded history epochs."""

        retrieval_mgr.cv = []
        retrieval_mgr.coords = []
        retrieval_mgr.dcd_paths = []
        retrieval_mgr.dcd_frame_idx = []
        retrieval_mgr.globals = []
        retrieval_mgr.keys = []
        retrieval_mgr.frame_idx = []
        retrieval_mgr.labels = None

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
            for k in retrieval_mgr.keys:
                b = _key_source_bucket(str(k))
                src_counts[b] = src_counts.get(b, 0) + 1
        except Exception:
            src_counts = {}

        print(
            f"[Retrieval Cache] Rebuilt from history ({tag}): frames={len(retrieval_mgr.cv)} "
            f"(cap={cap if cap > 0 else 'none'}, retrieval_cache_max_frames={getattr(config, 'retrieval_cache_max_frames', 0)}, "
            f"max_pca_train={getattr(config, 'max_pca_train', 0)})"
        )
        if src_counts:
            print(f"[Retrieval Cache] Sources ({tag}): {src_counts}")

    def _refresh_retrieval_cache_labels(tag: str, *, only_indices: Optional[np.ndarray] = None) -> None:
        _refresh_labels_impl(
            retrieval_mgr=retrieval_mgr,
            config=config,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector=geom2vec_projector,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_mgr=geom2vec_mgr,
            geom2vec_state_permutation_map=geom2vec_state_permutation_map,
            geom2vec_embed_stride=geom2vec_embed_stride,
            retrieval_cache_cap_fn=_retrieval_cache_cap,
            key_source_bucket_fn=_key_source_bucket,
            tag=tag,
            only_indices=only_indices,
        )

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
    
    # Initialize Geom2Vec Manager
    print("Initializing Geom2Vec manager...")
    geom2vec_mgr = Geom2VecManager(config, env_ref, torch_device)
    
    # Extract frequently-used references
    geom2vec_cache = geom2vec_mgr.cache
    geom2vec_topology_cache = geom2vec_mgr.topology_cache
    geom2vec_md_top_protein = geom2vec_mgr.md_top_protein
    geom2vec_device_str = geom2vec_mgr.device_str
    geom2vec_embed_stride = geom2vec_mgr.embed_stride
    geom2vec_embed_batch_size = geom2vec_mgr.embed_batch_size
    geom2vec_reduction = geom2vec_mgr.reduction
    geom2vec_center_ca = geom2vec_mgr.center_ca

    if geom2vec_cache is None:
        raise RuntimeError("[Geom2Vec] cache not initialized.")

    # Projectors (trained during refresh):
    # - VAMP (logits/soft assignments): discrete soft states for labels + ΔLOO-VAMP2
    # - VAC (linear):                  linear coordinates for VAMP-bin + MSM-IG + DGA basis
    vamp_projector = _LobeProjector(cache=geom2vec_cache, device=torch.device(geom2vec_device_str))
    vac_projector = _LobeProjector(cache=geom2vec_cache, device=torch.device(geom2vec_device_str))

    # Aliases used throughout the loop.
    geom2vec_projector = vamp_projector
    geom2vec_projector_linear = vac_projector

    # Endpoint CA references (token order) for boundary masks / committor features.
    endpoint_ca_ref_a: Optional[np.ndarray] = None
    endpoint_ca_ref_b: Optional[np.ndarray] = None
    # Endpoint CA references in the same *token order* as the geom2vec topology cache.
    #
    # Important: cached CA coords produced by geom2vec inference are in Å (because geom2vec expects Å),
    # while mdtraj loads PDBs in nm. We avoid any atom-index-based slicing between different PDB
    # topologies (which is fragile when hydrogen/atom ordering differs) by extracting CA positions
    # per residue token using residue order. RMSD computations later convert cached CA Å -> nm.
    endpoint_token_mask: Optional[np.ndarray] = None

    def _load_endpoint_ca_ref(pdb_path: str, *, label: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return load_endpoint_ca_ref(
            pdb_path, label=label,
            topology_cache=geom2vec_topology_cache,
            center_ca=geom2vec_center_ca,
        )

    endpoint_ca_ref_a, endpoint_valid_a = _load_endpoint_ca_ref(
        str(getattr(config, "pdb_a", config.pdb) or config.pdb), label="A"
    )
    endpoint_ca_ref_b, endpoint_valid_b = _load_endpoint_ca_ref(str(getattr(config, "pdb_b", "") or ""), label="B")
    
    # Log endpoint loading status for debugging
    if endpoint_ca_ref_a is not None:
        n_tokens_a = int(np.asarray(endpoint_ca_ref_a).shape[0])
        print(f"[Geom2Vec] Loaded endpoint A CA ref: {n_tokens_a} tokens from {getattr(config, 'pdb_a', config.pdb) or config.pdb}", flush=True)
    else:
        print("[Geom2Vec] Warning: endpoint A CA ref not loaded (required for VAMP boundary mode)", flush=True)
        
    if endpoint_ca_ref_b is not None:
        n_tokens_b = int(np.asarray(endpoint_ca_ref_b).shape[0])
        print(f"[Geom2Vec] Loaded endpoint B CA ref: {n_tokens_b} tokens from {getattr(config, 'pdb_b', '')}", flush=True)
    else:
        pdb_b_val = str(getattr(config, "pdb_b", "") or "").strip()
        if pdb_b_val:
            print(f"[Geom2Vec] Warning: endpoint B CA ref not loaded from {pdb_b_val} (required for VAMP boundary mode)", flush=True)
        else:
            print("[Geom2Vec] Note: pdb_b not specified; VAMP boundary mode will fall back to RMSD cutoffs", flush=True)
    
    try:
        if endpoint_valid_a is not None and endpoint_valid_b is not None:
            endpoint_token_mask = (np.asarray(endpoint_valid_a, dtype=bool) & np.asarray(endpoint_valid_b, dtype=bool)).reshape(-1)
        elif endpoint_valid_a is not None:
            endpoint_token_mask = np.asarray(endpoint_valid_a, dtype=bool).reshape(-1)
        else:
            endpoint_token_mask = None
    except Exception:
        endpoint_token_mask = None

    boundary_mode = str(getattr(config, "path_boundary_mode", "rmsd_cutoff") or "rmsd_cutoff").strip().lower()
    # Repo preference: keep A/B definitions stable. Even when learned state models drift,
    # RMSD-to-endpoints cutoffs remain interpretable and time-consistent.
    if boundary_mode != "rmsd_cutoff":
        print(
            f"[Path Boundaries] Note: forcing boundary_mode=rmsd_cutoff (requested: {boundary_mode}); "
            "this affects only the built-in boundary modes (plugin boundaries, if configured, still run).",
            flush=True,
        )
        boundary_mode = "rmsd_cutoff"

    boundary_plugin_spec = str(getattr(config, "path_boundary_plugin", "") or "").strip()
    boundary_plugin_fn: Optional[Callable[..., Any]] = None
    boundary_plugin_failed = False
    boundary_vac_fallback_warned = False
    boundary_vamp_state_fallback_warned = False

    # Optional plugin-provided boundary models may want access to a stable list of
    # cached trajectory keys for fitting (updated on each VAMP refresh).
    boundary_plugin_fit_keys: List[str] = []
    boundary_plugin_fit_nonce = 0

    _load_boundary_plugin = load_boundary_plugin

    if boundary_plugin_spec:
        try:
            boundary_plugin_fn = _load_boundary_plugin(boundary_plugin_spec)
            print(f"[Path Boundaries] Using boundary plugin: {boundary_plugin_spec}")
        except Exception as exc:
            boundary_plugin_fn = None
            boundary_plugin_failed = True
            print(f"[Path Boundaries] Warning: failed to load plugin {boundary_plugin_spec!r}: {exc}; using RMSD cutoffs.")

    _boundary_units_warned = [False]  # mutable container for warning state

    def _rmsd_to_endpoints(ca_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _cu.rmsd_to_endpoints(
            ca_coords,
            endpoint_ca_ref_a=endpoint_ca_ref_a,
            endpoint_ca_ref_b=endpoint_ca_ref_b,
            endpoint_token_mask=endpoint_token_mask,
            _units_warned=_boundary_units_warned,
        )

    # Optional: augment DGA committor features with policy global tokens (in addition to VAC/VAMP features).
    # This uses geom2vec token CA coordinates only and computes RMSD/E2E/Rg and (for ADK) theta/dist metrics.
    _adk_token_groups_cache: Dict[Tuple[str, int, Optional[Tuple[int, int]]], Any] = {}

    def _compute_globals_from_ca_tokens(ca_coords: Any) -> np.ndarray:
        if global_dim <= 0 or (not global_token_names):
            try:
                T0 = int(np.asarray(ca_coords).shape[0])
            except Exception:
                T0 = 0
            return np.zeros((max(0, T0), 0), dtype=np.float32)

        try:
            ca_np = ca_coords.detach().cpu().numpy() if isinstance(ca_coords, torch.Tensor) else ca_coords
        except Exception:
            ca_np = ca_coords
        ca_raw = np.asarray(ca_np, dtype=np.float64)
        if ca_raw.ndim != 3 or ca_raw.shape[-1] != 3:
            return np.zeros((0, int(global_dim)), dtype=np.float32)
        T0 = int(ca_raw.shape[0])
        n_tokens = int(ca_raw.shape[1])

        # Convert to nm (geom2vec cached CA coords are typically Å).
        ca_nm = ca_raw * 0.1
        try:
            if n_tokens >= 3:
                d = np.linalg.norm(ca_raw[:, 1:, :] - ca_raw[:, :-1, :], axis=2).reshape(-1)
                d = d[np.isfinite(d) & (d > 0)]
                med = float(np.median(d)) if d.size else float("nan")
                if np.isfinite(med) and med < 1.0:
                    ca_nm = ca_raw
        except Exception:
            ca_nm = ca_raw * 0.1

        # RMSD-to-endpoints if available (nm).
        rmsd_a_nm = None
        rmsd_b_nm = None
        try:
            ra, rb = _rmsd_to_endpoints(ca_raw)
            rmsd_a_nm = np.asarray(ra, dtype=np.float32).reshape(-1)
            rmsd_b_nm = np.asarray(rb, dtype=np.float32).reshape(-1)
        except Exception:
            rmsd_a_nm = None
            rmsd_b_nm = None

        # E2E and Rg over CA tokens (nm).
        e2e_nm = None
        rg_nm = None
        try:
            if n_tokens >= 2:
                e2e_nm = np.linalg.norm(ca_nm[:, 0, :] - ca_nm[:, -1, :], axis=1).astype(np.float32, copy=False)
        except Exception:
            e2e_nm = None
        try:
            centered = ca_nm - ca_nm.mean(axis=1, keepdims=True)
            rg_nm = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1)).astype(np.float32, copy=False)
        except Exception:
            rg_nm = None

        # ADK domain features from token CA coords (theta/dist); only computed if needed.
        need_adk = any(str(n) in set(ADK_GLOBAL_TOKEN_NAMES) for n in global_token_names) or any(
            str(n).lower().startswith(("theta_", "dist_")) for n in global_token_names
        )
        adk_theta_nmp = None
        adk_theta_lid = None
        adk_d145_52 = None
        adk_d169_55 = None
        if need_adk:
            try:
                pdb_a_path = str(getattr(config, "pdb_a", config.pdb) or config.pdb)
                mask_sig = None
                if endpoint_token_mask is not None:
                    try:
                        m = np.asarray(endpoint_token_mask, dtype=bool).reshape(-1)
                        if int(m.size) >= int(n_tokens):
                            mask_sig = (int(m[:n_tokens].sum()), int(m.size))
                    except Exception:
                        mask_sig = None
                ck = (pdb_a_path, int(n_tokens), mask_sig)
                groups = _adk_token_groups_cache.get(ck, None)
                if groups is None:
                    tr = md.load(str(pdb_a_path))
                    residues = list(tr.topology.residues)
                    resseq = []
                    for r in residues:
                        try:
                            rs = getattr(r, "resSeq", None)
                            rs_i = int(rs) if rs is not None else int(r.index)
                        except Exception:
                            rs_i = int(r.index)
                        resseq.append(int(rs_i))
                    resseq = resseq[: int(min(int(n_tokens), len(resseq)))]
                    resseq_to_token: Dict[int, int] = {}
                    for i, rs in enumerate(resseq):
                        if int(rs) not in resseq_to_token:
                            resseq_to_token[int(rs)] = int(i)

                    def _tok_for_res(resnum: int) -> int:
                        target = int(resnum)
                        if target in resseq_to_token:
                            return int(resseq_to_token[target])
                        if 0 <= target < int(len(resseq)):
                            return int(target)
                        if 0 <= (target - 1) < int(len(resseq)):
                            return int(target - 1)
                        return -1

                    m_tok = None
                    if endpoint_token_mask is not None:
                        try:
                            m = np.asarray(endpoint_token_mask, dtype=bool).reshape(-1)
                            if int(m.size) >= int(n_tokens):
                                m_tok = m[: int(n_tokens)]
                        except Exception:
                            m_tok = None

                    def _range(start: int, end: int) -> np.ndarray:
                        s = int(min(start, end))
                        e = int(max(start, end))
                        idx = np.asarray([i for i, rs in enumerate(resseq) if s <= int(rs) <= e], dtype=np.int64)
                        if m_tok is not None and idx.size:
                            idx = idx[m_tok[idx]]
                        return idx

                    groups = {
                        "core_lid": _range(115, 125),
                        "core1": _range(90, 100),
                        "nmp": _range(35, 55),
                        "core2": _range(179, 185),
                        "lid": _range(125, 153),
                        "res145": _tok_for_res(145),
                        "res52": _tok_for_res(52),
                        "res169": _tok_for_res(169),
                        "res55": _tok_for_res(55),
                    }
                    _adk_token_groups_cache[ck] = groups

                def _centroid(idx: np.ndarray) -> np.ndarray:
                    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
                    idx = idx[(idx >= 0) & (idx < int(n_tokens))]
                    if idx.size == 0:
                        return np.zeros((int(T0), 3), dtype=np.float32)
                    return ca_nm[:, idx, :].mean(axis=1).astype(np.float32, copy=False)

                def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
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

                def _dist(a: int, b: int) -> np.ndarray:
                    ai = int(a)
                    bi = int(b)
                    if ai < 0 or bi < 0 or ai >= int(n_tokens) or bi >= int(n_tokens) or ai == bi:
                        return np.zeros((int(T0),), dtype=np.float32)
                    return np.linalg.norm(ca_nm[:, ai, :] - ca_nm[:, bi, :], axis=1).astype(np.float32, copy=False)

                core_lid = _centroid(groups["core_lid"])
                core1 = _centroid(groups["core1"])
                nmp = _centroid(groups["nmp"])
                core2 = _centroid(groups["core2"])
                lid = _centroid(groups["lid"])
                adk_theta_nmp = _angle(core_lid, core1, nmp)
                adk_theta_lid = _angle(core2, core_lid, lid)
                adk_d145_52 = _dist(int(groups["res145"]), int(groups["res52"]))
                adk_d169_55 = _dist(int(groups["res169"]), int(groups["res55"]))
            except Exception:
                adk_theta_nmp = None
                adk_theta_lid = None
                adk_d145_52 = None
                adk_d169_55 = None

        out = np.zeros((int(T0), int(global_dim)), dtype=np.float32)
        for j, name in enumerate(global_token_names):
            key = str(name)
            key_l = key.lower()
            if key_l.startswith(("rmsd_a", "rmsda")):
                if rmsd_a_nm is not None and rmsd_a_nm.shape[0] == int(T0):
                    out[:, j] = rmsd_a_nm
            elif key_l.startswith(("rmsd_b", "rmsdb")):
                if rmsd_b_nm is not None and rmsd_b_nm.shape[0] == int(T0):
                    out[:, j] = rmsd_b_nm
            elif key_l.startswith("rmsd"):
                if rmsd_a_nm is not None and rmsd_a_nm.shape[0] == int(T0):
                    out[:, j] = rmsd_a_nm
            elif key_l.startswith("e2e"):
                if e2e_nm is not None and e2e_nm.shape[0] == int(T0):
                    out[:, j] = e2e_nm
            elif key_l.startswith("rg"):
                if rg_nm is not None and rg_nm.shape[0] == int(T0):
                    out[:, j] = rg_nm
            elif key == "theta_nmp_rad" and adk_theta_nmp is not None:
                out[:, j] = adk_theta_nmp
            elif key == "theta_lid_rad" and adk_theta_lid is not None:
                out[:, j] = adk_theta_lid
            elif key == "dist_145_52_nm" and adk_d145_52 is not None:
                out[:, j] = adk_d145_52
            elif key == "dist_169_55_nm" and adk_d169_55 is not None:
                out[:, j] = adk_d169_55

        out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
        return out

    def _augment_committor_dga_features(z_arr: np.ndarray, ca_coords: Any, *, use_globals: bool) -> np.ndarray:
        z = np.asarray(z_arr, dtype=np.float32)
        if (not use_globals) or global_dim <= 0:
            return z
        g = _compute_globals_from_ca_tokens(ca_coords)
        if g.ndim != 2 or z.ndim != 2 or int(g.shape[0]) != int(z.shape[0]):
            return z
        return np.concatenate([z, np.asarray(g, dtype=np.float32)], axis=1).astype(np.float32, copy=False)

    def _boundary_masks_from_ca(
        ca_coords: np.ndarray,
        *,
        cutoff_a_nm: float,
        cutoff_b_nm: float,
        graph_features: Optional[Any] = None,
        z_lin: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nonlocal boundary_vac_fallback_warned
        ca_input = ca_coords
        ca_np = ca_coords
        try:
            if isinstance(ca_coords, torch.Tensor):
                ca_np = ca_coords.detach().cpu().numpy()
        except Exception:
            ca_np = ca_coords

        # Cached CA coords are in Å (geom2vec input convention).
        # - Boundary plugins expect nm.
        # - Internal RMSD helpers (_rmsd_to_endpoints) convert Å -> nm themselves.
        ca_aa = np.asarray(ca_np, dtype=np.float64)
        if ca_aa.ndim != 3 or ca_aa.shape[-1] != 3:
            raise ValueError(f"ca_coords must be (T, tokens, 3), got {ca_aa.shape}")
        T = int(ca_aa.shape[0])
        ca_nm = ca_aa * 0.1

        if boundary_plugin_fn is not None and not boundary_plugin_failed:
            try:
                plugin_kwargs: Dict[str, Any] = {
                    "ca_coords": ca_nm,
                    "endpoint_ca_ref_a": endpoint_ca_ref_a,
                    "endpoint_ca_ref_b": endpoint_ca_ref_b,
                    "cutoff_a_nm": float(cutoff_a_nm),
                    "cutoff_b_nm": float(cutoff_b_nm),
                    "token_mask": endpoint_token_mask,
                    # Optional extra context for advanced plugins (all optional).
                    "z_lin": z_lin,
                    "fit_keys": list(boundary_plugin_fit_keys),
                    "fit_nonce": int(boundary_plugin_fit_nonce),
                    "embedding_cache": geom2vec_cache,
                    "projector_linear": geom2vec_projector_linear,
                    "config": config,
                    # Optional identifier for plugins that need to locate the underlying trajectory.
                    "cache_key": (str(cache_key) if cache_key is not None else None),
                }
                try:
                    import inspect

                    sig = inspect.signature(boundary_plugin_fn)
                    params = sig.parameters
                    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                    if not has_varkw:
                        plugin_kwargs = {k: v for k, v in plugin_kwargs.items() if k in params}
                except Exception:
                    pass
                out = boundary_plugin_fn(**plugin_kwargs)
                if not isinstance(out, tuple) or len(out) < 2:
                    raise ValueError("Boundary plugin must return (in_a, in_b) or (in_a, in_b, metric_a, metric_b).")
                in_a = np.asarray(out[0], dtype=bool).reshape(T)
                in_b = np.asarray(out[1], dtype=bool).reshape(T)
                if len(out) >= 4:
                    rmsd_a = np.asarray(out[2], dtype=np.float64).reshape(T)
                    rmsd_b = np.asarray(out[3], dtype=np.float64).reshape(T)
                else:
                    rmsd_a = np.full((T,), np.nan, dtype=np.float64)
                    rmsd_b = np.full((T,), np.nan, dtype=np.float64)

                both = in_a & in_b
                if np.any(both):
                    # Resolve overlaps; prefer using provided metrics if available.
                    if np.all(np.isfinite(rmsd_a)) and np.all(np.isfinite(rmsd_b)):
                        choose_a = rmsd_a <= rmsd_b
                        in_a = in_a & ~both
                        in_b = in_b & ~both
                        in_a[both] = choose_a[both]
                        in_b[both] = ~choose_a[both]
                    else:
                        in_b = in_b & ~both
                return in_a, in_b, rmsd_a, rmsd_b
            except Exception as exc:
                boundary_plugin_failed = True
                print(f"[Path Boundaries] Warning: boundary plugin failed once ({exc}); falling back to RMSD cutoffs.")

        # VAMP state-based boundaries: A and B are single VAMP states auto-detected from endpoints.
        if boundary_mode == "vamp_state":
            if (
                committor_state.boundary_vamp_state_valid
                and committor_state.boundary_vamp_a_state is not None
                and committor_state.boundary_vamp_b_state is not None
                and geom2vec_projector is not None
            ):
                try:
                    # Get VAMP state probabilities for this trajectory
                    probs_raw = None
                    if graph_features is not None:
                        if (
                            getattr(geom2vec_projector, "trained", None) is not None
                            or bool(getattr(geom2vec_projector, "trained_ensemble", None))
                        ):
                            probs_raw = geom2vec_projector.state_probs_from_embeddings(graph_features, ca_input)
                    if probs_raw is not None:
                        probs = np.asarray(probs_raw, dtype=np.float32)
                        if probs.ndim == 2 and int(probs.shape[0]) == T:
                            # Apply state permutation if available
                            if geom2vec_state_permutation_map is not None:
                                try:
                                    perm = np.asarray(geom2vec_state_permutation_map, dtype=np.int64)
                                    if perm.shape[0] == probs.shape[1]:
                                        probs = probs[:, perm]
                                except Exception:
                                    pass
                            labels = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                            in_a = (labels == int(committor_state.boundary_vamp_a_state))
                            in_b = (labels == int(committor_state.boundary_vamp_b_state))
                            rmsd_a, rmsd_b = _rmsd_to_endpoints(ca_aa)
                            # Resolve overlaps (shouldn't happen with single states, but be safe)
                            both = in_a & in_b
                            if np.any(both):
                                in_b = in_b & ~both
                            return (
                                in_a.astype(bool, copy=False),
                                in_b.astype(bool, copy=False),
                                rmsd_a,
                                rmsd_b,
                            )
                except Exception:
                    pass
            # Fall through to RMSD if VAMP state boundaries not ready
            if not boundary_vamp_state_fallback_warned:
                boundary_vamp_state_fallback_warned = True
                print(
                    "[Path Boundaries] VAMP state boundary mode requested but states not ready; "
                    "falling back to RMSD cutoffs for now.",
                    flush=True,
                )

        # VAC-kmeans boundaries (requires a fitted k-means model + linear projector).
        if boundary_mode in {"vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent"}:
            if (
                committor_state.boundary_vac_kmeans is not None
                and committor_state.boundary_vac_mu is not None
                and committor_state.boundary_vac_std is not None
                and committor_state.boundary_vac_a_clusters is not None
                and committor_state.boundary_vac_b_clusters is not None
                and bool(np.any(committor_state.boundary_vac_a_clusters))
                and bool(np.any(committor_state.boundary_vac_b_clusters))
            ):
                z_arr = None
                if z_lin is not None:
                    try:
                        z_arr = np.asarray(z_lin, dtype=np.float64)
                    except Exception:
                        z_arr = None
                if z_arr is None and graph_features is not None and geom2vec_projector_linear is not None:
                    try:
                        if (
                            getattr(geom2vec_projector_linear, "trained", None) is not None
                            or bool(getattr(geom2vec_projector_linear, "trained_ensemble", None))
                        ):
                            z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph_features, ca_input)
                            z_arr = np.asarray(z_raw, dtype=np.float64)
                    except Exception:
                        z_arr = None
                dim_km = int(committor_state.boundary_vac_mu.shape[0]) if committor_state.boundary_vac_mu is not None else 0
                if z_arr is not None and z_arr.ndim == 2 and int(z_arr.shape[0]) == T and int(z_arr.shape[1]) >= dim_km and dim_km > 0:
                    z_d = np.asarray(z_arr[:, :dim_km], dtype=np.float64)
                    mu = np.asarray(committor_state.boundary_vac_mu, dtype=np.float64).reshape(1, -1)
                    std = np.asarray(committor_state.boundary_vac_std, dtype=np.float64).reshape(1, -1)
                    std = np.where(std > 1e-8, std, 1.0)
                    z_norm = (z_d - mu) / std
                    try:
                        labels = committor_state.boundary_vac_kmeans.predict(z_norm)
                        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
                        in_a = np.asarray(committor_state.boundary_vac_a_clusters, dtype=bool)[labels]
                        in_b = np.asarray(committor_state.boundary_vac_b_clusters, dtype=bool)[labels]
                        rmsd_a, rmsd_b = _rmsd_to_endpoints(ca_aa)
                        both = in_a & in_b
                        if np.any(both):
                            in_b = in_b & ~both
                        return (
                            in_a.astype(bool, copy=False),
                            in_b.astype(bool, copy=False),
                            rmsd_a,
                            rmsd_b,
                        )
                    except Exception:
                        pass

            if not boundary_vac_fallback_warned:
                boundary_vac_fallback_warned = True
                print(
                    "[Path Boundaries] VAC-kmeans boundary mode requested but model is not ready; "
                    "falling back to RMSD cutoffs for now."
                )

        # Default: RMSD-to-endpoints cutoffs.
        rmsd_a, rmsd_b = _rmsd_to_endpoints(ca_aa)
        ca_cut_a = float(cutoff_a_nm)
        ca_cut_b = float(cutoff_b_nm)
        in_a = rmsd_a <= ca_cut_a
        in_b = rmsd_b <= ca_cut_b
        both = in_a & in_b
        if np.any(both):
            choose_a = rmsd_a < rmsd_b
            in_a = in_a & ~both
            in_b = in_b & ~both
            in_a[both] = choose_a[both]
            in_b[both] = ~choose_a[both]
        return in_a.astype(bool, copy=False), in_b.astype(bool, copy=False), rmsd_a, rmsd_b

    def _committor_connectivity_score(
                q_f: np.ndarray,
                lab_f: np.ndarray,
                *,
                in_a: Optional[np.ndarray] = None,
                in_b: Optional[np.ndarray] = None,
                return_diag: bool = False,
                tube_bins: int,
                K_tube: int,
                edge_delta: int,
                edge_bin_change_only: bool,
                edge_undirected: bool,
                connect_bonus: float,
                shortest_path_weight: float,
                flow_weight: float,
                flow_cap: int,
                component_merge_weight: float,
                gap_weight: float,
                comp_ids: Optional[np.ndarray] = None,
            ) -> Any:
            """Score a trajectory by how much it improves global A↔B connectivity in tube-cell graph."""
            diag: Dict[str, Any] = {}
            if committor_state.conn_adj is None:
                return (0.0, diag) if return_diag else 0.0
            q_f = np.asarray(q_f, dtype=np.float64).reshape(-1)
            lab_f = np.asarray(lab_f, dtype=np.int64).reshape(-1)
            ia_f = None
            ib_f = None
            if in_a is not None:
                try:
                    ia_f = np.asarray(in_a, dtype=bool).reshape(-1)
                except Exception:
                    ia_f = None
            if in_b is not None:
                try:
                    ib_f = np.asarray(in_b, dtype=bool).reshape(-1)
                except Exception:
                    ib_f = None
            n = int(min(int(q_f.shape[0]), int(lab_f.shape[0])))
            if ia_f is not None:
                n = int(min(int(n), int(ia_f.shape[0])))
            if ib_f is not None:
                n = int(min(int(n), int(ib_f.shape[0])))
            if n <= 0:
                return (0.0, diag) if return_diag else 0.0
            M = int(tube_bins)
            K = int(max(1, int(K_tube)))
            if M <= 1:
                return (0.0, diag) if return_diag else 0.0
            n_cells = int(M) * int(K)
            if n_cells <= 1:
                return (0.0, diag) if return_diag else 0.0

            q_u = np.clip(q_f[:n], 0.0, 1.0)
            lab_u = np.clip(lab_f[:n], 0, int(K) - 1)
            m_bins = np.floor(float(M) * q_u).astype(np.int64, copy=False)
            m_bins = np.clip(m_bins, 0, int(M) - 1)
            cell = m_bins * int(K) + lab_u

            # Trajectory-local A/B cell sets for source/target expansion.
            a_cells_local: set[int] = set()
            b_cells_local: set[int] = set()
            try:
                if ia_f is not None and ib_f is not None:
                    a_mask = np.asarray(ia_f[:n], dtype=bool)
                    b_mask = np.asarray(ib_f[:n], dtype=bool)
                else:
                    # Fallback: treat committor extremes as A/B when boundary masks are unavailable.
                    a_mask = m_bins == 0
                    b_mask = m_bins == (int(M) - 1)
                if bool(np.any(a_mask)):
                    a_cells_local = set(np.asarray(cell, dtype=np.int64)[a_mask].tolist())
                if bool(np.any(b_mask)):
                    b_cells_local = set(np.asarray(cell, dtype=np.int64)[b_mask].tolist())
            except Exception:
                a_cells_local = set()
                b_cells_local = set()

            sources = set(committor_state.conn_a_cells)
            sources.update(a_cells_local)
            targets = set(committor_state.conn_b_cells)
            targets.update(b_cells_local)
            if not sources or not targets:
                diag.update({"n_frames": int(n), "n_new_edges": 0, "note": "empty_sources_or_targets"})
                return (0.0, diag) if return_diag else 0.0

            dist_before = _conn_shortest_path_len(
                adj=committor_state.conn_adj,
                sources=list(sources),
                targets=targets,
                overlay=None,
            )
            flow_before = 0
            if float(flow_weight) != 0.0 and np.isfinite(dist_before):
                try:
                    flow_before = int(
                        _conn_max_flow_unit(
                            adj=committor_state.conn_adj,
                            sources=list(sources),
                            targets=targets,
                            overlay=None,
                            cap_max=int(flow_cap),
                        )
                    )
                except Exception:
                    flow_before = 0
            # Pre-connection completeness proxy: how far the A-side and B-side reachable components
            # have expanded along committor-bin index (works even when A and B are not connected).
            gap_before = None
            if float(gap_weight) != 0.0:
                try:
                    reach_a = _conn_reachable_nodes(adj=committor_state.conn_adj, sources=list(sources), overlay=None)
                    reach_b = _conn_reachable_nodes(adj=committor_state.conn_adj, sources=list(targets), overlay=None)
                    gap_before = _conn_bin_gap(reachable_a=reach_a, reachable_b=reach_b, K=K, M=M)[0]
                except Exception:
                    gap_before = None

            # Candidate new undirected edges (not already in the global graph).
            delta = int(max(1, int(edge_delta)))
            if n <= delta:
                # Still allow component merge reward if there are no edges (none).
                diag.update({"n_frames": int(n), "n_new_edges": 0, "note": "too_short"})
                return (0.0, diag) if return_diag else 0.0

            src = np.asarray(cell[: n - delta], dtype=np.int64)
            dst = np.asarray(cell[delta:n], dtype=np.int64)
            if edge_bin_change_only:
                mb0 = np.asarray(m_bins[: n - delta], dtype=np.int64)
                mb1 = np.asarray(m_bins[delta:n], dtype=np.int64)
                mask = mb0 != mb1
                if not bool(np.any(mask)):
                    diag.update({"n_frames": int(n), "n_new_edges": 0, "note": "no_bin_change"})
                    return (0.0, diag) if return_diag else 0.0
                src = src[mask]
                dst = dst[mask]
            if src.size == 0:
                diag.update({"n_frames": int(n), "n_new_edges": 0, "note": "empty_edges"})
                return (0.0, diag) if return_diag else 0.0

            if edge_undirected:
                a = np.minimum(src, dst)
                b = np.maximum(src, dst)
                src = a
                dst = b

            edge_idx = src.astype(np.int64) * int(n_cells) + dst.astype(np.int64)
            uniq = np.unique(edge_idx)
            overlay: Dict[int, List[int]] = {}
            merges = 0
            new_edges = 0
            if component_merge_weight != 0.0 and comp_ids is not None and int(comp_ids.shape[0]) == int(n_cells):
                parent: Dict[int, int] = {}

                def find(x: int) -> int:
                    p = parent.get(x, x)
                    while p != parent.get(p, p):
                        p = parent.get(p, p)
                    # Path compression (best-effort).
                    cur = x
                    while cur != p:
                        nxt = parent.get(cur, cur)
                        parent[cur] = p
                        if nxt == cur:
                            break
                        cur = nxt
                    return p

                def union(a: int, b: int) -> bool:
                    ra = find(a)
                    rb = find(b)
                    if ra == rb:
                        return False
                    parent[ra] = rb
                    return True
            else:
                parent = None  # type: ignore[assignment]
                find = None  # type: ignore[assignment]
                union = None  # type: ignore[assignment]

            new_any = False
            for e in uniq.tolist():
                ei = int(e)
                if ei in committor_state.conn_edges:
                    continue
                new_any = True
                new_edges += 1
                u = int(ei // int(n_cells))
                v = int(ei - u * int(n_cells))
                if 0 <= u < int(n_cells) and 0 <= v < int(n_cells) and u != v:
                    overlay.setdefault(u, []).append(v)
                    overlay.setdefault(v, []).append(u)
                    if component_merge_weight != 0.0 and comp_ids is not None:
                        cu = int(comp_ids[u])
                        cv = int(comp_ids[v])
                        if cu != cv and union is not None and union(cu, cv):
                            merges += 1

            if not new_any:
                diag.update({"n_frames": int(n), "n_new_edges": 0, "note": "no_new_edges"})
                return (0.0, diag) if return_diag else 0.0

                dist_after = _conn_shortest_path_len(
                    adj=committor_state.conn_adj,
                    sources=list(sources),
                    targets=targets,
                    overlay=overlay,
                )
                flow_after = 0
                if float(flow_weight) != 0.0 and np.isfinite(dist_after):
                    try:
                        flow_after = int(
                            _conn_max_flow_unit(
                                adj=committor_state.conn_adj,
                                sources=list(sources),
                                targets=targets,
                                overlay=overlay,
                                cap_max=int(flow_cap),
                            )
                        )
                    except Exception:
                        flow_after = 0
            gap_improve = 0.0
            gap_after = None
            if float(gap_weight) != 0.0 and gap_before is not None:
                try:
                    reach_a2 = _conn_reachable_nodes(adj=committor_state.conn_adj, sources=list(sources), overlay=overlay)
                    reach_b2 = _conn_reachable_nodes(adj=committor_state.conn_adj, sources=list(targets), overlay=overlay)
                    gap_after = int(_conn_bin_gap(reachable_a=reach_a2, reachable_b=reach_b2, K=K, M=M)[0])
                    gap_improve = float(max(0, int(gap_before) - int(gap_after)))
                except Exception:
                    gap_improve = 0.0

                connected_before = bool(np.isfinite(dist_before))
                connected_after = bool(np.isfinite(dist_after))
                bonus = float(connect_bonus) if ((not connected_before) and connected_after) else 0.0
                improve = 0.0
                if connected_before and connected_after:
                    improve = max(0.0, float(dist_before) - float(dist_after))
                flow_improve = 0.0
                if connected_before and connected_after:
                    flow_improve = float(max(0, int(flow_after) - int(flow_before)))

                raw = (
                    float(shortest_path_weight) * float(improve)
                    + float(bonus)
                    + float(flow_weight) * float(flow_improve)
                    + float(component_merge_weight) * float(merges)
                    + float(gap_weight) * float(gap_improve)
                )
            if not np.isfinite(raw):
                raw = 0.0

                if return_diag:
                    diag.update(
                        {
                        "n_frames": int(n),
                        "n_new_edges": int(new_edges),
                        "connected_before": bool(connected_before),
                        "connected_after": bool(connected_after),
                        "connect_bonus": float(bonus),
                        "path_improve": float(improve),
                        "gap_before": (None if gap_before is None else int(gap_before)),
                        "gap_after": (None if gap_after is None else int(gap_after)),
                        "gap_improve": float(gap_improve),
                        "merges": int(merges),
                            "dist_before": (None if not connected_before else float(dist_before)),
                            "dist_after": (None if not connected_after else float(dist_after)),
                            "flow_before": (None if not connected_before else int(flow_before)),
                            "flow_after": (None if not connected_after else int(flow_after)),
                            "flow_improve": float(flow_improve),
                        }
                    )
                    return float(raw), diag

                return float(raw)

    def _conn_max_flow_unit(*, adj, sources, targets, overlay=None, cap_max=10):
        return _graph.max_flow_unit_cap(adj=adj, sources=sources, targets=targets, overlay=overlay, cap_max=cap_max)


    def _refresh_vamp_projectors(reason: str) -> None:
        """Refresh VAMP + VAC projectors, boundary models, and committor. Delegated to component."""
        nonlocal geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, geom2vec_lagtime_embed
        nonlocal boundary_plugin_fit_keys, boundary_plugin_fit_nonce
        from .components.vamp_projector_refresh import _refresh_vamp_projectors as _impl
        # Build kwargs matching the closure's captured state
        result = _impl(
            reason,
            config=config,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector=geom2vec_projector,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_mgr=geom2vec_mgr,
            geom2vec_embed_stride=geom2vec_embed_stride,
            geom2vec_state_permutation_map=geom2vec_state_permutation_map,
            geom2vec_state_inverse_permutation_map=geom2vec_state_inverse_permutation_map,
            geom2vec_lagtime_embed=geom2vec_lagtime_embed,
            all_epoch_data=all_epoch_data,
            all_trajectory_lengths=all_trajectory_lengths,
            all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
            all_epoch_vamp_exclude=all_epoch_vamp_exclude,
            endpoint_ca_ref_a=endpoint_ca_ref_a,
            endpoint_ca_ref_b=endpoint_ca_ref_b,
            endpoint_token_mask=endpoint_token_mask,
            committor_state=committor_state,
            n_states=n_states,
            boundary_mode=boundary_mode,
            boundary_plugin_fn=boundary_plugin_fn,
            boundary_plugin_fit_keys=boundary_plugin_fit_keys,
            boundary_plugin_fit_nonce=boundary_plugin_fit_nonce,
            _rmsd_to_endpoints=_rmsd_to_endpoints,
            _boundary_masks_from_ca=_boundary_masks_from_ca,
            _invalidate_loo_vamp_stats=_invalidate_loo_vamp_stats,
            _invalidate_vamp_bin_stats=_invalidate_vamp_bin_stats,
            _invalidate_msm_ig_stats=_invalidate_msm_ig_stats,
            _invalidate_committor_state=_invalidate_committor_state,
            _invalidate_boundary_vac_kmeans=_invalidate_boundary_vac_kmeans,
            _refresh_retrieval_cache_labels=_refresh_retrieval_cache_labels,
            _ensure_msm_ig_history_counts=_ensure_msm_ig_history_counts,
            _ensure_committor_tube_history_counts=_ensure_committor_tube_history_counts,
            _iter_history_trajs=_iter_history_trajs,
        )
        if result is not None:
            geom2vec_state_permutation_map = result.get("perm", geom2vec_state_permutation_map)
            geom2vec_state_inverse_permutation_map = result.get("inv_perm", geom2vec_state_inverse_permutation_map)
            geom2vec_lagtime_embed = result.get("lagtime_embed", geom2vec_lagtime_embed)
            boundary_plugin_fit_keys = result.get("boundary_plugin_fit_keys", boundary_plugin_fit_keys)
            boundary_plugin_fit_nonce = result.get("boundary_plugin_fit_nonce", boundary_plugin_fit_nonce)


    def _refresh_geom2vec_vamp(reason: str) -> None:
        nonlocal geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, geom2vec_lagtime_embed

        def _reset_policy(tag):
            if getattr(config, "reset_policy_on_projector_refresh", False):
                if isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy):
                    with torch.no_grad():
                        if getattr(agent, "logits", None) is not None:
                            agent.logits.zero_()
                        if getattr(agent_old, "logits", None) is not None:
                            agent_old.logits.zero_()
                    for opt in policy_opts:
                        opt.state.clear()
                    print("[Geom2Vec Refresh] Reset StateLogitPolicy logits to uniform after refresh.")

        def _ensure_msm_ig_after_refresh(tag):
            _ensure_msm_ig_history_counts(
                upto_epoch_exclusive=len(all_epoch_data),
                n_states=int(getattr(config, "geom2vec_dim", 2) or 2),
                lag_embed=int(geom2vec_lagtime_embed),
                tag=tag,
            )

        perm, inv_perm, lag = _refresh_geom2vec_vamp_impl(
            reason,
            config=config,
            geom2vec_mgr=geom2vec_mgr,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector=geom2vec_projector,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_embed_stride=geom2vec_embed_stride,
            geom2vec_state_permutation_map=geom2vec_state_permutation_map,
            all_epoch_data=all_epoch_data,
            all_trajectory_lengths=all_trajectory_lengths,
            all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
            all_epoch_vamp_exclude=all_epoch_vamp_exclude,
            invalidate_callbacks=[
                _invalidate_loo_vamp_stats,
                _invalidate_vamp_bin_stats,
                _invalidate_msm_ig_stats,
                _invalidate_committor_state,
            ],
            refresh_retrieval_labels_fn=_refresh_retrieval_cache_labels,
            ensure_msm_ig_fn=_ensure_msm_ig_after_refresh,
            reset_policy_fn=_reset_policy,
        )
        geom2vec_state_permutation_map = perm
        geom2vec_state_inverse_permutation_map = inv_perm
        geom2vec_lagtime_embed = lag


    def _run_anchoring_round() -> None:
        from .components.anchoring import _run_anchoring_round as _anchor_impl
        _anchor_impl(
            config=config,
            envs=envs,
            sampler=sampler,
            agent=agent,
            agent_old=agent_old,
            env_ref=env_ref,
            gen_device=gen_device,
            torch_device=torch_device,
            all_epoch_data=all_epoch_data,
            all_trajectory_lengths=all_trajectory_lengths,
            all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
            all_epoch_traj_dcd_frame_indices=all_epoch_traj_dcd_frame_indices,
            all_epoch_vamp_exclude=all_epoch_vamp_exclude,
            append_retrieval_cache_fn=_append_retrieval_cache,
            rebalance_fn=_rebalance_retrieval_cache,
            refresh_geom2vec_fn=_refresh_geom2vec_vamp,
            feature_builder=feature_builder,
            grpo_opt=grpo_opt,
            geom2vec_mgr=geom2vec_mgr,
        )


    for ep in range(ep_start, config.episodes):
        print(f"\n{'='*60}")
        print(f"Starting Epoch {ep+1}/{config.episodes}")
        print(f"{'='*60}")
        epoch_start_time = time.time()

        # A. Refresh projector (every N epochs)
        refresh_interval = int(getattr(config, "pca_refresh_interval", 1) or 1)
        label_mode = str(getattr(config, "geom2vec_label_mode", "auto") or "auto").strip().lower()
        need_vamp = label_mode not in {"linear_kmeans", "vac_kmeans", "linear-kmeans", "kmeans"}
        model_ready = bool(
            geom2vec_projector_linear is not None
            and (
                getattr(geom2vec_projector_linear, "trained", None) is not None
                or bool(getattr(geom2vec_projector_linear, "trained_ensemble", None))
            )
            and (
                (not need_vamp)
                or (
                    geom2vec_projector is not None
                    and (
                        getattr(geom2vec_projector, "trained", None) is not None
                        or bool(getattr(geom2vec_projector, "trained_ensemble", None))
                    )
                )
            )
        )
        if refresh_interval <= 0:
            should_update = not model_ready
        else:
            should_update = ((ep + 1) % refresh_interval == 0) or (not model_ready)

        if should_update and all_epoch_data:
            print(f"[Epoch {ep+1}] Refreshing Geom2Vec VAMP lobe...")
            _refresh_geom2vec_vamp(reason=f"epoch {ep+1}")
            model_ready = bool(
                geom2vec_projector_linear is not None
                and (
                    getattr(geom2vec_projector_linear, "trained", None) is not None
                    or bool(getattr(geom2vec_projector_linear, "trained_ensemble", None))
                )
                and (
                    (not need_vamp)
                    or (
                        geom2vec_projector is not None
                        and (
                            getattr(geom2vec_projector, "trained", None) is not None
                            or bool(getattr(geom2vec_projector, "trained_ensemble", None))
                        )
                    )
                )
            )

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
                print(f"[LOO VAMP] Warning: failed to seed global stats ({exc_loo_seed})")

        # Enforce retrieval cache cap (uniform downsample; no VAMP-label balancing).
        if retrieval_mgr.cv and retrieval_mgr.coords:
            try:
                _rebalance_retrieval_cache(tag=f"epoch {ep+1}")
            except Exception as exc_reb:
                print(f"[Retrieval Cache] Warning: cache cap failed ({exc_reb})")

        # B. Sample torsion starts from autoregressive policy
        print(f"[Epoch {ep+1}] Phase B: Sampling torsion starts from TorsionAR policy...")

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
                has_rg_range = hasattr(config, "global_rg_range") and config.global_rg_range is not None
                token_range_map = getattr(config, "global_token_ranges", None)
                
                if token_range_map or has_rmsd_range or has_e2e_range or has_rg_range:
                    try:
                        n_gtok = int(getattr(agent_old, "n_global_tokens", 0) or 0)
                        if n_gtok <= 0:
                            global_ranges = None
                        else:
                            # Initialize with inf (no constraint by default)
                            global_ranges = torch.full((config.grp_size, n_gtok, 2), float("inf"), device=torch_device)
                            global_ranges[:, :, 0] = float("-inf")  # Set all mins to -inf initially

                        idx_rmsd_list: List[int] = []
                        idx_e2e = None
                        idx_rg = None
                        idx_explicit: set[int] = set()
                        try:
                            for i, name in enumerate(global_token_names):
                                key = str(name).lower()
                                if key.startswith("rmsd"):
                                    idx_rmsd_list.append(int(i))
                                if idx_e2e is None and key.startswith("e2e"):
                                    idx_e2e = int(i)
                                if idx_rg is None and key.startswith("rg"):
                                    idx_rg = int(i)
                        except Exception:
                            idx_rmsd_list = []
                            idx_e2e = None
                            idx_rg = None

                        range_msgs = []
                        # Apply explicit per-token ranges first.
                        if token_range_map and global_ranges is not None:
                            try:
                                for i, name in enumerate(global_token_names):
                                    key = str(name).strip().lower()
                                    if not key:
                                        continue
                                    if key not in token_range_map:
                                        continue
                                    mn, mx = token_range_map[key]
                                    if 0 <= int(i) < global_ranges.shape[1]:
                                        global_ranges[:, int(i), 0] = float(mn)
                                        global_ranges[:, int(i), 1] = float(mx)
                                        idx_explicit.add(int(i))
                                        range_msgs.append(f"{global_token_names[int(i)]}=[{float(mn):.3f}, {float(mx):.3f}]")
                            except Exception:
                                pass
                        if has_rmsd_range:
                            rmsd_min, rmsd_max = config.global_rmsd_range
                            if global_ranges is not None:
                                if not idx_rmsd_list:
                                    print(
                                        f"[TorsionAR][Epoch {ep+1}] Warning: RMSD range set but no RMSD token in globals; ignoring.",
                                        flush=True,
                                    )
                                else:
                                    for j in idx_rmsd_list:
                                        if 0 <= j < global_ranges.shape[1]:
                                            if int(j) in idx_explicit:
                                                continue
                                            global_ranges[:, j, 0] = float(rmsd_min)
                                            global_ranges[:, j, 1] = float(rmsd_max)
                                            label = global_token_names[j] if j < len(global_token_names) else "rmsd"
                                            range_msgs.append(f"{label}=[{rmsd_min:.3f}, {rmsd_max:.3f}]")
                        
                        if has_e2e_range:
                            e2e_min, e2e_max = config.global_e2e_range
                            if global_ranges is not None:
                                j = int(idx_e2e) if idx_e2e is not None else (1 if global_ranges.shape[1] >= 2 else -1)
                                if 0 <= j < global_ranges.shape[1]:
                                    if int(j) not in idx_explicit:
                                        global_ranges[:, j, 0] = float(e2e_min)
                                        global_ranges[:, j, 1] = float(e2e_max)
                                        label = global_token_names[j] if j < len(global_token_names) else "e2e"
                                        range_msgs.append(f"{label}=[{e2e_min:.3f}, {e2e_max:.3f}]")
                                else:
                                    print(
                                        f"[TorsionAR][Epoch {ep+1}] Warning: E2E range set but no E2E token in globals; ignoring.",
                                        flush=True,
                                    )
                        
                        if has_rg_range:
                            rg_min, rg_max = config.global_rg_range
                            if global_ranges is not None:
                                j = int(idx_rg) if idx_rg is not None else -1
                                if 0 <= j < global_ranges.shape[1]:
                                    if int(j) not in idx_explicit:
                                        global_ranges[:, j, 0] = float(rg_min)
                                        global_ranges[:, j, 1] = float(rg_max)
                                        label = global_token_names[j] if j < len(global_token_names) else "rg"
                                        range_msgs.append(f"{label}=[{rg_min:.3f}, {rg_max:.3f}]")
                                else:
                                    print(
                                        f"[TorsionAR][Epoch {ep+1}] Warning: Rg range set but no Rg token in globals; ignoring.",
                                        flush=True,
                                    )
                        
                        if range_msgs:
                            print(
                                f"[TorsionAR][Epoch {ep+1}] Using global ranges: {', '.join(range_msgs)}",
                                flush=True,
                            )
                    except Exception as exc_ranges:
                        print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to parse global ranges ({exc_ranges})")
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
                print(
                    f"[TorsionAR][Epoch {ep+1}] Proposed φ (rad): mean={phi_prop.mean().item():+.3f}, "
                    f"std={phi_prop.std().item():.3f}, min={phi_prop.min().item():+.3f}, max={phi_prop.max().item():+.3f}"
                )
                print(
                    f"[TorsionAR][Epoch {ep+1}] Proposed ψ (rad): mean={psi_prop.mean().item():+.3f}, "
                    f"std={psi_prop.std().item():.3f}, min={psi_prop.min().item():+.3f}, max={psi_prop.max().item():+.3f}"
                )
        except Exception as exc_prop:
            print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute proposed torsion stats ({exc_prop})")

        if start_globals is not None:
            try:
                g = start_globals.detach()
                ncols = int(g.shape[1]) if g.ndim == 2 else 0
                for j in range(ncols):
                    name = global_token_names[j] if j < len(global_token_names) else f"global_{j}"
                    key = str(name).lower()
                    unit = "nm" if (key.endswith("_nm") or key in ("rg", "rg_nm", "radius_of_gyration")) else ("rad" if key.endswith("_rad") else "")
                    v = g[:, j]
                    if unit:
                        print(
                            f"[TorsionAR][Epoch {ep+1}] Proposed {name} ({unit}): mean={v.mean().item():.3f}, "
                            f"std={v.std().item():.3f}, min={v.min().item():.3f}, max={v.max().item():.3f}"
                        )
                    else:
                        print(
                            f"[TorsionAR][Epoch {ep+1}] Proposed {name}: mean={v.mean().item():.3f}, "
                            f"std={v.std().item():.3f}, min={v.min().item():.3f}, max={v.max().item():.3f}"
                        )
            except Exception as exc_g:
                print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute global-token stats ({exc_g})")

        want_retrieval = bool(getattr(config, "use_replay_resets", True))
        have_history = bool(retrieval_mgr.cv) and bool(retrieval_mgr.coords)
        use_retrieval = bool(want_retrieval and have_history)
        replay_coords = None
        starts: torch.Tensor
        start_cluster_entropy: Optional[float] = None
        start_cluster_bonus: Optional[np.ndarray] = None

        if want_retrieval and not have_history:
            print(
                f"[Retrieval][Epoch {ep+1}] Replay-starts requested but no history cache is available yet "
                f"(retrieval_cache_frames={len(retrieval_mgr.cv)}). Falling back to PDB/generator starts.",
                flush=True,
            )

        if use_retrieval:
            try:
                labels = retrieval_mgr.labels
                n_clusters = int(max(1, n_discrete_states))
                if labels is None or labels.shape[0] != len(retrieval_mgr.cv):
                    _refresh_retrieval_cache_labels(tag=f"epoch {ep+1} pre-retrieval")
                    labels = retrieval_mgr.labels
                if labels is None or labels.shape[0] != len(retrieval_mgr.cv):
                    labels = np.zeros((len(retrieval_mgr.cv),), dtype=np.int64)
                else:
                    try:
                        missing_idx = np.nonzero(labels < 0)[0]
                    except Exception:
                        missing_idx = np.zeros((0,), dtype=np.int64)
                    if missing_idx.size:
                        _refresh_retrieval_cache_labels(tag=f"epoch {ep+1} fill-missing", only_indices=missing_idx)
                        labels = retrieval_mgr.labels
                        if labels is None or labels.shape[0] != len(retrieval_mgr.cv):
                            labels = np.zeros((len(retrieval_mgr.cv),), dtype=np.int64)

                m_candidates = int(getattr(config, "retrieval_candidates", 2048) or 2048)
                if m_candidates <= 0 or int(m_candidates) >= int(len(retrieval_mgr.cv)):
                    cand_idx = np.arange(int(len(retrieval_mgr.cv)), dtype=np.int64)
                else:
                    cand_idx = balanced_candidate_indices(
                        labels=labels,
                        n_clusters=n_clusters,
                        m=m_candidates,
                        rng=retrieval_mgr.rng,
                    )
                if cand_idx.size == 0:
                    raise RuntimeError("No retrieval candidates available.")

                cand_cv = np.asarray([retrieval_mgr.cv[int(i)] for i in cand_idx], dtype=np.float32)
                cand_coords = None
                try:
                    cand_coords = np.asarray([retrieval_mgr.coords[int(i)] for i in cand_idx], dtype=np.float32)
                except Exception:
                    cand_coords = None

                cand_tors = build_residue_torsions_from_cv_and_coords(
                    cand_cv,
                    cand_coords,
                    cv_residue_slots=np.asarray(env_ref.cv_residue_slots, dtype=np.int64),
                    cv_is_phi=np.asarray(env_ref.cv_is_phi, dtype=bool),
                    n_residues=int(env_ref.n_residues),
                    n_angles=int(n_angles),
                    chi_torsion_atoms=(chi_torsion_atoms if (int(n_angles) > 3 and chi_torsion_atoms.size) else None),
                    chi_residue_idx=(chi_residue_idx if (int(n_angles) > 3 and chi_residue_idx.size) else None),
                    chi_order_idx=(chi_order_idx if (int(n_angles) > 3 and chi_order_idx.size) else None),
                )
                cand_emb = embed_sincos_from_torsions_np(
                    cand_tors,
                    angle_exists_np,
                    angle_indices=torsion_embed_indices,
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
                    and retrieval_mgr.globals
                    and len(retrieval_mgr.globals) == len(retrieval_mgr.cv)
                    and int(getattr(agent_old, "n_global_tokens", 0) or 0) > 0
                ):
                    cand_glob = np.asarray([retrieval_mgr.globals[int(i)] for i in cand_idx], dtype=np.float32)
                    cand_y = np.log1p(np.clip(cand_glob, 0.0, None)).astype(np.float32, copy=False)
                    mu = cand_y.mean(axis=0).astype(np.float32, copy=False)
                    std = cand_y.std(axis=0).astype(np.float32, copy=False)
                    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
                    cand_glob_norm = ((cand_y - mu[None, :]) / std[None, :]) * float(global_w)
                    use_global_retrieval = True
                    global_norm_mu = mu
                    global_norm_std = std

                # Support thresholds for the "out-of-support" resample guard.
                support_thr_tors = compute_support_threshold(cand_emb, rng=retrieval_mgr.rng, sample_size=256, quantile=0.99)
                support_thr_glob = float("inf")
                if use_global_retrieval and cand_glob_norm is not None:
                    support_thr_glob = compute_support_threshold(
                        cand_glob_norm.astype(np.float32, copy=False),
                        rng=retrieval_mgr.rng,
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
                    coords_nm = np.asarray(retrieval_mgr.coords[hist_i], dtype=np.float32)
                    if not use_full_replay_coords:
                        return coords_nm
                    dcd_path = None
                    try:
                        if retrieval_mgr.dcd_paths and hist_i < len(retrieval_mgr.dcd_paths):
                            dcd_path = retrieval_mgr.dcd_paths[hist_i]
                    except Exception:
                        dcd_path = None
                    dcd_frame = None
                    try:
                        if retrieval_mgr.dcd_frame_idx and hist_i < len(retrieval_mgr.dcd_frame_idx):
                            dcd_frame = int(retrieval_mgr.dcd_frame_idx[hist_i])
                        elif retrieval_mgr.frame_idx and hist_i < len(retrieval_mgr.frame_idx):
                            dcd_frame = int(retrieval_mgr.frame_idx[hist_i])
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
                        q_tors_np = q_tors.detach().cpu().numpy().astype(np.float32, copy=False)
                        q_tors_emb = embed_sincos_from_torsions_np(
                            q_tors_np,
                            angle_exists_np,
                            angle_indices=torsion_embed_indices,
                        )  # [1, D]

                        # Stage 1: global-token sort to build a small pool.
                        # Stage 2: torsion distance within that pool.
                        if use_global_retrieval and cand_glob_norm is not None and start_globals is not None and global_norm_mu is not None and global_norm_std is not None:
                            try:
                                q_glob = start_globals[w : w + 1].detach().cpu().numpy().astype(np.float32, copy=False)
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
                            pick_local = sample_from_topk(topk_idx_local, topk_d2, rng=retrieval_mgr.rng, temperature=tau)
                            pos = int(stage1_pos[int(pick_local)])
                            hist_i = int(cand_idx[pos])
                            chosen_hist_idx[w] = hist_i
                            chosen_cv[w] = np.asarray(retrieval_mgr.cv[hist_i], dtype=np.float32)
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
                        pick = sample_from_topk(topk_idx, topk_d2, rng=retrieval_mgr.rng, temperature=tau)
                        hist_i = int(cand_idx[int(pick)])
                        chosen_hist_idx[w] = hist_i
                        chosen_cv[w] = np.asarray(retrieval_mgr.cv[hist_i], dtype=np.float32)
                        chosen_coords.append(_make_replay_payload(hist_i))
                        accepted = True
                        break

                    if not accepted:
                        if use_global_retrieval and cand_glob_norm is not None:
                            try:
                                q_glob = start_globals[w : w + 1].detach().cpu().numpy().astype(np.float32, copy=False)
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
                        chosen_cv[w] = np.asarray(retrieval_mgr.cv[hist_i], dtype=np.float32)
                        chosen_coords.append(_make_replay_payload(hist_i))

                # Use AR-proposed torsion targets for the CV conditioning; replay coordinates
                # provide the retrieved frame, and (optionally) targeted MD will steer toward
                # these CVs/globals.
                phi_psi = start_torsions[..., :2]
                phi_psi_np = phi_psi.detach().cpu().numpy()
                try:
                    cv_batch = env_ref.residue_batch_to_cv(phi_psi_np)
                except Exception as exc_cv:
                    print(f"[TorsionAR] Warning: residue→CV mapping failed ({exc_cv}); using replay CVs.")
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
                            print(f"[Retrieval][Epoch {ep+1}] Start clusters: H={start_cluster_entropy:.3f} nats, counts={counts.tolist()}")
                except Exception as exc_sc:
                    print(f"[Retrieval][Epoch {ep+1}] Warning: failed to compute start-cluster stats ({exc_sc})")
            except Exception as exc_retrieval:
                print(f"[Retrieval] Warning: retrieval starts failed ({exc_retrieval}); falling back to generator starts.")
                use_retrieval = False

        if not use_retrieval:
            phi_psi = start_torsions[..., :2]
            phi_psi_np = phi_psi.detach().cpu().numpy()
            try:
                cv_batch = env_ref.residue_batch_to_cv(phi_psi_np)
            except Exception as exc_cv:
                print(f"[TorsionAR] Warning: residue→CV mapping failed ({exc_cv}); falling back to random CV starts.")
                cv_batch = np.random.uniform(low=-math.pi, high=math.pi, size=(config.grp_size, env_ref.n_cvs)).astype(np.float32)
            starts = torch.as_tensor(cv_batch, device=torch_device, dtype=torch.float32)
            replay_coords = None

        # Diagnostics: measure how far the final sampled torsions deviate from a deterministic
        # "policy suggested" torsion sequence (greedy mixture-component μ under teacher forcing).
        mode_phi_mae: Optional[np.ndarray] = None
        mode_psi_mae: Optional[np.ndarray] = None
        mode_phi_psi_mae: Optional[np.ndarray] = None
        try:
            with torch.no_grad():
                try:
                    policy_dev = next(agent_old.parameters()).device
                except Exception:
                    policy_dev = torch_device
                tors = start_torsions.to(policy_dev)
                globals_for_mode = start_globals.to(policy_dev) if start_globals is not None else None
                residue_types_dev = residue_types_batch.to(policy_dev)
                angle_mask_dev = angle_mask_batch.to(policy_dev)
                mode = agent_old.mode_torsions(
                    tors,
                    residue_types=residue_types_dev,
                    angle_mask=angle_mask_dev,
                    global_values=globals_for_mode,
                )
                d = tors - mode
                d = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
                abs_d = torch.abs(d)
                m_phi = angle_mask_dev[..., 0].to(abs_d.dtype)
                m_psi = angle_mask_dev[..., 1].to(abs_d.dtype)
                denom_phi = m_phi.sum(dim=1).clamp_min(1.0)
                denom_psi = m_psi.sum(dim=1).clamp_min(1.0)
                phi_mae = (abs_d[..., 0] * m_phi).sum(dim=1) / denom_phi
                psi_mae = (abs_d[..., 1] * m_psi).sum(dim=1) / denom_psi
                denom_both = (m_phi + m_psi).sum(dim=1).clamp_min(1.0)
                both_mae = ((abs_d[..., 0] * m_phi) + (abs_d[..., 1] * m_psi)).sum(dim=1) / denom_both
                mode_phi_mae = phi_mae.detach().cpu().numpy().astype(np.float32)
                mode_psi_mae = psi_mae.detach().cpu().numpy().astype(np.float32)
                mode_phi_psi_mae = both_mae.detach().cpu().numpy().astype(np.float32)
        except Exception as exc_mode:
            print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute mode-deviation diagnostics ({exc_mode})")
            mode_phi_mae = None
            mode_psi_mae = None
            mode_phi_psi_mae = None

        rollout_token_entropy: Optional[np.ndarray] = None
        try:
            with torch.no_grad():
                try:
                    policy_dev = next(agent_old.parameters()).device
                except Exception:
                    policy_dev = torch_device
                ent_seq = agent_old.mean_token_entropy(
                    start_torsions.to(policy_dev),
                    residue_types=residue_types_batch.to(policy_dev),
                    angle_mask=angle_mask_batch.to(policy_dev),
                    include_omega=False,
                    global_values=start_globals.to(policy_dev) if start_globals is not None else None,
                )
                rollout_token_entropy = ent_seq.detach().cpu().numpy().astype(np.float32)
        except Exception as exc_ent:
            print(f"[TorsionAR][Epoch {ep+1}] Warning: failed to compute token entropy ({exc_ent})")
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
            replay_torsion_targets_residue=(start_torsions if replay_coords is not None else None),
            replay_global_targets=(start_globals if (replay_coords is not None and start_globals is not None) else None),
            replay_global_token_names=(
                list(global_token_names) if (replay_coords is not None and start_globals is not None) else None
            ),
        )

        launch_rewards = reward_tensors.get("launch", torch.zeros(config.grp_size, device=torch_device))
        if launch_rewards.numel() == 0:
            launch_rewards = torch.zeros(config.grp_size, device=torch_device)
        metrics["avg_launch"] = float(launch_rewards.detach().mean().item())

        all_epoch_data.append(current_epoch_data)
        all_epoch_vamp_exclude.append(False)
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
            print(f"[Retrieval Cache] Warning: append failed: {exc_append}")

        # D. Compute LOO VAMP rewards from the softmax VAMP lobe outputs (probabilities).
        loo_vamp_rewards = np.zeros(config.grp_size, dtype=np.float32)
        loo_vamp_contributions = None
        vamp_bin_bonus = None
        msm_ig_rewards = None
        msm_ig_unique_states = None
        committor_progress_raw = None
        committor_tube_raw = None
        committor_edge_raw = None
        committor_conn_raw = None
        committor_start_basin_bonus_raw = None
        committor_bin_entropy_raw = None
        if len(current_epoch_data) > 0 and "trajectory_lengths" in metrics:
            if model_ready and geom2vec_cache is not None and geom2vec_projector is not None:
                try:
                    lag_embed = int(geom2vec_lagtime_embed)
                    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)

                    vamp_bin_ent_coef = float(getattr(config, "vamp_bin_entropy_coef", -1.0) or -1.0)
                    if vamp_bin_ent_coef < 0:
                        vamp_bin_ent_coef = float(getattr(config, "ent_coef", 0.0) or 0.0)
                    use_vamp_bin_bonus = vamp_bin_ent_coef > 0

                    msm_ig_coef_cfg = float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0)
                    use_msm_ig = msm_ig_coef_cfg > 0.0
                    msm_ig_dim = int(getattr(config, "msm_ig_dim", 2) or 2)
                    msm_ig_state_mode = str(getattr(config, "msm_ig_state_mode", "kmeans") or "kmeans").strip().lower()
                    if msm_ig_state_mode not in {"kmeans", "vamp"}:
                        msm_ig_state_mode = "kmeans"

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

                        key = geom2vec_mgr.cache_key_for_history_traj(len(all_epoch_data) - 1, walker_idx, dcd_path)
                        cached_key = geom2vec_mgr.ensure_cached(
                            key=key,
                            dcd_path=dcd_path,
                            # Prefer embedding from DCD when available.
                            # Raw coords from the env may include solvent / differ in atom ordering
                            # and can break the geom2vec projector.
                            coords_nm=(None if dcd_path else coords_nm),
                            overwrite=False,
                            meta={
                                "epoch_idx": int(len(all_epoch_data) - 1),
                                "traj_idx": int(walker_idx),
                                "raw_frames": int(L_i),
                            },
                        )
                        cached_keys_epoch.append(cached_key)

                    # Seed global (history) sufficient statistics, excluding the just-appended epoch.
                    _ensure_loo_vamp_history_stats(
                        upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                        n_states=n_states,
                        lag_embed=lag_embed,
                        tag=f"epoch {ep+1} pre-reward",
                    )
                    feature_mode = "vamp"
                    sig_epoch = (int(n_states), int(lag_embed), False, str(feature_mode))
                    if loo_vamp_stats is None or loo_vamp_stats_signature != sig_epoch:
                        raise RuntimeError("Global LOO VAMP stats not initialized.")

                    if use_vamp_bin_bonus:
                        _ensure_vamp_bin_history_counts(
                            upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                            n_states=n_states,
                            tag=f"epoch {ep+1} pre-reward",
                        )

                    if use_msm_ig:
                        _ensure_msm_ig_history_counts(
                            upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                            n_states=n_states,
                            lag_embed=lag_embed,
                            tag=f"epoch {ep+1} pre-reward",
                        )

                    pair_counts = np.zeros(config.grp_size, dtype=np.int64)
                    traj_pairs: List[Optional[_VampPairRecord]] = [
                        None for _ in range(len(cached_keys_epoch))
                    ]
                    traj_in_a: List[Optional[np.ndarray]] = [None for _ in range(len(cached_keys_epoch))]
                    traj_in_b: List[Optional[np.ndarray]] = [None for _ in range(len(cached_keys_epoch))]
                    traj_rmsd_a: List[Optional[np.ndarray]] = [None for _ in range(len(cached_keys_epoch))]
                    traj_rmsd_b: List[Optional[np.ndarray]] = [None for _ in range(len(cached_keys_epoch))]

                    vamp_bin_ready = (
                        bool(use_vamp_bin_bonus)
                        and vamp_bin_mgr.counts is not None
                        and vamp_bin_mgr.mu is not None
                        and vamp_bin_mgr.std is not None
                        and geom2vec_projector_linear is not None
                        and getattr(geom2vec_projector_linear, "trained", None) is not None
                    )
                    if bool(use_msm_ig) and str(msm_ig_state_mode) == "vamp":
                        msm_ig_ready = (
                            msm_ig_mgr.counts is not None
                            and geom2vec_projector is not None
                            and (
                                getattr(geom2vec_projector, "trained", None) is not None
                                or bool(getattr(geom2vec_projector, "trained_ensemble", None))
                            )
                        )
                    else:
                        msm_ig_ready = (
                            bool(use_msm_ig)
                            and msm_ig_mgr.counts is not None
                            and msm_ig_mgr.mu is not None
                            and msm_ig_mgr.std is not None
                            and msm_ig_mgr.kmeans is not None
                            and geom2vec_projector_linear is not None
                            and getattr(geom2vec_projector_linear, "trained", None) is not None
                        )

                    dim_needed = 0
                    if vamp_bin_ready:
                        dim_needed = max(dim_needed, 2)
                    if use_msm_ig and str(msm_ig_state_mode) == "kmeans":
                        dim_needed = max(dim_needed, max(0, int(msm_ig_dim)))
                    committor_weight = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
                    committor_tube_weight = float(getattr(config, "committor_tube_diversity_reward_weight", 0.0) or 0.0)
                    committor_edge_weight = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0)
                    committor_conn_weight = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
                    committor_needed = (
                        (committor_weight > 0.0)
                        or (committor_tube_weight > 0.0)
                        or (committor_edge_weight > 0.0)
                        or (committor_conn_weight > 0.0)
                    )
                    committor_state_mode = str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
                    if committor_state_mode not in {"dga", "vamp"}:
                        committor_state_mode = "dga"
                    committor_ready = committor_needed and (
                        (str(committor_state_mode) == "vamp" and committor_state.vamp_model is not None)
                        or (str(committor_state_mode) != "vamp" and committor_state.dga_model is not None)
                    )
                    if committor_ready and str(committor_state_mode) != "vamp" and committor_state.dga_model is not None:
                        dim_needed = max(dim_needed, int(committor_state.dga_model.dim))
                    if (
                        boundary_mode in {"vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent"}
                        and committor_state.boundary_vac_kmeans is not None
                        and committor_state.boundary_vac_mu is not None
                        and committor_state.boundary_vac_std is not None
                        and committor_state.boundary_vac_a_clusters is not None
                        and committor_state.boundary_vac_b_clusters is not None
                        and bool(np.any(committor_state.boundary_vac_a_clusters))
                        and bool(np.any(committor_state.boundary_vac_b_clusters))
                    ):
                        dim_needed = max(dim_needed, int(committor_state.boundary_vac_mu.shape[0]))
                    dim_needed = int(min(int(dim_needed), int(n_states)))

                    traj_z_lin_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))]
                        if (
                            dim_needed > 0
                            and geom2vec_projector_linear is not None
                            and getattr(geom2vec_projector_linear, "trained", None) is not None
                        )
                        else None
                    )

                    traj_vamp_labels_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))]
                        if (
                            (use_msm_ig and str(msm_ig_state_mode) == "vamp")
                            or str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower() == "vamp"
                        )
                        else None
                    )

                    traj_vamp_probs_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))]
                        if (
                            committor_state.dga_model is not None
                            and getattr(committor_state.dga_model, "input_mode", "linear") == "vamp_probs"
                        )
                        else None
                    )

                    traj_dga_features_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))]
                        if (
                            committor_state.dga_model is not None
                            and bool(getattr(committor_state.dga_model, "include_global_tokens", False))
                        )
                        else None
                    )

                    for t_i, cached_key in enumerate(cached_keys_epoch):
                        if cached_key is None:
                            continue
                        try:
                            rec = geom2vec_cache.get(cached_key)
                            if int(getattr(rec, "frames", 0)) <= 0:
                                continue
                        except Exception:
                            continue
                        try:
                            graph, ca = geom2vec_cache.load(cached_key)
                        except Exception:
                            continue
                        z_full = None
                        if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "trained", None) is not None:
                            try:
                                if dim_needed > 0:
                                    z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                                    z_full = np.asarray(z_raw, dtype=np.float32)
                                    if traj_z_lin_best is not None and z_full.ndim == 2 and z_full.shape[0] > 0:
                                        traj_z_lin_best[t_i] = np.asarray(z_full[:, :dim_needed], dtype=np.float32)
                            except Exception:
                                if traj_z_lin_best is not None:
                                    traj_z_lin_best[t_i] = None
                                z_full = None
                        T = 0
                        try:
                            probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)
                            probs = np.asarray(probs_raw, dtype=np.float32)[:, geom2vec_state_permutation_map]
                            if probs.shape[1] != int(n_states):
                                if probs.shape[1] > int(n_states):
                                    probs = probs[:, : int(n_states)]
                                else:
                                    probs = np.pad(
                                        probs,
                                        ((0, 0), (0, int(n_states) - int(probs.shape[1]))),
                                        mode="constant",
                                    )

                            T = int(probs.shape[0])

                            # Cache DGA committor features if requested (optionally augmented with globals).
                            if traj_dga_features_best is not None and committor_state.dga_model is not None:
                                try:
                                    input_mode = getattr(committor_state.dga_model, "input_mode", "linear")
                                    base = probs if input_mode == "vamp_probs" else z_full
                                    if base is not None:
                                        base_arr = np.asarray(base, dtype=np.float32)
                                        if base_arr.ndim == 2 and int(base_arr.shape[0]) == int(T):
                                            traj_dga_features_best[t_i] = _augment_committor_dga_features(
                                                base_arr, ca, use_globals=True
                                            )
                                except Exception:
                                    traj_dga_features_best[t_i] = None

                            # Boundary masks / RMSDs do not depend on lag embedding; compute for any T>=1.
                            cutoff_a_nm = float(getattr(config, "path_boundary_cutoff_a_nm", 0.20) or 0.20)
                            cutoff_b_nm = float(getattr(config, "path_boundary_cutoff_b_nm", cutoff_a_nm) or cutoff_a_nm)
                            in_a, in_b, rmsd_a, rmsd_b = _boundary_masks_from_ca(
                                ca,
                                cutoff_a_nm=float(cutoff_a_nm),
                                cutoff_b_nm=float(cutoff_b_nm),
                                graph_features=graph,
                                z_lin=z_full,
                                cache_key=str(cached_key),
                            )
                            traj_in_a[t_i] = in_a
                            traj_in_b[t_i] = in_b
                            traj_rmsd_a[t_i] = rmsd_a
                            traj_rmsd_b[t_i] = rmsd_b

                            if T > lag_embed:
                                pair = _VampPairRecord(
                                    x=probs[:-lag_embed],
                                    y=probs[lag_embed:],
                                    ind_stop=None,
                                )
                                traj_pairs[t_i] = pair
                                pair_counts[t_i] = int(pair.x.shape[0])
                                if traj_vamp_labels_best is not None:
                                    traj_vamp_labels_best[t_i] = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                                if traj_vamp_probs_best is not None:
                                    traj_vamp_probs_best[t_i] = np.asarray(probs, dtype=np.float32)
                            else:
                                traj_pairs[t_i] = None
                                pair_counts[t_i] = 0
                        except Exception:
                            traj_pairs[t_i] = None
                            pair_counts[t_i] = 0

                    # If the VAMP/projection machinery isn't producing valid pairs/embeddings,
                    # it's easy to misread the resulting VAMP-2 and bin-bonus zeros as "disabled".
                    # Emit a compact, actionable diagnostic once per epoch.
                    try:
                        n_cached = int(sum(1 for k in cached_keys_epoch if k is not None))
                        n_pairs_traj = int(sum(1 for p in traj_pairs if p is not None))
                        total_pairs = int(sum(int(pc) for pc in pair_counts.tolist()))
                        n_z = int(
                            sum(
                                1
                                for z in (traj_z_lin_best or [])
                                if z is not None and getattr(z, "size", 0) > 0 and int(getattr(z, "shape", (0, 0))[1]) >= 2
                            )
                        )
                        if n_pairs_traj == 0 or total_pairs == 0:
                            max_T = 0
                            try:
                                # Use cached embedding lengths even if no valid pairs were formed.
                                max_T = int(
                                    max(
                                        int(getattr(geom2vec_cache.get(k), "frames", 0))
                                        for k in cached_keys_epoch
                                        if k is not None
                                    )
                                )
                            except Exception:
                                max_T = 0
                            print(
                                f"[LOO VAMP] Inactive this epoch: valid_trajs={n_pairs_traj}/{int(config.grp_size)} "
                                f"total_pairs={total_pairs} lag_embed={int(lag_embed)} cached={n_cached}/{int(config.grp_size)} "
                                f"max_T≈{max_T}. Check geom2vec_lagtime vs embed_stride (need T>lag).",
                                flush=True,
                            )
                        if use_vamp_bin_bonus and (traj_z_lin_best is None or n_z == 0):
                            trained_lin = bool(
                                geom2vec_projector_linear is not None
                                and getattr(geom2vec_projector_linear, "trained", None) is not None
                            )
                            print(
                                f"[VAMP Bin] Inactive this epoch: have_z2_trajs={n_z}/{int(config.grp_size)} "
                                f"trained_linear={trained_lin} cached={n_cached}/{int(config.grp_size)}. "
                                "If trained_linear is False, linear VAMP hasn't been fit yet (often due to no valid pairs).",
                                flush=True,
                            )
                    except Exception:
                        pass

                    # VAMP-bin count exploration bonus (first two VAMP coords).
                    if (
                        vamp_bin_ready
                        and traj_z_lin_best is not None
                        and vamp_bin_mgr.counts is not None
                        and vamp_bin_mgr.mu is not None
                        and vamp_bin_mgr.std is not None
                    ):
                        bins_v = int(getattr(config, "vamp_bin_bins", 50) or 50)
                        clip_v = float(getattr(config, "vamp_bin_clip", 3.0) or 3.0)
                        nbins = int(bins_v) * int(bins_v)
                        bonuses = np.zeros(config.grp_size, dtype=np.float32)
                        frame_counts = np.zeros(config.grp_size, dtype=np.int64)

                        for i, z_lin in enumerate(traj_z_lin_best):
                            if z_lin is None or z_lin.size == 0 or z_lin.shape[1] < 2:
                                continue
                            z2 = z_lin[:, :2]
                            z2s = (z2 - vamp_bin_mgr.mu[None, :]) / vamp_bin_mgr.std[None, :]
                            ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                            c = vamp_bin_mgr.counts[ix, iy].astype(np.float64, copy=False)
                            bonuses[i] = float((1.0 / np.sqrt(c + 1.0)).mean())
                            frame_counts[i] = int(ix.shape[0])

                        vamp_bin_bonus = bonuses

                        # Update histogram with current epoch trajectories (for future epochs).
                        for i, z_lin in enumerate(traj_z_lin_best):
                            cached_key = cached_keys_epoch[i] if i < len(cached_keys_epoch) else None
                            if cached_key is None or z_lin is None or z_lin.size == 0 or z_lin.shape[1] < 2:
                                continue
                            key_s = str(cached_key)
                            if key_s in vamp_bin_mgr.counted_keys:
                                continue
                            z2 = z_lin[:, :2]
                            z2s = (z2 - vamp_bin_mgr.mu[None, :]) / vamp_bin_mgr.std[None, :]
                            ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                            np.add.at(vamp_bin_mgr.counts, (ix, iy), 1)
                            vamp_bin_mgr.total += int(ix.shape[0])
                            vamp_bin_mgr.counted_keys.add(key_s)

                        try:
                            visited = int(np.count_nonzero(vamp_bin_mgr.counts))
                            print(
                                f"[VAMP Bins - Epoch {ep+1}] bonus_mean={float(bonuses.mean()):.4f} "
                                f"visited={visited}/{nbins} total_frames={int(vamp_bin_mgr.total)}"
                            )
                            print("  Per-trajectory bin bonuses (mean 1/sqrt(count+1)):")
                            for i in range(config.grp_size):
                                if frame_counts[i] <= 0:
                                    continue
                                print(
                                    f"    Traj {i:2d}: bonus={float(bonuses[i]):+.4f} frames={int(frame_counts[i])}"
                                )
                        except Exception:
                            pass

                    # MSM information-gain reward (MSM over discrete states).
                    if (
                        bool(use_msm_ig)
                        and bool(msm_ig_ready)
                        and msm_ig_mgr.counts is not None
                        and (
                            (str(msm_ig_state_mode) == "vamp" and traj_vamp_labels_best is not None)
                            or (
                                str(msm_ig_state_mode) != "vamp"
                                and traj_z_lin_best is not None
                                and msm_ig_mgr.mu is not None
                                and msm_ig_mgr.std is not None
                                and msm_ig_mgr.kmeans is not None
                            )
                        )
                    ):
                        prior_alpha = float(getattr(config, "msm_ig_prior_alpha", 1.0) or 1.0)
                        if prior_alpha <= 0.0:
                            print("[MSM IG] Warning: msm_ig_prior_alpha must be > 0; skipping.", flush=True)
                        else:
                            K_msm = int(msm_ig_mgr.counts.shape[0])
                            lag_msm = int(lag_embed)

                            rewards_msm = np.zeros(config.grp_size, dtype=np.float32)
                            uniq_states = np.zeros(config.grp_size, dtype=np.int64)
                            n_trajs = len(traj_pairs)
                            delta_counts_epoch: List[Optional[np.ndarray]] = [None for _ in range(n_trajs)]
                            delta_pairs_epoch = np.zeros(n_trajs, dtype=np.int64)

                            if str(msm_ig_state_mode) == "vamp":
                                for i, labels_raw in enumerate(traj_vamp_labels_best or []):
                                    if (
                                        labels_raw is None
                                        or getattr(labels_raw, "size", 0) == 0
                                        or int(labels_raw.shape[0]) <= lag_msm
                                    ):
                                        continue
                                    labels = np.asarray(labels_raw, dtype=np.int64).reshape(-1)
                                    labels = np.clip(labels, 0, int(K_msm) - 1)
                                    try:
                                        uniq_states[i] = int(np.unique(labels).size)
                                    except Exception:
                                        uniq_states[i] = 0
                                    counts_i, n_pairs_i = _msm_transition_counts(labels, n_states=K_msm, lag=lag_msm)
                                    if int(n_pairs_i) <= 0:
                                        continue
                                    delta_counts_epoch[i] = counts_i
                                    delta_pairs_epoch[i] = int(n_pairs_i)
                                    try:
                                        rewards_msm[i] = float(
                                            _msm_ig_reward_from_counts(
                                                base_counts=msm_ig_mgr.counts,
                                                delta_counts=counts_i,
                                                prior_alpha=float(prior_alpha),
                                            )
                                        )
                                    except Exception:
                                        rewards_msm[i] = 0.0
                            else:
                                dim_msm = int(getattr(config, "msm_ig_dim", 2) or 2)
                                dim_msm = int(max(1, min(int(dim_msm), int(n_states))))

                                mu_msm = np.asarray(msm_ig_mgr.mu, dtype=np.float32).reshape(-1)[:dim_msm]
                                std_msm = np.asarray(msm_ig_mgr.std, dtype=np.float32).reshape(-1)[:dim_msm]
                                std_msm = np.where(std_msm > 1e-8, std_msm, 1.0).astype(np.float32, copy=False)

                                for i, z_lin in enumerate(traj_z_lin_best or []):
                                    if z_lin is None or z_lin.size == 0 or int(z_lin.shape[0]) <= lag_msm:
                                        continue
                                    if int(z_lin.shape[1]) < int(dim_msm):
                                        continue
                                    z_d = np.asarray(z_lin[:, :dim_msm], dtype=np.float32)
                                    z_norm = (z_d - mu_msm[None, :]) / std_msm[None, :]
                                    try:
                                        labels = msm_ig_mgr.kmeans.predict(np.asarray(z_norm, dtype=np.float64))  # type: ignore[union-attr]
                                    except Exception:
                                        continue
                                    try:
                                        uniq_states[i] = int(np.unique(labels).size)
                                    except Exception:
                                        uniq_states[i] = 0
                                    counts_i, n_pairs_i = _msm_transition_counts(labels, n_states=K_msm, lag=lag_msm)
                                    if int(n_pairs_i) <= 0:
                                        continue
                                    delta_counts_epoch[i] = counts_i
                                    delta_pairs_epoch[i] = int(n_pairs_i)
                                    try:
                                        rewards_msm[i] = float(
                                            _msm_ig_reward_from_counts(
                                                base_counts=msm_ig_mgr.counts,
                                                delta_counts=counts_i,
                                                prior_alpha=float(prior_alpha),
                                            )
                                        )
                                    except Exception:
                                        rewards_msm[i] = 0.0

                            msm_ig_rewards = rewards_msm
                            msm_ig_unique_states = uniq_states

                            # Update global MSM counts once per key (for future epochs).
                            added_epoch = 0
                            for i, delta_i in enumerate(delta_counts_epoch):
                                cached_key = cached_keys_epoch[i] if i < len(cached_keys_epoch) else None
                                if cached_key is None or delta_i is None:
                                    continue
                                key_s = str(cached_key)
                                if key_s in msm_ig_mgr.counted_keys:
                                    continue
                                msm_ig_mgr.counts += np.asarray(delta_i, dtype=np.float64)
                                msm_ig_mgr.total_pairs += int(delta_pairs_epoch[i])
                                msm_ig_mgr.counted_keys.add(key_s)
                                added_epoch += 1

                            try:
                                metrics["msm_ig_mgr.total_pairs"] = int(msm_ig_mgr.total_pairs)
                                metrics["msm_ig_nonzero_edges"] = int(np.count_nonzero(msm_ig_mgr.counts))
                                metrics["msm_ig_k"] = int(K_msm)
                            except Exception:
                                pass

                            try:
                                print(
                                    f"[MSM IG - Epoch {ep+1}] ig_mean={float(np.mean(rewards_msm)):.4f} "
                                    f"ig_std={float(np.std(rewards_msm)):.4f} "
                                    f"uniq_states_mean={float(np.mean(uniq_states)):.2f} "
                                    f"mode={str(msm_ig_state_mode)} K={int(K_msm)} lag_embed={int(lag_msm)} added={int(added_epoch)}",
                                    flush=True,
                                )
                            except Exception:
                                pass

                    # Committor-based rewards (DGA on VAC coords, or discrete committor on VAMP clusters).
                    if committor_needed:
                        committor_state_mode_local = (
                            str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
                        )
                        if committor_state_mode_local not in {"dga", "vamp"}:
                            committor_state_mode_local = "dga"
                        tube_state_mode = str(getattr(config, "committor_tube_state_mode", "model") or "model").strip().lower()
                        if tube_state_mode in {"dga", "vamp"}:
                            tube_state_mode = "model"
                        if tube_state_mode not in {"model", "q_only", "rmsd_2d"}:
                            tube_state_mode = "model"
                        rmsd_bins = int(getattr(config, "committor_rmsd_state_bins", 10) or 10)
                        rmsd_bins = int(max(1, rmsd_bins))
                        try:
                            rmsd_max_nm = float(getattr(config, "committor_rmsd_state_max_nm", 2.0) or 2.0)
                        except Exception:
                            rmsd_max_nm = 2.0
                        if not np.isfinite(rmsd_max_nm) or rmsd_max_nm <= 0.0:
                            rmsd_max_nm = 2.0
                        force_rmsd_proxy = bool(getattr(config, "committor_use_rmsd_proxy", False))
                        model_ready = bool(
                            (
                                str(committor_state_mode_local) == "vamp"
                                and committor_state.vamp_model is not None
                                and traj_vamp_labels_best is not None
                            )
                            or (
                                str(committor_state_mode_local) != "vamp"
                                and committor_state.dga_model is not None
                                and traj_z_lin_best is not None
                            )
                        )

                        if (not model_ready) and (not force_rmsd_proxy):
                            if str(committor_state_mode_local) == "vamp":
                                if committor_state.vamp_model is None:
                                    print(
                                        f"[Committor - Epoch {ep+1}] Warning: committor-based reward(s) enabled "
                                        "but VAMP committor model is not ready; using Q=0.",
                                        flush=True,
                                    )
                                elif traj_vamp_labels_best is None:
                                    print(
                                        f"[Committor - Epoch {ep+1}] Warning: VAMP labels unavailable; using Q=0.",
                                        flush=True,
                                    )
                            else:
                                if committor_state.dga_model is None:
                                    print(
                                        f"[Committor - Epoch {ep+1}] Warning: committor-based reward(s) enabled "
                                        "but DGA model is not ready; using Q=0.",
                                        flush=True,
                                    )
                                elif traj_z_lin_best is None:
                                    print(
                                        f"[Committor - Epoch {ep+1}] Warning: linear projector outputs unavailable; using Q=0.",
                                        flush=True,
                                    )

                        conn_epoch_diag: Optional[List[Dict[str, Any]]] = None
                        if model_ready or force_rmsd_proxy:
                            Q_vals = np.zeros(config.grp_size, dtype=np.float32)
                            tube_vals = (
                                np.zeros(config.grp_size, dtype=np.float32) if committor_tube_weight > 0.0 else None
                            )
                            edge_vals = (
                                np.zeros(config.grp_size, dtype=np.float32) if committor_edge_weight > 0.0 else None
                            )
                            comm_bin_ent_coef_cfg = float(getattr(config, "committor_bin_entropy_reward_weight", 0.0) or 0.0)
                            bin_ent_vals = (
                                np.zeros(config.grp_size, dtype=np.float32) if comm_bin_ent_coef_cfg > 0.0 else None
                            )
                            comm_start_basin_coef_cfg = float(
                                getattr(config, "committor_start_basin_entropy_reward_weight", 0.0) or 0.0
                            )
                            start_basin_bonus_vals = (
                                np.zeros(config.grp_size, dtype=np.float32) if comm_start_basin_coef_cfg > 0.0 else None
                            )
                            committor_start_basin_entropy_p_a: Optional[float] = None
                            committor_start_basin_entropy_p_b: Optional[float] = None
                            committor_start_basin_epoch_a = 0
                            committor_start_basin_epoch_b = 0
                            committor_conn_weight = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
                            conn_vals = (
                                np.zeros(config.grp_size, dtype=np.float32) if committor_conn_weight > 0.0 else None
                            )
                            if conn_vals is not None and committor_conn_weight > 0.0:
                                conn_epoch_diag = []
                            need_qlab = (
                                (committor_tube_weight > 0.0)
                                or (committor_edge_weight > 0.0)
                                or (committor_conn_weight > 0.0)
                            )
                            qlab_epoch: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = (
                                [None for _ in range(config.grp_size)] if need_qlab else None
                            )
                            tube_bins = int(getattr(config, "committor_tube_diversity_q_bins", 20) or 20)
                            tube_alpha = float(getattr(config, "committor_tube_diversity_alpha", 1.0) or 1.0)
                            tube_log_eps = float(getattr(config, "committor_tube_diversity_log_eps", 1e-12) or 1e-12)
                            if tube_alpha <= 0.0:
                                tube_alpha = 1e-6
                            tube_log_eps = max(0.0, float(tube_log_eps))
                            if str(tube_state_mode) == "q_only":
                                K_tube = 1
                            elif str(tube_state_mode) == "rmsd_2d":
                                K_tube = int(rmsd_bins) * int(rmsd_bins)
                            else:
                                if model_ready and str(committor_state_mode_local) == "vamp":
                                    K_tube = int(
                                        np.asarray(getattr(committor_state.vamp_model, "q_state", np.zeros((0,)))).reshape(-1).shape[0]
                                    )
                                elif model_ready:
                                    K_tube = int(getattr(committor_state.dga_model, "coeffs", np.zeros((0,))).shape[0])
                                else:
                                    # Tube/edge states require the committor model in this mode; keep K=1 to avoid NaNs.
                                    K_tube = 1
                            if (committor_tube_weight > 0.0) or (committor_edge_weight > 0.0) or (committor_conn_weight > 0.0):
                                _ensure_committor_tube_history_counts(
                                    upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                                    n_bins=int(tube_bins),
                                    tag=f"epoch {ep+1} pre-reward",
                                )
                            denom_tube = float(committor_state.tube_total_weight) + float(tube_alpha) * float(
                                int(tube_bins) * max(1, K_tube)
                            )
                            edge_delta = int(getattr(config, "committor_tube_edge_novelty_lag", 1) or 1)
                            edge_delta = int(max(1, edge_delta))
                            edge_undirected = bool(getattr(config, "committor_tube_edge_novelty_undirected", True))
                            edge_bin_change_only = bool(getattr(config, "committor_tube_edge_novelty_bin_change_only", False))
                            n_cells = int(tube_bins) * int(max(1, K_tube))
                            conn_bonus_cfg = float(getattr(config, "committor_connectivity_connect_bonus", 1.0) or 1.0)
                            conn_sp_weight = float(getattr(config, "committor_connectivity_shortest_path_weight", 1.0) or 1.0)
                            conn_flow_weight = float(getattr(config, "committor_connectivity_flow_weight", 0.0) or 0.0)
                            conn_flow_cap = int(getattr(config, "committor_connectivity_flow_cap", 10) or 10)
                            conn_flow_cap = int(max(1, conn_flow_cap))
                            conn_merge_weight = float(getattr(config, "committor_connectivity_component_merge_weight", 0.0) or 0.0)
                            conn_gap_weight = float(getattr(config, "committor_connectivity_gap_weight", 0.0) or 0.0)
                            conn_auto_tune = bool(getattr(config, "committor_connectivity_auto_tune", False))

                            # Auto-tune: interpret the configured weights as maxima and scale them based on current
                            # assembled-graph progress (gap/dist). This balances "exploration" (merging/expanding
                            # components) vs "bridging" (closing the A/B gap and then shortening paths).
                            if (
                                conn_vals is not None
                                and committor_conn_weight > 0.0
                                and conn_auto_tune
                                and committor_state.conn_adj is not None
                                and committor_state.conn_a_cells
                                and committor_state.conn_b_cells
                                and int(tube_bins) > 1
                            ):
                                try:
                                    dist_ab0 = _conn_shortest_path_len(
                                        adj=committor_state.conn_adj,
                                        sources=list(committor_state.conn_a_cells),
                                        targets=set(committor_state.conn_b_cells),
                                        overlay=None,
                                    )
                                    connected0 = bool(np.isfinite(dist_ab0))
                                    reach_a0 = _conn_reachable_nodes(
                                        adj=committor_state.conn_adj,
                                        sources=list(committor_state.conn_a_cells),
                                        overlay=None,
                                    )
                                    reach_b0 = _conn_reachable_nodes(
                                        adj=committor_state.conn_adj,
                                        sources=list(committor_state.conn_b_cells),
                                        overlay=None,
                                    )
                                    gap0, amax0, bmin0 = _conn_bin_gap(
                                        reachable_a=reach_a0,
                                        reachable_b=reach_b0,
                                        K=int(max(1, K_tube)),
                                        M=int(tube_bins),
                                    )
                                    denom = float(max(1, int(tube_bins) - 1))
                                    gap_frac = float(np.clip(float(gap0) / denom, 0.0, 1.0))
                                    # When disconnected:
                                    # - Large gap => emphasize exploration (merge) to discover new cells/edges.
                                    # - Small gap => emphasize bridging (gap) to close the remaining gap.
                                    if connected0:
                                        sp_scale = 1.0
                                        gap_scale = 0.0
                                        merge_scale = 0.1
                                    else:
                                        sp_scale = 0.0
                                        merge_scale = gap_frac
                                        gap_scale = float(np.clip(0.25 + 0.75 * (1.0 - gap_frac), 0.0, 1.0))

                                    conn_sp_weight = float(conn_sp_weight) * float(sp_scale)
                                    conn_gap_weight = float(conn_gap_weight) * float(gap_scale)
                                    conn_merge_weight = float(conn_merge_weight) * float(merge_scale)

                                    print(
                                        f"[Committor Conn] Auto-tune (epoch {ep+1}): "
                                        f"connected={connected0} A→B_dist={'inf' if not np.isfinite(dist_ab0) else int(dist_ab0)} "
                                        f"gap={int(gap0)} (A_max_bin={int(amax0)} B_min_bin={int(bmin0)}), "
                                        f"weights: gap={conn_gap_weight:.3f} sp={conn_sp_weight:.3f} "
                                        f"flow={float(conn_flow_weight):.3f}(cap={int(conn_flow_cap)}) "
                                        f"merge={conn_merge_weight:.3f}",
                                        flush=True,
                                    )
                                except Exception:
                                    pass
                            conn_comp_ids: Optional[np.ndarray] = None
                            if (
                                conn_vals is not None
                                and committor_conn_weight > 0.0
                                and committor_state.conn_adj is not None
                                and conn_merge_weight != 0.0
                            ):
                                try:
                                    conn_comp_ids = _conn_component_ids(committor_state.conn_adj)
                                except Exception:
                                    conn_comp_ids = None
                            comm_diag = {
                                "trajs": 0,
                                "has_a": 0,
                                "has_b": 0,
                                "mean_in_a": [],
                                "mean_in_b": [],
                                "rmsd_a0": [],
                                "rmsd_b0": [],
                                "rmsd_a_min": [],
                                "rmsd_b_min": [],
                                "q0": [],
                                "qmax": [],
                                "progress": [],
                            }
                            per_traj_lines: List[str] = []
                            try:
                                cut_a_nm = float(getattr(config, "path_boundary_cutoff_a_nm", 0.20) or 0.20)
                                cut_b_nm = float(getattr(config, "path_boundary_cutoff_b_nm", cut_a_nm) or cut_a_nm)
                            except Exception:
                                cut_a_nm = float("nan")
                                cut_b_nm = float("nan")
                            if model_ready:
                                if str(committor_state_mode_local) == "vamp":
                                    use_rmsd_fallback_global = bool(getattr(committor_state.vamp_model, "degenerate", False))
                                    delay_n_comm = 0
                                    delay_delta_comm = 1
                                else:
                                    use_rmsd_fallback_global = bool(getattr(committor_state.dga_model, "degenerate", False))
                                    delay_n_comm = int(getattr(committor_state.dga_model, "delay_n_embed", 0) or 0)
                                    delay_delta_comm = int(getattr(committor_state.dga_model, "delay_delta", 1) or 1)
                            else:
                                use_rmsd_fallback_global = True
                                delay_n_comm = 0
                                delay_delta_comm = 1
                            rmsd_fallback_warned = False
                            if model_ready and str(committor_state_mode_local) == "vamp":
                                for i, labels_in in enumerate(traj_vamp_labels_best or []):
                                    if labels_in is None or getattr(labels_in, "size", 0) == 0:
                                        continue
                                    ia = traj_in_a[i] if i < len(traj_in_a) else None
                                    ib = traj_in_b[i] if i < len(traj_in_b) else None
                                    ra = traj_rmsd_a[i] if i < len(traj_rmsd_a) else None
                                    rb = traj_rmsd_b[i] if i < len(traj_rmsd_b) else None
                                    if ia is None or ib is None:
                                        continue

                                    try:
                                        ia_b = np.asarray(ia, dtype=bool).reshape(-1)
                                        ib_b = np.asarray(ib, dtype=bool).reshape(-1)
                                        q, labels = committor_state.vamp_model.predict_q_and_labels(  # type: ignore[union-attr]
                                            np.asarray(labels_in, dtype=np.int64),
                                            in_a=ia_b,
                                            in_b=ib_b,
                                        )
                                        q_use = np.asarray(q, dtype=np.float64).reshape(-1)
                                        lab_use = np.asarray(labels, dtype=np.int64).reshape(-1)
                                        q_src = "vamp"
                                    except Exception:
                                        continue

                                    try:
                                        per_traj_flat = float(np.std(q_use)) if q_use.size else 0.0
                                        # Only fall back when explicitly requested or when the *model* is known
                                        # to be unreliable (degenerate). A per-trajectory flat q is expected
                                        # when the trajectory stays in one region and should not trigger a
                                        # global fallback.
                                        need_fallback = bool(force_rmsd_proxy) or bool(use_rmsd_fallback_global)
                                        if need_fallback and ra is not None and rb is not None:
                                            q_r = _committor_from_rmsd(ra, rb)
                                            q_r = np.asarray(q_r, dtype=np.float64).reshape(-1)
                                            n_q = int(min(int(q_r.shape[0]), int(lab_use.shape[0])))
                                            if n_q > 0:
                                                q_use = np.asarray(q_r[:n_q], dtype=np.float64)
                                                lab_use = np.asarray(lab_use[:n_q], dtype=np.int64)
                                                ia_l = np.asarray(ia_b, dtype=bool).reshape(-1)[:n_q]
                                                ib_l = np.asarray(ib_b, dtype=bool).reshape(-1)[:n_q]
                                                q_use = np.clip(q_use, 0.0, 1.0)
                                                q_use[ia_l] = 0.0
                                                q_use[ib_l] = 1.0
                                                q_src = "rmsd"
                                                if not rmsd_fallback_warned:
                                                    print(
                                                        (
                                                            f"[Committor - Epoch {ep+1}] Note: forcing RMSD committor proxy for "
                                                            "Q/tube/edge this epoch."
                                                            if force_rmsd_proxy
                                                            else f"[Committor - Epoch {ep+1}] Note: committor is unstable/degenerate; "
                                                            "using RMSD committor fallback for Q/tube/edge this epoch."
                                                        ),
                                                        flush=True,
                                                    )
                                                    rmsd_fallback_warned = True
                                    except Exception:
                                        pass

                                    # Guard: if trajectory hits both A and B but visits very few states (<=4),
                                    # it's likely a tunneling/short-circuit artifact. Zero out the reward.
                                    if np.any(ia_b) and np.any(ib_b):
                                        if np.unique(labels_in).size <= 4:
                                            q_use.fill(0.0)

                                    prog = float(_committor_progress(q_use))
                                    Q_vals[i] = prog

                                    # Committor-bin entropy (coarse bins in committor space).
                                    try:
                                        if bin_ent_vals is not None and int(tube_bins) > 1 and q_use.size:
                                            q_ent = np.asarray(q_use, dtype=np.float64).reshape(-1)
                                            q_ent = np.clip(q_ent, 0.0, 1.0)
                                            w_hat_ent = 4.0 * q_ent * (1.0 - q_ent)
                                            w_hat_ent = np.where(np.isfinite(w_hat_ent), w_hat_ent, 0.0)
                                            w_hat_ent = np.clip(w_hat_ent, 0.0, 1.0)
                                            if float(np.sum(w_hat_ent)) <= 1e-12:
                                                bin_ent_vals[i] = 0.0
                                            else:
                                                m_ent = np.floor(float(tube_bins) * q_ent).astype(np.int64, copy=False)
                                                m_ent = np.clip(m_ent, 0, int(tube_bins) - 1)
                                                hist = np.bincount(m_ent, weights=w_hat_ent, minlength=int(tube_bins)).astype(np.float64)
                                                tot = float(np.sum(hist))
                                                if tot <= 0.0 or (not np.isfinite(tot)):
                                                    bin_ent_vals[i] = 0.0
                                                else:
                                                    p = hist / tot
                                                    p_nz = p[p > 0]
                                                    H = float(-(p_nz * np.log(p_nz)).sum()) if p_nz.size else 0.0
                                                    denom = float(np.log(float(tube_bins)))
                                                    Hn = float(H / denom) if denom > 0.0 else float(H)
                                                    bin_ent_vals[i] = float(Hn)
                                    except Exception:
                                        pass

                                    # Start-basin entropy bonus (A vs B) computed from boundary membership at t=0.
                                    # Uses running counts across epochs so the rarer start basin gets higher reward,
                                    # encouraging bidirectional dataset coverage.
                                    try:
                                        if start_basin_bonus_vals is not None and ia_b.size and ib_b.size:
                                            start_in_a = bool(ia_b[0])
                                            start_in_b = (not start_in_a) and bool(ib_b[0])
                                            # If the boundary masks don't classify the first frame, fall back to q0 side.
                                            if (not start_in_a) and (not start_in_b) and q_use.size:
                                                start_in_a = bool(float(q_use[0]) <= 0.5)
                                                start_in_b = not start_in_a
                                            if start_in_a or start_in_b:
                                                alpha = 1.0
                                                ca = float(committor_state.start_basin_counts_a)
                                                cb = float(committor_state.start_basin_counts_b)
                                                denom = ca + cb + 2.0 * alpha
                                                p_a = (ca + alpha) / denom if denom > 0.0 else 0.5
                                                p_b = (cb + alpha) / denom if denom > 0.0 else 0.5
                                                committor_start_basin_entropy_p_a = float(p_a)
                                                committor_start_basin_entropy_p_b = float(p_b)
                                                if start_in_a:
                                                    start_basin_bonus_vals[i] = float(-math.log(max(p_a, 1e-12)))
                                                    committor_start_basin_epoch_a += 1
                                                else:
                                                    start_basin_bonus_vals[i] = float(-math.log(max(p_b, 1e-12)))
                                                    committor_start_basin_epoch_b += 1
                                    except Exception:
                                        pass
                                    if (
                                        qlab_epoch is not None
                                        or (
                                            tube_vals is not None
                                            and committor_tube_weight > 0.0
                                            and committor_state.tube_counts is not None
                                        )
                                        or (
                                            edge_vals is not None
                                            and committor_edge_weight > 0.0
                                            and committor_state.edge_counts is not None
                                        )
                                        or (
                                            conn_vals is not None
                                            and committor_conn_weight > 0.0
                                            and committor_state.conn_adj is not None
                                        )
                                    ):
                                        q_f = np.asarray(q_use, dtype=np.float64).reshape(-1)
                                        if str(tube_state_mode) == "q_only":
                                            lab_f = np.zeros((int(q_f.shape[0]),), dtype=np.int64)
                                        elif str(tube_state_mode) == "rmsd_2d" and ra is not None and rb is not None:
                                            lab_f = _rmsd_state_labels_2d(
                                                np.asarray(ra, dtype=np.float64).reshape(-1),
                                                np.asarray(rb, dtype=np.float64).reshape(-1),
                                                bins=int(rmsd_bins),
                                                max_nm=float(rmsd_max_nm),
                                            )
                                        else:
                                            lab_f = np.asarray(lab_use, dtype=np.int64).reshape(-1)

                                        n = int(min(int(q_f.shape[0]), int(lab_f.shape[0])))
                                        if n > 0:
                                            q_f = np.clip(np.asarray(q_f[:n], dtype=np.float64), 0.0, 1.0)
                                            lab_f = np.clip(
                                                np.asarray(lab_f[:n], dtype=np.int64), 0, max(0, K_tube - 1)
                                            )
                                            if qlab_epoch is not None:
                                                qlab_epoch[i] = (
                                                    np.asarray(q_f, dtype=np.float32),
                                                    np.asarray(lab_f, dtype=np.int64),
                                                )

                                            # Tube diversity: -log p(m,s) with symmetric tube focus weights.
                                            if tube_vals is not None and committor_tube_weight > 0.0 and committor_state.tube_counts is not None:
                                                w_hat = 4.0 * q_f * (1.0 - q_f)
                                                w_hat = np.where(np.isfinite(w_hat), w_hat, 0.0)
                                                w_hat = np.clip(w_hat, 0.0, 1.0)
                                                m_bins = np.floor(float(tube_bins) * q_f).astype(np.int64, copy=False)
                                                m_bins = np.clip(m_bins, 0, int(tube_bins) - 1)
                                                cell = m_bins * int(max(1, K_tube)) + lab_f
                                                p = (np.asarray(committor_state.tube_counts, dtype=np.float64)[cell] + float(tube_alpha)) / float(
                                                    denom_tube if denom_tube > 0.0 else 1.0
                                                )
                                                p = np.where(np.isfinite(p), p, 0.0)
                                                novelty = -np.log(p + float(tube_log_eps))
                                                tube_vals[i] = float(np.mean(w_hat * novelty)) if w_hat.size else 0.0

                                            # Tube edge novelty: 1/(N_edge+1) on edges between tube cells.
                                            if edge_vals is not None and committor_edge_weight > 0.0 and committor_state.edge_counts is not None:
                                                if int(n_cells) > 1 and n > int(edge_delta):
                                                    q0 = q_f[: n - int(edge_delta)]
                                                    q1 = q_f[int(edge_delta) : n]
                                                    q_bar = 0.5 * (q0 + q1)
                                                    w_edge = 4.0 * q_bar * (1.0 - q_bar)
                                                    w_edge = np.where(np.isfinite(w_edge), w_edge, 0.0)
                                                    w_edge = np.clip(w_edge, 0.0, 1.0)

                                                    m0 = np.floor(float(tube_bins) * q0).astype(np.int64, copy=False)
                                                    m1 = np.floor(float(tube_bins) * q1).astype(np.int64, copy=False)
                                                    m0 = np.clip(m0, 0, int(tube_bins) - 1)
                                                    m1 = np.clip(m1, 0, int(tube_bins) - 1)
                                                    if edge_bin_change_only:
                                                        mask = m0 != m1
                                                        if not bool(np.any(mask)):
                                                            edge_vals[i] = 0.0
                                                            m0 = m0[:0]
                                                            m1 = m1[:0]
                                                            w_edge = w_edge[:0]
                                                            src_lab = lab_f[:0]
                                                            dst_lab = lab_f[:0]
                                                        else:
                                                            m0 = m0[mask]
                                                            m1 = m1[mask]
                                                            w_edge = w_edge[mask]
                                                            src_lab = lab_f[: n - int(edge_delta)][mask]
                                                            dst_lab = lab_f[int(edge_delta) : n][mask]
                                                    else:
                                                        src_lab = lab_f[: n - int(edge_delta)]
                                                        dst_lab = lab_f[int(edge_delta) : n]

                                                    src_cell = m0.astype(np.int64) * int(max(1, K_tube)) + src_lab.astype(np.int64)
                                                    dst_cell = m1.astype(np.int64) * int(max(1, K_tube)) + dst_lab.astype(np.int64)
                                                    if edge_undirected:
                                                        a = np.minimum(src_cell, dst_cell)
                                                        b = np.maximum(src_cell, dst_cell)
                                                        src_cell = a
                                                        dst_cell = b
                                                    edge_idx = src_cell.astype(np.int64) * int(n_cells) + dst_cell.astype(np.int64)

                                                    if isinstance(committor_state.edge_counts, np.ndarray):
                                                        c = np.asarray(committor_state.edge_counts, dtype=np.float64)[edge_idx]
                                                    else:
                                                        d = committor_state.edge_counts
                                                        c = np.fromiter(
                                                            (float(d.get(int(e), 0.0)) for e in edge_idx.tolist()),
                                                            dtype=np.float64,
                                                            count=int(edge_idx.size),
                                                        )
                                                    c = np.where(np.isfinite(c), c, 0.0)
                                                    novelty = 1.0 / (c + 1.0)
                                                    edge_vals[i] = float(np.mean(w_edge * novelty)) if w_edge.size else 0.0

                                            if (
                                                conn_vals is not None
                                                and committor_conn_weight > 0.0
                                                and committor_state.conn_adj is not None
                                            ):
                                                ia_conn = ia_b
                                                ib_conn = ib_b
                                                try:
                                                    if delay_n_comm > 0 and int(q_f.shape[0]) != int(ia_b.shape[0]):
                                                        ia_conn = _lift_center_for_delay(
                                                            ia_b,
                                                            n_embed=int(delay_n_comm),
                                                            lag=int(delay_delta_comm),
                                                            target_len=int(q_f.shape[0]),
                                                        ).astype(bool, copy=False)
                                                        ib_conn = _lift_center_for_delay(
                                                            ib_b,
                                                            n_embed=int(delay_n_comm),
                                                            lag=int(delay_delta_comm),
                                                            target_len=int(q_f.shape[0]),
                                                        ).astype(bool, copy=False)
                                                except Exception:
                                                    ia_conn = ia_b
                                                    ib_conn = ib_b
                                                try:
                                                    score, diag = _committor_connectivity_score(
                                                        q_f,
                                                        lab_f,
                                                        in_a=ia_conn,
                                                        in_b=ib_conn,
                                                        return_diag=True,
                                                        tube_bins=int(tube_bins),
                                                        K_tube=int(max(1, K_tube)),
                                                        edge_delta=int(edge_delta),
                                                        edge_bin_change_only=bool(edge_bin_change_only),
                                                        edge_undirected=bool(edge_undirected),
                                                        connect_bonus=float(conn_bonus_cfg),
                                                        shortest_path_weight=float(conn_sp_weight),
                                                        flow_weight=float(conn_flow_weight),
                                                        flow_cap=int(conn_flow_cap),
                                                        component_merge_weight=float(conn_merge_weight),
                                                        gap_weight=float(conn_gap_weight),
                                                        comp_ids=conn_comp_ids,
                                                    )
                                                    conn_vals[i] = float(score)
                                                    if conn_epoch_diag is not None:
                                                        diag["traj"] = int(i)
                                                        diag["score"] = float(score)
                                                        conn_epoch_diag.append(diag)
                                                except Exception:
                                                    conn_vals[i] = 0.0
                                    # Continue to shared block below.
                            elif model_ready:
                                for i, z_lin in enumerate(traj_z_lin_best):
                                    if z_lin is None or z_lin.size == 0:
                                        continue
                                    
                                    # Select input features
                                    use_globals = bool(getattr(committor_state.dga_model, "include_global_tokens", False))
                                    if (
                                        use_globals
                                        and traj_dga_features_best is not None
                                        and traj_dga_features_best[i] is not None
                                    ):
                                        z_in = traj_dga_features_best[i]
                                    else:
                                        input_mode = getattr(committor_state.dga_model, "input_mode", "linear")
                                        if input_mode == "vamp_probs":
                                            if traj_vamp_probs_best is None or traj_vamp_probs_best[i] is None:
                                                continue
                                            z_in = traj_vamp_probs_best[i]
                                        else:
                                            z_in = z_lin

                                    if z_in is None or z_in.size == 0:
                                        continue

                                    ia = traj_in_a[i] if i < len(traj_in_a) else None
                                    ib = traj_in_b[i] if i < len(traj_in_b) else None
                                    ra = traj_rmsd_a[i] if i < len(traj_rmsd_a) else None
                                    rb = traj_rmsd_b[i] if i < len(traj_rmsd_b) else None
                                    if ia is None or ib is None:
                                        continue
                                    try:
                                        ia_b = np.asarray(ia, dtype=bool).reshape(-1)
                                        ib_b = np.asarray(ib, dtype=bool).reshape(-1)
                                        q, labels = committor_state.dga_model.predict_q_and_labels(
                                            np.asarray(z_in, dtype=np.float64),
                                            in_a=ia_b,
                                            in_b=ib_b,
                                        )
                                        q_use = np.asarray(q, dtype=np.float64).reshape(-1)
                                        lab_use = np.asarray(labels, dtype=np.int64).reshape(-1)
                                        q_src = "dga"
                                    except Exception:
                                        continue

                                    # Guardrail: if the DGA committor is degenerate/unstable (common early
                                    # when VAC space collapses), fall back to an RMSD-based committor for
                                    # Q progress and tube weights while retaining DGA labels for microstate
                                    # diversity bookkeeping.
                                    try:
                                        per_traj_flat = float(np.std(q_use)) if q_use.size else 0.0
                                        need_fallback = bool(force_rmsd_proxy) or bool(use_rmsd_fallback_global)
                                        if need_fallback and ra is not None and rb is not None:
                                            q_r = _committor_from_rmsd(ra, rb)
                                            q_r = np.asarray(q_r, dtype=np.float64).reshape(-1)
                                            ia_l = ia_b
                                            ib_l = ib_b
                                            if delay_n_comm > 0:
                                                q_r = _lift_center_for_delay(
                                                    q_r, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q_use.size)
                                                )
                                                ia_l = _lift_center_for_delay(
                                                    ia_b, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q_use.size)
                                                ).astype(bool, copy=False)
                                                ib_l = _lift_center_for_delay(
                                                    ib_b, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q_use.size)
                                                ).astype(bool, copy=False)
                                            n_q = int(min(int(q_r.shape[0]), int(lab_use.shape[0])))
                                            if n_q > 0:
                                                q_use = np.asarray(q_r[:n_q], dtype=np.float64)
                                                lab_use = np.asarray(lab_use[:n_q], dtype=np.int64)
                                                ia_l = np.asarray(ia_l, dtype=bool).reshape(-1)[:n_q]
                                                ib_l = np.asarray(ib_l, dtype=bool).reshape(-1)[:n_q]
                                                q_use = np.clip(q_use, 0.0, 1.0)
                                                q_use[ia_l] = 0.0
                                                q_use[ib_l] = 1.0
                                                q_src = "rmsd"
                                                if not rmsd_fallback_warned:
                                                    print(
                                                        (
                                                            f"[Committor - Epoch {ep+1}] Note: forcing RMSD committor proxy for "
                                                            "Q/tube/edge this epoch."
                                                            if force_rmsd_proxy
                                                            else f"[Committor - Epoch {ep+1}] Note: DGA committor is unstable/degenerate; "
                                                            "using RMSD committor fallback for Q/tube/edge this epoch."
                                                        ),
                                                        flush=True,
                                                    )
                                                    rmsd_fallback_warned = True
                                    except Exception:
                                        pass

                                    prog = float(_committor_progress(q_use))
                                    Q_vals[i] = prog
                                    if (
                                        qlab_epoch is not None
                                        or (
                                            tube_vals is not None
                                            and committor_tube_weight > 0.0
                                            and committor_state.tube_counts is not None
                                        )
                                        or (
                                            edge_vals is not None
                                            and committor_edge_weight > 0.0
                                            and committor_state.edge_counts is not None
                                        )
                                        or (
                                            conn_vals is not None
                                            and committor_conn_weight > 0.0
                                            and committor_state.conn_adj is not None
                                        )
                                    ):
                                        q_f = np.asarray(q_use, dtype=np.float64).reshape(-1)
                                        if str(tube_state_mode) == "q_only":
                                            lab_f = np.zeros((int(q_f.shape[0]),), dtype=np.int64)
                                        elif str(tube_state_mode) == "rmsd_2d" and ra is not None and rb is not None:
                                            ra_use = np.asarray(ra, dtype=np.float64).reshape(-1)
                                            rb_use = np.asarray(rb, dtype=np.float64).reshape(-1)
                                            if delay_n_comm > 0:
                                                ra_use = _lift_center_for_delay(
                                                    ra_use,
                                                    n_embed=int(delay_n_comm),
                                                    lag=int(delay_delta_comm),
                                                    target_len=int(q_f.shape[0]),
                                                )
                                                rb_use = _lift_center_for_delay(
                                                    rb_use,
                                                    n_embed=int(delay_n_comm),
                                                    lag=int(delay_delta_comm),
                                                    target_len=int(q_f.shape[0]),
                                                )
                                            lab_f = _rmsd_state_labels_2d(
                                                ra_use,
                                                rb_use,
                                                bins=int(rmsd_bins),
                                                max_nm=float(rmsd_max_nm),
                                            )
                                        else:
                                            lab_f = np.asarray(lab_use, dtype=np.int64).reshape(-1)

                                        n = int(min(int(q_f.shape[0]), int(lab_f.shape[0])))
                                        if n > 0:
                                            q_f = np.clip(np.asarray(q_f[:n], dtype=np.float64), 0.0, 1.0)
                                            lab_f = np.clip(
                                                np.asarray(lab_f[:n], dtype=np.int64), 0, max(0, K_tube - 1)
                                            )
                                            if qlab_epoch is not None:
                                                qlab_epoch[i] = (
                                                    np.asarray(q_f, dtype=np.float32),
                                                    np.asarray(lab_f, dtype=np.int64),
                                                )

                                            # Tube diversity: -log p(m,s) with symmetric tube focus weights.
                                            if tube_vals is not None and committor_tube_weight > 0.0 and committor_state.tube_counts is not None:
                                                w_hat = 4.0 * q_f * (1.0 - q_f)
                                                w_hat = np.where(np.isfinite(w_hat), w_hat, 0.0)
                                                w_hat = np.clip(w_hat, 0.0, 1.0)
                                                m_bins = np.floor(float(tube_bins) * q_f).astype(np.int64, copy=False)
                                                m_bins = np.clip(m_bins, 0, int(tube_bins) - 1)
                                                cell = m_bins * int(max(1, K_tube)) + lab_f
                                                p = (np.asarray(committor_state.tube_counts, dtype=np.float64)[cell] + float(tube_alpha)) / float(
                                                    denom_tube if denom_tube > 0.0 else 1.0
                                                )
                                                p = np.where(np.isfinite(p), p, 0.0)
                                                novelty = -np.log(p + float(tube_log_eps))
                                                tube_vals[i] = float(np.mean(w_hat * novelty)) if w_hat.size else 0.0

                                            # Tube edge novelty: 1/(N_edge+1) on edges between tube cells.
                                            if edge_vals is not None and committor_edge_weight > 0.0 and committor_state.edge_counts is not None:
                                                if int(n_cells) > 1 and n > int(edge_delta):
                                                    q0 = q_f[: n - int(edge_delta)]
                                                    q1 = q_f[int(edge_delta) : n]
                                                    q_bar = 0.5 * (q0 + q1)
                                                    w_edge = 4.0 * q_bar * (1.0 - q_bar)
                                                    w_edge = np.where(np.isfinite(w_edge), w_edge, 0.0)
                                                    w_edge = np.clip(w_edge, 0.0, 1.0)

                                                    m0 = np.floor(float(tube_bins) * q0).astype(np.int64, copy=False)
                                                    m1 = np.floor(float(tube_bins) * q1).astype(np.int64, copy=False)
                                                    m0 = np.clip(m0, 0, int(tube_bins) - 1)
                                                    m1 = np.clip(m1, 0, int(tube_bins) - 1)
                                                    if edge_bin_change_only:
                                                        mask = m0 != m1
                                                        if not bool(np.any(mask)):
                                                            edge_vals[i] = 0.0
                                                            m0 = m0[:0]
                                                            m1 = m1[:0]
                                                            w_edge = w_edge[:0]
                                                            src_lab = lab_f[:0]
                                                            dst_lab = lab_f[:0]
                                                        else:
                                                            m0 = m0[mask]
                                                            m1 = m1[mask]
                                                            w_edge = w_edge[mask]
                                                            src_lab = lab_f[: n - int(edge_delta)][mask]
                                                            dst_lab = lab_f[int(edge_delta) : n][mask]
                                                    else:
                                                        src_lab = lab_f[: n - int(edge_delta)]
                                                        dst_lab = lab_f[int(edge_delta) : n]

                                                    src_cell = m0.astype(np.int64) * int(max(1, K_tube)) + src_lab.astype(np.int64)
                                                    dst_cell = m1.astype(np.int64) * int(max(1, K_tube)) + dst_lab.astype(np.int64)
                                                    if edge_undirected:
                                                        a = np.minimum(src_cell, dst_cell)
                                                        b = np.maximum(src_cell, dst_cell)
                                                        src_cell = a
                                                        dst_cell = b
                                                    edge_idx = src_cell.astype(np.int64) * int(n_cells) + dst_cell.astype(np.int64)

                                                    if isinstance(committor_state.edge_counts, np.ndarray):
                                                        c = np.asarray(committor_state.edge_counts, dtype=np.float64)[edge_idx]
                                                    else:
                                                        d = committor_state.edge_counts
                                                        c = np.fromiter(
                                                            (float(d.get(int(e), 0.0)) for e in edge_idx.tolist()),
                                                            dtype=np.float64,
                                                            count=int(edge_idx.size),
                                                        )
                                                    c = np.where(np.isfinite(c), c, 0.0)
                                                    novelty = 1.0 / (c + 1.0)
                                                    edge_vals[i] = float(np.mean(w_edge * novelty)) if w_edge.size else 0.0

                                            if (
                                                conn_vals is not None
                                                and committor_conn_weight > 0.0
                                                and committor_state.conn_adj is not None
                                            ):
                                                ia_conn = ia_b
                                                ib_conn = ib_b
                                                try:
                                                    if delay_n_comm > 0 and int(q_f.shape[0]) != int(ia_b.shape[0]):
                                                        ia_conn = _lift_center_for_delay(
                                                            ia_b,
                                                            n_embed=int(delay_n_comm),
                                                            lag=int(delay_delta_comm),
                                                            target_len=int(q_f.shape[0]),
                                                        ).astype(bool, copy=False)
                                                        ib_conn = _lift_center_for_delay(
                                                            ib_b,
                                                            n_embed=int(delay_n_comm),
                                                            lag=int(delay_delta_comm),
                                                            target_len=int(q_f.shape[0]),
                                                        ).astype(bool, copy=False)
                                                except Exception:
                                                    ia_conn = ia_b
                                                    ib_conn = ib_b
                                                try:
                                                    score, diag = _committor_connectivity_score(
                                                        q_f,
                                                        lab_f,
                                                        in_a=ia_conn,
                                                        in_b=ib_conn,
                                                        return_diag=True,
                                                        tube_bins=int(tube_bins),
                                                        K_tube=int(max(1, K_tube)),
                                                        edge_delta=int(edge_delta),
                                                        edge_bin_change_only=bool(edge_bin_change_only),
                                                        edge_undirected=bool(edge_undirected),
                                                        connect_bonus=float(conn_bonus_cfg),
                                                        shortest_path_weight=float(conn_sp_weight),
                                                        flow_weight=float(conn_flow_weight),
                                                        flow_cap=int(conn_flow_cap),
                                                        component_merge_weight=float(conn_merge_weight),
                                                        gap_weight=float(conn_gap_weight),
                                                        comp_ids=conn_comp_ids,
                                                    )
                                                    conn_vals[i] = float(score)
                                                    if conn_epoch_diag is not None:
                                                        diag["traj"] = int(i)
                                                        diag["score"] = float(score)
                                                        conn_epoch_diag.append(diag)
                                                except Exception:
                                                    conn_vals[i] = 0.0
                            else:
                                # Pure RMSD committor proxy (no committor model needed).
                                model_state_warned = False
                                for i in range(int(config.grp_size)):
                                    ia = traj_in_a[i] if i < len(traj_in_a) else None
                                    ib = traj_in_b[i] if i < len(traj_in_b) else None
                                    ra = traj_rmsd_a[i] if i < len(traj_rmsd_a) else None
                                    rb = traj_rmsd_b[i] if i < len(traj_rmsd_b) else None
                                    if ia is None or ib is None or ra is None or rb is None:
                                        continue
                                    ia_b = np.asarray(ia, dtype=bool).reshape(-1)
                                    ib_b = np.asarray(ib, dtype=bool).reshape(-1)
                                    q_use = _committor_from_rmsd(ra, rb)
                                    q_use = np.asarray(q_use, dtype=np.float64).reshape(-1)
                                    n_q = int(min(int(q_use.shape[0]), int(ia_b.shape[0]), int(ib_b.shape[0])))
                                    if n_q <= 0:
                                        continue
                                    q_use = np.clip(q_use[:n_q], 0.0, 1.0)
                                    ia_l = ia_b[:n_q]
                                    ib_l = ib_b[:n_q]
                                    q_use[ia_l] = 0.0
                                    q_use[ib_l] = 1.0
                                    q_src = "rmsd"

                                    prog = float(_committor_progress(q_use))
                                    Q_vals[i] = prog

                                    if (
                                        qlab_epoch is not None
                                        or (
                                            tube_vals is not None
                                            and committor_tube_weight > 0.0
                                            and committor_state.tube_counts is not None
                                        )
                                        or (
                                            edge_vals is not None
                                            and committor_edge_weight > 0.0
                                            and committor_state.edge_counts is not None
                                        )
                                        or (
                                            conn_vals is not None
                                            and committor_conn_weight > 0.0
                                            and committor_state.conn_adj is not None
                                        )
                                    ):
                                        q_f = np.asarray(q_use, dtype=np.float64).reshape(-1)
                                        if str(tube_state_mode) == "q_only":
                                            lab_f = np.zeros((int(q_f.shape[0]),), dtype=np.int64)
                                        elif str(tube_state_mode) == "rmsd_2d":
                                            lab_f = _rmsd_state_labels_2d(
                                                np.asarray(ra, dtype=np.float64).reshape(-1)[: int(q_f.shape[0])],
                                                np.asarray(rb, dtype=np.float64).reshape(-1)[: int(q_f.shape[0])],
                                                bins=int(rmsd_bins),
                                                max_nm=float(rmsd_max_nm),
                                            )
                                        else:
                                            if not model_state_warned:
                                                print(
                                                    f"[Committor - Epoch {ep+1}] Warning: committor_tube_state_mode=model "
                                                    "requires a committor model; falling back to q_only for tube/edge.",
                                                    flush=True,
                                                )
                                                model_state_warned = True
                                            lab_f = np.zeros((int(q_f.shape[0]),), dtype=np.int64)

                                        n = int(min(int(q_f.shape[0]), int(lab_f.shape[0])))
                                        if n > 0:
                                            q_f = np.clip(np.asarray(q_f[:n], dtype=np.float64), 0.0, 1.0)
                                            lab_f = np.clip(np.asarray(lab_f[:n], dtype=np.int64), 0, max(0, K_tube - 1))
                                            if qlab_epoch is not None:
                                                qlab_epoch[i] = (
                                                    np.asarray(q_f, dtype=np.float32),
                                                    np.asarray(lab_f, dtype=np.int64),
                                                )

                                            if tube_vals is not None and committor_tube_weight > 0.0 and committor_state.tube_counts is not None:
                                                w_hat = 4.0 * q_f * (1.0 - q_f)
                                                w_hat = np.where(np.isfinite(w_hat), w_hat, 0.0)
                                                w_hat = np.clip(w_hat, 0.0, 1.0)
                                                m_bins = np.floor(float(tube_bins) * q_f).astype(np.int64, copy=False)
                                                m_bins = np.clip(m_bins, 0, int(tube_bins) - 1)
                                                cell = m_bins * int(max(1, K_tube)) + lab_f
                                                p = (np.asarray(committor_state.tube_counts, dtype=np.float64)[cell] + float(tube_alpha)) / float(
                                                    denom_tube if denom_tube > 0.0 else 1.0
                                                )
                                                p = np.where(np.isfinite(p), p, 0.0)
                                                novelty = -np.log(p + float(tube_log_eps))
                                                tube_vals[i] = float(np.mean(w_hat * novelty)) if w_hat.size else 0.0

                                            if edge_vals is not None and committor_edge_weight > 0.0 and committor_state.edge_counts is not None:
                                                if int(n_cells) > 1 and n > int(edge_delta):
                                                    q0 = q_f[: n - int(edge_delta)]
                                                    q1 = q_f[int(edge_delta) : n]
                                                    q_bar = 0.5 * (q0 + q1)
                                                    w_edge = 4.0 * q_bar * (1.0 - q_bar)
                                                    w_edge = np.where(np.isfinite(w_edge), w_edge, 0.0)
                                                    w_edge = np.clip(w_edge, 0.0, 1.0)

                                                    m0 = np.floor(float(tube_bins) * q0).astype(np.int64, copy=False)
                                                    m1 = np.floor(float(tube_bins) * q1).astype(np.int64, copy=False)
                                                    m0 = np.clip(m0, 0, int(tube_bins) - 1)
                                                    m1 = np.clip(m1, 0, int(tube_bins) - 1)
                                                    if edge_bin_change_only:
                                                        mask = m0 != m1
                                                        if not bool(np.any(mask)):
                                                            edge_vals[i] = 0.0
                                                            m0 = m0[:0]
                                                            m1 = m1[:0]
                                                            w_edge = w_edge[:0]
                                                            src_lab = lab_f[:0]
                                                            dst_lab = lab_f[:0]
                                                        else:
                                                            m0 = m0[mask]
                                                            m1 = m1[mask]
                                                            w_edge = w_edge[mask]
                                                            src_lab = lab_f[: n - int(edge_delta)][mask]
                                                            dst_lab = lab_f[int(edge_delta) : n][mask]
                                                    else:
                                                        src_lab = lab_f[: n - int(edge_delta)]
                                                        dst_lab = lab_f[int(edge_delta) : n]

                                                    src_cell = m0.astype(np.int64) * int(max(1, K_tube)) + src_lab.astype(np.int64)
                                                    dst_cell = m1.astype(np.int64) * int(max(1, K_tube)) + dst_lab.astype(np.int64)
                                                    if edge_undirected:
                                                        a = np.minimum(src_cell, dst_cell)
                                                        b = np.maximum(src_cell, dst_cell)
                                                        src_cell = a
                                                        dst_cell = b
                                                    edge_idx = src_cell.astype(np.int64) * int(n_cells) + dst_cell.astype(np.int64)

                                                    if isinstance(committor_state.edge_counts, np.ndarray):
                                                        c = np.asarray(committor_state.edge_counts, dtype=np.float64)[edge_idx]
                                                    else:
                                                        d = committor_state.edge_counts
                                                        c = np.fromiter(
                                                            (float(d.get(int(e), 0.0)) for e in edge_idx.tolist()),
                                                            dtype=np.float64,
                                                            count=int(edge_idx.size),
                                                        )
                                                    c = np.where(np.isfinite(c), c, 0.0)
                                                    novelty = 1.0 / (c + 1.0)
                                                    edge_vals[i] = float(np.mean(w_edge * novelty)) if w_edge.size else 0.0

                                            if (
                                                conn_vals is not None
                                                and committor_conn_weight > 0.0
                                                and committor_state.conn_adj is not None
                                            ):
                                                ia_conn = ia_b
                                                ib_conn = ib_b
                                                try:
                                                    if delay_n_comm > 0 and int(q_f.shape[0]) != int(ia_b.shape[0]):
                                                        ia_conn = _lift_center_for_delay(
                                                            ia_b,
                                                            n_embed=int(delay_n_comm),
                                                            lag=int(delay_delta_comm),
                                                            target_len=int(q_f.shape[0]),
                                                        ).astype(bool, copy=False)
                                                        ib_conn = _lift_center_for_delay(
                                                            ib_b,
                                                            n_embed=int(delay_n_comm),
                                                            lag=int(delay_delta_comm),
                                                            target_len=int(q_f.shape[0]),
                                                        ).astype(bool, copy=False)
                                                except Exception:
                                                    ia_conn = ia_b
                                                    ib_conn = ib_b
                                                try:
                                                    score, diag = _committor_connectivity_score(
                                                        q_f,
                                                        lab_f,
                                                        in_a=ia_conn,
                                                        in_b=ib_conn,
                                                        return_diag=True,
                                                        tube_bins=int(tube_bins),
                                                        K_tube=int(max(1, K_tube)),
                                                        edge_delta=int(edge_delta),
                                                        edge_bin_change_only=bool(edge_bin_change_only),
                                                        edge_undirected=bool(edge_undirected),
                                                        connect_bonus=float(conn_bonus_cfg),
                                                        shortest_path_weight=float(conn_sp_weight),
                                                        flow_weight=float(conn_flow_weight),
                                                        flow_cap=int(conn_flow_cap),
                                                        component_merge_weight=float(conn_merge_weight),
                                                        gap_weight=float(conn_gap_weight),
                                                        comp_ids=conn_comp_ids,
                                                    )
                                                    conn_vals[i] = float(score)
                                                    if conn_epoch_diag is not None:
                                                        diag["traj"] = int(i)
                                                        diag["score"] = float(score)
                                                        conn_epoch_diag.append(diag)
                                                except Exception:
                                                    conn_vals[i] = 0.0
                                    try:
                                        comm_diag["trajs"] += 1
                                        comm_diag["has_a"] += int(bool(np.any(ia_b)))
                                        comm_diag["has_b"] += int(bool(np.any(ib_b)))
                                        comm_diag["mean_in_a"].append(float(np.mean(ia_b.astype(np.float32))))
                                        comm_diag["mean_in_b"].append(float(np.mean(ib_b.astype(np.float32))))
                                        if ra is not None:
                                            ra_f = np.asarray(ra, dtype=np.float64).reshape(-1)
                                            if ra_f.size:
                                                comm_diag["rmsd_a0"].append(float(ra_f[0]))
                                                comm_diag["rmsd_a_min"].append(float(np.min(ra_f)))
                                        if rb is not None:
                                            rb_f = np.asarray(rb, dtype=np.float64).reshape(-1)
                                            if rb_f.size:
                                                comm_diag["rmsd_b0"].append(float(rb_f[0]))
                                                comm_diag["rmsd_b_min"].append(float(np.min(rb_f)))
                                        q_f = np.asarray(q_use, dtype=np.float64).reshape(-1)
                                        if q_f.size:
                                            comm_diag["q0"].append(float(q_f[0]))
                                            comm_diag["qmax"].append(float(np.max(q_f)))
                                            comm_diag["progress"].append(float(prog))
                                            try:
                                                frac_a = float(np.mean(ia_b.astype(np.float32))) if ia_b.size else 0.0
                                                frac_b = float(np.mean(ib_b.astype(np.float32))) if ib_b.size else 0.0
                                                ra0 = (
                                                    float(np.asarray(ra, dtype=np.float64).reshape(-1)[0])
                                                    if ra is not None
                                                    else float("nan")
                                                )
                                                rb0 = (
                                                    float(np.asarray(rb, dtype=np.float64).reshape(-1)[0])
                                                    if rb is not None
                                                    else float("nan")
                                                )
                                                ramin = float(np.min(np.asarray(ra, dtype=np.float64))) if ra is not None else float("nan")
                                                rbmin = float(np.min(np.asarray(rb, dtype=np.float64))) if rb is not None else float("nan")
                                                q0 = float(q_f[0])
                                                qmax = float(np.max(q_f))
                                                per_traj_lines.append(
                                                    f"  Traj {i:02d}: T={int(q_f.size):4d} "
                                                    f"inA={frac_a:.3f} inB={frac_b:.3f} "
                                                    f"rmsdA0={ra0:.3f} rmsdB0={rb0:.3f} "
                                                    f"minA={ramin:.3f} minB={rbmin:.3f} "
                                                    f"q0={q0:.3f} qmax={qmax:.3f} prog={float(prog):.4f} src={q_src}"
                                                )
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                            committor_progress_raw = Q_vals
                            if tube_vals is not None and committor_tube_weight > 0.0:
                                committor_tube_raw = tube_vals
                            if edge_vals is not None and committor_edge_weight > 0.0:
                                committor_edge_raw = edge_vals
                            if conn_vals is not None and committor_conn_weight > 0.0:
                                committor_conn_raw = conn_vals
                            if start_basin_bonus_vals is not None and np.any(np.abs(start_basin_bonus_vals) > 1e-8):
                                committor_start_basin_bonus_raw = start_basin_bonus_vals
                                # Update running start counts after assigning bonuses (so all trajs in this epoch
                                # share the same baseline rarity probabilities).
                                committor_state.start_basin_counts_a += int(committor_start_basin_epoch_a)
                                committor_state.start_basin_counts_b += int(committor_start_basin_epoch_b)
                            if bin_ent_vals is not None and np.any(np.abs(bin_ent_vals) > 1e-8):
                                committor_bin_entropy_raw = bin_ent_vals
                            try:
                                metrics["avg_committor_Q_raw"] = float(np.mean(Q_vals))
                                metrics["std_committor_Q_raw"] = float(np.std(Q_vals))
                            except Exception:
                                pass
                            if committor_bin_entropy_raw is not None:
                                try:
                                    metrics["avg_committor_bin_entropy_raw"] = float(
                                        np.mean(np.asarray(committor_bin_entropy_raw, dtype=np.float64))
                                    )
                                    metrics["std_committor_bin_entropy_raw"] = float(
                                        np.std(np.asarray(committor_bin_entropy_raw, dtype=np.float64))
                                    )
                                except Exception:
                                    pass
                            if committor_start_basin_bonus_raw is not None:
                                try:
                                    metrics["avg_committor_start_basin_bonus_raw"] = float(
                                        np.mean(np.asarray(committor_start_basin_bonus_raw, dtype=np.float64))
                                    )
                                    metrics["committor_start_basin_epoch_a"] = int(committor_start_basin_epoch_a)
                                    metrics["committor_start_basin_epoch_b"] = int(committor_start_basin_epoch_b)
                                except Exception:
                                    pass
                            if committor_tube_raw is not None:
                                try:
                                    metrics["avg_committor_tube_raw"] = float(np.mean(np.asarray(committor_tube_raw, dtype=np.float64)))
                                    metrics["std_committor_tube_raw"] = float(np.std(np.asarray(committor_tube_raw, dtype=np.float64)))
                                except Exception:
                                    pass
                            if committor_edge_raw is not None:
                                try:
                                    metrics["avg_committor_edge_raw"] = float(np.mean(np.asarray(committor_edge_raw, dtype=np.float64)))
                                    metrics["std_committor_edge_raw"] = float(np.std(np.asarray(committor_edge_raw, dtype=np.float64)))
                                except Exception:
                                    pass
                            if committor_conn_raw is not None:
                                try:
                                    metrics["avg_committor_conn_raw"] = float(np.mean(np.asarray(committor_conn_raw, dtype=np.float64)))
                                    metrics["std_committor_conn_raw"] = float(np.std(np.asarray(committor_conn_raw, dtype=np.float64)))
                                except Exception:
                                    pass
                            try:
                                n = int(comm_diag["trajs"])
                                if n > 0:
                                    def _m(xs):
                                        return float(np.mean(xs)) if xs else 0.0
                                    def _mn(xs):
                                        return float(np.min(xs)) if xs else 0.0
                                    print(
                                        f"[Committor - Epoch {ep+1}] diag: trajs={n} anyA={int(comm_diag['has_a'])}/{n} anyB={int(comm_diag['has_b'])}/{n} "
                                        f"mean_inA={_m(comm_diag['mean_in_a']):.3f} mean_inB={_m(comm_diag['mean_in_b']):.3f} "
                                        f"rmsdA0={_m(comm_diag['rmsd_a0']):.3f} rmsdB0={_m(comm_diag['rmsd_b0']):.3f} "
                                        f"rmsdA_min={_m(comm_diag['rmsd_a_min']):.3f} rmsdB_min={_m(comm_diag['rmsd_b_min']):.3f} "
                                        f"q0={_m(comm_diag['q0']):.3f} qmax={_m(comm_diag['qmax']):.3f} progress={_m(comm_diag['progress']):.4f}",
                                        flush=True,
                                    )
                                    if int(comm_diag["has_b"]) == 0:
                                        print(
                                            f"[Committor - Epoch {ep+1}] diag hint: no frames classified as boundary-B under current cutoffs; "
                                            "q will not be clamped to 1, so the last tube bins may be under-visited. "
                                            "Consider loosening --path-boundary-cutoff-b-nm or verifying endpoint B alignment.",
                                            flush=True,
                                        )
                            except Exception:
                                pass
                            try:
                                comm_prog_coef_cfg = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
                                comm_conn_coef_cfg = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
                                if comm_prog_coef_cfg != 0.0:
                                    print(
                                        f"[Committor - Epoch {ep+1}] Q_mean={float(np.mean(Q_vals)):.4f} "
                                        f"Q_std={float(np.std(Q_vals)):.4f}",
                                        flush=True,
                                    )
                                if comm_conn_coef_cfg != 0.0 and committor_conn_raw is not None:
                                    print(
                                        f"[Committor Conn - Epoch {ep+1}] C_mean={float(np.mean(committor_conn_raw)):.4f} "
                                        f"C_std={float(np.std(committor_conn_raw)):.4f}",
                                        flush=True,
                                    )
                                    try:
                                        if conn_epoch_diag:
                                            n_traj = int(len(conn_epoch_diag))
                                            new_edges_sum = int(
                                                sum(int(d.get("n_new_edges", 0) or 0) for d in conn_epoch_diag)
                                            )
                                            new_edges_trajs = int(
                                                sum(
                                                    1
                                                    for d in conn_epoch_diag
                                                    if int(d.get("n_new_edges", 0) or 0) > 0
                                                )
                                            )
                                            connect_events = int(
                                                sum(
                                                    1
                                                    for d in conn_epoch_diag
                                                    if float(d.get("connect_bonus", 0.0) or 0.0) > 0.0
                                                )
                                            )
                                            merges_sum = int(
                                                sum(int(d.get("merges", 0) or 0) for d in conn_epoch_diag)
                                            )
                                            flow_improve_sum = float(
                                                sum(float(d.get("flow_improve", 0.0) or 0.0) for d in conn_epoch_diag)
                                            )
                                            gap_improve_sum = float(
                                                sum(float(d.get("gap_improve", 0.0) or 0.0) for d in conn_epoch_diag)
                                            )
                                            path_improve_sum = float(
                                                sum(float(d.get("path_improve", 0.0) or 0.0) for d in conn_epoch_diag)
                                            )
                                            connected_before = int(
                                                sum(1 for d in conn_epoch_diag if bool(d.get("connected_before", False)))
                                            )
                                            connected_after = int(
                                                sum(1 for d in conn_epoch_diag if bool(d.get("connected_after", False)))
                                            )
                                            top = sorted(
                                                conn_epoch_diag,
                                                key=lambda d: float(d.get("score", 0.0) or 0.0),
                                                reverse=True,
                                            )[: min(3, n_traj)]
                                            top_s = ", ".join(
                                                [
                                                    (
                                                        f"t{int(d.get('traj', -1))}:"
                                                        f"s={float(d.get('score', 0.0) or 0.0):.3g}"
                                                        f" e+={int(d.get('n_new_edges', 0) or 0)}"
                                                        f" bonus={float(d.get('connect_bonus', 0.0) or 0.0):.3g}"
                                                        f" Δd={float(d.get('path_improve', 0.0) or 0.0):.3g}"
                                                        f" Δf={float(d.get('flow_improve', 0.0) or 0.0):.3g}"
                                                        f" Δgap={float(d.get('gap_improve', 0.0) or 0.0):.3g}"
                                                    )
                                                    for d in top
                                                ]
                                            )
                                            print(
                                                f"[Committor Conn - Epoch {ep+1}] trajs={n_traj} with_new_edges={new_edges_trajs} "
                                                f"new_edges_sum={new_edges_sum} connect_events={connect_events} "
                                                f"connected_before/after={connected_before}/{connected_after} merges_sum={merges_sum} "
                                                f"path_improve_sum={path_improve_sum:.3g} flow_improve_sum={flow_improve_sum:.3g} "
                                                f"gap_improve_sum={gap_improve_sum:.3g} "
                                                f"top={top_s}",
                                                flush=True,
                                            )
                                    except Exception:
                                        pass
                                    try:
                                        if committor_state.conn_adj is not None and committor_state.conn_a_cells and committor_state.conn_b_cells:
                                            dist_ab = _conn_shortest_path_len(
                                                adj=committor_state.conn_adj,
                                                sources=list(committor_state.conn_a_cells),
                                                targets=set(committor_state.conn_b_cells),
                                                overlay=None,
                                            )
                                            dist_s = "inf" if not np.isfinite(dist_ab) else str(int(dist_ab))
                                            reach_a = _conn_reachable_nodes(
                                                adj=committor_state.conn_adj,
                                                sources=list(committor_state.conn_a_cells),
                                                overlay=None,
                                            )
                                            reach_b = _conn_reachable_nodes(
                                                adj=committor_state.conn_adj,
                                                sources=list(committor_state.conn_b_cells),
                                                overlay=None,
                                            )
                                            gap_i, amax_i, bmin_i = _conn_bin_gap(
                                                reachable_a=reach_a,
                                                reachable_b=reach_b,
                                                K=int(max(1, K_tube)),
                                                M=int(tube_bins),
                                            )
                                            print(
                                                f"[Committor Conn - Epoch {ep+1}] graph: A→B_dist={dist_s} gap={int(gap_i)} "
                                                f"A_max_bin={int(amax_i)} B_min_bin={int(bmin_i)} "
                                                f"cells={int(len(committor_state.conn_adj))} edges={int(len(committor_state.conn_edges))}",
                                                flush=True,
                                            )
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            if committor_tube_raw is not None:
                                try:
                                    print(
                                        f"[Committor Tube - Epoch {ep+1}] P_mean={float(np.mean(committor_tube_raw)):.4f} "
                                        f"P_std={float(np.std(committor_tube_raw)):.4f} M={int(tube_bins)} K={int(K_tube)}",
                                        flush=True,
                                    )
                                except Exception:
                                    pass
                            if committor_edge_raw is not None:
                                try:
                                    print(
                                        f"[Committor Edge - Epoch {ep+1}] E_mean={float(np.mean(committor_edge_raw)):.4f} "
                                        f"E_std={float(np.std(committor_edge_raw)):.4f} delta={int(edge_delta)} "
                                        f"undirected={bool(edge_undirected)} bin_change_only={bool(edge_bin_change_only)} "
                                        f"cells={int(n_cells)}",
                                        flush=True,
                                    )
                                except Exception:
                                    pass
                            try:
                                comm_prog_coef_cfg = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
                                if comm_prog_coef_cfg != 0.0 and per_traj_lines:
                                    print(
                                        f"[Committor - Epoch {ep+1}] Per-trajectory stats "
                                        f"(cutA={cut_a_nm:.3f}nm cutB={cut_b_nm:.3f}nm):",
                                        flush=True,
                                    )
                                    for line in per_traj_lines:
                                        print(line, flush=True)
                            except Exception:
                                pass
                            # Visualization: committor on (z1,z2) of the linear VAMP space.
                            try:
                                if config.snapshot_dir and traj_z_lin_best is not None:
                                    out_path = Path(config.snapshot_dir) / f"committor_vac_ep{ep+1:03d}.png"
                                    max_pts = int(os.environ.get("GESPO_COMMITTOR_PLOT_MAX_POINTS", "20000"))
                                    xs = []
                                    ys = []
                                    cs = []
                                    a_mask = []
                                    b_mask = []
                                    used_fallback_any = False
                                    force_rmsd_proxy = bool(getattr(config, "committor_use_rmsd_proxy", False))
                                    comm_plot_mode = str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
                                    if comm_plot_mode not in {"dga", "vamp"}:
                                        comm_plot_mode = "dga"
                                    if comm_plot_mode == "vamp" and (committor_state.vamp_model is None or traj_vamp_labels_best is None):
                                        raise RuntimeError("committor_state_mode=vamp but vamp model/labels not available")
                                    if comm_plot_mode == "dga" and committor_state.dga_model is None:
                                        raise RuntimeError("committor_state_mode=dga but DGA model not available")
                                    for i, z_lin in enumerate(traj_z_lin_best):
                                        if z_lin is None or z_lin.size == 0:
                                            continue
                                        z_arr = np.asarray(z_lin, dtype=np.float32)
                                        if z_arr.ndim != 2 or z_arr.shape[0] <= 0 or z_arr.shape[1] < 2:
                                            continue
                                        ia = traj_in_a[i] if i < len(traj_in_a) else None
                                        ib = traj_in_b[i] if i < len(traj_in_b) else None
                                        if ia is None or ib is None:
                                            continue
                                        ia_b = np.asarray(ia, dtype=bool).reshape(-1)
                                        ib_b = np.asarray(ib, dtype=bool).reshape(-1)
                                        delay_n_comm = 0
                                        delay_delta_comm = 1
                                        if comm_plot_mode == "vamp":
                                            labels_in = traj_vamp_labels_best[i] if i < len(traj_vamp_labels_best) else None
                                            if labels_in is None or getattr(labels_in, "size", 0) == 0:
                                                continue
                                            q, _labels_out = committor_state.vamp_model.predict_q_and_labels(  # type: ignore[union-attr]
                                                np.asarray(labels_in, dtype=np.int64),
                                                in_a=ia_b,
                                                in_b=ib_b,
                                            )
                                            q = np.asarray(q, dtype=np.float64).reshape(-1)
                                            q_flat = float(np.std(q)) if q.size else 0.0
                                            need_fallback = bool(force_rmsd_proxy) or bool(getattr(committor_state.vamp_model, "degenerate", False))
                                        else:
                                            q = committor_state.dga_model.predict_q(  # type: ignore[union-attr]
                                                np.asarray(z_arr, dtype=np.float64),
                                                in_a=ia_b,
                                                in_b=ib_b,
                                            )
                                            q = np.asarray(q, dtype=np.float64).reshape(-1)
                                            need_fallback = bool(force_rmsd_proxy) or bool(getattr(committor_state.dga_model, "degenerate", False))
                                            delay_n_comm = int(getattr(committor_state.dga_model, "delay_n_embed", 0) or 0)
                                            delay_delta_comm = int(getattr(committor_state.dga_model, "delay_delta", 1) or 1)
                                        # Lift boundary masks to the embedded axis for plotting/override.
                                        ia_plot = ia_b
                                        ib_plot = ib_b
                                        if delay_n_comm > 0 and int(q.shape[0]) > 0:
                                            ia_plot = _lift_center_for_delay(
                                                ia_b, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q.shape[0])
                                            ).astype(bool, copy=False)
                                            ib_plot = _lift_center_for_delay(
                                                ib_b, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q.shape[0])
                                            ).astype(bool, copy=False)
                                        if need_fallback:
                                            ra = traj_rmsd_a[i] if i < len(traj_rmsd_a) else None
                                            rb = traj_rmsd_b[i] if i < len(traj_rmsd_b) else None
                                            if ra is not None and rb is not None:
                                                q_r = _committor_from_rmsd(ra, rb)
                                                q_r = np.asarray(q_r, dtype=np.float64).reshape(-1)
                                                if delay_n_comm > 0:
                                                    q_r = _lift_center_for_delay(
                                                        q_r, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q.shape[0])
                                                    )
                                                    ia_plot = _lift_center_for_delay(
                                                        ia_b, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q.shape[0])
                                                    ).astype(bool, copy=False)
                                                    ib_plot = _lift_center_for_delay(
                                                        ib_b, n_embed=delay_n_comm, lag=delay_delta_comm, target_len=int(q.shape[0])
                                                    ).astype(bool, copy=False)
                                                n_q = int(min(int(q_r.shape[0]), int(q.shape[0])))
                                                if n_q > 0:
                                                    q = np.asarray(q_r[:n_q], dtype=np.float64)
                                                    ia_plot = np.asarray(ia_plot, dtype=bool).reshape(-1)[:n_q]
                                                    ib_plot = np.asarray(ib_plot, dtype=bool).reshape(-1)[:n_q]
                                                    q = np.clip(q, 0.0, 1.0)
                                                    q[ia_plot] = 0.0
                                                    q[ib_plot] = 1.0
                                                    used_fallback_any = True
                                        else:
                                            # Ensure boundaries are clamped in the plotted q as well.
                                            try:
                                                q = np.clip(q, 0.0, 1.0)
                                                if int(ia_plot.shape[0]) == int(q.shape[0]):
                                                    q[ia_plot] = 0.0
                                                if int(ib_plot.shape[0]) == int(q.shape[0]):
                                                    q[ib_plot] = 1.0
                                            except Exception:
                                                pass
                                        q = np.asarray(q, dtype=np.float32).reshape(-1)
                                        # Align z to the same embedded axis as q when delay embedding is used.
                                        z_plot = z_arr
                                        if delay_n_comm > 0:
                                            total = int(delay_delta_comm) * int(delay_n_comm)
                                            pad_left = int(total // 2)
                                            pad_right = int(total - pad_left)
                                            if int(z_plot.shape[0]) > int(total):
                                                z_plot = z_plot[pad_left : int(z_plot.shape[0]) - pad_right]
                                            else:
                                                z_plot = np.zeros((0, int(z_plot.shape[1])), dtype=z_plot.dtype)

                                        T = int(min(int(z_plot.shape[0]), int(q.shape[0])))
                                        if T <= 0:
                                            continue
                                        xs.append(z_plot[:T, 0])
                                        ys.append(z_plot[:T, 1])
                                        cs.append(q[:T])
                                        a_mask.append(np.asarray(ia_plot, dtype=bool).reshape(-1)[:T])
                                        b_mask.append(np.asarray(ib_plot, dtype=bool).reshape(-1)[:T])

                                    if xs and ys and cs:
                                        x = np.concatenate(xs, axis=0)
                                        y = np.concatenate(ys, axis=0)
                                        c = np.concatenate(cs, axis=0)
                                        ma = np.concatenate(a_mask, axis=0) if a_mask else None
                                        mb = np.concatenate(b_mask, axis=0) if b_mask else None
                                        n_tot = int(x.shape[0])
                                        if n_tot > max_pts:
                                            rng = np.random.default_rng(0)
                                            idx = rng.choice(n_tot, size=max_pts, replace=False)
                                            x = x[idx]
                                            y = y[idx]
                                            c = c[idx]
                                            if ma is not None:
                                                ma = ma[idx]
                                            if mb is not None:
                                                mb = mb[idx]

                                        import matplotlib.pyplot as plt  # type: ignore

                                        fig = plt.figure(figsize=(6.5, 5.5))
                                        ax = fig.add_subplot(1, 1, 1)
                                        sc = ax.scatter(x, y, c=c, s=4, cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.7, linewidths=0)
                                        if ma is not None and bool(np.any(ma)):
                                            ax.scatter(x[ma], y[ma], s=12, c="tab:blue", alpha=0.8, marker="x", linewidths=0.8, label="in A")
                                        if mb is not None and bool(np.any(mb)):
                                            ax.scatter(x[mb], y[mb], s=12, c="tab:red", alpha=0.8, marker="x", linewidths=0.8, label="in B")
                                        title = f"Committor q on linear VAMP space (epoch {ep+1})"
                                        if used_fallback_any:
                                            title += " [proxy fallback]"
                                        ax.set_title(title)
                                        ax.set_xlabel("z1 (linear)")
                                        ax.set_ylabel("z2 (linear)")
                                        cb = fig.colorbar(sc, ax=ax)
                                        cb_label = f"q (committor; mode={comm_plot_mode})"
                                        if used_fallback_any:
                                            cb_label += " [proxy fallback]"
                                        cb.set_label(cb_label)
                                        if (ma is not None and bool(np.any(ma))) or (mb is not None and bool(np.any(mb))):
                                            ax.legend(loc="best", frameon=False)
                                        fig.tight_layout()
                                        fig.savefig(out_path, dpi=200)
                                        plt.close(fig)
                                        print(f"[Committor] Saved VAC committor plot: {out_path}", flush=True)
                            except Exception as exc_plot:
                                print(f"[Committor] Warning: failed to plot committor VAC scatter ({exc_plot})", flush=True)

                            # Visualization: ADK feature pairs (domain angles + key distances).
                            # Best-effort and auto-skips when the endpoint topology doesn't match ADK residue numbering.
                            try:
                                if not config.snapshot_dir:
                                    raise RuntimeError("no snapshot_dir")
                                # Opt-out knob: set GESPO_PLOT_ADK_PAIRS=0.
                                if str(os.environ.get("GESPO_PLOT_ADK_PAIRS", "1")).strip().lower() in {"0", "false", "no", "n"}:
                                    raise RuntimeError("disabled")
                                if endpoint_ca_ref_a is None or endpoint_ca_ref_b is None:
                                    raise RuntimeError("missing endpoint CA refs")
                                if geom2vec_cache is None:
                                    raise RuntimeError("missing geom2vec cache")
                                if cached_keys_epoch is None:
                                    raise RuntimeError("missing cached keys")

                                pdb_a_path = str(getattr(config, "pdb_a", "") or getattr(config, "pdb", "") or "").strip()
                                if not pdb_a_path:
                                    raise RuntimeError("missing pdb_a path")
                                n_tokens = int(np.asarray(endpoint_ca_ref_a).shape[0])

                                def _build_adk_groups_from_endpoint_pdb(
                                    pdb_path: str,
                                    *,
                                    n_tokens: int,
                                    token_mask: Optional[np.ndarray],
                                ) -> Dict[str, Any]:
                                    tr = md.load(str(pdb_path))
                                    residues = list(tr.topology.residues)
                                    if not residues:
                                        raise ValueError("No residues in endpoint PDB.")

                                    resseq: List[int] = []
                                    for r in residues:
                                        try:
                                            rs = getattr(r, "resSeq", None)
                                            rs_i = int(rs) if rs is not None else int(r.index)
                                        except Exception:
                                            rs_i = int(r.index)
                                        resseq.append(int(rs_i))

                                    n_use = int(min(int(n_tokens), len(resseq)))
                                    resseq = resseq[:n_use]

                                    mask = None
                                    if token_mask is not None:
                                        try:
                                            m = np.asarray(token_mask, dtype=bool).reshape(-1)
                                            if int(m.size) == int(n_tokens):
                                                mask = m[:n_use]
                                        except Exception:
                                            mask = None

                                    def _range(start: int, end: int) -> np.ndarray:
                                        start_i = int(min(start, end))
                                        end_i = int(max(start, end))
                                        idx = np.asarray(
                                            [i for i, rs in enumerate(resseq) if start_i <= int(rs) <= end_i],
                                            dtype=np.int64,
                                        )
                                        if mask is not None and idx.size:
                                            idx = idx[mask[idx]]
                                        return idx

                                    resseq_to_token: Dict[int, int] = {}
                                    for i, rs in enumerate(resseq):
                                        if int(rs) not in resseq_to_token:
                                            resseq_to_token[int(rs)] = int(i)

                                    def _res(resnum: int) -> int:
                                        target = int(resnum)
                                        if target in resseq_to_token:
                                            i = int(resseq_to_token[target])
                                            if mask is None or (0 <= i < int(mask.size) and bool(mask[i])):
                                                return i
                                        i = int(target)
                                        if 0 <= i < n_use and (mask is None or bool(mask[i])):
                                            return i
                                        i = int(target - 1)
                                        if 0 <= i < n_use and (mask is None or bool(mask[i])):
                                            return i
                                        return -1

                                    return {
                                        "core_lid": _range(115, 125),
                                        "core1": _range(90, 100),
                                        "nmp": _range(35, 55),
                                        "core2": _range(179, 185),
                                        "lid": _range(125, 153),
                                        "res145": _res(145),
                                        "res52": _res(52),
                                        "res169": _res(169),
                                        "res55": _res(55),
                                    }

                                def _centroid(ca_nm: np.ndarray, token_idx: np.ndarray) -> np.ndarray:
                                    ca_nm = np.asarray(ca_nm, dtype=np.float64)
                                    token_idx = np.asarray(token_idx, dtype=np.int64).reshape(-1)
                                    if ca_nm.ndim != 3 or ca_nm.shape[-1] != 3:
                                        raise ValueError(f"ca_nm must be (T, tokens, 3), got {ca_nm.shape}")
                                    if token_idx.size == 0:
                                        return np.zeros((int(ca_nm.shape[0]), 3), dtype=np.float32)
                                    token_idx = token_idx[(token_idx >= 0) & (token_idx < int(ca_nm.shape[1]))]
                                    if token_idx.size == 0:
                                        return np.zeros((int(ca_nm.shape[0]), 3), dtype=np.float32)
                                    return ca_nm[:, token_idx, :].mean(axis=1).astype(np.float32, copy=False)

                                def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
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

                                def _distance(ca_nm: np.ndarray, a: int, b: int) -> np.ndarray:
                                    ca_nm = np.asarray(ca_nm, dtype=np.float64)
                                    if ca_nm.ndim != 3 or ca_nm.shape[-1] != 3:
                                        raise ValueError(f"ca_nm must be (T, tokens, 3), got {ca_nm.shape}")
                                    T, n_tok, _ = ca_nm.shape
                                    a_i = int(a)
                                    b_i = int(b)
                                    if a_i < 0 or b_i < 0 or a_i >= int(n_tok) or b_i >= int(n_tok) or a_i == b_i:
                                        return np.zeros((int(T),), dtype=np.float32)
                                    return np.linalg.norm(ca_nm[:, a_i, :] - ca_nm[:, b_i, :], axis=1).astype(np.float32, copy=False)

                                groups = _build_adk_groups_from_endpoint_pdb(pdb_a_path, n_tokens=n_tokens, token_mask=endpoint_token_mask)
                                # Heuristic guard: only plot if we can locate the ADK residues/groups.
                                ok_groups = (
                                    int(np.asarray(groups["core_lid"]).size) > 0
                                    and int(np.asarray(groups["core1"]).size) > 0
                                    and int(np.asarray(groups["nmp"]).size) > 0
                                    and int(np.asarray(groups["core2"]).size) > 0
                                    and int(np.asarray(groups["lid"]).size) > 0
                                    and int(groups["res145"]) >= 0
                                    and int(groups["res52"]) >= 0
                                    and int(groups["res169"]) >= 0
                                    and int(groups["res55"]) >= 0
                                )
                                if not ok_groups:
                                    raise RuntimeError("ADK residue mapping not found; skipping")

                                theta_nmp_deg_all: List[np.ndarray] = []
                                theta_lid_deg_all: List[np.ndarray] = []
                                dist_145_52_A_all: List[np.ndarray] = []
                                dist_169_55_A_all: List[np.ndarray] = []

                                for ck in cached_keys_epoch:
                                    if ck is None:
                                        continue
                                    try:
                                        _, ca_aa = geom2vec_cache.load(ck)
                                    except Exception:
                                        continue
                                    try:
                                        if isinstance(ca_aa, torch.Tensor):
                                            ca_aa_np = ca_aa.detach().cpu().numpy()
                                        else:
                                            ca_aa_np = ca_aa
                                    except Exception:
                                        ca_aa_np = ca_aa

                                    ca_aa_np = np.asarray(ca_aa_np, dtype=np.float64)
                                    if ca_aa_np.ndim != 3 or int(ca_aa_np.shape[0]) <= 0 or int(ca_aa_np.shape[1]) != int(n_tokens):
                                        continue

                                    ca_nm = ca_aa_np * 0.1
                                    core_lid = _centroid(ca_nm, groups["core_lid"])
                                    core1 = _centroid(ca_nm, groups["core1"])
                                    nmp = _centroid(ca_nm, groups["nmp"])
                                    core2 = _centroid(ca_nm, groups["core2"])
                                    lid = _centroid(ca_nm, groups["lid"])

                                    theta_nmp = _angle(core_lid, core1, nmp).astype(np.float64, copy=False)
                                    theta_lid = _angle(core2, core_lid, lid).astype(np.float64, copy=False)
                                    dist_145_52_nm = _distance(ca_nm, int(groups["res145"]), int(groups["res52"])).astype(np.float64, copy=False)
                                    dist_169_55_nm = _distance(ca_nm, int(groups["res169"]), int(groups["res55"])).astype(np.float64, copy=False)

                                    deg = 180.0 / math.pi
                                    theta_nmp_deg_all.append((theta_nmp * deg).astype(np.float32, copy=False))
                                    theta_lid_deg_all.append((theta_lid * deg).astype(np.float32, copy=False))
                                    dist_145_52_A_all.append((dist_145_52_nm * 10.0).astype(np.float32, copy=False))
                                    dist_169_55_A_all.append((dist_169_55_nm * 10.0).astype(np.float32, copy=False))

                                if not theta_nmp_deg_all:
                                    raise RuntimeError("no ADK frames available for plotting")

                                theta_nmp_deg = np.concatenate(theta_nmp_deg_all, axis=0)
                                theta_lid_deg = np.concatenate(theta_lid_deg_all, axis=0)
                                dist_145_52_A = np.concatenate(dist_145_52_A_all, axis=0)
                                dist_169_55_A = np.concatenate(dist_169_55_A_all, axis=0)

                                # Optional: overlay history density for context (helps diagnose missing A/B coverage).
                                overlay_history = str(os.environ.get("GESPO_ADK_PAIR_PLOT_OVERLAY_HISTORY", "1")).strip().lower() not in {
                                    "0",
                                    "false",
                                    "no",
                                    "n",
                                }
                                theta_nmp_deg_hist = None
                                theta_lid_deg_hist = None
                                dist_145_52_A_hist = None
                                dist_169_55_A_hist = None
                                if overlay_history:
                                    try:
                                        epoch_key_set = {str(k) for k in cached_keys_epoch if k}
                                        hist_keys_all: List[str] = []
                                        try:
                                            if boundary_plugin_fit_keys:
                                                hist_keys_all = [str(k) for k in boundary_plugin_fit_keys if k]
                                        except Exception:
                                            hist_keys_all = []
                                        # Fallback: no key list available.
                                        if hist_keys_all:
                                            hist_keys_all = [k for k in hist_keys_all if k not in epoch_key_set]
                                            max_hist_keys = int(os.environ.get("GESPO_ADK_PAIR_PLOT_MAX_HISTORY_KEYS", "200"))
                                            if max_hist_keys > 0 and len(hist_keys_all) > max_hist_keys:
                                                rng = np.random.default_rng(123)
                                                idx = rng.choice(len(hist_keys_all), size=max_hist_keys, replace=False)
                                                hist_keys = [hist_keys_all[int(i)] for i in idx.tolist()]
                                            else:
                                                hist_keys = hist_keys_all

                                            thn_all: List[np.ndarray] = []
                                            thl_all: List[np.ndarray] = []
                                            d1_all: List[np.ndarray] = []
                                            d2_all: List[np.ndarray] = []
                                            for ck in hist_keys:
                                                try:
                                                    _, ca_aa = geom2vec_cache.load(ck)
                                                except Exception:
                                                    continue
                                                try:
                                                    if isinstance(ca_aa, torch.Tensor):
                                                        ca_aa_np = ca_aa.detach().cpu().numpy()
                                                    else:
                                                        ca_aa_np = ca_aa
                                                except Exception:
                                                    ca_aa_np = ca_aa
                                                ca_aa_np = np.asarray(ca_aa_np, dtype=np.float64)
                                                if ca_aa_np.ndim != 3 or int(ca_aa_np.shape[0]) <= 0 or int(ca_aa_np.shape[1]) != int(n_tokens):
                                                    continue
                                                ca_nm = ca_aa_np * 0.1
                                                core_lid = _centroid(ca_nm, groups["core_lid"])
                                                core1 = _centroid(ca_nm, groups["core1"])
                                                nmp = _centroid(ca_nm, groups["nmp"])
                                                core2 = _centroid(ca_nm, groups["core2"])
                                                lid = _centroid(ca_nm, groups["lid"])

                                                theta_nmp = _angle(core_lid, core1, nmp).astype(np.float64, copy=False)
                                                theta_lid = _angle(core2, core_lid, lid).astype(np.float64, copy=False)
                                                dist_145_52_nm = _distance(ca_nm, int(groups["res145"]), int(groups["res52"])).astype(np.float64, copy=False)
                                                dist_169_55_nm = _distance(ca_nm, int(groups["res169"]), int(groups["res55"])).astype(np.float64, copy=False)

                                                deg = 180.0 / math.pi
                                                thn_all.append((theta_nmp * deg).astype(np.float32, copy=False))
                                                thl_all.append((theta_lid * deg).astype(np.float32, copy=False))
                                                d1_all.append((dist_145_52_nm * 10.0).astype(np.float32, copy=False))
                                                d2_all.append((dist_169_55_nm * 10.0).astype(np.float32, copy=False))

                                            if thn_all:
                                                theta_nmp_deg_hist = np.concatenate(thn_all, axis=0)
                                                theta_lid_deg_hist = np.concatenate(thl_all, axis=0)
                                                dist_145_52_A_hist = np.concatenate(d1_all, axis=0)
                                                dist_169_55_A_hist = np.concatenate(d2_all, axis=0)
                                    except Exception:
                                        theta_nmp_deg_hist = None
                                        theta_lid_deg_hist = None
                                        dist_145_52_A_hist = None
                                        dist_169_55_A_hist = None

                                # Filter NaNs/Infs + subsample for speed.
                                m_ang = np.isfinite(theta_nmp_deg) & np.isfinite(theta_lid_deg)
                                m_dst = np.isfinite(dist_145_52_A) & np.isfinite(dist_169_55_A)
                                theta_nmp_deg = theta_nmp_deg[m_ang]
                                theta_lid_deg = theta_lid_deg[m_ang]
                                dist_145_52_A = dist_145_52_A[m_dst]
                                dist_169_55_A = dist_169_55_A[m_dst]

                                max_pts = int(os.environ.get("GESPO_ADK_PAIR_PLOT_MAX_POINTS", "50000"))
                                if int(theta_nmp_deg.shape[0]) > max_pts:
                                    rng = np.random.default_rng(0)
                                    idx = rng.choice(int(theta_nmp_deg.shape[0]), size=max_pts, replace=False)
                                    theta_nmp_deg = theta_nmp_deg[idx]
                                    theta_lid_deg = theta_lid_deg[idx]
                                if int(dist_145_52_A.shape[0]) > max_pts:
                                    rng = np.random.default_rng(1)
                                    idx = rng.choice(int(dist_145_52_A.shape[0]), size=max_pts, replace=False)
                                    dist_145_52_A = dist_145_52_A[idx]
                                    dist_169_55_A = dist_169_55_A[idx]

                                if theta_nmp_deg_hist is not None and theta_lid_deg_hist is not None:
                                    m_h = np.isfinite(theta_nmp_deg_hist) & np.isfinite(theta_lid_deg_hist)
                                    theta_nmp_deg_hist = theta_nmp_deg_hist[m_h]
                                    theta_lid_deg_hist = theta_lid_deg_hist[m_h]
                                if dist_145_52_A_hist is not None and dist_169_55_A_hist is not None:
                                    m_h = np.isfinite(dist_145_52_A_hist) & np.isfinite(dist_169_55_A_hist)
                                    dist_145_52_A_hist = dist_145_52_A_hist[m_h]
                                    dist_169_55_A_hist = dist_169_55_A_hist[m_h]
                                max_hist_pts = int(os.environ.get("GESPO_ADK_PAIR_PLOT_MAX_POINTS_HISTORY", str(max_pts)))
                                if (
                                    theta_nmp_deg_hist is not None
                                    and theta_lid_deg_hist is not None
                                    and int(theta_nmp_deg_hist.shape[0]) > max_hist_pts
                                ):
                                    rng = np.random.default_rng(2)
                                    idx = rng.choice(int(theta_nmp_deg_hist.shape[0]), size=max_hist_pts, replace=False)
                                    theta_nmp_deg_hist = theta_nmp_deg_hist[idx]
                                    theta_lid_deg_hist = theta_lid_deg_hist[idx]
                                if (
                                    dist_145_52_A_hist is not None
                                    and dist_169_55_A_hist is not None
                                    and int(dist_145_52_A_hist.shape[0]) > max_hist_pts
                                ):
                                    rng = np.random.default_rng(3)
                                    idx = rng.choice(int(dist_145_52_A_hist.shape[0]), size=max_hist_pts, replace=False)
                                    dist_145_52_A_hist = dist_145_52_A_hist[idx]
                                    dist_169_55_A_hist = dist_169_55_A_hist[idx]

                                def _endpoint_point(ca_nm_ref: np.ndarray) -> Tuple[float, float, float, float]:
                                    ca_nm_ref = np.asarray(ca_nm_ref, dtype=np.float64).reshape(1, int(n_tokens), 3)
                                    core_lid_e = _centroid(ca_nm_ref, groups["core_lid"])
                                    core1_e = _centroid(ca_nm_ref, groups["core1"])
                                    nmp_e = _centroid(ca_nm_ref, groups["nmp"])
                                    core2_e = _centroid(ca_nm_ref, groups["core2"])
                                    lid_e = _centroid(ca_nm_ref, groups["lid"])
                                    th_nmp = float(_angle(core_lid_e, core1_e, nmp_e).reshape(-1)[0]) * (180.0 / math.pi)
                                    th_lid = float(_angle(core2_e, core_lid_e, lid_e).reshape(-1)[0]) * (180.0 / math.pi)
                                    d1 = float(_distance(ca_nm_ref, int(groups["res145"]), int(groups["res52"])).reshape(-1)[0]) * 10.0
                                    d2 = float(_distance(ca_nm_ref, int(groups["res169"]), int(groups["res55"])).reshape(-1)[0]) * 10.0
                                    return th_nmp, th_lid, d1, d2

                                a_pt = _endpoint_point(np.asarray(endpoint_ca_ref_a, dtype=np.float64))
                                b_pt = _endpoint_point(np.asarray(endpoint_ca_ref_b, dtype=np.float64))

                                out_png = Path(config.snapshot_dir) / f"adk_feature_pairs_ep{ep+1:03d}.png"
                                out_npz = Path(config.snapshot_dir) / f"adk_feature_pairs_ep{ep+1:03d}.npz"

                                try:
                                    np.savez_compressed(
                                        str(out_npz),
                                        theta_nmp_deg=theta_nmp_deg,
                                        theta_lid_deg=theta_lid_deg,
                                        dist_145_52_A=dist_145_52_A,
                                        dist_169_55_A=dist_169_55_A,
                                        theta_nmp_deg_hist=(
                                            np.asarray(theta_nmp_deg_hist, dtype=np.float32)
                                            if theta_nmp_deg_hist is not None
                                            else np.zeros((0,), dtype=np.float32)
                                        ),
                                        theta_lid_deg_hist=(
                                            np.asarray(theta_lid_deg_hist, dtype=np.float32)
                                            if theta_lid_deg_hist is not None
                                            else np.zeros((0,), dtype=np.float32)
                                        ),
                                        dist_145_52_A_hist=(
                                            np.asarray(dist_145_52_A_hist, dtype=np.float32)
                                            if dist_145_52_A_hist is not None
                                            else np.zeros((0,), dtype=np.float32)
                                        ),
                                        dist_169_55_A_hist=(
                                            np.asarray(dist_169_55_A_hist, dtype=np.float32)
                                            if dist_169_55_A_hist is not None
                                            else np.zeros((0,), dtype=np.float32)
                                        ),
                                        endpoint_a=np.asarray(a_pt, dtype=np.float32),
                                        endpoint_b=np.asarray(b_pt, dtype=np.float32),
                                        epoch=np.asarray([int(ep + 1)], dtype=np.int64),
                                    )
                                except Exception:
                                    pass

                                import matplotlib.pyplot as plt  # type: ignore
                                from matplotlib.colors import LogNorm  # type: ignore

                                fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
                                ax0, ax1 = axs[0], axs[1]

                                if theta_nmp_deg_hist is not None and theta_lid_deg_hist is not None and theta_nmp_deg_hist.size:
                                    ax0.hexbin(
                                        theta_nmp_deg_hist,
                                        theta_lid_deg_hist,
                                        gridsize=140,
                                        mincnt=1,
                                        cmap="Greys",
                                        norm=LogNorm(),
                                        alpha=0.25,
                                    )
                                hb0 = ax0.hexbin(
                                    theta_nmp_deg,
                                    theta_lid_deg,
                                    gridsize=140,
                                    mincnt=1,
                                    cmap="magma",
                                    norm=LogNorm(),
                                )
                                ax0.scatter([a_pt[0]], [a_pt[1]], c="tab:blue", s=50, marker="o", edgecolors="k", linewidths=0.5, label="A")
                                ax0.scatter([b_pt[0]], [b_pt[1]], c="tab:red", s=50, marker="o", edgecolors="k", linewidths=0.5, label="B")
                                ax0.set_xlabel("theta_nmp_deg")
                                ax0.set_ylabel("theta_lid_deg")
                                ax0.set_title(f"ADK domain angles (epoch {ep+1}{'; +history' if overlay_history else ''})")
                                cb0 = fig.colorbar(hb0, ax=ax0)
                                cb0.set_label("count (log)")
                                ax0.legend(loc="best", frameon=False)

                                if dist_145_52_A_hist is not None and dist_169_55_A_hist is not None and dist_145_52_A_hist.size:
                                    ax1.hexbin(
                                        dist_145_52_A_hist,
                                        dist_169_55_A_hist,
                                        gridsize=140,
                                        mincnt=1,
                                        cmap="Greys",
                                        norm=LogNorm(),
                                        alpha=0.25,
                                    )
                                hb1 = ax1.hexbin(
                                    dist_145_52_A,
                                    dist_169_55_A,
                                    gridsize=140,
                                    mincnt=1,
                                    cmap="magma",
                                    norm=LogNorm(),
                                )
                                ax1.scatter([a_pt[2]], [a_pt[3]], c="tab:blue", s=50, marker="o", edgecolors="k", linewidths=0.5, label="A")
                                ax1.scatter([b_pt[2]], [b_pt[3]], c="tab:red", s=50, marker="o", edgecolors="k", linewidths=0.5, label="B")
                                ax1.set_xlabel("dist_145_52_A")
                                ax1.set_ylabel("dist_169_55_A")
                                ax1.set_title(f"ADK distances (epoch {ep+1}{'; +history' if overlay_history else ''})")
                                cb1 = fig.colorbar(hb1, ax=ax1)
                                cb1.set_label("count (log)")
                                ax1.legend(loc="best", frameon=False)

                                if overlay_history:
                                    fig.suptitle("ADK feature pairs (epoch in color; history in gray)")
                                else:
                                    fig.suptitle("ADK feature pairs (from cached epoch trajectories)")
                                fig.tight_layout()
                                fig.savefig(out_png, dpi=200)
                                plt.close(fig)
                                print(f"[ADK Pairs] Saved feature-pair plot: {out_png}", flush=True)
                            except Exception as exc_adk:
                                # Keep silent-ish; ADK plots are optional and should not fail training.
                                msg = str(exc_adk)
                                if msg and msg != "disabled":
                                    print(f"[ADK Pairs] Note: skipped ({exc_adk})", flush=True)

                            # Update tube/edge/connectivity stats with current epoch trajectories (for future epochs).
                            edge_coef_cfg = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0)
                            conn_coef_cfg = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
                            need_update_tube = (committor_tube_weight > 0.0) and (committor_state.tube_counts is not None)
                            need_update_edge = (edge_coef_cfg > 0.0) and (committor_state.edge_counts is not None)
                            need_update_conn = (conn_coef_cfg > 0.0) and (committor_state.conn_adj is not None)
                            if (
                                (need_update_tube or need_update_edge or need_update_conn)
                                and qlab_epoch is not None
                                and int(tube_bins) > 1
                            ):
                                edge_delta_u = int(getattr(config, "committor_tube_edge_novelty_lag", 1) or 1)
                                edge_delta_u = int(max(1, edge_delta_u))
                                edge_undirected_u = bool(getattr(config, "committor_tube_edge_novelty_undirected", True))
                                edge_bin_change_only_u = bool(getattr(config, "committor_tube_edge_novelty_bin_change_only", False))
                                n_cells_u = int(tube_bins) * int(max(1, K_tube))

                                added_tube_trajs = 0
                                added_edge_trajs = 0
                                added_edges = 0
                                added_conn_trajs = 0
                                added_conn_edges = 0
                                for i, cached_key in enumerate(cached_keys_epoch):
                                    if cached_key is None:
                                        continue
                                    key_s = str(cached_key)

                                    tube_done = (not need_update_tube) or (key_s in committor_state.tube_counted_keys)
                                    edge_done = (not need_update_edge) or (key_s in committor_state.edge_counted_keys)
                                    conn_done = (not need_update_conn) or (key_s in committor_state.conn_counted_keys)
                                    if tube_done and edge_done and conn_done:
                                        continue

                                    pair = qlab_epoch[i] if i < len(qlab_epoch) else None
                                    if pair is None:
                                        if need_update_tube and not tube_done:
                                            committor_state.tube_counted_keys.add(key_s)
                                        if need_update_edge and not edge_done:
                                            committor_state.edge_counted_keys.add(key_s)
                                        if need_update_conn and not conn_done:
                                            committor_state.conn_counted_keys.add(key_s)
                                        continue

                                    q_f, lab_f = pair
                                    q64 = np.asarray(q_f, dtype=np.float64).reshape(-1)
                                    lab64 = np.asarray(lab_f, dtype=np.int64).reshape(-1)
                                    n = int(min(int(q64.shape[0]), int(lab64.shape[0])))
                                    if n <= 0:
                                        if need_update_tube and not tube_done:
                                            committor_state.tube_counted_keys.add(key_s)
                                        if need_update_edge and not edge_done:
                                            committor_state.edge_counted_keys.add(key_s)
                                        if need_update_conn and not conn_done:
                                            committor_state.conn_counted_keys.add(key_s)
                                        continue

                                    q64 = np.clip(q64[:n], 0.0, 1.0)
                                    lab64 = np.clip(lab64[:n], 0, int(K_tube) - 1)
                                    m_bins = np.floor(float(tube_bins) * q64).astype(np.int64, copy=False)
                                    m_bins = np.clip(m_bins, 0, int(tube_bins) - 1)
                                    cell = m_bins * int(K_tube) + lab64

                                    if need_update_tube and not tube_done:
                                        w_hat = 4.0 * q64 * (1.0 - q64)
                                        w_hat = np.where(np.isfinite(w_hat), w_hat, 0.0)
                                        w_hat = np.clip(w_hat, 0.0, 1.0)
                                        if np.any(w_hat > 0.0):
                                            np.add.at(committor_state.tube_counts, cell, w_hat)
                                            committor_state.tube_total_weight += float(np.sum(w_hat))
                                            added_tube_trajs += 1
                                        committor_state.tube_counted_keys.add(key_s)

                                    if need_update_edge and not edge_done:
                                        if int(n_cells_u) > 1 and n > int(edge_delta_u):
                                            src = np.asarray(cell[: n - int(edge_delta_u)], dtype=np.int64)
                                            dst = np.asarray(cell[int(edge_delta_u) : n], dtype=np.int64)
                                            if edge_bin_change_only_u:
                                                mb0 = np.asarray(m_bins[: n - int(edge_delta_u)], dtype=np.int64)
                                                mb1 = np.asarray(m_bins[int(edge_delta_u) : n], dtype=np.int64)
                                                mask = mb0 != mb1
                                                src = src[mask]
                                                dst = dst[mask]
                                            if src.size:
                                                if edge_undirected_u:
                                                    a = np.minimum(src, dst)
                                                    b = np.maximum(src, dst)
                                                    src = a
                                                    dst = b
                                                edge_idx = src.astype(np.int64) * int(n_cells_u) + dst.astype(np.int64)
                                                if isinstance(committor_state.edge_counts, np.ndarray):
                                                    np.add.at(committor_state.edge_counts, edge_idx, 1.0)
                                                else:
                                                    d = committor_state.edge_counts
                                                    uniq, cnts = np.unique(edge_idx, return_counts=True)
                                                    for u, c in zip(uniq.tolist(), cnts.tolist()):
                                                        d[int(u)] = float(d.get(int(u), 0.0)) + float(c)
                                                committor_state.edge_total_count += float(edge_idx.size)
                                                added_edge_trajs += 1
                                                added_edges += int(edge_idx.size)
                                        committor_state.edge_counted_keys.add(key_s)

                                    if need_update_conn and not conn_done and committor_state.conn_adj is not None:
                                        # Basin-side cell bookkeeping.
                                        try:
                                            a_mask = m_bins == 0
                                            b_mask = m_bins == (int(tube_bins) - 1)
                                            if bool(np.any(a_mask)):
                                                committor_state.conn_a_cells.update(np.asarray(cell, dtype=np.int64)[a_mask].tolist())
                                            if bool(np.any(b_mask)):
                                                committor_state.conn_b_cells.update(np.asarray(cell, dtype=np.int64)[b_mask].tolist())
                                        except Exception:
                                            pass

                                        if int(n_cells_u) > 1 and n > int(edge_delta_u):
                                            src = np.asarray(cell[: n - int(edge_delta_u)], dtype=np.int64)
                                            dst = np.asarray(cell[int(edge_delta_u) : n], dtype=np.int64)
                                            if edge_bin_change_only_u:
                                                mb0 = np.asarray(m_bins[: n - int(edge_delta_u)], dtype=np.int64)
                                                mb1 = np.asarray(m_bins[int(edge_delta_u) : n], dtype=np.int64)
                                                mask = mb0 != mb1
                                                src = src[mask]
                                                dst = dst[mask]
                                            if src.size:
                                                if edge_undirected_u:
                                                    a = np.minimum(src, dst)
                                                    b = np.maximum(src, dst)
                                                    src = a
                                                    dst = b
                                                edge_idx = src.astype(np.int64) * int(n_cells_u) + dst.astype(np.int64)
                                                uniq = np.unique(edge_idx)
                                                new = 0
                                                for e in uniq.tolist():
                                                    ei = int(e)
                                                    if ei in committor_state.conn_edges:
                                                        continue
                                                    committor_state.conn_edges.add(ei)
                                                    u = int(ei // int(n_cells_u))
                                                    v = int(ei - u * int(n_cells_u))
                                                    if 0 <= u < int(n_cells_u) and 0 <= v < int(n_cells_u):
                                                        committor_state.conn_adj[u].add(v)
                                                        committor_state.conn_adj[v].add(u)
                                                    new += 1
                                                if new:
                                                    added_conn_edges += int(new)
                                        committor_state.conn_counted_keys.add(key_s)
                                        added_conn_trajs += 1

                                if added_tube_trajs:
                                    print(
                                        f"[Committor Tube - Epoch {ep+1}] Updated history counts: added_trajs={added_tube_trajs} "
                                        f"total_w={float(committor_state.tube_total_weight):.1f}",
                                        flush=True,
                                    )
                                if added_edge_trajs:
                                    mode = "dense" if isinstance(committor_state.edge_counts, np.ndarray) else "sparse"
                                    print(
                                        f"[Committor Edge - Epoch {ep+1}] Updated history edges: added_trajs={added_edge_trajs} "
                                        f"added_edges={int(added_edges)} mode={mode} total_edges={float(committor_state.edge_total_count):.1f}",
                                        flush=True,
                                    )
                                if added_conn_trajs:
                                    dist_s = "inf"
                                    gap_s = "?"
                                    amax_s = "?"
                                    bmin_s = "?"
                                    try:
                                        if committor_state.conn_adj is not None and committor_state.conn_a_cells and committor_state.conn_b_cells:
                                            dist_ab = _conn_shortest_path_len(
                                                adj=committor_state.conn_adj,
                                                sources=list(committor_state.conn_a_cells),
                                                targets=set(committor_state.conn_b_cells),
                                                overlay=None,
                                            )
                                            dist_s = "inf" if not np.isfinite(dist_ab) else str(int(dist_ab))
                                            reach_a = _conn_reachable_nodes(
                                                adj=committor_state.conn_adj,
                                                sources=list(committor_state.conn_a_cells),
                                                overlay=None,
                                            )
                                            reach_b = _conn_reachable_nodes(
                                                adj=committor_state.conn_adj,
                                                sources=list(committor_state.conn_b_cells),
                                                overlay=None,
                                            )
                                            gap_i, amax_i, bmin_i = _conn_bin_gap(
                                                reachable_a=reach_a,
                                                reachable_b=reach_b,
                                                K=int(max(1, K_tube)),
                                                M=int(tube_bins),
                                            )
                                            gap_s = str(int(gap_i))
                                            amax_s = str(int(amax_i))
                                            bmin_s = str(int(bmin_i))
                                    except Exception:
                                        dist_s = "inf"
                                    print(
                                        f"[Committor Conn - Epoch {ep+1}] Updated history graph: added_trajs={added_conn_trajs} "
                                        f"added_edges={int(added_conn_edges)} A→B_dist={dist_s} gap={gap_s} "
                                        f"A_max_bin={amax_s} B_min_bin={bmin_s}",
                                        flush=True,
                                    )

                    # Add current epoch trajectories to global stats (once per key).
                    for t_i, cached_key in enumerate(cached_keys_epoch):
                        if cached_key is None:
                            continue
                        key_s = str(cached_key)
                        if key_s in loo_vamp_stats_keys:
                            continue
                        pair = traj_pairs[t_i] if t_i < len(traj_pairs) else None
                        if pair is not None and loo_vamp_stats is not None:
                            try:
                                loo_vamp_stats.add_trajectory(pair.x, pair.y, pair.ind_stop)  # type: ignore[arg-type]
                                loo_vamp_stats_keys.add(key_s)
                            except Exception:
                                pass

                    if loo_vamp_stats is None:
                        raise RuntimeError("Global LOO VAMP stats not initialized.")

                    vamp_baseline = float(loo_vamp_stats.compute_vamp2_score())
                    contribs = np.zeros(config.grp_size, dtype=np.float32)

                    for i, pair in enumerate(traj_pairs):
                        if pair is None:
                            continue
                        loo_vamp_stats.remove_trajectory(pair.x, pair.y, pair.ind_stop)  # type: ignore[arg-type]
                        vamp_wo = float(loo_vamp_stats.compute_vamp2_score())
                        contribs[i] = float(vamp_baseline - vamp_wo)
                        loo_vamp_stats.add_trajectory(pair.x, pair.y, pair.ind_stop)  # type: ignore[arg-type]

                    loo_vamp_rewards = contribs.astype(np.float32, copy=False)
                    loo_vamp_contributions = contribs.astype(np.float32, copy=False)

                    print(f"\n[LOO VAMP Statistics - Epoch {ep+1}]")
                    print(f"  Baseline VAMP-2 (history+epoch trajectories): {vamp_baseline:.4f}")

                    total_pairs = int(getattr(loo_vamp_stats, "n_samples", 0) or 0)
                    if total_pairs > 0:
                        try:
                            stride_eff = int(geom2vec_embed_stride)
                        except Exception:
                            stride_eff = 0
                        print(
                            f"  Total transition pairs (T-lag): {total_pairs} "
                            f"(lag_embed={int(lag_embed)}, embed_stride={stride_eff})"
                        )

                    print("  Per-trajectory contributions (delta_i = S_all - S_minus_i):")
                    for i in range(config.grp_size):
                        if pair_counts[i] <= 0:
                            continue
                        frac = float(pair_counts[i]) / float(total_pairs) if total_pairs > 0 else 0.0
                        raw_per_pair = float(contribs[i]) / float(pair_counts[i]) if pair_counts[i] > 0 else 0.0
                        print(
                            f"    Traj {i:2d}: raw={float(contribs[i]):+.4f}  "
                            f"pairs={int(pair_counts[i])} ({frac:.2%})  raw/pair={raw_per_pair:+.2e}  "
                            f"score={float(loo_vamp_rewards[i]):+.4f}"
                        )
                except Exception as exc:
                    print(f"[LOO VAMP] Warning: Failed to compute geom2vec LOO rewards: {exc}")
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

            term_loo_vamp = torch.zeros_like(launch_rewards)
            term_start_cluster = torch.zeros_like(launch_rewards)
            term_start_basin = torch.zeros_like(launch_rewards)
            term_vamp_bin = torch.zeros_like(launch_rewards)
            term_msm_ig = torch.zeros_like(launch_rewards)
            term_committor = torch.zeros_like(launch_rewards)
            term_committor_tube = torch.zeros_like(launch_rewards)
            term_committor_bin_entropy = torch.zeros_like(launch_rewards)
            term_off_action = torch.zeros_like(launch_rewards)
            off_action_discount = torch.ones_like(launch_rewards)

            if loo_vamp_rewards is not None and np.any(np.abs(loo_vamp_rewards) > 1e-8):
                loo_vamp_tensor = torch.tensor(loo_vamp_rewards, device=torch_device, dtype=torch.float32)
                term_loo_vamp = loo_weight * loo_vamp_tensor
                combined_rewards = term_loo_vamp
                print(f"[Rewards] Using LOO VAMP only (weight={loo_weight:.2f}): loo_mean={loo_vamp_tensor.mean():.4f}")
            else:
                combined_rewards = torch.zeros_like(launch_rewards)

            # Active mask used to ignore invalid/failed walkers when aggregating reward terms.
            active_mask = metrics.get(
                "active_mask",
                torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
            )
            if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                active_mask = active_mask.to(torch_device)
            if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                active_mask = active_mask.bool()

            if start_cluster_ent_coef > 0 and start_cluster_bonus is not None:
                bonus_tensor = torch.as_tensor(start_cluster_bonus, device=torch_device, dtype=torch.float32)
                term_start_cluster = start_cluster_ent_coef * bonus_tensor
                combined_rewards = combined_rewards + term_start_cluster
                try:
                    h_val = float(start_cluster_entropy) if start_cluster_entropy is not None else float("nan")
                    print(
                        f"[Rewards] Added start-cluster entropy bonus: coef={start_cluster_ent_coef:.2e}, "
                        f"H={h_val:.3f}, bonus_mean={bonus_tensor.mean().item():.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added start-cluster entropy bonus (coef={start_cluster_ent_coef:.2e}).")

            comm_start_basin_coef = float(getattr(config, "committor_start_basin_entropy_reward_weight", 0.0) or 0.0)
            if comm_start_basin_coef > 0.0 and committor_start_basin_bonus_raw is not None:
                bonus_tensor = torch.as_tensor(committor_start_basin_bonus_raw, device=torch_device, dtype=torch.float32)
                term_start_basin = comm_start_basin_coef * bonus_tensor
                combined_rewards = combined_rewards + term_start_basin
                try:
                    a_n = int(metrics.get("committor_start_basin_epoch_a", 0) or 0)
                    b_n = int(metrics.get("committor_start_basin_epoch_b", 0) or 0)
                    print(
                        f"[Rewards] Added committor start-basin entropy bonus: coef={comm_start_basin_coef:.2e}, "
                        f"starts(A,B)=({a_n},{b_n}) bonus_mean={bonus_tensor.mean().item():.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added committor start-basin entropy bonus (coef={comm_start_basin_coef:.2e}).")

            comm_bin_ent_coef = float(getattr(config, "committor_bin_entropy_reward_weight", 0.0) or 0.0)
            if comm_bin_ent_coef != 0.0 and committor_bin_entropy_raw is not None:
                H_raw = np.asarray(committor_bin_entropy_raw, dtype=np.float32).reshape(-1)
                H_raw = np.where(np.isfinite(H_raw), H_raw, 0.0).astype(np.float32, copy=False)

                H_mean = float(np.mean(H_raw)) if H_raw.size else 0.0
                H_std = float(np.std(H_raw)) if H_raw.size else 0.0

                eps = 1e-8
                ema_beta = float(getattr(config, "committor_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = H_mean
                cur_m2 = float(np.mean(H_raw * H_raw)) if H_raw.size else 0.0

                if use_ema:
                    if committor_state.bin_entropy_norm_ema_m1 is None or committor_state.bin_entropy_norm_ema_m2 is None:
                        committor_state.bin_entropy_norm_ema_m1 = float(cur_m1)
                        committor_state.bin_entropy_norm_ema_m2 = float(cur_m2)
                    else:
                        committor_state.bin_entropy_norm_ema_m1 = float(ema_beta) * float(committor_state.bin_entropy_norm_ema_m1) + float(
                            1.0 - ema_beta
                        ) * float(cur_m1)
                        committor_state.bin_entropy_norm_ema_m2 = float(ema_beta) * float(committor_state.bin_entropy_norm_ema_m2) + float(
                            1.0 - ema_beta
                        ) * float(cur_m2)
                    mean_use = float(committor_state.bin_entropy_norm_ema_m1)
                    var_use = float(max(float(committor_state.bin_entropy_norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(H_std)

                if np.isfinite(std_use) and std_use > eps:
                    H_z = (H_raw - mean_use) / std_use
                else:
                    H_z = H_raw * 0.0

                loo_std = 0.0
                try:
                    loo_std = float(np.std(np.asarray(loo_vamp_rewards, dtype=np.float32)))
                except Exception:
                    loo_std = 0.0
                target_scale_cur = float(loo_weight) * loo_std if (np.isfinite(loo_std) and loo_std > eps) else 0.0
                if not (np.isfinite(target_scale_cur) and target_scale_cur > eps):
                    try:
                        cur_std = float(combined_rewards[active_mask].std(unbiased=False).item())
                        if np.isfinite(cur_std) and cur_std > eps:
                            target_scale_cur = cur_std
                    except Exception:
                        pass
                if not (np.isfinite(target_scale_cur) and target_scale_cur > eps):
                    target_scale_cur = 1.0

                H_norm = H_z * float(target_scale_cur)
                h_tensor = torch.as_tensor(H_norm, device=torch_device, dtype=torch.float32)
                term_committor_bin_entropy = comm_bin_ent_coef * h_tensor
                term_committor_bin_entropy = torch.where(
                    active_mask, term_committor_bin_entropy, torch.zeros_like(term_committor_bin_entropy)
                )
                combined_rewards = combined_rewards + term_committor_bin_entropy

                try:
                    metrics["avg_committor_bin_entropy_raw"] = float(H_mean)
                    metrics["std_committor_bin_entropy_raw"] = float(H_std)
                    metrics["avg_committor_bin_entropy_norm_mean"] = float(mean_use)
                    metrics["avg_committor_bin_entropy_norm_std"] = float(std_use)
                    metrics["avg_committor_bin_entropy_norm_scale"] = float(target_scale_cur)
                    h_active = term_committor_bin_entropy[active_mask]
                    metrics["avg_committor_bin_entropy"] = float(h_active.mean().item()) if h_active.numel() else 0.0
                    print(
                        f"[Rewards] Added committor-bin entropy: coef={comm_bin_ent_coef:.2e}, "
                        f"H_mean={H_mean:.4f} H_std={H_std:.4f} ema_beta={ema_beta:.2f} "
                        f"H_tilde_mean={metrics['avg_committor_bin_entropy']:+.4f}",
                        flush=True,
                    )
                except Exception:
                    print(f"[Rewards] Added committor-bin entropy term (coef={comm_bin_ent_coef:.2e}).")

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
                    if vamp_bin_mgr.norm_ema_m1 is None or vamp_bin_mgr.norm_ema_m2 is None:
                        vamp_bin_mgr.norm_ema_m1 = float(cur_m1)
                        vamp_bin_mgr.norm_ema_m2 = float(cur_m2)
                    else:
                        vamp_bin_mgr.norm_ema_m1 = float(ema_beta) * float(vamp_bin_mgr.norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        vamp_bin_mgr.norm_ema_m2 = float(ema_beta) * float(vamp_bin_mgr.norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)

                    mean_use = float(vamp_bin_mgr.norm_ema_m1)
                    var_use = float(max(float(vamp_bin_mgr.norm_ema_m2) - mean_use * mean_use, 0.0))
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
                    if vamp_bin_mgr.norm_ema_scale is None:
                        vamp_bin_mgr.norm_ema_scale = float(target_scale_cur)
                    else:
                        vamp_bin_mgr.norm_ema_scale = float(ema_beta) * float(vamp_bin_mgr.norm_ema_scale) + float(1.0 - ema_beta) * float(target_scale_cur)
                    target_scale = float(vamp_bin_mgr.norm_ema_scale)
                else:
                    target_scale = float(target_scale_cur)

                bonus_norm = bonus_z * float(target_scale)

                bonus_tensor = torch.as_tensor(bonus_norm, device=torch_device, dtype=torch.float32)
                term_vamp_bin = vamp_bin_ent_coef * bonus_tensor
                combined_rewards = combined_rewards + term_vamp_bin
                try:
                    metrics["avg_vamp_bin_bonus"] = float(bonus_tensor.mean().item())
                    metrics["std_vamp_bin_bonus"] = float(bonus_tensor.std(unbiased=False).item())
                    metrics["avg_vamp_bin_bonus_norm_mean"] = float(mean_use)
                    metrics["avg_vamp_bin_bonus_norm_std"] = float(std_use)
                    metrics["avg_vamp_bin_bonus_norm_scale"] = float(target_scale)
                except Exception:
                    pass
                try:
                    print(
                        f"[Rewards] Added VAMP-bin bonus: coef={vamp_bin_ent_coef:.2e}, "
                        f"raw_mean={bonus_raw_mean:.4f} raw_std={bonus_raw_std:.4f} "
                        f"ema_beta={ema_beta:.2f} norm_std={std_use:.4f} "
                        f"scale={target_scale:.3e} norm_mean={bonus_tensor.mean().item():+.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added VAMP-bin bonus (coef={vamp_bin_ent_coef:.2e}).")

            msm_ig_coef = float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0)
            if msm_ig_coef > 0.0 and msm_ig_rewards is not None:
                active_mask = metrics.get(
                    "active_mask",
                    torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                )
                if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                    active_mask = active_mask.to(torch_device)
                if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                    active_mask = active_mask.bool()

                try:
                    active_cpu = (
                        active_mask.detach().cpu().numpy().astype(bool)
                        if torch.is_tensor(active_mask)
                        else np.ones((config.grp_size,), dtype=bool)
                    )
                except Exception:
                    active_cpu = np.ones((config.grp_size,), dtype=bool)

                try:
                    msm_raw = np.asarray(msm_ig_rewards, dtype=np.float32)
                    msm_raw = np.where(np.isfinite(msm_raw), msm_raw, 0.0).astype(np.float32, copy=False)
                except Exception:
                    msm_raw = np.zeros((config.grp_size,), dtype=np.float32)

                msm_raw_active = msm_raw[active_cpu] if msm_raw.size == int(config.grp_size) else msm_raw
                msm_raw_mean = float(np.mean(msm_raw_active)) if msm_raw_active.size else 0.0
                msm_raw_std = float(np.std(msm_raw_active)) if msm_raw_active.size else 0.0

                # Normalize MSM-IG to match the typical LOO-VAMP scale (like the VAMP-bin bonus).
                eps = 1e-8
                ema_beta = float(getattr(config, "msm_ig_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = float(msm_raw_mean)
                try:
                    cur_m2 = float(np.mean(msm_raw_active.astype(np.float64) ** 2)) if msm_raw_active.size else 0.0
                except Exception:
                    cur_m2 = float(cur_m1 * cur_m1 + msm_raw_std * msm_raw_std)

                if use_ema:
                    if msm_ig_mgr.norm_ema_m1 is None or msm_ig_mgr.norm_ema_m2 is None:
                        msm_ig_mgr.norm_ema_m1 = float(cur_m1)
                        msm_ig_mgr.norm_ema_m2 = float(cur_m2)
                    else:
                        msm_ig_mgr.norm_ema_m1 = float(ema_beta) * float(msm_ig_mgr.norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        msm_ig_mgr.norm_ema_m2 = float(ema_beta) * float(msm_ig_mgr.norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)

                    mean_use = float(msm_ig_mgr.norm_ema_m1)
                    var_use = float(max(float(msm_ig_mgr.norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(msm_raw_std)

                if np.isfinite(std_use) and std_use > eps:
                    msm_z = (msm_raw - float(mean_use)) / float(std_use)
                else:
                    msm_z = msm_raw * 0.0

                loo_std = 0.0
                try:
                    loo_std = float(np.std(np.asarray(loo_vamp_rewards, dtype=np.float32)))
                except Exception:
                    loo_std = 0.0

                target_scale_cur = float(loo_weight) * loo_std if (np.isfinite(loo_std) and loo_std > eps) else 0.0
                if not (np.isfinite(target_scale_cur) and target_scale_cur > eps):
                    try:
                        cur_std = float(combined_rewards[active_mask].std(unbiased=False).item())
                        if np.isfinite(cur_std) and cur_std > eps:
                            target_scale_cur = cur_std
                    except Exception:
                        pass
                if not (np.isfinite(target_scale_cur) and target_scale_cur > eps):
                    target_scale_cur = 1.0
                if use_ema:
                    if msm_ig_mgr.norm_ema_scale is None:
                        msm_ig_mgr.norm_ema_scale = float(target_scale_cur)
                    else:
                        msm_ig_mgr.norm_ema_scale = float(ema_beta) * float(msm_ig_mgr.norm_ema_scale) + float(1.0 - ema_beta) * float(target_scale_cur)
                    target_scale = float(msm_ig_mgr.norm_ema_scale)
                else:
                    target_scale = float(target_scale_cur)

                msm_norm = msm_z * float(target_scale)
                try:
                    msm_tensor = torch.as_tensor(msm_norm, device=torch_device, dtype=torch.float32)
                except Exception:
                    msm_tensor = torch.zeros_like(launch_rewards)

                term_msm_ig = msm_ig_coef * msm_tensor
                term_msm_ig = torch.where(active_mask, term_msm_ig, torch.zeros_like(term_msm_ig))
                combined_rewards = combined_rewards + term_msm_ig

                try:
                    metrics["avg_msm_ig_raw"] = float(msm_raw_mean)
                    metrics["std_msm_ig_raw"] = float(msm_raw_std)
                    metrics["avg_msm_ig_norm_mean"] = float(mean_use)
                    metrics["avg_msm_ig_norm_std"] = float(std_use)
                    metrics["avg_msm_ig_norm_scale"] = float(target_scale)
                    term_active = term_msm_ig[active_mask]
                    metrics["avg_msm_ig"] = float(term_active.mean().item()) if term_active.numel() else 0.0
                    if msm_ig_unique_states is not None:
                        uniq = np.asarray(msm_ig_unique_states, dtype=np.float32)
                        uniq = np.where(np.isfinite(uniq), uniq, 0.0)
                        metrics["avg_msm_ig_unique_states"] = float(np.mean(uniq))
                    print(
                        f"[Rewards] Added MSM-IG reward: coef={msm_ig_coef:.2e}, "
                        f"raw_mean={metrics['avg_msm_ig_raw']:.4f} raw_std={metrics['std_msm_ig_raw']:.4f} "
                        f"ema_beta={ema_beta:.2f} norm_std={std_use:.4f} "
                        f"scale={target_scale:.3e} norm_mean={msm_tensor.mean().item():+.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added MSM-IG reward (coef={msm_ig_coef:.2e}).")

            committor_conn_coef = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
            use_committor_connectivity = (committor_conn_coef != 0.0) and (committor_conn_raw is not None)
            if use_committor_connectivity:
                print(
                    f"[Rewards] Committor connectivity enabled (coef={committor_conn_coef:.2e}); "
                    "ignoring committor_progress/tube/edge rewards.",
                    flush=True,
                )

            committor_coef = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
            if (not use_committor_connectivity) and committor_coef != 0.0 and committor_progress_raw is not None:
                Q_raw = np.asarray(committor_progress_raw, dtype=np.float32).reshape(-1)
                Q_raw = np.where(np.isfinite(Q_raw), Q_raw, 0.0).astype(np.float32, copy=False)

                Q_mean = float(np.mean(Q_raw)) if Q_raw.size else 0.0
                Q_std = float(np.std(Q_raw)) if Q_raw.size else 0.0

                eps = 1e-8
                ema_beta = float(getattr(config, "committor_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = Q_mean
                cur_m2 = float(np.mean(Q_raw * Q_raw)) if Q_raw.size else 0.0

                if use_ema:
                    if committor_state.norm_ema_m1 is None or committor_state.norm_ema_m2 is None:
                        committor_state.norm_ema_m1 = float(cur_m1)
                        committor_state.norm_ema_m2 = float(cur_m2)
                    else:
                        committor_state.norm_ema_m1 = float(ema_beta) * float(committor_state.norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        committor_state.norm_ema_m2 = float(ema_beta) * float(committor_state.norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)
                    mean_use = float(committor_state.norm_ema_m1)
                    var_use = float(max(float(committor_state.norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(Q_std)

                denom = float(std_use + eps) if (np.isfinite(std_use) and std_use > 0.0) else 1.0
                Q_tilde = (Q_raw - float(mean_use)) / denom
                Q_tilde = np.where(np.isfinite(Q_tilde), Q_tilde, 0.0).astype(np.float32, copy=False)

                active_mask = metrics.get(
                    "active_mask",
                    torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                )
                if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                    active_mask = active_mask.to(torch_device)
                if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                    active_mask = active_mask.bool()

                q_tensor = torch.as_tensor(Q_tilde, device=torch_device, dtype=torch.float32)
                term_committor = committor_coef * q_tensor
                term_committor = torch.where(active_mask, term_committor, torch.zeros_like(term_committor))
                combined_rewards = combined_rewards + term_committor

                try:
                    metrics["avg_committor_Q"] = float(Q_mean)
                    metrics["std_committor_Q"] = float(Q_std)
                    metrics["avg_committor_Q_norm_mean"] = float(mean_use)
                    metrics["avg_committor_Q_norm_std"] = float(std_use)
                    q_active = term_committor[active_mask]
                    metrics["avg_committor_Q_tilde"] = float(q_active.mean().item()) if q_active.numel() else 0.0
                    print(
                        f"[Rewards] Added committor-progress: coef={committor_coef:.2e}, "
                        f"Q_mean={Q_mean:.4f} Q_std={Q_std:.4f} ema_beta={ema_beta:.2f} "
                        f"Q_tilde_mean={metrics['avg_committor_Q_tilde']:+.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added committor-progress term (coef={committor_coef:.2e}).")

            committor_tube_coef = float(getattr(config, "committor_tube_diversity_reward_weight", 0.0) or 0.0)
            if (not use_committor_connectivity) and committor_tube_coef != 0.0 and committor_tube_raw is not None:
                P_raw = np.asarray(committor_tube_raw, dtype=np.float32).reshape(-1)
                P_raw = np.where(np.isfinite(P_raw), P_raw, 0.0).astype(np.float32, copy=False)

                P_mean = float(np.mean(P_raw)) if P_raw.size else 0.0
                P_std = float(np.std(P_raw)) if P_raw.size else 0.0

                eps = 1e-8
                ema_beta = float(getattr(config, "committor_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = P_mean
                cur_m2 = float(np.mean(P_raw * P_raw)) if P_raw.size else 0.0

                if use_ema:
                    if committor_state.tube_norm_ema_m1 is None or committor_state.tube_norm_ema_m2 is None:
                        committor_state.tube_norm_ema_m1 = float(cur_m1)
                        committor_state.tube_norm_ema_m2 = float(cur_m2)
                    else:
                        committor_state.tube_norm_ema_m1 = float(ema_beta) * float(committor_state.tube_norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        committor_state.tube_norm_ema_m2 = float(ema_beta) * float(committor_state.tube_norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)
                    mean_use = float(committor_state.tube_norm_ema_m1)
                    var_use = float(max(float(committor_state.tube_norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(P_std)

                denom = float(std_use + eps) if (np.isfinite(std_use) and std_use > 0.0) else 1.0
                P_tilde = (P_raw - float(mean_use)) / denom
                P_tilde = np.where(np.isfinite(P_tilde), P_tilde, 0.0).astype(np.float32, copy=False)

                active_mask = metrics.get(
                    "active_mask",
                    torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                )
                if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                    active_mask = active_mask.to(torch_device)
                if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                    active_mask = active_mask.bool()

                p_tensor = torch.as_tensor(P_tilde, device=torch_device, dtype=torch.float32)
                term_committor_tube = committor_tube_coef * p_tensor
                term_committor_tube = torch.where(active_mask, term_committor_tube, torch.zeros_like(term_committor_tube))
                combined_rewards = combined_rewards + term_committor_tube

                try:
                    metrics["avg_committor_tube_raw"] = float(P_mean)
                    metrics["std_committor_tube_raw"] = float(P_std)
                    metrics["avg_committor_tube_norm_mean"] = float(mean_use)
                    metrics["avg_committor_tube_norm_std"] = float(std_use)
                    p_active = term_committor_tube[active_mask]
                    metrics["avg_committor_tube_tilde"] = float(p_active.mean().item()) if p_active.numel() else 0.0
                    print(
                        f"[Rewards] Added committor-tube diversity: coef={committor_tube_coef:.2e}, "
                        f"P_mean={P_mean:.4f} P_std={P_std:.4f} ema_beta={ema_beta:.2f} "
                        f"P_tilde_mean={metrics['avg_committor_tube_tilde']:+.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added committor-tube diversity term (coef={committor_tube_coef:.2e}).")

            committor_edge_coef = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0)
            if (not use_committor_connectivity) and committor_edge_coef != 0.0 and committor_edge_raw is not None:
                E_raw = np.asarray(committor_edge_raw, dtype=np.float32).reshape(-1)
                E_raw = np.where(np.isfinite(E_raw), E_raw, 0.0).astype(np.float32, copy=False)

                E_mean = float(np.mean(E_raw)) if E_raw.size else 0.0
                E_std = float(np.std(E_raw)) if E_raw.size else 0.0

                eps = 1e-8
                ema_beta = float(getattr(config, "committor_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = E_mean
                cur_m2 = float(np.mean(E_raw * E_raw)) if E_raw.size else 0.0

                if use_ema:
                    if committor_state.edge_norm_ema_m1 is None or committor_state.edge_norm_ema_m2 is None:
                        committor_state.edge_norm_ema_m1 = float(cur_m1)
                        committor_state.edge_norm_ema_m2 = float(cur_m2)
                    else:
                        committor_state.edge_norm_ema_m1 = float(ema_beta) * float(committor_state.edge_norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        committor_state.edge_norm_ema_m2 = float(ema_beta) * float(committor_state.edge_norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)
                    mean_use = float(committor_state.edge_norm_ema_m1)
                    var_use = float(max(float(committor_state.edge_norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(E_std)

                denom = float(std_use + eps) if (np.isfinite(std_use) and std_use > 0.0) else 1.0
                E_tilde = (E_raw - float(mean_use)) / denom
                E_tilde = np.where(np.isfinite(E_tilde), E_tilde, 0.0).astype(np.float32, copy=False)

                active_mask = metrics.get(
                    "active_mask",
                    torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                )
                if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                    active_mask = active_mask.to(torch_device)
                if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                    active_mask = active_mask.bool()

                e_tensor = torch.as_tensor(E_tilde, device=torch_device, dtype=torch.float32)
                term_committor_edge = committor_edge_coef * e_tensor
                term_committor_edge = torch.where(active_mask, term_committor_edge, torch.zeros_like(term_committor_edge))
                combined_rewards = combined_rewards + term_committor_edge

                try:
                    metrics["avg_committor_edge_raw"] = float(E_mean)
                    metrics["std_committor_edge_raw"] = float(E_std)
                    metrics["avg_committor_edge_norm_mean"] = float(mean_use)
                    metrics["avg_committor_edge_norm_std"] = float(std_use)
                    e_active = term_committor_edge[active_mask]
                    metrics["avg_committor_edge_tilde"] = float(e_active.mean().item()) if e_active.numel() else 0.0
                    print(
                        f"[Rewards] Added committor-edge novelty: coef={committor_edge_coef:.2e}, "
                        f"E_mean={E_mean:.4f} E_std={E_std:.4f} ema_beta={ema_beta:.2f} "
                        f"E_tilde_mean={metrics['avg_committor_edge_tilde']:+.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Added committor-edge novelty term (coef={committor_edge_coef:.2e}).")

            if committor_conn_coef != 0.0 and committor_conn_raw is not None:
                C_raw = np.asarray(committor_conn_raw, dtype=np.float32).reshape(-1)
                C_raw = np.where(np.isfinite(C_raw), C_raw, 0.0).astype(np.float32, copy=False)

                C_mean = float(np.mean(C_raw)) if C_raw.size else 0.0
                C_std = float(np.std(C_raw)) if C_raw.size else 0.0

                eps = 1e-8
                ema_beta = float(getattr(config, "committor_norm_ema_beta", 0.9) or 0.9)
                ema_beta = max(0.0, min(1.0, ema_beta))
                use_ema = ema_beta > 0.0

                cur_m1 = C_mean
                cur_m2 = float(np.mean(C_raw * C_raw)) if C_raw.size else 0.0

                if use_ema:
                    if committor_state.conn_norm_ema_m1 is None or committor_state.conn_norm_ema_m2 is None:
                        committor_state.conn_norm_ema_m1 = float(cur_m1)
                        committor_state.conn_norm_ema_m2 = float(cur_m2)
                    else:
                        committor_state.conn_norm_ema_m1 = float(ema_beta) * float(committor_state.conn_norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                        committor_state.conn_norm_ema_m2 = float(ema_beta) * float(committor_state.conn_norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)
                    mean_use = float(committor_state.conn_norm_ema_m1)
                    var_use = float(max(float(committor_state.conn_norm_ema_m2) - mean_use * mean_use, 0.0))
                    std_use = float(np.sqrt(var_use))
                else:
                    mean_use = float(cur_m1)
                    std_use = float(C_std)

                denom = float(std_use + eps) if (np.isfinite(std_use) and std_use > 0.0) else 1.0
                C_tilde = (C_raw - float(mean_use)) / denom
                C_tilde = np.where(np.isfinite(C_tilde), C_tilde, 0.0).astype(np.float32, copy=False)

                active_mask = metrics.get(
                    "active_mask",
                    torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                )
                if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                    active_mask = active_mask.to(torch_device)
                if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                    active_mask = active_mask.bool()

                c_tensor = torch.as_tensor(C_tilde, device=torch_device, dtype=torch.float32)
                term_committor_conn = committor_conn_coef * c_tensor
                term_committor_conn = torch.where(active_mask, term_committor_conn, torch.zeros_like(term_committor_conn))
                combined_rewards = combined_rewards + term_committor_conn

                try:
                    metrics["avg_committor_conn_raw"] = float(C_mean)
                    metrics["std_committor_conn_raw"] = float(C_std)
                    metrics["avg_committor_conn_norm_mean"] = float(mean_use)
                    metrics["avg_committor_conn_norm_std"] = float(std_use)
                    c_active = term_committor_conn[active_mask]
                    metrics["avg_committor_conn_tilde"] = float(c_active.mean().item()) if c_active.numel() else 0.0
                    print(
                        f"[Rewards] Added committor-connectivity: coef={committor_conn_coef:.2e}, "
                        f"C_mean={C_mean:.4f} C_std={C_std:.4f} ema_beta={ema_beta:.2f} "
                        f"C_tilde_mean={metrics['avg_committor_conn_tilde']:+.4f}",
                        flush=True,
                    )
                except Exception:
                    print(f"[Rewards] Added committor-connectivity term (coef={committor_conn_coef:.2e}).")

            # Optional off-action discount: downweight rewards for starts whose
            # requested torsion/CV targets cannot be realized by reset/steering.
            off_coef = float(getattr(config, "off_action_penalty_coef", 0.0) or 0.0)
            if off_coef > 0.0:
                reset_mae = metrics.get("reset_cv_mae", [float("nan")] * config.grp_size)
                try:
                    reset_mae_t = torch.as_tensor(reset_mae, device=torch_device, dtype=torch.float32)
                    reset_mae_t = torch.nan_to_num(reset_mae_t, nan=0.0, posinf=0.0, neginf=0.0)
                    reset_mae_t = torch.clamp(reset_mae_t, min=0.0)
                except Exception:
                    reset_mae_t = torch.zeros_like(launch_rewards)

                active_mask = metrics.get(
                    "active_mask",
                    torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                )
                if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                    active_mask = active_mask.to(torch_device)
                if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                    active_mask = active_mask.bool()

                reward_before = combined_rewards
                discount = torch.exp((-off_coef) * reset_mae_t)
                discount = torch.clamp(discount, min=0.0, max=1.0)
                discount = torch.where(active_mask, discount, torch.ones_like(discount))
                off_action_discount = discount
                combined_rewards = combined_rewards * off_action_discount
                term_off_action = combined_rewards - reward_before
                try:
                    mae_active = reset_mae_t[active_mask]
                    disc_active = off_action_discount[active_mask]
                    metrics["avg_reset_cv_mae"] = float(mae_active.mean().item()) if mae_active.numel() else 0.0
                    metrics["avg_off_action_discount"] = float(disc_active.mean().item()) if disc_active.numel() else 1.0
                    print(
                        f"[Rewards] Off-action discount (coef={off_coef:.2e}): "
                        f"reset_mae_mean={metrics['avg_reset_cv_mae']:.4f} rad, "
                        f"discount_mean={metrics['avg_off_action_discount']:.4f}"
                    )
                except Exception:
                    print(f"[Rewards] Off-action discount enabled (coef={off_coef:.2e}).")

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

            if diag_path is not None:
                try:
                    active_mask = metrics.get(
                        "active_mask",
                        torch.ones(config.grp_size, dtype=torch.bool, device=combined_rewards.device),
                    )
                    if torch.is_tensor(active_mask) and active_mask.device != combined_rewards.device:
                        active_mask = active_mask.to(combined_rewards.device)

                    traj_reward = reward_tensors.get("hill", torch.zeros_like(combined_rewards))
                    if traj_reward.device != combined_rewards.device:
                        traj_reward = traj_reward.to(combined_rewards.device)

                    launch_reward_raw = launch_rewards
                    if launch_reward_raw.device != combined_rewards.device:
                        launch_reward_raw = launch_reward_raw.to(combined_rewards.device)

                    vendi_w = metrics.get("vendi_per_walker", [0.0] * config.grp_size)
                    pca_w = metrics.get("pca_per_walker", [0.0] * config.grp_size)
                    state_w = metrics.get("state_per_walker", [0.0] * config.grp_size)

                    reset_mae = metrics.get("reset_cv_mae", [float("nan")] * config.grp_size)
                    reset_rmse = metrics.get("reset_cv_rmse", [float("nan")] * config.grp_size)
                    reset_maxabs = metrics.get("reset_cv_maxabs", [float("nan")] * config.grp_size)

                    term_loo_cpu = term_loo_vamp.detach().cpu().numpy().astype(np.float32)
                    term_sc_cpu = term_start_cluster.detach().cpu().numpy().astype(np.float32)
                    term_vb_cpu = term_vamp_bin.detach().cpu().numpy().astype(np.float32)
                    term_msm_cpu = term_msm_ig.detach().cpu().numpy().astype(np.float32)
                    term_off_cpu = term_off_action.detach().cpu().numpy().astype(np.float32)
                    off_disc_cpu = off_action_discount.detach().cpu().numpy().astype(np.float32)
                    combined_cpu = combined_rewards.detach().cpu().numpy().astype(np.float32)
                    launch_cpu = launch_reward_raw.detach().cpu().numpy().astype(np.float32)
                    traj_cpu = traj_reward.detach().cpu().numpy().astype(np.float32)
                    valid_cpu = valid_mask.detach().cpu().numpy().astype(bool)
                    active_cpu = active_mask.detach().cpu().numpy().astype(bool)
                    msm_raw_cpu = None
                    if msm_ig_rewards is not None:
                        try:
                            msm_raw_cpu = np.asarray(msm_ig_rewards, dtype=np.float32)
                        except Exception:
                            msm_raw_cpu = None

                    with diag_path.open("a", encoding="utf-8") as handle:
                        for w in range(int(config.grp_size)):
                            rec = {
                                "epoch": int(ep + 1),
                                "walker": int(w),
                                "active": bool(active_cpu[w]),
                                "valid": bool(valid_cpu[w]),
                                "reward": {
                                    "combined": float(combined_cpu[w]),
                                    "loo_vamp": float(term_loo_cpu[w]),
                                    "start_cluster": float(term_sc_cpu[w]),
                                    "vamp_bin": float(term_vb_cpu[w]),
                                    "msm_ig": float(term_msm_cpu[w]),
                                    "msm_ig_raw": float(msm_raw_cpu[w]) if msm_raw_cpu is not None and w < len(msm_raw_cpu) else None,
                                    "off_action_penalty": float(term_off_cpu[w]),
                                    "off_action_discount": float(off_disc_cpu[w]),
                                    "launch_raw": float(launch_cpu[w]),
                                    "traj_hill": float(traj_cpu[w]),
                                    "traj_vendi": float(vendi_w[w]) if w < len(vendi_w) else 0.0,
                                    "traj_pca": float(pca_w[w]) if w < len(pca_w) else 0.0,
                                    "traj_state": float(state_w[w]) if w < len(state_w) else 0.0,
                                },
                                "deviation": {
                                    "torsion_sample_vs_mode_phi_mae": float(mode_phi_mae[w]) if mode_phi_mae is not None else None,
                                    "torsion_sample_vs_mode_psi_mae": float(mode_psi_mae[w]) if mode_psi_mae is not None else None,
                                    "torsion_sample_vs_mode_phi_psi_mae": float(mode_phi_psi_mae[w]) if mode_phi_psi_mae is not None else None,
                                    "reset_cv_mae": float(reset_mae[w]) if w < len(reset_mae) else None,
                                    "reset_cv_rmse": float(reset_rmse[w]) if w < len(reset_rmse) else None,
                                    "reset_cv_maxabs": float(reset_maxabs[w]) if w < len(reset_maxabs) else None,
                                },
                            }
                            handle.write(json.dumps(rec) + "\n")
                except Exception as exc_diag:
                    print(f"[Diagnostics] Warning: failed to write generative diagnostics log ({exc_diag})")

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
                print(f"[Policy] Synced old policy at epoch {ep+1}")
        else:
            update_stats = {}

        # ========== Per-Trajectory Path Stats Summary ==========
        path_epoch_summary: Dict[str, Any] = {
            "n_trajs_valid": 0,
            "n_hit_a": 0,
            "n_hit_b": 0,
            "n_transition": 0,
            "min_metric_a": float("inf"),
            "min_metric_b": float("inf"),
        }
        try:
            path_stats_lines: List[str] = []
            path_stats_lines.append(f"\n[Path Stats - Epoch {ep+1}] Per-trajectory summary:")
            boundary_plugin_active = False
            try:
                boundary_plugin_active = bool(str(getattr(config, "path_boundary_plugin", "") or "").strip()) and bool(
                    boundary_plugin_fn is not None and (not boundary_plugin_failed)
                )
            except Exception:
                boundary_plugin_active = False
            col_a = "minMetric_A" if boundary_plugin_active else "minRMSD_A"
            col_b = "minMetric_B" if boundary_plugin_active else "minRMSD_B"
            show_q_prog = False
            show_conn = False
            try:
                show_q_prog = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0) != 0.0
            except Exception:
                show_q_prog = False
            try:
                show_conn = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0) != 0.0
            except Exception:
                show_conn = False

            header = f"  {'Traj':>4} | {col_a:>9} | {col_b:>9} | {'hitA':>4} | {'hitB':>4} | "
            if show_q_prog:
                header += f"{'Q_prog':>6} | "
            if show_conn:
                header += f"{'Conn':>6} | "
            header += "states_visited"
            path_stats_lines.append(header)
            path_stats_lines.append("  " + "-" * 110)
            
            n_trajs_valid = 0
            n_hit_a = 0
            n_hit_b = 0
            n_transition = 0  # hit both A and B
            min_rmsd_a_all = float("inf")
            min_rmsd_b_all = float("inf")
            global_state_counts: Dict[int, int] = {}  # Global state visit counts
            
            for i in range(config.grp_size):
                ra = traj_rmsd_a[i] if i < len(traj_rmsd_a) and traj_rmsd_a[i] is not None else None
                rb = traj_rmsd_b[i] if i < len(traj_rmsd_b) and traj_rmsd_b[i] is not None else None
                ia = traj_in_a[i] if i < len(traj_in_a) and traj_in_a[i] is not None else None
                ib = traj_in_b[i] if i < len(traj_in_b) and traj_in_b[i] is not None else None
                q_prog = Q_vals[i] if show_q_prog and 'Q_vals' in dir() and Q_vals is not None and i < len(Q_vals) else 0.0
                c_raw = (
                    float(committor_conn_raw[i])
                    if (committor_conn_raw is not None and i < len(committor_conn_raw))
                    else 0.0
                )
                
                # Get VAMP labels for this trajectory
                vamp_labels = None
                if 'traj_vamp_labels_best' in dir() and traj_vamp_labels_best is not None:
                    if i < len(traj_vamp_labels_best) and traj_vamp_labels_best[i] is not None:
                        vamp_labels = traj_vamp_labels_best[i]
                
                # Compute min RMSD
                min_ra = float(np.min(ra)) if ra is not None and ra.size > 0 else float("nan")
                min_rb = float(np.min(rb)) if rb is not None and rb.size > 0 else float("nan")
                
                # Check boundary hits
                hit_a = bool(np.any(ia)) if ia is not None else False
                hit_b = bool(np.any(ib)) if ib is not None else False
                
                # Compute visited states with counts
                states_visited_str = "-"
                if vamp_labels is not None and vamp_labels.size > 0:
                    unique_states, counts = np.unique(vamp_labels, return_counts=True)
                    # Sort by count (descending)
                    sort_idx = np.argsort(-counts)
                    unique_states = unique_states[sort_idx]
                    counts = counts[sort_idx]
                    # Format as "state:count" pairs
                    state_strs = []
                    for s, c in zip(unique_states[:5], counts[:5]):  # Show top 5
                        state_strs.append(f"{int(s)}:{int(c)}")
                        # Track global counts
                        global_state_counts[int(s)] = global_state_counts.get(int(s), 0) + int(c)
                    if len(unique_states) > 5:
                        state_strs.append(f"...+{len(unique_states)-5}")
                    states_visited_str = ", ".join(state_strs)
                
                # Track stats
                if np.isfinite(min_ra) or np.isfinite(min_rb):
                    n_trajs_valid += 1
                if hit_a:
                    n_hit_a += 1
                if hit_b:
                    n_hit_b += 1
                if hit_a and hit_b:
                    n_transition += 1
                if np.isfinite(min_ra):
                    min_rmsd_a_all = min(min_rmsd_a_all, min_ra)
                if np.isfinite(min_rb):
                    min_rmsd_b_all = min(min_rmsd_b_all, min_rb)
                
                row = f"  {i:4d} | {min_ra:9.4f} | {min_rb:9.4f} | {str(hit_a):>4} | {str(hit_b):>4} | "
                if show_q_prog:
                    row += f"{float(q_prog):6.3f} | "
                if show_conn:
                    row += f"{float(c_raw):6.3f} | "
                row += f"{states_visited_str}"
                path_stats_lines.append(row)
            
            path_stats_lines.append("  " + "-" * 110)
            path_stats_lines.append(
                f"  Summary: {n_trajs_valid} valid trajs, {n_hit_a} hit A, {n_hit_b} hit B, "
                f"{n_transition} transitions (A→B or B→A)"
            )
            if np.isfinite(min_rmsd_a_all):
                unit = "" if boundary_plugin_active else " nm"
                path_stats_lines.append(f"  Global min A metric: {min_rmsd_a_all:.4f}{unit}")
            if np.isfinite(min_rmsd_b_all):
                unit = "" if boundary_plugin_active else " nm"
                path_stats_lines.append(f"  Global min B metric: {min_rmsd_b_all:.4f}{unit}")
            if committor_state.boundary_vamp_state_valid:
                path_stats_lines.append(
                    f"  Boundary states: A=state_{committor_state.boundary_vamp_a_state}, B=state_{committor_state.boundary_vamp_b_state}"
                )
            else:
                plugin_spec = str(getattr(config, "path_boundary_plugin", "") or "").strip()
                if boundary_plugin_active:
                    boundary_mode_msg = "plugin"
                else:
                    boundary_mode_msg = "RMSD cutoff"
                    if plugin_spec:
                        boundary_mode_msg += " (plugin inactive)"
                path_stats_lines.append(f"  Boundary mode: {boundary_mode_msg}")
            
            # Global state distribution summary
            if global_state_counts:
                total_frames = sum(global_state_counts.values())
                sorted_states = sorted(global_state_counts.items(), key=lambda x: -x[1])
                state_dist_str = ", ".join(
                    f"s{s}:{c} ({100.0*c/total_frames:.1f}%)" 
                    for s, c in sorted_states[:8]
                )
                if len(sorted_states) > 8:
                    state_dist_str += f", ...+{len(sorted_states)-8} more"
                path_stats_lines.append(f"  Global state distribution: {state_dist_str}")
            
            print("\n".join(path_stats_lines), flush=True)

            path_epoch_summary = {
                "n_trajs_valid": int(n_trajs_valid),
                "n_hit_a": int(n_hit_a),
                "n_hit_b": int(n_hit_b),
                "n_transition": int(n_transition),
                "min_metric_a": float(min_rmsd_a_all),
                "min_metric_b": float(min_rmsd_b_all),
                "boundary_plugin_active": bool(boundary_plugin_active),
            }
        except Exception as exc_path_stats:
            print(f"[Path Stats] Warning: failed to print path stats ({exc_path_stats})", flush=True)

        # ========== Committor Graph Status (end of epoch) ==========
        try:
            if bool(getattr(config, "path_sampling", False)):
                comm_state_mode = str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
                tube_state_mode = str(getattr(config, "committor_tube_state_mode", "model") or "model").strip().lower()
                force_rmsd_proxy = bool(getattr(config, "committor_use_rmsd_proxy", False))
                w_prog = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
                w_tube = float(getattr(config, "committor_tube_diversity_reward_weight", 0.0) or 0.0)
                w_edge = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0)
                w_conn = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
                plugin_spec_s = str(getattr(config, "path_boundary_plugin", "") or "").strip()
                plugin_loaded = bool(plugin_spec_s) and bool(boundary_plugin_fn is not None and (not boundary_plugin_failed))

                model_ok = (
                    (comm_state_mode == "vamp" and committor_state.vamp_model is not None)
                    or (comm_state_mode != "vamp" and committor_state.dga_model is not None)
                )
                dga_deg = None
                dga_qstd = None
                dga_K = None
                if comm_state_mode != "vamp" and committor_state.dga_model is not None:
                    try:
                        dga_deg = bool(getattr(committor_state.dga_model, "degenerate", False))
                    except Exception:
                        dga_deg = None
                    try:
                        dga_qstd = float(getattr(committor_state.dga_model, "q_domain_std", float("nan")))
                    except Exception:
                        dga_qstd = None
                    try:
                        dga_K = int(getattr(committor_state.dga_model, "coeffs", np.zeros((0,), dtype=np.float32)).shape[0])
                    except Exception:
                        dga_K = None

                tube_cells = None
                tube_visited = None
                if committor_state.tube_counts is not None:
                    try:
                        tube_cells = int(np.asarray(committor_state.tube_counts).size)
                        tube_visited = int(np.count_nonzero(np.asarray(committor_state.tube_counts) > 0.0))
                    except Exception:
                        tube_cells = None
                        tube_visited = None

                edge_n = None
                if committor_state.edge_counts is not None:
                    try:
                        if isinstance(committor_state.edge_counts, dict):
                            edge_n = int(len(committor_state.edge_counts))
                        else:
                            edge_n = int(np.count_nonzero(np.asarray(committor_state.edge_counts) > 0.0))
                    except Exception:
                        edge_n = None

                conn_cells = None
                conn_edges = None
                conn_a = None
                conn_b = None
                conn_dist = None
                conn_gap = None
                conn_components = None
                if committor_state.conn_adj is not None:
                    try:
                        conn_cells = int(len(committor_state.conn_adj))
                        conn_edges = int(len(committor_state.conn_edges))
                        conn_a = int(len(committor_state.conn_a_cells))
                        conn_b = int(len(committor_state.conn_b_cells))
                        if committor_state.conn_a_cells and committor_state.conn_b_cells:
                            dist_ab = _conn_shortest_path_len(
                                adj=committor_state.conn_adj,
                                sources=list(committor_state.conn_a_cells),
                                targets=set(committor_state.conn_b_cells),
                                overlay=None,
                            )
                            conn_dist = None if not np.isfinite(dist_ab) else int(dist_ab)
                            reach_a = _conn_reachable_nodes(adj=committor_state.conn_adj, sources=list(committor_state.conn_a_cells), overlay=None)
                            reach_b = _conn_reachable_nodes(adj=committor_state.conn_adj, sources=list(committor_state.conn_b_cells), overlay=None)
                            g_i, _amax, _bmin = _conn_bin_gap(
                                reachable_a=reach_a,
                                reachable_b=reach_b,
                                K=int(max(1, K_tube)),
                                M=int(tube_bins),
                            )
                            conn_gap = int(g_i)
                        # Count connected components among nodes that have any edge (degree>0) or are in A/B.
                        deg_nodes = set()
                        for u, nbrs in enumerate(committor_state.conn_adj):
                            if nbrs:
                                deg_nodes.add(int(u))
                        deg_nodes |= set(int(x) for x in committor_state.conn_a_cells)
                        deg_nodes |= set(int(x) for x in committor_state.conn_b_cells)
                        seen = set()
                        comps = 0
                        for s in list(deg_nodes):
                            if s in seen:
                                continue
                            comps += 1
                            stack = [s]
                            seen.add(s)
                            while stack:
                                cur = stack.pop()
                                for v in committor_state.conn_adj[int(cur)]:
                                    vi = int(v)
                                    if vi in deg_nodes and vi not in seen:
                                        seen.add(vi)
                                        stack.append(vi)
                        conn_components = int(comps)
                    except Exception:
                        pass

                print(
                    f"[Committor Graph - Epoch {ep+1}] mode={comm_state_mode} model_ok={int(bool(model_ok))} "
                    f"use_rmsd_proxy={int(force_rmsd_proxy)} tube_state_mode={tube_state_mode} "
                    f"weights(prog={w_prog:.2e}, tube={w_tube:.2e}, edge={w_edge:.2e}, conn={w_conn:.2e})",
                    flush=True,
                )
                if plugin_spec_s:
                    print(
                        f"[Committor Graph - Epoch {ep+1}] boundary_plugin={plugin_spec_s} loaded={int(bool(plugin_loaded))}",
                        flush=True,
                    )
                if dga_K is not None or dga_deg is not None or dga_qstd is not None:
                    print(
                        f"[Committor Graph - Epoch {ep+1}] dga: K={dga_K} degenerate={dga_deg} q_domain_std={dga_qstd} "
                        f"globals={int(bool(getattr(committor_state.dga_model, 'include_global_tokens', False)))} "
                        f"feat_dim_total={getattr(committor_state.dga_model, 'feature_dim_total', None)}",
                        flush=True,
                    )
                if tube_cells is not None:
                    print(
                        f"[Committor Graph - Epoch {ep+1}] tube: cells={tube_cells} visited={tube_visited} "
                        f"total_w={float(committor_state.tube_total_weight):.1f} counted_keys={len(committor_state.tube_counted_keys)}",
                        flush=True,
                    )
                if edge_n is not None:
                    print(
                        f"[Committor Graph - Epoch {ep+1}] edge: edges_visited={edge_n} "
                        f"total_count={float(committor_state.edge_total_count):.1f} counted_keys={len(committor_state.edge_counted_keys)}",
                        flush=True,
                    )
                if conn_cells is not None:
                    dist_s = "inf" if conn_dist is None else str(int(conn_dist))
                    gap_s = "?" if conn_gap is None else str(int(conn_gap))
                    comps_s = "?" if conn_components is None else str(int(conn_components))
                    print(
                        f"[Committor Graph - Epoch {ep+1}] conn: cells={conn_cells} edges={conn_edges} "
                        f"A_cells={conn_a} B_cells={conn_b} A→B_dist={dist_s} gap={gap_s} components={comps_s} "
                        f"counted_keys={len(committor_state.conn_counted_keys)}",
                        flush=True,
                    )

                # Also print a compact epoch-level path hit summary (for quick sanity checks).
                try:
                    print(
                        f"[Committor Graph - Epoch {ep+1}] epoch_hits: hitA={path_epoch_summary.get('n_hit_a')} "
                        f"hitB={path_epoch_summary.get('n_hit_b')} transitions={path_epoch_summary.get('n_transition')}",
                        flush=True,
                    )
                except Exception:
                    pass
        except Exception as exc_graph:
            print(f"[Committor Graph] Warning: failed to print graph status ({exc_graph})", flush=True)

        print(f"[Epoch {ep+1}] Phase E: Logging & checkpointing...")
        if "avg_hill" in metrics:
            print(f"Epoch {ep+1} Complete. Avg Reward: {float(metrics.get('avg_hill', 0.0)):.4f}")
        else:
            print(
                f"Epoch {ep+1} Complete. Avg Reward: 0.0000 "
                "(reward not computed this epoch; projector/reward state not ready yet)"
            )

        if rollout_token_entropy is not None and rollout_token_entropy.size > 0:
            try:
                ent_mean = float(np.mean(rollout_token_entropy))
                ent_std = float(np.std(rollout_token_entropy))
                print(f"[TorsionAR][Epoch {ep+1}] Token entropy (torsions, excl ω; nats/token): mean={ent_mean:.3f}, std={ent_std:.3f}")
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
            ckpt_manager.save(ep, agent, agent_old, policy_opts, {}, None, vars(args))
            try:
                _save_torsion_ar_policy(ep + 1)
            except Exception as exc_ar_save:
                print(f"[TorsionAR] Warning: failed to save torsion AR checkpoint: {exc_ar_save}")

        if config.plot_freq > 0 and (ep + 1) % config.plot_freq == 0:
            all_cv_samples = [item[0] for epoch_data in all_epoch_data for item in epoch_data]
            all_cv_array = np.asarray(all_cv_samples, dtype=np.float32)
            plot_ramachandran_by_residue(all_cv_array, envs[0].cv_names, f"ramachandran_ep{ep+1:03d}.png")

            try:
                plot_analysis_summary(
                    all_epoch_data[round_epoch_data_start:],
                    None,  # pca_scorer removed
                    envs[0].cv_names,
                    config.pdb,
                    f"analysis_summary_ep{ep+1:03d}.png",
                    feature_builder=feature_builder,
                    vamp_bin_counts=vamp_bin_mgr.counts,
                    vamp_bin_clip=float(getattr(config, "vamp_bin_clip", 3.0) or 3.0),
                    vamp_bin_total=int(vamp_bin_mgr.total),
                    pdb_a=str(getattr(config, "pdb_a", None) or getattr(config, "pdb", None)),
                    pdb_b=str(getattr(config, "pdb_b", None) or ""),
                )
            except Exception as exc_plot:
                print(f"[Plot] Warning: summary plot failed: {exc_plot}")

        elapsed = time.time() - epoch_start_time
        print(f"[Epoch {ep+1}] Complete in {elapsed:.1f}s.")

        gc.collect()
