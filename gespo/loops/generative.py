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
from gespo.training.vamp_stats import SufficientStatistics
from gespo.training.components.geom2vec_manager import Geom2VecManager, align_states_by_overlap
from gespo.rewards.loo_vamp import LOOVAMPRewards
from gespo.training.components.retrieval_manager import RetrievalManager
from gespo.training.components.policy_io import save_torsion_ar_policy
from gespo.loops._helpers import iter_history_trajs as _shared_iter_history_trajs
from gespo.utils.log import epoch_phase, warn as log_warn
from gespo.analysis.metrics import compute_globals_from_coords as _compute_globals_impl
from gespo.rewards.vamp_bin import VampBinBonus
from gespo.training.components.loo_vamp_seeder import ensure_loo_vamp_history_stats as _ensure_loo_vamp_impl
from gespo.training.components.retrieval_labels import refresh_retrieval_cache_labels as _refresh_labels_impl
from gespo.rewards.msm_ig import MSMIGBonus
from gespo.projections.vamp_refresh import refresh_geom2vec_vamp as _refresh_geom2vec_vamp_impl

import argparse
import gc
import json


def _log_ligand_pocket_stats(
    epoch: int,
    traj_dcd_paths,
    sim_top_path: str,
    *,
    lig_names=("UNK", "BEN", "LIG"),
    pocket_resname="ASP",
    pocket_seq_pos=170,
    bound_cutoff_A=6.0,
    dissociated_cutoff_A=15.0,
):
    """Log per-walker ligand-pocket distance stats after each epoch.

    Computes minimum heavy-atom distance between ligand and Asp189 from DCD files.
    Prints a concise summary: mean/min/max distance, % bound, % dissociated.
    """
    try:
        import mdtraj as _md
    except ImportError:
        return
    if not traj_dcd_paths or not sim_top_path:
        return
    import os as _os
    if not _os.path.isfile(sim_top_path):
        return

    try:
        ref = _md.load(sim_top_path)
        top = ref.topology
        prot_res = [r for r in top.residues if r.is_protein]
        if pocket_seq_pos >= len(prot_res):
            return
        pocket_resseq = prot_res[pocket_seq_pos].resSeq
        lig_heavy = np.array([a.index for a in top.atoms
                              if a.residue.name in lig_names and a.element.symbol != "H"], dtype=int)
        pocket_heavy = np.array([a.index for a in top.atoms
                                 if a.residue.resSeq == pocket_resseq and a.element.symbol != "H"], dtype=int)
        if lig_heavy.size == 0 or pocket_heavy.size == 0:
            return
        pairs = np.array([(i, j) for i in lig_heavy for j in pocket_heavy])
    except Exception:
        return

    all_dists = []
    per_walker = []
    for w_idx, dcd_path in enumerate(traj_dcd_paths):
        if dcd_path is None or not _os.path.isfile(str(dcd_path)):
            per_walker.append(None)
            continue
        try:
            traj = _md.load(str(dcd_path), top=sim_top_path)
            dists_nm = _md.compute_distances(traj, pairs)
            min_dists_A = dists_nm.min(axis=1) * 10.0
            all_dists.append(min_dists_A)
            per_walker.append(min_dists_A)
        except Exception:
            per_walker.append(None)

    if not all_dists:
        return

    all_d = np.concatenate(all_dists)
    n_total = len(all_d)
    pct_bound = np.sum(all_d < bound_cutoff_A) / n_total * 100
    pct_dissoc = np.sum(all_d > dissociated_cutoff_A) / n_total * 100

    print(
        f"[Ligand-Pocket - Epoch {epoch}] "
        f"BEN-Asp189 min dist (Å): mean={np.mean(all_d):.1f}, "
        f"median={np.median(all_d):.1f}, "
        f"min={np.min(all_d):.1f}, max={np.max(all_d):.1f} | "
        f"bound(<{bound_cutoff_A}Å)={pct_bound:.1f}%, "
        f"dissoc(>{dissociated_cutoff_A}Å)={pct_dissoc:.1f}%",
        flush=True,
    )

    # Per-walker breakdown
    walker_strs = []
    for w_idx, wd in enumerate(per_walker):
        if wd is None:
            walker_strs.append(f"W{w_idx:02d}=N/A")
        else:
            walker_strs.append(
                f"W{w_idx:02d}: mean={np.mean(wd):.1f} min={np.min(wd):.1f} max={np.max(wd):.1f}"
            )
    for ws in walker_strs:
        print(f"  {ws}", flush=True)
import math
import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from gespo.utils.tmem63_metrics import (
    ACTIVE_TMEM63_RESIDUE_CONFIG,
    TMEM63_GLOBAL_TOKEN_NAMES,
    build_tmem63_token_indices,
)
from gespo.utils.ppi_metrics import PPI_GLOBAL_TOKEN_NAMES, PPITokenIndices, build_ppi_token_indices, compute_ppi_globals_from_frames, set_native_contacts
from gespo.utils.geometry import _dihedral, calculate_rmsd, find_all_backbone_dihedrals
from gespo.utils.openmm_utils import resolve_device
from gespo.utils.plotting import plot_analysis_summary, plot_ramachandran_by_residue
from gespo.utils.selection import infer_protein_atom_indices
from gespo.training.checkpointing import CheckpointManager
from gespo.training.features import FeatureBuilder
from gespo.projections.geom2vec import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector, infer_and_cache
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
            import copy

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
        ref_full = md.load(config.pdb)
        try:
            prot_idx = infer_protein_atom_indices(ref_full.topology, include_caps=True, allow_empty=True)
            if prot_idx is not None and len(prot_idx) > 0 and len(prot_idx) < int(ref_full.n_atoms):
                ref_top = ref_full.atom_slice(np.asarray(prot_idx, dtype=np.int64)).topology
            else:
                ref_top = ref_full.topology
        except Exception:
            ref_top = ref_full.topology
        res_codes = [three_to_one.get(res.name.upper(), "X") for res in ref_top.residues]
        residue_types_1d = np.array([aa_to_idx.get(code, fallback_idx) for code in res_codes], dtype=np.int64)
    except Exception as exc_rt:
        print(f"[TorsionAR] Warning: failed to build residue type map from PDB: {exc_rt}")
        residue_types_1d = np.zeros((env_ref.n_residues,), dtype=np.int64)

    # Note: residue_types_1d may be larger than env.n_residues when the AR model
    # includes ligand torsion slots. Final size is determined after AR checkpoint
    # is loaded (see num_residues below). For now, keep the full PDB-derived types.
    _pdb_residue_types = residue_types_1d  # save for later resizing

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
                global_token_names = list(ADK_GLOBAL_TOKEN_NAMES[1:])
            elif n_global_tokens == len(ADK_GLOBAL_TOKEN_NAMES):
                global_token_names = list(ADK_GLOBAL_TOKEN_NAMES)
            else:
                global_token_names = [f"global_{i}" for i in range(max(0, n_global_tokens))]
        if num_residues < env_ref.n_residues:
            raise RuntimeError(
                f"TorsionAR checkpoint has fewer residues ({num_residues}) than env ({env_ref.n_residues})."
            )
        if num_residues > env_ref.n_residues:
            print(
                f"[TorsionAR] Model has {num_residues} residues, env has {env_ref.n_residues}; "
                f"extra {num_residues - env_ref.n_residues} residue(s) are ligand torsion slots."
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
    # Finalize residue_types to match the AR model's num_residues.
    # ------------------------------------------------------------------
    residue_types_1d = _pdb_residue_types
    if residue_types_1d.shape[0] < num_residues:
        pad = np.full((num_residues - residue_types_1d.shape[0],), fallback_idx, dtype=np.int64)
        residue_types_1d = np.concatenate([residue_types_1d, pad])
    elif residue_types_1d.shape[0] > num_residues:
        residue_types_1d = residue_types_1d[:num_residues]
    residue_types_tensor = torch.as_tensor(residue_types_1d, device=torch_device, dtype=torch.long)
    residue_types_batch = residue_types_tensor.unsqueeze(0).expand(config.grp_size, -1)

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

    # Pad angle_exists to match AR model's num_residues (extra slots = ligand rotatable bonds).
    # NEW LAYOUT: each ligand rotatable dihedral occupies its OWN row in angle_exists,
    # with only angle index 0 valid (others masked off).
    if num_residues > env_ref.n_residues:
        n_extra = num_residues - env_ref.n_residues
        lig_mask = np.zeros((n_extra, n_angles), dtype=bool)
        # Prefer SMILES-based RDKit detection.
        _lig_smiles = str(getattr(args, "ligand_smiles", "") or "")
        if _lig_smiles:
            from gespo.utils.geometry import find_ligand_rotatable_dihedrals_smiles
            try:
                lig_dihedrals = find_ligand_rotatable_dihedrals_smiles(env_ref.md_topology, _lig_smiles)
            except Exception:
                from gespo.utils.geometry import find_ligand_rotatable_dihedrals
                lig_dihedrals = find_ligand_rotatable_dihedrals(env_ref.md_topology)
        else:
            from gespo.utils.geometry import find_ligand_rotatable_dihedrals
            lig_dihedrals = find_ligand_rotatable_dihedrals(env_ref.md_topology)
        # One position per dihedral; angle 0 is valid at each ligand position.
        n_lig_positions = min(len(lig_dihedrals), n_extra)
        for pos_idx in range(n_lig_positions):
            lig_mask[pos_idx, 0] = True
        angle_exists_np = np.concatenate([angle_exists_np, lig_mask], axis=0)

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
    # Per-epoch flag: True = include in VAMP training but EXCLUDE from retrieval pool.
    # Use case: unfiltered prior-round data for better VAMP CVs without polluting
    # retrieval with dissociated/unbound starting frames.
    all_epoch_retrieval_exclude: List[bool] = []
    ep_start = 0
    pca_state = None


    # Retrieval cache manager for torsion-query replay starts
    retrieval_cache_max = int(getattr(config, "retrieval_cache_max_frames", 0) or 0)
    if retrieval_cache_max <= 0:
        retrieval_cache_max = int(getattr(config, "max_pca_train", 0) or 0)
    
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

    # Precompute CA mapping (and optional ADK or TMEM indices) for fast global-metric extraction
    # from protein coords.
    retrieval_ca_local: np.ndarray = np.zeros((0,), dtype=np.int64)
    retrieval_ref_ca_nm: np.ndarray = np.zeros((0, 3), dtype=np.float64)
    retrieval_protein_indices: np.ndarray = np.zeros((0,), dtype=np.int64)
    adk_token_idx = None
    tmem63_token_idx = None
    tmem63_ref_rmsd_nm = None
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
        if global_dim == len(ADK_GLOBAL_TOKEN_NAMES) or any(n in set(ADK_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                adk_token_idx = build_adk_token_indices(env_ref.md_topology, protein_indices=retrieval_protein_indices)
            except Exception:
                adk_token_idx = None
        if global_dim == len(TMEM63_GLOBAL_TOKEN_NAMES) or any(n in set(TMEM63_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                # ACTIVE_TMEM63_RESIDUE_CONFIG (gespo/utils/tmem63_metrics.py) is the
                # single switch point for which TMEM63 construct GESPO trains against
                # -- reassign it there, not here, to point at TMEM63A or a custom config.
                tmem63_token_idx = build_tmem63_token_indices(
                    env_ref.md_topology,
                    ACTIVE_TMEM63_RESIDUE_CONFIG,
                    token_names=[n for n in global_token_names if n in set(TMEM63_GLOBAL_TOKEN_NAMES)],
                    protein_indices=retrieval_protein_indices,
                )
                if tmem63_token_idx.rmsd_atom_idx is not None:
                    ref_full_nm = np.asarray(env_ref._ref_positions_nm_full, dtype=np.float64)
                    tmem63_ref_rmsd_nm = ref_full_nm[
                        retrieval_protein_indices[tmem63_token_idx.rmsd_atom_idx]
                    ]
            except Exception as exc:
                log_warn(
                    f"[generative] build_tmem63_token_indices failed, TMEM63 global "
                    f"tokens will be all-zero this run: {type(exc).__name__}: {exc}"
                )
                tmem63_token_idx = None
        if any(n in set(PPI_GLOBAL_TOKEN_NAMES) for n in global_token_names):
            try:
                ppi_token_idx = build_ppi_token_indices(env_ref.md_topology, protein_indices=retrieval_protein_indices)
                # Populate native contacts from the reference CA coords (endpoint A or equilibrated).
                if retrieval_ref_ca_nm.shape[0] > 0:
                    ppi_token_idx = set_native_contacts(ppi_token_idx, retrieval_ref_ca_nm)
            except Exception:
                ppi_token_idx = None
    except Exception:
        retrieval_ca_local = np.zeros((0,), dtype=np.int64)
        retrieval_ref_ca_nm = np.zeros((0, 3), dtype=np.float64)
        retrieval_protein_indices = np.zeros((0,), dtype=np.int64)
        adk_token_idx = None
        tmem63_token_idx = None
        tmem63_ref_rmsd_nm = None
        ppi_token_idx = None

    # Auto-detect pocket indices for pocket_rmsd / n_contacts global tokens.
    _pocket_ca_local: Optional[np.ndarray] = None
    _pocket_ref_ca_nm: Optional[np.ndarray] = None
    _prot_heavy_local: Optional[np.ndarray] = None
    _lig_heavy_local: Optional[np.ndarray] = None
    print(f"[POCKET_DEBUG] global_token_names={global_token_names}, checking pocket condition...", flush=True)
    if any(n in ("pocket_rmsd_nm", "n_contacts_scaled", "n_contacts", "lig_pocket_dist_nm") for n in (global_token_names or [])):
        print(f"[Pocket] Pocket globals requested. Detecting pocket residues...", flush=True)
        try:
            import mdtraj as _md_pocket
            _POCKET_NM = 0.5
            _LIG_NAMES = {"UNK", "BEN", "LIG"}
            _AA_P = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP",
                     "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","ACE","NME"}
            # Prefer an explicit pocket reference PDB (e.g. holo structure when running apo GESPO),
            # then fall back to simulation_topology.pdb, then to the main --pdb.
            _pocket_ref = str(getattr(config, "pocket_ref_pdb", "") or "")
            if _pocket_ref and os.path.isfile(_pocket_ref):
                _sim_top_path = _pocket_ref
                print(f"[Pocket] Using pocket_ref_pdb: {_sim_top_path}", flush=True)
            else:
                _sim_top_path = os.path.join(str(getattr(config, "snapshot_dir", "")), "simulation_topology.pdb")
                if not os.path.isfile(_sim_top_path):
                    _sim_top_path = str(getattr(config, "pdb", ""))
            _ref_traj = _md_pocket.load(_sim_top_path)
            ref_xyz = _ref_traj.xyz[0]
            # Identify ligand and protein residues in full topology
            _full_prot_res = [r for r in _ref_traj.topology.residues if r.name.upper() in _AA_P]
            _lig_heavy_full = np.array([a.index for a in _ref_traj.topology.atoms
                                        if a.residue.name in _LIG_NAMES and a.element.symbol != "H"], dtype=np.int64)
            print(f"[Pocket] {_lig_heavy_full.size} lig heavy, {len(_full_prot_res)} prot res, "
                  f"retrieval_ca_local.size={retrieval_ca_local.size}", flush=True)
            if _lig_heavy_full.size > 0 and retrieval_ca_local.size > 0:
                ref_lig = ref_xyz[_lig_heavy_full]
                pocket_res_mask = np.zeros(len(_full_prot_res), dtype=bool)
                for ri, r in enumerate(_full_prot_res):
                    r_h = [a.index for a in r.atoms if a.element.symbol != "H"]
                    if not r_h: continue
                    d = np.min(np.linalg.norm(ref_xyz[r_h][:, None, :] - ref_lig[None, :, :], axis=-1))
                    if d < _POCKET_NM:
                        pocket_res_mask[ri] = True
                # Map to protein-only CA frame: retrieval_ca_local[i] is the i-th CA
                # pocket_res_mask[i] is True if the i-th protein residue is in the pocket
                n_ca = min(len(pocket_res_mask), retrieval_ca_local.size, retrieval_ref_ca_nm.shape[0])
                pocket_mask_ca = pocket_res_mask[:n_ca]
                if pocket_mask_ca.any():
                    _pocket_ca_local = retrieval_ca_local[pocket_mask_ca].copy()
                    _pocket_ref_ca_nm = retrieval_ref_ca_nm[pocket_mask_ca].copy()
                    print(f"[Pocket] Auto-detected {int(pocket_mask_ca.sum())} pocket CA "
                          f"(residues {list(np.where(pocket_mask_ca)[0][:5])}...) in protein-only frame", flush=True)
                else:
                    print(f"[Pocket] Warning: no pocket residues found within {_POCKET_NM*10:.0f}Å", flush=True)
            else:
                print(f"[Pocket] Warning: no ligand atoms or empty ca_local", flush=True)
        except Exception as exc_pk:
            import traceback
            print(f"[Pocket] Warning: auto-detection failed ({exc_pk})", flush=True)
            traceback.print_exc()

    def _compute_globals_from_coords(coords_list: List[np.ndarray]) -> np.ndarray:
        return _compute_globals_impl(
            coords_list,
            global_dim=global_dim,
            global_token_names=global_token_names,
            ca_local=retrieval_ca_local,
            ref_ca_nm=retrieval_ref_ca_nm,
            adk_token_indices=adk_token_idx,
            tmem63_token_indices=tmem63_token_idx,
            tmem63_ref_rmsd_nm=tmem63_ref_rmsd_nm,
            ppi_token_indices=ppi_token_idx,
            pocket_ca_local=_pocket_ca_local,
            pocket_ref_ca_nm=_pocket_ref_ca_nm,
            prot_heavy_local=_prot_heavy_local,
            lig_heavy_local=_lig_heavy_local,
        )

    # LOO-VAMP Rewards Manager
    n_models = 1  # Single model for generative loop
    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
    loo_vamp_mgr = LOOVAMPRewards(n_models=n_models, n_states=n_states, lag_embed=1)
    
    # Keep references for backward compatibility
    loo_vamp_stats = None       # Will be managed by loo_vamp_mgr
    loo_vamp_stats_keys = set()  # Will be managed by loo_vamp_mgr
    loo_vamp_stats_signature = None  # Will be managed by loo_vamp_mgr

    # VAMP-bin exploration bonus manager (2D bin-count statistics).
    vamp_bin_mgr = VampBinBonus()
    vamp_bin_counted_keys: set = set()

    # MSM information-gain exploration bonus manager.
    msm_ig_mgr = MSMIGBonus()

    # Start-distance (PCCA macrostate-centroid) reward manager
    from gespo.rewards.frontier import StartDistanceReward
    start_dist_mgr = StartDistanceReward(
        n_macrostates=int(getattr(config, "pcca_n_macrostates", 7) or 7),
    )


    def _invalidate_vamp_bin_stats(tag: str) -> None:
        vamp_bin_mgr.invalidate(tag)

    def _invalidate_msm_ig_stats(tag: str) -> None:
        msm_ig_mgr.invalidate(tag)

    def _iter_history_trajs(upto_epoch_exclusive: int):
        """Yield history trajectories (epoch_idx, traj_idx, L_i, dcd_path, coords_nm).

        Thin closure over the shared ``gespo.loops._helpers.iter_history_trajs``
        that supplies this loop's history buffers. Used by multiple reward
        helpers (MSM-IG, VAMP-bin) to populate history statistics.
        """
        return _shared_iter_history_trajs(
            all_epoch_data,
            all_trajectory_lengths,
            all_epoch_traj_dcd_paths,
            all_epoch_vamp_exclude,
            upto_epoch_exclusive,
        )

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
        # Fit PCCA macrostates on the MSM-IG k-means + transition matrix.
        # Used only for the per-trajectory start-distance reward (no masking).
        try:
            if (
                float(getattr(config, "start_distance_reward_coef", 0.0) or 0.0) > 0
                and msm_ig_mgr.kmeans is not None
                and getattr(msm_ig_mgr, "counts", None) is not None
            ):
                centroids = np.asarray(msm_ig_mgr.kmeans.cluster_centers_, dtype=np.float32)
                trans = np.asarray(msm_ig_mgr.counts, dtype=np.float64)
                if trans.shape[0] == centroids.shape[0] and trans.shape[0] >= 4:
                    ok = start_dist_mgr.fit(centroids, trans)
                    n_macro = start_dist_mgr.n_macrostates if ok else 0
                    print(
                        f"[StartDistance] PCCA refit: K={centroids.shape[0]} macrostates={n_macro} "
                        f"success={ok}",
                        flush=True,
                    )
                    # Diagnostic: macrostate visit counts (informational only — no masking).
                    if ok and start_dist_mgr.M is not None:
                        try:
                            mss_visits_micro = np.asarray(msm_ig_mgr.counts, dtype=np.float64).sum(axis=1)
                            macro_visits = (mss_visits_micro[:, None] * start_dist_mgr.M).sum(axis=0)
                            macro_visits = macro_visits.astype(np.int64, copy=False)
                            print(
                                f"[StartDistance] Macrostate visits (info): {macro_visits.tolist()}",
                                flush=True,
                            )
                        except Exception as _exc_vis:
                            print(f"[StartDistance] Warning: visit-count diagnostic failed: {_exc_vis}", flush=True)
        except Exception as _exc_pcca:
            print(f"[StartDistance] PCCA refit failed: {_exc_pcca}", flush=True)

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
        # Extract ligand torsions from 5-tuple form (cv, coords, energy, bias, lig_tors).
        # Older 4-tuple form (cv, coords, energy, bias) gets empty arrays.
        new_lig_torsions = []
        for item in epoch_data:
            if len(item) >= 5 and item[4] is not None:
                try:
                    new_lig_torsions.append(np.asarray(item[4], dtype=np.float32).reshape(-1))
                except Exception:
                    new_lig_torsions.append(np.zeros((0,), dtype=np.float32))
            else:
                new_lig_torsions.append(np.zeros((0,), dtype=np.float32))
        retrieval_mgr.cv.extend(new_cv)
        retrieval_mgr.coords.extend(new_coords)
        retrieval_mgr.lig_torsions.extend(new_lig_torsions)
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
            f"(cap={cap}, retrieval_cache_max_frames={getattr(config, 'retrieval_cache_max_frames', 0)}, "
            f"max_pca_train={getattr(config, 'max_pca_train', 0)})"
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
            if epoch_idx < len(all_epoch_retrieval_exclude) and all_epoch_retrieval_exclude[epoch_idx]:
                continue
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
    geom2vec_projector = geom2vec_mgr.projector
    geom2vec_projector_linear = geom2vec_mgr.projector_linear
    geom2vec_topology_cache = geom2vec_mgr.topology_cache
    geom2vec_md_top_protein = geom2vec_mgr.md_top_protein
    geom2vec_device_str = geom2vec_mgr.device_str
    geom2vec_embed_stride = geom2vec_mgr.embed_stride
    geom2vec_embed_batch_size = geom2vec_mgr.embed_batch_size
    geom2vec_reduction = geom2vec_mgr.reduction
    geom2vec_center_ca = geom2vec_mgr.center_ca

    # Initialize effective VAMP lag in embedding frames.
    # geom2vec_lagtime is specified in saved frames; convert to embedded-frame lag.
    try:
        lag_raw_init = int(getattr(config, "geom2vec_lagtime", 1) or 1)
        geom2vec_lagtime_embed = max(1, int(round(max(1, lag_raw_init) / max(1, int(geom2vec_embed_stride)))))
    except Exception:
        geom2vec_lagtime_embed = 1
    
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
                loo_vamp_mgr.invalidate,
                _invalidate_vamp_bin_stats,
                _invalidate_msm_ig_stats,
            ],
            refresh_retrieval_labels_fn=_refresh_retrieval_cache_labels,
            ensure_msm_ig_fn=_ensure_msm_ig_after_refresh,
            reset_policy_fn=_reset_policy,
        )
        geom2vec_state_permutation_map = perm
        geom2vec_state_inverse_permutation_map = inv_perm
        geom2vec_lagtime_embed = lag

    # Load Checkpoint
    latest_ckpt, ckpt_data = ckpt_manager.load_latest()
    if config.resume and ckpt_data:
        ep_start, _, pca_state = ckpt_manager.restore_states(agent, agent_old, policy_opts, ckpt_data, torch_device)
        imported_frames, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.snapshot_dir or ".",
            ligand_smiles=str(getattr(config, "ligand_smiles", "") or ""),
        )
        if imported_frames > 0:
            print(f"[Resume] Rebuilt history from {imported_frames} frames (DCD).")
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(False)
            all_epoch_retrieval_exclude.append(False)
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
                print(f"[Resume] Rebuilt history from {restored} NPZ epochs.")
                all_trajectory_lengths.extend(restored_lens)
                for lens in restored_lens:
                    all_epoch_traj_dcd_paths.append([None] * len(lens))
                    all_epoch_traj_dcd_frame_indices.append([None] * len(lens))
                    all_epoch_vamp_exclude.append(False)
                    all_epoch_retrieval_exclude.append(False)

    # Bootstrap history from external DCDs if provided.
    if getattr(config, "history_dcd_dir", ""):
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.history_dcd_dir,
            ligand_smiles=str(getattr(config, "ligand_smiles", "") or ""),
        )
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(False)  # history-dcd-dir: unbiased prior-round data, include in VAMP
            all_epoch_retrieval_exclude.append(False)
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
            print(
                f"[History DCD] Imported {imported} frames; retrieval cache will be rebuilt from all loaded history before training."
            )

    # Bootstrap replay-only history from external DCDs (e.g. pretrain trajs with
    # different time spacing).  These frames are available for discrete-state
    # start retrieval but are EXCLUDED from VAMP training & LOO-VAMP rewards.
    if getattr(config, "replay_only_dcd_dir", ""):
        print("[Replay-Only History] Loading pretrain trajs for replay pool (VAMP-excluded)...")
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.replay_only_dcd_dir,
            ligand_smiles=str(getattr(config, "ligand_smiles", "") or ""),
        )
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(True)  # quarantine: exclude from VAMP
            all_epoch_retrieval_exclude.append(False)
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
            print(f"[Replay-Only History] Imported {imported} frames (VAMP-excluded).")

    # Bootstrap VAMP-only history: included in VAMP training but excluded from retrieval pool.
    # Use case: unfiltered prior-round trajectories for richer CV coverage, while keeping
    # the retrieval pool restricted to bound/valid starting frames.
    if getattr(config, "vamp_dcd_dir", ""):
        print("[VAMP-Only History] Loading trajs for VAMP training (retrieval-excluded)...")
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.vamp_dcd_dir,
            ligand_smiles=str(getattr(config, "ligand_smiles", "") or ""),
        )
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(False)   # included in VAMP
            all_epoch_retrieval_exclude.append(True)  # excluded from retrieval pool
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
            print(f"[VAMP-Only History] Imported {imported} frames (retrieval-excluded).")

    # Discrete state dimension (number of VAMP states)
    n_tica_components = int(getattr(config, "geom2vec_dim", 2) or 2)
    config.n_discrete_states = n_tica_components
    n_discrete_states = n_tica_components
    print("Initializing geom2vec VAMP backbone...")

    feature_builder = FeatureBuilder(config, generator=envs[0].generator, env=envs[0])

    sampler = Sampler(config, envs, reward_calculator=None, device=torch_device)

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

    diag_path = Path(config.generative_diagnostics_log) if getattr(config, "generative_diagnostics_log", "") else None
    if diag_path is not None:
        diag_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_anchoring_round() -> None:
        from .components.anchoring import _run_anchoring_round as _anchor_impl # NOTE: Why doesn't this fail?
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


    # Populate retrieval cache from any bootstrapped history (metad seeds, resumed DCDs).
    if all_epoch_data and len(retrieval_mgr.cv) == 0:
        _rebuild_retrieval_cache_from_history(tag="bootstrap")
        print(f"[Retrieval] Cache rebuilt from history: {len(retrieval_mgr.cv)} frames")

    # Pre-seed retrieval labels from a precomputed file (e.g., VAMP cluster labels).
    _preseed_path = str(getattr(config, "retrieval_preseed_labels_npz", "") or "")
    if _preseed_path and os.path.isfile(_preseed_path) and retrieval_mgr.labels is None:
        try:
            _preseed = np.load(_preseed_path, allow_pickle=True)
            _preseed_labels = np.asarray(_preseed["labels"], dtype=np.int64)
            n_pool = len(retrieval_mgr.cv)
            if _preseed_labels.shape[0] <= n_pool:
                pad = np.zeros(n_pool - _preseed_labels.shape[0], dtype=np.int64)
                retrieval_mgr.labels = np.concatenate([_preseed_labels, pad])
            else:
                retrieval_mgr.labels = _preseed_labels[:n_pool]
            print(f"[Retrieval] Pre-seeded {retrieval_mgr.labels.shape[0]} labels from {_preseed_path} "
                  f"(unique: {len(np.unique(retrieval_mgr.labels))})", flush=True)
        except Exception as exc_ps:
            print(f"[Retrieval] Warning: failed to pre-seed labels ({exc_ps})", flush=True)

    for ep in range(ep_start, config.episodes):
        print(f"\n{'='*60}")
        print(f"Starting Epoch {ep+1}/{config.episodes}")
        print(f"{'='*60}")
        epoch_start_time = time.time()

        # A. Refresh projector (every N epochs)
        epoch_phase(ep + 1, config.episodes, "A")
        refresh_interval = int(getattr(config, "pca_refresh_interval", 1) or 1)
        model_ready = geom2vec_mgr.is_trained()
        if refresh_interval <= 0:
            should_update = not model_ready
        else:
            should_update = ((ep + 1) % refresh_interval == 0) or (not model_ready)

        # Check if there's any VAMP-eligible data (not all replay-only / quarantined).
        _vamp_eligible_epochs = [
            i for i in range(len(all_epoch_data))
            if i >= len(all_epoch_vamp_exclude) or not all_epoch_vamp_exclude[i]
        ]
        _has_vamp_data = len(_vamp_eligible_epochs) > 0

        if should_update and all_epoch_data and _has_vamp_data:
            print(f"[Epoch {ep+1}] Refreshing Geom2Vec VAMP lobe...")
            _refresh_geom2vec_vamp(reason=f"epoch {ep+1}")
            model_ready = geom2vec_mgr.is_trained()
        elif should_update and all_epoch_data and not _has_vamp_data:
            print(f"[Epoch {ep+1}] Skipping Geom2Vec VAMP refresh: no VAMP-eligible data yet "
                  f"(all {len(all_epoch_data)} epoch(s) are replay-only / VAMP-excluded)")

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
        epoch_phase(ep + 1, config.episodes, "B", "Sampling torsion starts from TorsionAR policy")

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
        start_cluster_ids_epoch: Optional[np.ndarray] = None
        retrieval_labels_epoch: Optional[np.ndarray] = None

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
                cand_idx = balanced_candidate_indices(labels=labels, n_clusters=n_clusters, m=m_candidates, rng=retrieval_mgr.rng)
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
                # Pad candidate torsions to match AR model num_residues (ligand slots).
                if cand_tors.shape[1] < num_residues:
                    _pad = np.zeros((cand_tors.shape[0], num_residues - cand_tors.shape[1], cand_tors.shape[2]), dtype=np.float32)
                    cand_tors = np.concatenate([cand_tors, _pad], axis=1)

                # Fill ligand torsion positions from cached lig_torsions if available.
                # NEW LAYOUT: ligand dihedral d goes at row (env.n_residues + d), col 0.
                if num_residues > int(env_ref.n_residues):
                    try:
                        cand_lig = []
                        any_nonempty = False
                        for i in cand_idx:
                            lt = retrieval_mgr.lig_torsions[int(i)] if int(i) < len(retrieval_mgr.lig_torsions) else np.zeros((0,), dtype=np.float32)
                            lt = np.asarray(lt, dtype=np.float32).reshape(-1)
                            if lt.size > 0:
                                any_nonempty = True
                            cand_lig.append(lt)
                        if any_nonempty:
                            n_extra_slots = int(num_residues) - int(env_ref.n_residues)
                            for ci, lt in enumerate(cand_lig):
                                n_use = int(min(lt.size, n_extra_slots))
                                for d in range(n_use):
                                    cand_tors[ci, int(env_ref.n_residues) + d, 0] = float(lt[d])
                    except Exception as exc_lig_fill:
                        print(f"[Retrieval] Warning: ligand torsion fill failed ({exc_lig_fill})")
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
                retrieval_distances_sq = []  # Track L2² torsion distance between AR proposal and retrieved frame
                retrieval_distances_glob_sq = []  # Track L2² global (RMSD/E2E) distance
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
                            global_pool = min(int(d2_glob.size), int(topk))
                            stage1_pos = glob_order[:global_pool]
                            d2_tors = pairwise_sq_l2(q_tors_emb, cand_emb[stage1_pos])[0]  # [global_pool]
                            d_min_glob = float(d2_glob[glob_order[0]]) if glob_order.size else float("inf")
                            d_min_tors = float(np.min(d2_tors)) if d2_tors.size else float("inf")
                            # Debug: log AR's proposed globals and stage1 pool composition
                            if w < 3 or w == config.grp_size - 1:
                                _q_raw = start_globals[w].detach().cpu().numpy()
                                _stage1_globs = np.asarray([retrieval_mgr.globals[int(cand_idx[int(p)])] for p in stage1_pos], dtype=np.float32)
                                print(f"  [Retrieval Debug w{w:02d}] AR globals={_q_raw}, "
                                      f"stage1 glob range: {_stage1_globs.min(0)} to {_stage1_globs.max(0)}, "
                                      f"d_min_glob={d_min_glob:.4f}, d_min_tors={d_min_tors:.1f}", flush=True)

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
                            retrieval_distances_sq.append(float(d2_tors[int(pick_local)]))
                            retrieval_distances_glob_sq.append(float(d2_glob[pos]))
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
                        pick_idx_in_d2 = int(pick)  # pick is index into cand_idx
                        hist_i = int(cand_idx[pick_idx_in_d2])
                        chosen_hist_idx[w] = hist_i
                        chosen_cv[w] = np.asarray(retrieval_mgr.cv[hist_i], dtype=np.float32)
                        chosen_coords.append(_make_replay_payload(hist_i))
                        retrieval_distances_sq.append(float(d2[pick_idx_in_d2]))
                        retrieval_distances_glob_sq.append(float("nan"))  # No globals in this path
                        accepted = True
                        break

                    if not accepted:
                        d2_glob_fallback = None
                        if use_global_retrieval and cand_glob_norm is not None:
                            try:
                                q_glob = start_globals[w : w + 1].detach().cpu().numpy().astype(np.float32, copy=False)
                                q_y = np.log1p(np.clip(q_glob, 0.0, None)).astype(np.float32, copy=False)
                                q_norm = ((q_y - global_norm_mu[None, :]) / global_norm_std[None, :]) * float(global_w)
                                d2_glob_fallback = pairwise_sq_l2(q_norm.astype(np.float32, copy=False), cand_glob_norm)[0]
                                glob_order = np.argsort(d2_glob_fallback)
                                pos = int(glob_order[0]) if glob_order.size else 0
                                hist_i = int(cand_idx[pos])
                            except Exception:
                                hist_i = int(cand_idx[0])
                                pos = 0
                        else:
                            hist_i = int(cand_idx[0])
                            pos = 0
                        chosen_hist_idx[w] = hist_i
                        chosen_cv[w] = np.asarray(retrieval_mgr.cv[hist_i], dtype=np.float32)
                        chosen_coords.append(_make_replay_payload(hist_i))
                        # For fallback, use d2 from last resample attempt
                        retrieval_distances_sq.append(float(d2[pos]) if d2.size > pos else float("nan"))
                        if d2_glob_fallback is not None and d2_glob_fallback.size > pos:
                            retrieval_distances_glob_sq.append(float(d2_glob_fallback[pos]))
                        else:
                            retrieval_distances_glob_sq.append(float("nan"))

                # Use AR-proposed torsion targets for the CV conditioning; replay coordinates
                # provide the retrieved frame, and (optionally) targeted MD will steer toward
                # these CVs/globals.
                phi_psi = start_torsions[:, :env_ref.n_residues, :2]
                phi_psi_np = phi_psi.detach().cpu().numpy()
                try:
                    cv_batch = env_ref.residue_batch_to_cv(phi_psi_np)
                except Exception as exc_cv:
                    print(f"[TorsionAR] Warning: residue→CV mapping failed ({exc_cv}); using replay CVs.")
                    cv_batch = chosen_cv
                starts = torch.as_tensor(cv_batch, device=torch_device, dtype=torch.float32)
                replay_coords = chosen_coords

                # Log retrieval distance statistics (both global and torsion)
                if retrieval_distances_glob_sq:
                    valid_glob = [d for d in retrieval_distances_glob_sq if not np.isnan(d)]
                    if valid_glob:
                        mean_g = float(np.mean(valid_glob))
                        max_g = float(np.max(valid_glob))
                        print(f"[Retrieval - Epoch {ep+1}] Global (RMSD/E2E) L2² dist: "
                              f"mean={mean_g:.4f}, max={max_g:.4f}, RMS={np.sqrt(mean_g):.4f}")
                if retrieval_distances_sq:
                    valid_dists = [d for d in retrieval_distances_sq if not np.isnan(d)]
                    if valid_dists:
                        mean_d2 = float(np.mean(valid_dists))
                        max_d2 = float(np.max(valid_dists))
                        print(f"[Retrieval - Epoch {ep+1}] Torsion L2² dist (AR → retrieved): "
                              f"mean={mean_d2:.4f}, max={max_d2:.4f}, RMS={np.sqrt(mean_d2):.4f}")

                # Optional start-cluster entropy bonus (requires labels)
                try:
                    if labels is not None and chosen_hist_idx.size == config.grp_size:
                        start_cluster_ids = labels[chosen_hist_idx.clip(0, labels.shape[0] - 1)]
                        start_cluster_ids_epoch = start_cluster_ids.copy()
                        retrieval_labels_epoch = labels.copy()
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
            # Trim ligand torsion slots: CV mapping only uses protein residues.
            phi_psi = start_torsions[:, :env_ref.n_residues, :2]
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

        epoch_phase(ep + 1, config.episodes, "C")
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

        # Log reset CV MAE: measures |θ_AR - θ_actual| after steering
        if "reset_cv_mae" in metrics and metrics["reset_cv_mae"]:
            mae_vals = [float(x) for x in metrics["reset_cv_mae"] if not np.isnan(x)]
            if mae_vals:
                print(f"[Reset CV MAE - Epoch {ep+1}] mean={np.mean(mae_vals):.4f}, "
                      f"max={np.max(mae_vals):.4f}, min={np.min(mae_vals):.4f} "
                      f"(|AR proposal - actual torsions| after steering, radians)")

        all_epoch_data.append(current_epoch_data)
        all_epoch_vamp_exclude.append(False)
        all_epoch_retrieval_exclude.append(False)
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

        # Log ligand-pocket distance stats for this epoch
        if config.snapshot_dir and getattr(config, "ligand_smiles", None):
            _sim_top_for_lig = os.path.join(str(config.snapshot_dir), "simulation_topology.pdb")
            try:
                _log_ligand_pocket_stats(
                    ep + 1, traj_dcd_paths, _sim_top_for_lig,
                )
            except Exception as _exc_lig:
                print(f"[Ligand-Pocket] Warning: logging failed: {_exc_lig}", flush=True)

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
        epoch_phase(ep + 1, config.episodes, "D")
        loo_vamp_rewards = np.zeros(config.grp_size, dtype=np.float32)
        loo_vamp_contributions = None
        vamp_bin_bonus = None
        msm_ig_rewards = None
        msm_ig_unique_states = None
        cached_keys_epoch_for_plots: Optional[List[Optional[str]]] = None
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
                    msm_ig_state_mode = (
                        str(getattr(config, "msm_ig_state_mode", "kmeans") or "kmeans").strip().lower()
                    )
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

                    cached_keys_epoch_for_plots = list(cached_keys_epoch)

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

                    if use_msm_ig:
                        _ensure_msm_ig_history_counts(
                            upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                            n_states=n_states,
                            lag_embed=lag_embed,
                            tag=f"epoch {ep+1} pre-reward",
                        )

                    pair_counts = np.zeros(config.grp_size, dtype=np.int64)
                    traj_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
                        None for _ in range(len(cached_keys_epoch))
                    ]

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
                    dim_needed = int(min(int(dim_needed), int(n_states)))

                    traj_vamp_labels_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))]
                        if (use_msm_ig and str(msm_ig_state_mode) == "vamp")
                        else None
                    )

                    traj_z_lin_best: Optional[List[Optional[np.ndarray]]] = (
                        [None for _ in range(len(cached_keys_epoch))]
                        if (
                            dim_needed > 0
                            and geom2vec_projector_linear is not None
                            and getattr(geom2vec_projector_linear, "trained", None) is not None
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
                            if T > lag_embed:
                                traj_pairs[t_i] = (probs[:-lag_embed], probs[lag_embed:])
                                pair_counts[t_i] = max(1, int(traj_pairs[t_i][0].shape[0]))  # type: ignore[index]
                                if traj_vamp_labels_best is not None:
                                    traj_vamp_labels_best[t_i] = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                        except Exception:
                            traj_pairs[t_i] = None

                        if traj_z_lin_best is None or geom2vec_projector_linear is None:
                            continue
                        try:
                            z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                            z = np.asarray(z_raw, dtype=np.float32)
                            if dim_needed > 0 and z.shape[0] > 0:
                                traj_z_lin_best[t_i] = np.asarray(z[:, :dim_needed], dtype=np.float32)
                        except Exception:
                            traj_z_lin_best[t_i] = None

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
                            if key_s in vamp_bin_counted_keys:
                                continue
                            z2 = z_lin[:, :2]
                            z2s = (z2 - vamp_bin_mgr.mu[None, :]) / vamp_bin_mgr.std[None, :]
                            ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                            np.add.at(vamp_bin_mgr.counts, (ix, iy), 1)
                            vamp_bin_mgr.total += int(ix.shape[0])
                            vamp_bin_counted_keys.add(key_s)

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
                                    counts_i, n_pairs_i = _msm_transition_counts(labels, n_states=K_msm, lag=lag_msm)
                                    try:
                                        uniq_states[i] = int(np.unique(labels).size)
                                    except Exception:
                                        uniq_states[i] = 0
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
        epoch_phase(ep + 1, config.episodes, "E")
        if len(current_epoch_data) > 0:
            hill_data = None

            launch_rewards = reward_tensors.get("launch", torch.zeros(config.grp_size, device=torch_device))
            if launch_rewards.device != torch_device:
                launch_rewards = launch_rewards.to(torch_device)
                reward_tensors["launch"] = launch_rewards

            # ── Start-distance reward per trajectory ────────────────
            # Per-walker reward = L2 distance (in normalized linear-VAMP space) from
            # the first trajectory frame to the nearest PCCA macrostate centroid.
            # Rewards proposing starts that escape known basins; no MD relaxation
            # dependence and no hard masking.
            start_distance_rewards: Optional[np.ndarray] = None
            if (
                float(getattr(config, "start_distance_reward_coef", 0.0) or 0.0) > 0
                and start_dist_mgr.is_ready()
                and traj_z_lin_best is not None
                and msm_ig_mgr.mu is not None
                and msm_ig_mgr.std is not None
            ):
                try:
                    dim_f = int(getattr(config, "msm_ig_dim", 2) or 2)
                    mu_f = np.asarray(msm_ig_mgr.mu, dtype=np.float32).reshape(-1)[:dim_f]
                    std_f = np.asarray(msm_ig_mgr.std, dtype=np.float32).reshape(-1)[:dim_f]
                    std_f = np.where(std_f > 1e-8, std_f, 1.0).astype(np.float32, copy=False)

                    start_distance_rewards = np.zeros(config.grp_size, dtype=np.float32)
                    for _ti, _zl in enumerate(traj_z_lin_best or []):
                        if _zl is None or _zl.size == 0 or int(_zl.shape[1]) < int(dim_f):
                            continue
                        z_start = np.asarray(_zl[0:1, :dim_f], dtype=np.float32)
                        z_start_norm = (z_start - mu_f[None, :]) / std_f[None, :]
                        d = start_dist_mgr.start_distance(z_start_norm)
                        start_distance_rewards[_ti] = float(d[0]) if d.size > 0 else 0.0
                    print(
                        f"[StartDistance - Epoch {ep+1}] mean={float(np.mean(start_distance_rewards)):.4f} "
                        f"max={float(np.max(start_distance_rewards)):.4f} "
                        f"per_walker={[round(float(x), 3) for x in start_distance_rewards.tolist()]}",
                        flush=True,
                    )
                except Exception as _exc_sd:
                    print(f"[StartDistance] Warning: reward computation failed: {_exc_sd}", flush=True)
                    start_distance_rewards = None

            # ── Combine rewards via RewardCombiner ──────────────────
            from gespo.rewards.combiner import RewardCombiner, RewardTerms

            if not hasattr(train, "_reward_combiner"):
                train._reward_combiner = RewardCombiner(config, torch_device)
            _rc = train._reward_combiner

            _raw_terms = RewardTerms(
                loo_vamp=loo_vamp_rewards,
                start_cluster=start_cluster_bonus if start_cluster_bonus is not None else None,
                vamp_bin=vamp_bin_bonus if vamp_bin_bonus is not None else None,
                start_distance=start_distance_rewards,
                msm_ig=msm_ig_rewards if msm_ig_rewards is not None else None,
                off_action_mae=(
                    np.asarray(metrics.get("reset_cv_mae", [0.0] * config.grp_size), dtype=np.float32)
                    if float(getattr(config, "off_action_penalty_coef", 0.0) or 0.0) > 0 else None
                ),
            )

            active_mask = metrics.get(
                "active_mask",
                torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
            )
            if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                active_mask = active_mask.to(torch_device)

            combined_rewards, reward_metrics = _rc.combine(
                _raw_terms, launch_rewards, active_mask,
            )
            metrics.update(reward_metrics)

            # Backward-compat term tensors for diagnostics logging
            loo_weight = _rc.loo_weight
            term_loo_vamp = torch.zeros_like(launch_rewards)
            term_start_cluster = torch.zeros_like(launch_rewards)
            term_vamp_bin = torch.zeros_like(launch_rewards)
            term_msm_ig = torch.zeros_like(launch_rewards)
            term_off_action = torch.zeros_like(launch_rewards)
            off_action_discount = torch.ones_like(launch_rewards)
            if loo_vamp_rewards is not None:
                term_loo_vamp = loo_weight * torch.tensor(loo_vamp_rewards, device=torch_device, dtype=torch.float32)

            # Inverse-population weighting: downweight walkers in overrepresented states.
            if retrieval_labels_epoch is not None and start_cluster_ids_epoch is not None:
                try:
                    _gl = retrieval_labels_epoch
                    _gc = np.bincount(_gl, minlength=int(_gl.max()) + 1).astype(np.float64)
                    _gp = _gc / max(_gc.sum(), 1)
                    _pw = np.array([1.0 / max(_gp[int(s)], 0.01)
                                    for s in start_cluster_ids_epoch[:config.grp_size]], dtype=np.float32)
                    _pw = _pw / _pw.mean()
                    combined_rewards = combined_rewards * torch.tensor(_pw, device=torch_device, dtype=torch.float32)
                    print(f"[Reward] Inverse-population weights: min={_pw.min():.2f} max={_pw.max():.2f} "
                          f"mean={_pw.mean():.2f}", flush=True)
                except Exception as exc_pw:
                    print(f"[Reward] Warning: population weighting failed ({exc_pw})", flush=True)

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

        epoch_phase(ep + 1, config.episodes, "F", "Logging & checkpointing")
        print(f"Epoch {ep+1} Complete. Avg Reward: {metrics.get('avg_hill', 0.0):.4f}")

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
                    all_epoch_data,
                    None,  # pca_scorer removed
                    envs[0].cv_names,
                    config.pdb,
                    f"analysis_summary_ep{ep+1:03d}.png",
                    feature_builder=feature_builder,
                    vamp_bin_counts=vamp_bin_mgr.counts,
                    vamp_bin_clip=float(getattr(config, "vamp_bin_clip", 3.0) or 3.0),
                    vamp_bin_total=int(vamp_bin_mgr.total),
                )
            except Exception as exc_plot:
                print(f"[Plot] Warning: summary plot failed: {exc_plot}")

            # Optional per-epoch analysis/monitoring plugin (e.g. a custom
            # feature-pair diagnostic). Off by default; opt in with
            # --diagnostics-plugin '/path/file.py:callable'. Best-effort: any
            # failure (including a plugin that itself needs the geom2vec cache
            # and doesn't find one yet) is reported and skipped, not fatal.
            plugin_spec = str(getattr(config, "diagnostics_plugin", "") or "").strip()
            if plugin_spec and config.snapshot_dir:
                try:
                    from gespo.training.components.endpoint_loader import load_boundary_plugin

                    plugin_fn = load_boundary_plugin(plugin_spec)
                    plugin_fn(
                        config=config,
                        epoch=ep + 1,
                        geom2vec_cache=geom2vec_cache,
                        cached_keys=cached_keys_epoch_for_plots,
                    )
                except Exception as exc_plugin:
                    print(f"[Diagnostics Plugin] Note: skipped ({exc_plugin})", flush=True)

        elapsed = time.time() - epoch_start_time
        print(f"[Epoch {ep+1}] Complete in {elapsed:.1f}s.")

        gc.collect()
