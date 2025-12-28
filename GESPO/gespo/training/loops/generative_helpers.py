from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import torch

from gespo.utils.logging import log_print

from ...config import TrainingConfig
from ...sampling import PeptideSampler
from ...utils.adk_metrics import ADK_GLOBAL_TOKEN_NAMES
from ...utils.openmm_utils import resolve_device
from ..rl import TorsionARPolicy
from ..stats import SufficientStatistics


def prepare_generative_config(args: argparse.Namespace) -> TrainingConfig:
    """Build and validate the TrainingConfig for the generative loop."""
    config = TrainingConfig.from_args(args)
    proj_mode = str(getattr(config, "projection", "geom2vec_vamp")).lower()
    if proj_mode != "geom2vec_vamp":
        raise ValueError(
            "The generative loop requires --projection=geom2vec_vamp. "
            f"Got --projection={proj_mode!r}."
        )
    if not str(getattr(config, "geom2vec_checkpoint", "") or "").strip():
        raise ValueError("--geom2vec-ckpt is required when --projection=geom2vec_vamp.")
    config.discrete_state_starts = False
    if not hasattr(config, "max_pca_train") or config.max_pca_train < 100000:
        config.max_pca_train = 100000
    if not hasattr(config, "history_reward_weight"):
        config.history_reward_weight = 5.0
    return config


def resolve_devices(config: TrainingConfig) -> tuple[str, torch.device]:
    """Resolve torch and generator devices based on config and CUDA availability."""
    torch_device = "cuda" if torch.cuda.is_available() and config.platform != "CPU" else "cpu"
    gen_device = resolve_device(config.gen_device)
    return torch_device, gen_device


def log_startup(config: TrainingConfig, torch_device: str, gen_device: torch.device) -> None:
    """Emit startup logging describing the generative training configuration."""
    log_print("GESPO Training (generative loop) with geom2vec VAMP + LOO rewards")
    log_print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    log_print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")


def init_envs(args: argparse.Namespace, config: TrainingConfig, gen_device: torch.device) -> list[PeptideSampler]:
    """Instantiate peptide-sampler environments for each walker."""
    log_print("Initializing environments...")
    return [PeptideSampler(args, generator_device=gen_device) for _ in range(config.grp_size)]


def build_residue_metadata(
    config: TrainingConfig, env_ref: PeptideSampler, torch_device: str
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Load residue type indices and torsion masks for torsion-AR policies."""
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
    ref_top = md.load(config.pdb).topology
    res_codes = [three_to_one.get(res.name.upper(), "X") for res in ref_top.residues]
    residue_types_1d = np.array([aa_to_idx.get(code, fallback_idx) for code in res_codes], dtype=np.int64)

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
    angle_mask_1d = torch.stack([phi_exists, psi_exists, omega_exists], dim=-1)
    angle_mask_batch = angle_mask_1d.unsqueeze(0).expand(config.grp_size, -1, -1)
    return residue_types_1d, residue_types_batch, angle_mask_batch


def load_torsion_policy(
    args: argparse.Namespace,
    config: TrainingConfig,
    env_ref: PeptideSampler,
    residue_types_1d: np.ndarray,
    torch_device: str,
) -> tuple[TorsionARPolicy, TorsionARPolicy, Dict[str, Any], List[str], str]:
    """Load or initialize the torsion autoregressive policy and metadata."""
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
        names_cfg = ar_cfg.get("global_token_names")
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

    return agent, agent_old, ar_cfg, global_token_names, torsion_ckpt


def setup_training(args: argparse.Namespace):
    """Create config, devices, envs, metadata, and policies needed by the loop."""
    config = prepare_generative_config(args)
    torch_device, gen_device = resolve_devices(config)
    log_startup(config, torch_device, gen_device)
    envs = init_envs(args, config, gen_device)
    env_ref = envs[0]
    residue_types_1d, residue_types_batch, angle_mask_batch = build_residue_metadata(
        config, env_ref, torch_device
    )
    agent, agent_old, ar_cfg, global_token_names, torsion_ckpt = load_torsion_policy(
        args, config, env_ref, residue_types_1d, torch_device
    )
    return (
        config,
        torch_device,
        gen_device,
        envs,
        env_ref,
        residue_types_1d,
        residue_types_batch,
        angle_mask_batch,
        agent,
        agent_old,
        ar_cfg,
        global_token_names,
        torsion_ckpt,
    )


def init_history_buffers() -> tuple[list, list, list, list, int]:
    """Initialize epoch/trajectory history containers for the loop."""
    all_epoch_data: List[List[Any]] = []
    all_trajectory_lengths: List[List[int]] = []
    all_epoch_traj_dcd_paths: List[List[Optional[str]]] = []
    all_epoch_traj_dcd_frame_indices: List[List[Optional[np.ndarray]]] = []
    ep_start = 0
    return (
        all_epoch_data,
        all_trajectory_lengths,
        all_epoch_traj_dcd_paths,
        all_epoch_traj_dcd_frame_indices,
        ep_start,
    )


def init_retrieval_history() -> tuple[list, list, list, list, list, list, list, Optional[np.ndarray], np.random.Generator]:
    """Initialize replay retrieval caches and RNG state."""
    retrieval_hist_cv: List[np.ndarray] = []
    retrieval_hist_coords: List[np.ndarray] = []
    retrieval_hist_dcd_paths: List[Optional[str]] = []
    retrieval_hist_dcd_frame_idx: List[int] = []
    retrieval_hist_globals: List[np.ndarray] = []
    retrieval_hist_keys: List[str] = []
    retrieval_hist_frame_idx: List[int] = []
    retrieval_hist_labels: Optional[np.ndarray] = None
    retrieval_rng = np.random.default_rng(0)
    return (
        retrieval_hist_cv,
        retrieval_hist_coords,
        retrieval_hist_dcd_paths,
        retrieval_hist_dcd_frame_idx,
        retrieval_hist_globals,
        retrieval_hist_keys,
        retrieval_hist_frame_idx,
        retrieval_hist_labels,
        retrieval_rng,
    )


def init_vamp_stats_state():
    """Initialize LOO-VAMP and VAMP-bin tracking state."""
    loo_vamp_stats: Optional[SufficientStatistics] = None
    loo_vamp_stats_keys: set[str] = set()
    loo_vamp_stats_signature: Optional[Tuple[int, int]] = None

    vamp_bin_mu: Optional[np.ndarray] = None
    vamp_bin_std: Optional[np.ndarray] = None
    vamp_bin_counts: Optional[np.ndarray] = None
    vamp_bin_total: int = 0
    vamp_bin_counted_keys: set[str] = set()
    vamp_bin_signature: Optional[Tuple[int, float, int, int]] = None
    vamp_bin_norm_ema_m1: Optional[float] = None
    vamp_bin_norm_ema_m2: Optional[float] = None
    vamp_bin_norm_ema_scale: Optional[float] = None
    return (
        loo_vamp_stats,
        loo_vamp_stats_keys,
        loo_vamp_stats_signature,
        vamp_bin_mu,
        vamp_bin_std,
        vamp_bin_counts,
        vamp_bin_total,
        vamp_bin_counted_keys,
        vamp_bin_signature,
        vamp_bin_norm_ema_m1,
        vamp_bin_norm_ema_m2,
        vamp_bin_norm_ema_scale,
    )
