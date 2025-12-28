from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gespo.utils.logging import log_print

from ...config import TrainingConfig
from ...sampling import PeptideSampler
from ...utils.openmm_utils import resolve_device
from ..optimization import GRPOOptimizer
from ..setup import initialize_policy
from ..geom2vec_backbone import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector
from ..rl import StateLogitPolicy


def prepare_discrete_config(args: argparse.Namespace) -> TrainingConfig:
    """Build and validate the TrainingConfig for the discrete-start loop."""
    config = TrainingConfig.from_args(args)
    proj_mode = str(getattr(config, "projection", "geom2vec_vamp")).lower()
    if proj_mode != "geom2vec_vamp":
        raise ValueError(
            "This discrete-start loop requires a discrete-state projector. "
            f"Got --projection={proj_mode!r}; supported: geom2vec_vamp."
        )
    if not str(getattr(config, "geom2vec_checkpoint", "") or "").strip():
        raise ValueError("--geom2vec-ckpt is required when --projection=geom2vec_vamp.")
    config.discrete_state_starts = True
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


def init_envs(args: argparse.Namespace, config: TrainingConfig, gen_device: torch.device) -> list[PeptideSampler]:
    """Instantiate peptide-sampler environments for each walker."""
    log_print("Initializing environments...")
    return [PeptideSampler(args, generator_device=gen_device) for _ in range(config.grp_size)]


def init_discrete_policy(
    config: TrainingConfig, envs: list[PeptideSampler], torch_device: str
) -> tuple[StateLogitPolicy, StateLogitPolicy, GRPOOptimizer]:
    """Initialize discrete state policies and the optimizer wrapper."""
    log_print("Initializing policy...")
    agent, agent_old, policy_opts, _ = initialize_policy(config, envs, torch_device)
    if not (isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy)):
        raise RuntimeError("Discrete state starts require StateLogitPolicy for both current and old policies")
    optimizer = GRPOOptimizer(config, agent, agent_old, policy_opts)
    return agent, agent_old, optimizer


def log_startup(config: TrainingConfig, torch_device: str, gen_device: torch.device) -> None:
    """Emit startup logging describing the training configuration."""
    log_print("GESPO Training with LOO VAMP (Geom2Vec, single-model)")
    log_print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    log_print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")
    log_print("Geom2Vec VAMP lobe: single model (no UCB)")


def setup_training(args: argparse.Namespace):
    """Create config, devices, envs, and policies needed by the loop."""
    config = prepare_discrete_config(args)
    torch_device, gen_device = resolve_devices(config)
    log_startup(config, torch_device, gen_device)
    envs = init_envs(args, config, gen_device)
    env_ref = envs[0]
    agent, agent_old, optimizer = init_discrete_policy(config, envs, torch_device)
    return config, torch_device, gen_device, envs, env_ref, agent, agent_old, optimizer


def init_history_buffers() -> tuple[list, list, list, list, int]:
    """Initialize epoch/trajectory history containers for the loop."""
    all_epoch_data = []
    all_trajectory_lengths = []
    all_epoch_traj_dcd_paths: list[list[Optional[str]]] = []
    all_epoch_traj_dcd_frame_indices: list[list[Optional[np.ndarray]]] = []
    ep_start = 0
    return (
        all_epoch_data,
        all_trajectory_lengths,
        all_epoch_traj_dcd_paths,
        all_epoch_traj_dcd_frame_indices,
        ep_start,
    )


def init_geom2vec_state(config: TrainingConfig):
    """Initialize Geom2Vec-related state placeholders and defaults."""
    geom2vec_cache: Optional[EmbeddingCache] = None
    geom2vec_projector: Optional[Geom2VecVAMPProjector] = None
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
    return (
        geom2vec_cache,
        geom2vec_projector,
        geom2vec_projector_linear,
        geom2vec_state_permutation_map,
        geom2vec_state_inverse_permutation_map,
        geom2vec_lagtime_embed,
        geom2vec_embed_stride,
        geom2vec_embed_batch_size,
        geom2vec_reduction,
        geom2vec_center_ca,
        geom2vec_hidden_channels_opt,
        geom2vec_md_top_protein,
        geom2vec_topology_cache,
        geom2vec_cache_root_path,
        geom2vec_device_str,
    )
