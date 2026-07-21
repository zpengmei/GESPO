"""Geom2Vec backbone and VAMP lobe configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Geom2VecConfig:
    """Configuration for the geom2vec representation and VAMP lobe training."""

    # Projection settings (used when projection == "geom2vec_vamp").
    geom2vec_lagtime: int = 20
    geom2vec_dim: int = 2
    geom2vec_label_mode: str = "linear_kmeans"

    # Backbone checkpoint + inference.
    geom2vec_checkpoint: str = ""
    geom2vec_cache_root: str = ""
    geom2vec_manifest_name: str = "geom2vec_manifest.jsonl"
    geom2vec_device: str = ""
    geom2vec_data_parallel: bool = False
    geom2vec_data_parallel_devices: str = ""
    geom2vec_reserve_gpus: bool = False
    geom2vec_embed_stride: int = 2
    geom2vec_embed_batch_size: int = 32
    geom2vec_hidden_channels: int = 0
    geom2vec_reduction: str = "sum"
    geom2vec_center_ca: bool = True
    geom2vec_exclude_hydrogens: bool = True

    # VAMP lobe training.
    geom2vec_vamp_pair_stride: int = 1
    geom2vec_vamp_train_fraction: float = 0.8
    geom2vec_vamp_seed_base: int = 42
    geom2vec_vamp_epsilon: float = 1e-6
    geom2vec_vamp_inverse_mode: str = "regularize"
    geom2vec_vamp_epochs: int = 10
    geom2vec_vamp_batch_size: int = 256
    geom2vec_vamp_learning_rate: float = 5e-4
    geom2vec_vamp_weight_decay: float = 0.0
    geom2vec_vamp_optimizer: str = "Adam"
    geom2vec_vamp_softmax_output: bool = True

    # Lobe architecture.
    geom2vec_vamp_merger: bool = False
    geom2vec_vamp_merger_window: int = 6
    geom2vec_vamp_double_lobe: bool = False
    geom2vec_lobe_invariant: bool = False

    CLI_META: ClassVar[dict] = {
        "geom2vec_lagtime": "Lagtime (in saved frames) for VAMP time-lagged pairs.",
        "geom2vec_dim": {
            "help": "Number of discrete states (K) for the geom2vec VAMP lobe.",
            "aliases": ["--tica-dim", "--n-states"],
        },
        "geom2vec_label_mode": {
            "help": "Discrete label mode: 'auto', 'vamp' (argmax), or 'linear_kmeans'.",
            "choices": ["auto", "vamp", "linear_kmeans"],
        },
        "geom2vec_checkpoint": {
            "help": "Path to a geom2vec representation checkpoint.",
            "flag": "--geom2vec-ckpt",
        },
        "geom2vec_cache_root": "Root directory for cached geom2vec embeddings.",
        "geom2vec_manifest_name": "Manifest filename under cache root.",
        "geom2vec_device": "Device for geom2vec inference and VAMP training (default: cuda if available).",
        "geom2vec_data_parallel": {
            "help": "Enable torch.nn.DataParallel for geom2vec inference/training.",
            "no_help": "Disable DataParallel for geom2vec (default).",
        },
        "geom2vec_data_parallel_devices": (
            "Comma-separated CUDA device indices for DataParallel (empty=all)."
        ),
        "geom2vec_reserve_gpus": {
            "help": "Exclude geom2vec CUDA devices from OpenMM walker pool.",
            "no_help": "Share geom2vec GPUs with OpenMM walkers (default).",
        },
        "geom2vec_embed_stride": "Stride for subsampling DCD frames before embedding.",
        "geom2vec_embed_batch_size": "Batch size for geom2vec embedding inference.",
        "geom2vec_hidden_channels": "Override geom2vec hidden channels H (0=infer from model).",
        "geom2vec_reduction": {
            "help": "Coarse-graining reduction when pooling atom reps to residue tokens.",
            "choices": ["sum", "mean"],
        },
        "geom2vec_center_ca": {
            "help": "Center per-frame CA coords across tokens (default).",
            "no_help": "Disable CA centering for the geom2vec lobe input.",
        },
        "geom2vec_exclude_hydrogens": {
            "help": "Exclude hydrogens when building geom2vec inputs (default).",
            "no_help": "Include hydrogens when building geom2vec inputs.",
        },
        "geom2vec_vamp_pair_stride": "Stride for time-lagged pair sampling in VAMP dataset.",
        "geom2vec_vamp_train_fraction": "Trajectory-level train/val split fraction for VAMP lobe training.",
        "geom2vec_vamp_seed_base": "Base RNG seed for VAMP lobe ensemble (member i uses seed_base + 1000*i).",
        "geom2vec_vamp_epsilon": "Numerical stabilizer epsilon for VAMP covariance inverses.",
        "geom2vec_vamp_inverse_mode": {
            "help": "Mode for VAMP covariance inversion.",
            "choices": ["regularize", "trunc"],
        },
        "geom2vec_vamp_epochs": "Epochs for geom2vec VAMP lobe training.",
        "geom2vec_vamp_batch_size": "Batch size for geom2vec VAMP lobe training.",
        "geom2vec_vamp_learning_rate": "Learning rate for VAMP lobe training.",
        "geom2vec_vamp_weight_decay": "Weight decay for VAMP lobe training.",
        "geom2vec_vamp_optimizer": "Optimizer name for VAMP lobe training.",
        "geom2vec_vamp_softmax_output": {
            "help": "Use Softmax output activation in geom2vec lobe (default).",
            "no_help": "Disable Softmax output activation (treat lobe outputs as logits).",
        },
        "geom2vec_vamp_merger": {
            "help": "Enable token merger in VAMP lobe to reduce GPU memory.",
            "no_help": "Disable token merger (default).",
        },
        "geom2vec_vamp_merger_window": "Merger window size: merge every N tokens into 1.",
        "geom2vec_vamp_double_lobe": {
            "help": "Use a separate lagged lobe instead of weight-sharing.",
            "no_help": "Disable separate lagged lobe (default).",
        },
        "geom2vec_lobe_invariant": {
            "help": "Use invariant lobe mixer (equi_rep=False).",
            "no_flag": "--geom2vec-lobe-equivariant",
            "no_help": "Use equivariant lobe mixer (default).",
        },
    }
