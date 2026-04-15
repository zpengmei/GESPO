"""Programmatic API for GESPO training workflows.

Example usage::

    from gespo.api import train, pretrain_torsion_ar
    from gespo.config import load_config

    # Load config from YAML
    config = load_config("config.yaml")

    # Modify programmatically
    config.episodes = 10
    config.snapshot_dir = "runs/round1"

    # Train
    train(config)

    # Or pretrain a TorsionAR policy
    pretrain_torsion_ar(
        source_path="history_dcds/",
        out_path="checkpoints/r0.pt",
        pdb_path="endpoint_A.pdb",
    )
"""
from __future__ import annotations

from typing import Optional


def train(config) -> None:
    """Run GESPO training with the given configuration.

    Parameters
    ----------
    config : TrainingConfig or dict or str
        - ``TrainingConfig``: use directly.
        - ``dict``: passed to ``TrainingConfig.from_dict()``.
        - ``str``: treated as a YAML file path and loaded via ``load_config()``.
    """
    from .config import TrainingConfig, load_config
    from .training import train as _train
    import argparse

    if isinstance(config, str):
        config = load_config(config)
    elif isinstance(config, dict):
        config = TrainingConfig.from_dict(config)

    # Convert to argparse.Namespace for backward compatibility with training loops.
    args = argparse.Namespace(**config.to_dict())
    _train(args)


def pretrain_torsion_ar(
    source_path: str,
    out_path: str,
    pdb_path: str,
    *,
    pdb_b_path: Optional[str] = None,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[str] = None,
    stride: int = 1,
    global_token_set: str = "rmsd_e2e",
    torsion_mode: str = "backbone",
) -> None:
    """Pretrain a TorsionAR policy from trajectory data.

    Parameters
    ----------
    source_path : str
        Path to DCD/trajectory directory or file.
    out_path : str
        Output path for the pretrained checkpoint.
    pdb_path : str
        PDB file for topology.
    """
    from .training.pretrain_torsion_ar import pretrain_torsion_ar as _pretrain

    _pretrain(
        source_path=source_path,
        out_path=out_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        pdb_path=pdb_path,
        pdb_b_path=pdb_b_path,
        stride=stride,
        global_token_set=global_token_set,
        torsion_mode=torsion_mode,
    )
