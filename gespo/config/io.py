"""I/O, checkpointing, and history configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class IOConfig:
    """Configuration for checkpointing, logging, and history loading."""

    checkpoint_dir: str = ""
    checkpoint_interval: int = 0
    resume: bool = False
    snapshot_dir: str = ""
    stats_log: str = ""
    generative_diagnostics_log: str = ""
    plot_freq: int = 1
    history_dcd_dir: str = ""
    replay_only_dcd_dir: str = ""
    history_dcd_stride: int = 1
    history_dcd_max_frames: int = 0
    history_dcd_batch_size: int = 500
    rmsd_filter: float = 0.0
    save_epoch_coords: bool = True
    anchoring_source_dir: str = ""

    CLI_META: ClassVar[dict] = {
        "checkpoint_dir": "Directory to save checkpoints.",
        "checkpoint_interval": "Epochs between checkpoints (0=disabled).",
        "resume": {"help": "Resume from latest checkpoint.", "flag_only": True},
        "snapshot_dir": "Directory for epoch snapshots.",
        "stats_log": "Path to CSV stats log.",
        "generative_diagnostics_log": "Path to JSONL diagnostics log (generative loop).",
        "plot_freq": "Epochs between plots.",
        "history_dcd_dir": "Directory of trajectory files (.dcd/.xtc) to bootstrap history.",
        "replay_only_dcd_dir": (
            "External trajectories for replay pool only (excluded from VAMP training)."
        ),
        "history_dcd_stride": "Stride for history trajectories.",
        "history_dcd_max_frames": "Max frames to load from history (0=all).",
        "history_dcd_batch_size": (
            "Frames per mdtraj.iterload chunk when bootstrapping history "
            "(lower reduces peak memory)."
        ),
        "rmsd_filter": "RMSD threshold (nm) for filtering.",
        "save_epoch_coords": {
            "help": "Save coordinates in epoch NPZs (default).",
            "no_help": "Disable saving coordinates in epoch NPZs.",
        },
        "anchoring_source_dir": (
            "Directory with existing anchoring trajectories to bootstrap history."
        ),
    }
