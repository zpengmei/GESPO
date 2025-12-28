"""Helpers for anchoring-round setup in training loops."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from gespo.utils.logging import log_print

# Local imports
from ..core.checkpointing import CheckpointManager
from ...config import TrainingConfig
from ...sampling import PeptideSampler


def run_anchoring_round(
    *,
    config: TrainingConfig,
    ckpt_manager: CheckpointManager,
    all_epoch_data: list,
    all_trajectory_lengths: List[List[int]],
    all_epoch_traj_dcd_paths: List[List[Optional[str]]],
    all_epoch_traj_dcd_frame_indices: List[List[Optional[np.ndarray]]],
    env_ref: PeptideSampler,
) -> None:
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
            imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
                all_epoch_data,
                env_ref,
                base_dir=source_dir,
            )
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
            log_print(
                f"[Anchoring] Detected existing anchoring DCDs in {source_dir}; "
                f"skipped rerun ({imported} frames loaded)."
            )
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
            log_print(
                f"[Anchoring] Detected existing anchoring data in {source_dir}; "
                f"skipped rerun ({restored} NPZ epochs loaded)."
            )
            return

    log_print("[Anchoring] Running initial anchoring trajectories...")
    anchor_env = env_ref
    n_anchors = int(getattr(config, "n_anchoring_trajectories", 2) or 2)
    anchor_steps = int(getattr(config, "anchoring_steps", 2000) or 2000)
    anchor_stride = int(getattr(config, "anchoring_stride", 10) or 10)

    for i in range(n_anchors):
        anchor_env.reset()
        traj = []
        for t in range(anchor_steps):
            anchor_env.step(None)
            if t % anchor_stride == 0:
                traj.append(anchor_env.state_snapshot())
        all_epoch_data.append(traj)
        all_trajectory_lengths.append([len(traj)])
        all_epoch_traj_dcd_paths.append([None])
        all_epoch_traj_dcd_frame_indices.append([None])

    if anchor_dir:
        ckpt_manager.save_history_npz(all_epoch_data, base_dir=anchor_dir)
        log_print(f"[Anchoring] Saved anchoring trajectories to {anchor_dir}")
