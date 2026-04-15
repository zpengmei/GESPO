"""Shared helper functions for GESPO training loops.

These were originally identical closures inside train() in both
loop_generative.py and loop_generative_path.py. They are now
standalone functions that take explicit parameters instead of
capturing train()-local variables.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple

import numpy as np


# ── History iteration ───────────────────────────────────────────────

def iter_history_trajs(
    all_epoch_data: List[List[Any]],
    all_trajectory_lengths: List[List[int]],
    all_epoch_traj_dcd_paths: List[List[Optional[str]]],
    all_epoch_vamp_exclude: List[bool],
    upto_epoch_exclusive: int,
) -> Iterator[Tuple[int, int, int, Optional[str], Optional[np.ndarray]]]:
    """Yield ``(epoch_idx, traj_idx, L_i, dcd_path, coords_nm)`` from history."""
    upto = max(0, min(int(upto_epoch_exclusive), len(all_epoch_data)))
    for epoch_idx in range(upto):
        if epoch_idx < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[epoch_idx]:
            continue
        epoch_data = all_epoch_data[epoch_idx]
        traj_lens = (
            all_trajectory_lengths[epoch_idx]
            if epoch_idx < len(all_trajectory_lengths)
            else []
        )
        traj_paths = (
            all_epoch_traj_dcd_paths[epoch_idx]
            if epoch_idx < len(all_epoch_traj_dcd_paths)
            else []
        )
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
                    coords_list = [
                        np.asarray(fr[1], dtype=np.float32)
                        for fr in epoch_data[cursor:cursor_next]
                    ]
                    if coords_list and len(coords_list) == L_i:
                        coords_nm = np.asarray(coords_list, dtype=np.float32)
                except Exception:
                    coords_nm = None
            cursor = cursor_next
            yield int(epoch_idx), int(traj_idx), int(L_i), dcd_path, coords_nm


# ── Retrieval cache utilities ───────────────────────────────────────

def retrieval_cache_cap(config) -> int:
    """Get the max retrieval cache size from config (0 = unlimited)."""
    return int(getattr(config, "retrieval_cache_max_frames", 0) or 0)


def key_source_bucket(key: str) -> str:
    """Classify an embedding cache key into its source bucket."""
    key_lower = key.lower()
    if "anchoring" in key_lower:
        return "anchoring"
    if "history" in key_lower or "external" in key_lower:
        return "history"
    return "epoch"


# ── Invalidation helpers ────────────────────────────────────────────

def invalidate_vamp_bin(vamp_bin_mgr, tag: str) -> None:
    """Reset VAMP-bin histogram state."""
    if vamp_bin_mgr is not None:
        vamp_bin_mgr.invalidate(tag)


def invalidate_msm_ig(msm_ig_mgr, tag: str) -> None:
    """Reset MSM-IG count state."""
    if msm_ig_mgr is not None:
        msm_ig_mgr.invalidate(tag)


# ── Seeding helpers ─────────────────────────────────────────────────

def ensure_vamp_bin_history_counts(
    *,
    upto_epoch_exclusive: int,
    n_states: int,
    tag: str,
    config,
    vamp_bin_mgr,
    geom2vec_cache,
    geom2vec_projector_linear,
    geom2vec_mgr,
    all_epoch_data,
    all_trajectory_lengths,
    all_epoch_traj_dcd_paths,
    all_epoch_vamp_exclude,
) -> None:
    """Seed VAMP-bin histogram from history trajectories."""
    if vamp_bin_mgr is None:
        return
    iter_fn = lambda upto: iter_history_trajs(
        all_epoch_data, all_trajectory_lengths,
        all_epoch_traj_dcd_paths, all_epoch_vamp_exclude, upto,
    )
    vamp_bin_mgr.ensure_history_counts(
        upto_epoch_exclusive=upto_epoch_exclusive,
        n_states=n_states,
        tag=tag,
        config=config,
        geom2vec_cache=geom2vec_cache,
        geom2vec_projector_linear=geom2vec_projector_linear,
        geom2vec_mgr=geom2vec_mgr,
        iter_history_trajs_fn=iter_fn,
    )


def ensure_msm_ig_history_counts(
    *,
    upto_epoch_exclusive: int,
    n_states: int,
    lag_embed: int,
    tag: str,
    config,
    msm_ig_mgr,
    geom2vec_cache,
    geom2vec_projector_linear,
    geom2vec_mgr,
    all_epoch_data,
    all_trajectory_lengths,
    all_epoch_traj_dcd_paths,
    all_epoch_vamp_exclude,
) -> None:
    """Seed MSM-IG transition counts from history trajectories."""
    if msm_ig_mgr is None:
        return
    iter_fn = lambda upto: iter_history_trajs(
        all_epoch_data, all_trajectory_lengths,
        all_epoch_traj_dcd_paths, all_epoch_vamp_exclude, upto,
    )
    msm_ig_mgr.ensure_history_counts(
        upto_epoch_exclusive=upto_epoch_exclusive,
        n_states=n_states,
        lag_embed=lag_embed,
        tag=tag,
        config=config,
        geom2vec_cache=geom2vec_cache,
        geom2vec_projector_linear=geom2vec_projector_linear,
        geom2vec_mgr=geom2vec_mgr,
        iter_history_trajs_fn=iter_fn,
    )


# ── Save policy checkpoint ──────────────────────────────────────────

def save_torsion_ar_policy_checkpoint(
    ep_idx_1based: int,
    config,
    agent,
    agent_old,
    ar_cfg: dict,
    torsion_ckpt: str,
    env_ref,
    residue_types_1d,
    global_token_names: list,
) -> None:
    """Save TorsionAR policy checkpoint (delegates to policy_io)."""
    from gespo.training.components.policy_io import save_torsion_ar_policy

    save_torsion_ar_policy(
        agent=agent,
        agent_old=agent_old,
        ar_cfg=ar_cfg,
        torsion_ckpt=torsion_ckpt,
        env_ref=env_ref,
        residue_types_1d=residue_types_1d,
        global_token_names=global_token_names,
        ep_idx_1based=ep_idx_1based,
        config=config,
    )
