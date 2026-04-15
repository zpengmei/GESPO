"""LOO-VAMP history seeding from geom2vec cache.

Seeds the LOO-VAMP sufficient statistics from all cached historical
trajectories, using the geom2vec projector for state probability
computation.
"""
from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple

import numpy as np

from ..vamp_stats import SufficientStatistics


def ensure_loo_vamp_history_stats(
    *,
    loo_vamp_stats_ref: list,  # mutable [SufficientStatistics or None]
    loo_vamp_stats_keys_ref: list,  # mutable [set]
    loo_vamp_stats_signature_ref: list,  # mutable [tuple or None]
    upto_epoch_exclusive: int,
    n_states: int,
    lag_embed: int,
    tag: str,
    geom2vec_cache: Any,
    geom2vec_projector: Any,
    geom2vec_mgr: Any,
    all_epoch_data: List[List[Any]],
    all_trajectory_lengths: List[List[int]],
    all_epoch_traj_dcd_paths: List[List[Optional[str]]],
    all_epoch_vamp_exclude: List[bool],
) -> None:
    """Ensure LOO-VAMP sufficient statistics are seeded from cached history.

    Uses mutable list containers for the stats/keys/signature so the caller's
    nonlocal variables are updated in place.
    """
    if geom2vec_cache is None or geom2vec_projector is None or not geom2vec_mgr.is_trained():
        return

    sig = (int(n_states), int(lag_embed))
    if loo_vamp_stats_ref[0] is None or loo_vamp_stats_signature_ref[0] != sig:
        loo_vamp_stats_ref[0] = SufficientStatistics(int(n_states))
        loo_vamp_stats_keys_ref[0] = set()
        loo_vamp_stats_signature_ref[0] = sig

    stats = loo_vamp_stats_ref[0]
    keys_seen: Set[str] = loo_vamp_stats_keys_ref[0]

    upto = max(0, min(int(upto_epoch_exclusive), len(all_epoch_data)))
    if upto == 0:
        return

    added = 0
    skipped_short = 0
    skipped_missing = 0

    for epoch_idx in range(upto):
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
            if L_i <= 0:
                continue
            dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
            key = geom2vec_mgr.cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)
            cursor_next = cursor + L_i

            if key in keys_seen:
                cursor = cursor_next
                continue

            cached_key = None
            if geom2vec_mgr.has_cached_key(key):
                cached_key = key
            else:
                coords_nm = None
                if not dcd_path:
                    try:
                        coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor:cursor_next]]
                        coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                    except Exception:
                        coords_nm = None
                cached_key = geom2vec_mgr.ensure_cached(
                    key=key, dcd_path=dcd_path, coords_nm=coords_nm, overwrite=False,
                    meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                )

            cursor = cursor_next

            if cached_key is None:
                skipped_missing += 1
                continue
            if cached_key in keys_seen:
                continue

            try:
                rec = geom2vec_cache.get(cached_key)
                if int(rec.frames) <= int(lag_embed):
                    keys_seen.add(str(cached_key))
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
                probs = np.asarray(probs_raw, dtype=np.float32)
                T = int(probs.shape[0])
                if T > lag_embed:
                    stats.add_trajectory(probs[:-lag_embed], probs[lag_embed:])
            except Exception:
                skipped_missing += 1
                continue

            keys_seen.add(str(cached_key))
            added += 1

    if added:
        print(
            f"[LOO VAMP] Seeded global stats ({tag}): added_trajs={added}, "
            f"skipped_short={skipped_short}, skipped_missing={skipped_missing}"
        )
