"""VAMP lobe refresh: retrain geom2vec projectors and realign state indices."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gespo.training.components.geom2vec_manager import Geom2VecManager, align_states_by_overlap


def _natural_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
    parts = re.split(r"(\d+)", text)
    key: list = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part.lower()))
    return tuple(key)


def _key_source_bucket(key: str) -> str:
    """Extract a human-readable source bucket from a cache key."""
    s = str(key)
    if "/" in s:
        parts = s.rsplit("/", 1)
        return parts[0].rsplit("/", 1)[-1] if "/" in parts[0] else parts[0]
    return "unknown"


def refresh_geom2vec_vamp(
    reason: str,
    *,
    config: Any,
    geom2vec_mgr: Geom2VecManager,
    geom2vec_cache: Any,
    geom2vec_projector: Any,
    geom2vec_projector_linear: Any,
    geom2vec_embed_stride: int,
    geom2vec_state_permutation_map: np.ndarray,
    all_epoch_data: List[List[Any]],
    all_trajectory_lengths: List[List[int]],
    all_epoch_traj_dcd_paths: List[List[Optional[str]]],
    all_epoch_vamp_exclude: List[bool],
    invalidate_callbacks: List[Callable[[str], None]],
    refresh_retrieval_labels_fn: Optional[Callable[[str], None]] = None,
    ensure_msm_ig_fn: Optional[Callable] = None,
    reset_policy_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Retrain geom2vec VAMP lobe and realign state permutation.

    Returns
    -------
    perm_map : ndarray
        Updated state permutation map.
    inv_perm_map : ndarray
        Updated inverse permutation map.
    lagtime_embed : int
        Updated lagtime in embedding frames.
    """
    if not all_epoch_data:
        print(f"[Geom2Vec Refresh] Skipped ({reason}): no history frames available.")
        return geom2vec_state_permutation_map, np.arange(len(geom2vec_state_permutation_map)), 1

    lag_raw_i = int(getattr(config, "geom2vec_lagtime", 1) or 1)
    embed_stride_i = max(1, int(geom2vec_embed_stride))
    lagtime_embed = max(1, int(round(max(1, lag_raw_i) / embed_stride_i)))
    lag_embed = int(lagtime_embed)

    # Ensure embeddings exist for all historical trajectories.
    all_keys: List[str] = []
    for epoch_idx, epoch_data in enumerate(all_epoch_data):
        vamp_excluded = epoch_idx < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[epoch_idx]
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
            if geom2vec_mgr.has_cached_key(key):
                if not vamp_excluded:
                    all_keys.append(str(key))
                cursor += L_i
                continue
            coords_nm = None
            try:
                coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor:cursor + L_i]]
                coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
            except Exception:
                coords_nm = None
            cached_key = geom2vec_mgr.ensure_cached(
                key=key, dcd_path=dcd_path, coords_nm=coords_nm, overwrite=False,
                meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
            )
            if cached_key is not None and not vamp_excluded:
                all_keys.append(str(cached_key))
            cursor += L_i

    if not all_keys:
        print(f"[Geom2Vec Refresh] Skipped ({reason}): no trajectories could be cached.")
        return geom2vec_state_permutation_map, np.arange(len(geom2vec_state_permutation_map)), lagtime_embed

    keys_sorted = sorted(set(all_keys), key=_natural_sort_key)

    max_train_raw = int(getattr(config, "max_pca_train", 0) or 0)
    max_train_embed = int(math.ceil(max_train_raw / max(1, int(geom2vec_embed_stride)))) if max_train_raw > 0 else 0

    train_keys: List[str] = []
    total_embed_frames = 0
    for key in reversed(keys_sorted):
        try:
            rec = geom2vec_cache.get(key)
        except KeyError:
            continue
        if int(rec.frames) <= lag_embed:
            continue
        if max_train_embed > 0 and total_embed_frames + int(rec.frames) > max_train_embed and train_keys:
            break
        train_keys.append(key)
        total_embed_frames += int(rec.frames)
    train_keys = list(reversed(train_keys))

    if not train_keys:
        print(f"[Geom2Vec Refresh] Skipped ({reason}): no cached trajectories with frames > lag ({lag_embed}).")
        return geom2vec_state_permutation_map, np.arange(len(geom2vec_state_permutation_map)), lagtime_embed

    # Capture old labels for state alignment after retraining.
    old_labels_by_key: Optional[Dict[str, np.ndarray]] = None
    if geom2vec_mgr.is_trained():
        old_labels_by_key = {}
        for key in train_keys:
            try:
                probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                probs_canon = probs_raw[:, geom2vec_state_permutation_map]
                old_labels_by_key[key] = np.argmax(probs_canon, axis=1).astype(np.int64, copy=False)
            except Exception:
                continue

    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
    pair_stride = 1
    train_fraction = float(getattr(config, "geom2vec_vamp_train_fraction", 0.8) or 0.8)
    n_epochs = int(getattr(config, "geom2vec_vamp_epochs", 10) or 10)
    batch_size = int(getattr(config, "geom2vec_vamp_batch_size", 256) or 256)
    learning_rate = float(getattr(config, "geom2vec_vamp_learning_rate", 5e-4) or 5e-4)
    weight_decay = float(getattr(config, "geom2vec_vamp_weight_decay", 0.0) or 0.0)
    optimizer_name = str(getattr(config, "geom2vec_vamp_optimizer", "Adam") or "Adam")

    try:
        src_counts: Dict[str, int] = {}
        for k in train_keys:
            b = _key_source_bucket(str(k))
            src_counts[b] = int(src_counts.get(b, 0)) + 1
    except Exception:
        src_counts = {}

    src_msg = f", sources={src_counts}" if src_counts else ""
    print(
        f"[Geom2Vec Refresh] {reason}: trajs={len(train_keys)}, embed_frames~{total_embed_frames}, "
        f"lag_embed={lag_embed}, pair_stride={pair_stride}{src_msg}"
    )

    # Train softmax lobe.
    try:
        seed_base = int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42)
        epsilon = float(getattr(config, "geom2vec_vamp_epsilon", 1e-6) or 1e-6)
        inverse_mode = str(getattr(config, "geom2vec_vamp_inverse_mode", "regularize") or "regularize")
        vamp_merger = bool(getattr(config, "geom2vec_vamp_merger", False))
        vamp_merger_window = int(getattr(config, "geom2vec_vamp_merger_window", 6) or 6)
        vamp_double_lobe = bool(getattr(config, "geom2vec_vamp_double_lobe", False))
        lobe_invariant = bool(getattr(config, "geom2vec_lobe_invariant", False))

        lobe_kwargs: Dict[str, Any] = {}
        try:
            hidden_channels = int(getattr(config, "geom2vec_hidden_channels", 0) or 0)
            if hidden_channels > 0:
                lobe_kwargs["hidden_channels"] = int(hidden_channels)
        except Exception:
            pass

        geom2vec_projector.fit_from_cache(
            keys=train_keys, n_states=n_states, lag_time=lag_embed, seed=int(seed_base),
            pair_stride=pair_stride, train_fraction=train_fraction, batch_size=batch_size,
            n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
            optimizer=optimizer_name, epsilon=epsilon, inverse_mode=inverse_mode,
            softmax_output=True, vamp_merger=vamp_merger, vamp_merger_window=vamp_merger_window,
            lobe_kwargs=lobe_kwargs, double_lobe=vamp_double_lobe, lobe_invariant=lobe_invariant,
        )
        # Linear lobe.
        if geom2vec_projector_linear is not None:
            try:
                geom2vec_projector_linear.fit_from_cache(
                    keys=train_keys, n_states=n_states, lag_time=lag_embed, seed=int(seed_base) + 1000,
                    pair_stride=pair_stride, train_fraction=train_fraction, batch_size=batch_size,
                    n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay,
                    optimizer=optimizer_name, epsilon=epsilon, inverse_mode=inverse_mode,
                    softmax_output=False, vamp_merger=vamp_merger, vamp_merger_window=vamp_merger_window,
                    lobe_kwargs=lobe_kwargs, double_lobe=vamp_double_lobe, lobe_invariant=lobe_invariant,
                )
            except Exception as exc_lin:
                print(f"[Geom2Vec Refresh] Warning: failed to train linear VAMP lobe (keeping previous): {exc_lin}")
    except Exception as exc:
        print(f"[Geom2Vec Refresh] CRITICAL ERROR: Failed to train geom2vec softmax VAMP lobe: {exc}")
        import traceback
        traceback.print_exc()
        return geom2vec_state_permutation_map, np.arange(len(geom2vec_state_permutation_map)), lagtime_embed

    # Log fit outcomes.
    try:
        outcomes_sm = getattr(geom2vec_projector, "last_fit_outcomes", None)
        if outcomes_sm:
            rec = outcomes_sm[0]
            print(
                f"[Geom2Vec Refresh] Softmax fit (softmax_output={rec.get('softmax_output')}): "
                f"val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
            )
        if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "last_fit_outcomes", None):
            rec = geom2vec_projector_linear.last_fit_outcomes[0]
            print(
                f"[Geom2Vec Refresh] Linear fit (softmax_output={rec.get('softmax_output')}): "
                f"val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
            )
    except Exception:
        pass

    # Realign state labels.
    perm_map: np.ndarray
    inv_perm_map: np.ndarray
    if old_labels_by_key is not None:
        old_all: List[np.ndarray] = []
        new_all: List[np.ndarray] = []
        for key in train_keys:
            old_lbl = old_labels_by_key.get(key)
            if old_lbl is None:
                continue
            try:
                probs_new_raw = geom2vec_projector.state_probs_from_cache_key(key)
            except Exception:
                continue
            new_lbl = np.argmax(probs_new_raw, axis=1).astype(np.int64, copy=False)
            if new_lbl.shape[0] != old_lbl.shape[0]:
                continue
            old_all.append(old_lbl)
            new_all.append(new_lbl)

        if old_all and new_all:
            old_flat = np.concatenate(old_all, axis=0)
            new_flat = np.concatenate(new_all, axis=0)
            try:
                perm_map = align_states_by_overlap(old_labels=old_flat, new_labels=new_flat, n_states=n_states)
                inv_perm_map = np.zeros_like(perm_map)
                inv_perm_map[perm_map] = np.arange(len(perm_map))
                print(f"[Geom2Vec Refresh] Applied confusion-matrix alignment: {perm_map}")
            except Exception as exc_align:
                print(f"[Geom2Vec Refresh] Warning: overlap alignment failed: {exc_align}; using identity mapping.")
                perm_map = np.arange(n_states, dtype=int)
                inv_perm_map = np.arange(n_states, dtype=int)
        else:
            perm_map = np.arange(n_states, dtype=int)
            inv_perm_map = np.arange(n_states, dtype=int)
    else:
        perm_map = np.arange(n_states, dtype=int)
        inv_perm_map = np.arange(n_states, dtype=int)

    # Save lobe checkpoints.
    if config.snapshot_dir:
        out_dir = Path(config.snapshot_dir) / "geom2vec_vamp"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            lobe_softmax = geom2vec_projector._require_lobe()
            torch.save(lobe_softmax.state_dict(), out_dir / "lobe_state_dict.pt")
            torch.save(lobe_softmax.state_dict(), out_dir / "lobe_state_dict_softmax.pt")
            if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "trained", None) is not None:
                lobe_lin = geom2vec_projector_linear._require_lobe()
                torch.save(lobe_lin.state_dict(), out_dir / "lobe_state_dict_linear.pt")
            torch.save(
                {
                    "train_keys": train_keys,
                    "lag_embed": int(lag_embed),
                    "embed_stride": int(geom2vec_embed_stride),
                    "state_permutation_map": np.asarray(perm_map, dtype=np.int64),
                    "state_inverse_permutation_map": np.asarray(inv_perm_map, dtype=np.int64),
                    "softmax_fit_outcomes": getattr(geom2vec_projector, "last_fit_outcomes", None),
                    "linear_fit_outcomes": (
                        getattr(geom2vec_projector_linear, "last_fit_outcomes", None)
                        if geom2vec_projector_linear is not None else None
                    ),
                },
                out_dir / "lobe_meta.pt",
            )
            print(f"[Geom2Vec Refresh] Saved lobe to {out_dir}")
        except Exception as exc_save:
            print(f"[Geom2Vec Refresh] Warning: failed to save lobe weights/meta: {exc_save}")

    # Invalidate downstream caches.
    tag = f"geom2vec refresh: {reason}"
    for cb in invalidate_callbacks:
        try:
            cb(tag)
        except Exception:
            pass

    if refresh_retrieval_labels_fn is not None:
        try:
            refresh_retrieval_labels_fn(tag)
        except Exception as exc_lbl:
            print(f"[Retrieval Cache] Warning: failed to refresh VAMP labels ({exc_lbl})")

    if ensure_msm_ig_fn is not None:
        try:
            if float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0) > 0.0:
                ensure_msm_ig_fn(tag)
        except Exception as exc_msm:
            print(f"[MSM IG] Warning: failed to refit/seed after refresh ({exc_msm})", flush=True)

    if reset_policy_fn is not None:
        try:
            reset_policy_fn(tag)
        except Exception:
            pass

    return perm_map, inv_perm_map, lagtime_embed
