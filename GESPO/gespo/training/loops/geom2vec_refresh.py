"""Helpers for training/refreshing Geom2Vec VAMP lobes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch

from gespo.utils.logging import log_print

from ..geom2vec_utils import align_states_by_overlap
from ..rl import StateLogitPolicy
from ..geom2vec_backbone import EmbeddingCache, Geom2VecVAMPProjector
from ...config import TrainingConfig


@dataclass
class PolicyResetHandles:
    agent: StateLogitPolicy
    agent_old: StateLogitPolicy
    policy_opts: Sequence[torch.optim.Optimizer]


def refresh_geom2vec_vamp(
    *,
    reason: str,
    config: TrainingConfig,
    all_epoch_data: List[List[Tuple[np.ndarray, np.ndarray]]],
    all_trajectory_lengths: List[List[int]],
    all_epoch_traj_dcd_paths: List[List[Optional[str]]],
    geom2vec_cache: Optional[EmbeddingCache],
    geom2vec_projector: Optional[Geom2VecVAMPProjector],
    geom2vec_projector_linear: Optional[Geom2VecVAMPProjector],
    geom2vec_embed_stride: int,
    geom2vec_state_permutation_map: np.ndarray,
    geom2vec_state_inverse_permutation_map: np.ndarray,
    geom2vec_cache_key_for_history_traj: Callable[[int, int, Optional[str]], str],
    geom2vec_cache_has_key: Callable[[str], bool],
    geom2vec_ensure_cached: Callable[..., Optional[str]],
    invalidate_callbacks: Optional[Iterable[Callable[[str], None]]] = None,
    refresh_label_callbacks: Optional[Iterable[Callable[[str], None]]] = None,
    policy_reset: Optional[PolicyResetHandles] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Refresh the geom2vec VAMP lobes and return updated state maps + lag."""
    if geom2vec_cache is None or geom2vec_projector is None:
        log_print(f"[Geom2Vec Refresh] Skipped ({reason}): projector/cache not initialized.")
        return geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, int(
            getattr(config, "geom2vec_lagtime", 1) or 1
        )
    if not all_epoch_data:
        log_print(f"[Geom2Vec Refresh] Skipped ({reason}): no history frames available.")
        return geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, int(
            getattr(config, "geom2vec_lagtime", 1) or 1
        )

    lag_raw = int(getattr(config, "geom2vec_lagtime", 1) or 1)
    lag_embed = max(1, int(round(lag_raw / max(1, int(geom2vec_embed_stride)))))

    def _natural_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
        import re

        parts = re.split(r"(\d+)", text)
        key: List[Tuple[int, object]] = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append((1, int(part)))
            else:
                key.append((0, part.lower()))
        return tuple(key)

    all_keys: List[str] = []
    for epoch_idx, epoch_data in enumerate(all_epoch_data):
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
            key = geom2vec_cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)

            if geom2vec_cache_has_key(key):
                all_keys.append(str(key))
                cursor += L_i
                continue

            coords_nm = None
            try:
                coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor : cursor + L_i]]
                coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
            except Exception:
                coords_nm = None

            cached_key = geom2vec_ensure_cached(
                key=key,
                dcd_path=dcd_path,
                coords_nm=coords_nm,
                overwrite=False,
                meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
            )
            if cached_key is not None:
                all_keys.append(str(cached_key))

            cursor += L_i

    if not all_keys:
        log_print(f"[Geom2Vec Refresh] Skipped ({reason}): no trajectories could be cached.")
        return geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, lag_embed

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
        log_print(
            f"[Geom2Vec Refresh] Skipped ({reason}): no cached trajectories with frames > lag ({lag_embed})."
        )
        return geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, lag_embed

    old_labels_by_key: Optional[dict[str, np.ndarray]] = None
    if getattr(geom2vec_projector, "trained", None) is not None:
        old_labels_by_key = {}
        for key in train_keys:
            try:
                probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                probs_canon = probs_raw[:, geom2vec_state_permutation_map]
                old_labels_by_key[key] = np.argmax(probs_canon, axis=1).astype(np.int64, copy=False)
            except Exception:
                continue

    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
    pair_stride = int(getattr(config, "geom2vec_vamp_pair_stride", 1) or 1)
    train_fraction = float(getattr(config, "geom2vec_vamp_train_fraction", 0.8) or 0.8)
    n_epochs = int(getattr(config, "geom2vec_vamp_epochs", 10) or 10)
    batch_size = int(getattr(config, "geom2vec_vamp_batch_size", 256) or 256)
    learning_rate = float(getattr(config, "geom2vec_vamp_learning_rate", 5e-4) or 5e-4)
    weight_decay = float(getattr(config, "geom2vec_vamp_weight_decay", 0.0) or 0.0)
    optimizer_name = str(getattr(config, "geom2vec_vamp_optimizer", "Adam") or "Adam")

    log_print(
        f"[Geom2Vec Refresh] {reason}: trajs={len(train_keys)}, embed_frames~{total_embed_frames}, "
        f"lag_embed={lag_embed}, pair_stride={pair_stride}"
    )

    try:
        seed_base = int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42)
        epsilon = float(getattr(config, "geom2vec_vamp_epsilon", 1e-6) or 1e-6)
        inverse_mode = str(getattr(config, "geom2vec_vamp_inverse_mode", "regularize") or "regularize")
        geom2vec_projector.fit_from_cache(
            keys=train_keys,
            n_states=n_states,
            lag_time=lag_embed,
            seed=int(seed_base),
            pair_stride=pair_stride,
            train_fraction=train_fraction,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer_name,
            epsilon=epsilon,
            inverse_mode=inverse_mode,
            softmax_output=True,
        )
        if geom2vec_projector_linear is not None:
            try:
                geom2vec_projector_linear.fit_from_cache(
                    keys=train_keys,
                    n_states=n_states,
                    lag_time=lag_embed,
                    seed=int(seed_base) + 1000,
                    pair_stride=pair_stride,
                    train_fraction=train_fraction,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    optimizer=optimizer_name,
                    epsilon=epsilon,
                    inverse_mode=inverse_mode,
                    softmax_output=False,
                )
            except Exception as exc_lin:
                log_print(
                    f"[Geom2Vec Refresh] Warning: failed to train linear VAMP lobe (keeping previous): {exc_lin}"
                )
    except Exception as exc:
        log_print(f"[Geom2Vec Refresh] CRITICAL ERROR: Failed to train geom2vec softmax VAMP lobe: {exc}")
        try:
            outcomes = getattr(geom2vec_projector, "last_fit_outcomes", None)
            if outcomes:
                log_print("[Geom2Vec Refresh] Fit outcomes:")
                for rec in outcomes:
                    if rec.get("ok"):
                        continue
                    midx = rec.get("model_idx", "?")
                    err = rec.get("error", rec)
                    log_print(f"  - model {midx}: {err}")
        except Exception:
            pass
        import traceback

        traceback.print_exc()
        return geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, lag_embed

    try:
        outcomes_sm = getattr(geom2vec_projector, "last_fit_outcomes", None)
        if outcomes_sm:
            rec = outcomes_sm[0]
            log_print(
                f"[Geom2Vec Refresh] Softmax fit: val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
            )
        if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "last_fit_outcomes", None):
            rec = geom2vec_projector_linear.last_fit_outcomes[0]
            log_print(
                f"[Geom2Vec Refresh] Linear fit: val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
            )
    except Exception:
        pass

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
                perm = align_states_by_overlap(old_labels=old_flat, new_labels=new_flat, n_states=n_states)
                geom2vec_state_permutation_map = perm
                inv_perm = np.zeros_like(perm)
                inv_perm[perm] = np.arange(len(perm))
                geom2vec_state_inverse_permutation_map = inv_perm
                log_print(f"[Geom2Vec Refresh] Applied confusion-matrix alignment: {perm}")
            except Exception as exc_align:
                log_print(
                    f"[Geom2Vec Refresh] Warning: overlap alignment failed: {exc_align}; using identity mapping."
                )
                identity = np.arange(n_states, dtype=int)
                geom2vec_state_permutation_map = identity
                geom2vec_state_inverse_permutation_map = identity
    else:
        identity = np.arange(n_states, dtype=int)
        geom2vec_state_permutation_map = identity
        geom2vec_state_inverse_permutation_map = identity

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
                    "state_permutation_map": np.asarray(geom2vec_state_permutation_map, dtype=np.int64),
                    "state_inverse_permutation_map": np.asarray(geom2vec_state_inverse_permutation_map, dtype=np.int64),
                    "softmax_fit_outcomes": getattr(geom2vec_projector, "last_fit_outcomes", None),
                    "linear_fit_outcomes": (
                        getattr(geom2vec_projector_linear, "last_fit_outcomes", None)
                        if geom2vec_projector_linear is not None
                        else None
                    ),
                },
                out_dir / "lobe_meta.pt",
            )
            log_print(f"[Geom2Vec Refresh] Saved lobe to {out_dir}")
        except Exception as exc_save:
            log_print(f"[Geom2Vec Refresh] Warning: failed to save lobe weights/meta: {exc_save}")

    tag = f"geom2vec refresh: {reason}"
    if invalidate_callbacks:
        for callback in invalidate_callbacks:
            try:
                callback(tag)
            except Exception:
                continue
    if refresh_label_callbacks:
        for callback in refresh_label_callbacks:
            try:
                callback(tag)
            except Exception:
                continue

    if getattr(config, "reset_policy_on_projector_refresh", False) and policy_reset is not None:
        try:
            if isinstance(policy_reset.agent, StateLogitPolicy) and isinstance(policy_reset.agent_old, StateLogitPolicy):
                with torch.no_grad():
                    if getattr(policy_reset.agent, "logits", None) is not None:
                        policy_reset.agent.logits.zero_()
                    if getattr(policy_reset.agent_old, "logits", None) is not None:
                        policy_reset.agent_old.logits.zero_()
                for opt in policy_reset.policy_opts:
                    opt.state.clear()
                log_print("[Geom2Vec Refresh] Reset StateLogitPolicy logits to uniform after refresh.")
        except Exception as exc_reset:
            log_print(f"[Geom2Vec Refresh] Warning: failed to reset policy after refresh: {exc_reset}")

    return geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, lag_embed
