"""Retrieval cache label refresh logic.

Assigns discrete state labels to the in-memory replay buffer for balanced
sampling. Supports VAMP argmax labels with an optional linear-VAMP+kmeans
fallback when VAMP labels collapse.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def refresh_retrieval_cache_labels(
    *,
    retrieval_mgr: Any,
    config: Any,
    geom2vec_cache: Any,
    geom2vec_projector: Any,
    geom2vec_projector_linear: Any,
    geom2vec_mgr: Any,
    geom2vec_state_permutation_map: np.ndarray,
    geom2vec_embed_stride: int,
    retrieval_cache_cap_fn: Callable[[], int],
    key_source_bucket_fn: Callable[[str], str],
    tag: str,
    only_indices: Optional[np.ndarray] = None,
) -> None:
    """Assign discrete state labels to the retrieval cache.

    When ``only_indices`` is provided, only updates those positions in-place.
    """
    if not retrieval_mgr.cv:
        return
    if geom2vec_cache is None:
        return

    label_mode = str(getattr(config, "geom2vec_label_mode", "auto") or "auto").strip().lower()
    force_kmeans = label_mode in {"linear_kmeans", "vac_kmeans", "linear-kmeans", "kmeans"}
    force_vamp = label_mode == "vamp"

    if force_kmeans:
        if geom2vec_projector_linear is None or getattr(geom2vec_projector_linear, "trained", None) is None:
            return
    else:
        if geom2vec_projector is None or not geom2vec_mgr.is_trained():
            return
    if len(retrieval_mgr.keys) != len(retrieval_mgr.cv) or len(retrieval_mgr.frame_idx) != len(retrieval_mgr.cv):
        retrieval_mgr.labels = np.zeros((len(retrieval_mgr.cv),), dtype=np.int64)
        print(f"[Retrieval Cache] Warning: missing replay key metadata; labels reset to zeros ({tag}).")
        return

    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
    perm = np.asarray(geom2vec_state_permutation_map, dtype=int)
    if perm.ndim != 1 or int(perm.shape[0]) != int(n_states):
        perm = np.arange(n_states, dtype=int)

    groups: Dict[str, List[int]] = {}
    idx_arr: Optional[np.ndarray] = None
    if only_indices is not None and not force_kmeans:
        try:
            idx_arr = np.asarray(only_indices, dtype=np.int64).reshape(-1)
            idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(retrieval_mgr.cv))]
        except Exception:
            idx_arr = None
    if idx_arr is None:
        for i, key in enumerate(retrieval_mgr.keys):
            if key:
                groups.setdefault(key, []).append(int(i))
    else:
        for pos in idx_arr.tolist():
            key = retrieval_mgr.keys[int(pos)]
            if key:
                groups.setdefault(key, []).append(int(pos))

    if not groups:
        if retrieval_mgr.labels is None or retrieval_mgr.labels.shape[0] != len(retrieval_mgr.cv):
            retrieval_mgr.labels = np.zeros((len(retrieval_mgr.cv),), dtype=np.int64)
        return

    if retrieval_mgr.labels is None or retrieval_mgr.labels.shape[0] != len(retrieval_mgr.cv) or idx_arr is None:
        labels_out = np.zeros((len(retrieval_mgr.cv),), dtype=np.int64)
    else:
        labels_out = np.asarray(retrieval_mgr.labels, dtype=np.int64).copy()
        labels_out[idx_arr] = 0
    labels_mode_used = "linear_kmeans" if force_kmeans else "vamp"
    missing = 0
    if not force_kmeans:
        for key, idxs in groups.items():
            try:
                rec = geom2vec_cache.get(key)
            except KeyError:
                missing += 1
                continue
            stride_eff = int(rec.meta.get("stride", geom2vec_embed_stride) or geom2vec_embed_stride)
            stride_eff = max(1, stride_eff)

            try:
                probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                probs = probs_raw[:, perm]
                if probs.shape[1] != n_states:
                    if probs.shape[1] > n_states:
                        probs = probs[:, :n_states]
                    else:
                        probs = np.pad(probs, ((0, 0), (0, n_states - probs.shape[1])), mode="constant")
                labels_embed = np.argmax(probs, axis=1).astype(np.int64, copy=False)
            except Exception as exc_probs:
                print(f"[Retrieval Cache] Warning: failed to compute VAMP labels for key={key} ({exc_probs})")
                continue

            T = int(labels_embed.shape[0])
            if T <= 0:
                continue
            for pos in idxs:
                raw_i = int(retrieval_mgr.frame_idx[pos])
                embed_i = raw_i // stride_eff
                if embed_i < 0:
                    embed_i = 0
                elif embed_i >= T:
                    embed_i = T - 1
                labels_out[pos] = int(labels_embed[embed_i])

    # Fallback: if VAMP labels collapse, use k-means on linear-VAMP coords.
    if idx_arr is None and int(n_states) >= 2 and not force_vamp:
        counts0 = np.zeros((int(n_states),), dtype=np.int64)
        total0 = 0
        occ0 = 0
        frac0 = 1.0
        collapsed = False
        try:
            counts0 = np.bincount(labels_out.clip(0, n_states - 1), minlength=n_states)
            total0 = int(counts0.sum())
            occ0 = int(np.count_nonzero(counts0))
            frac0 = float(counts0.max() / float(total0)) if total0 > 0 else 1.0
            collapsed = (total0 > 0) and (occ0 < 2 or frac0 > 0.995)
        except Exception:
            collapsed = False

        if (force_kmeans or (collapsed and (not force_vamp))) and geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "trained", None) is not None:
            try:
                try:
                    from sklearn.cluster import MiniBatchKMeans
                except Exception:
                    MiniBatchKMeans = None

                z_blocks: List[np.ndarray] = []
                key_meta: List[Tuple[str, List[int], int, int]] = []
                d_min: Optional[int] = None
                for key, idxs in groups.items():
                    try:
                        rec = geom2vec_cache.get(key)
                        stride_eff = int(rec.meta.get("stride", geom2vec_embed_stride) or geom2vec_embed_stride)
                        stride_eff = max(1, stride_eff)
                    except Exception:
                        stride_eff = max(1, int(geom2vec_embed_stride))
                    try:
                        z = np.asarray(geom2vec_projector_linear.state_output_from_cache_key(key), dtype=np.float32)
                    except Exception:
                        continue
                    if z.ndim != 2 or z.shape[0] <= 0:
                        continue
                    d_min = int(z.shape[1]) if d_min is None else int(min(d_min, int(z.shape[1])))
                    z_blocks.append(z)
                    key_meta.append((str(key), list(idxs), int(stride_eff), int(z.shape[0])))

                if z_blocks and key_meta and d_min is not None and d_min > 0:
                    z_cat = np.concatenate([z[:, :d_min] for z in z_blocks], axis=0).astype(np.float32, copy=False)
                    if z_cat.shape[0] >= int(n_states):
                        mu = z_cat.mean(axis=0, keepdims=True)
                        std = z_cat.std(axis=0, keepdims=True)
                        std = np.where(std < 1e-6, 1.0, std)
                        z_norm = ((z_cat - mu) / std).astype(np.float32, copy=False)

                        max_fit = int(getattr(config, "max_pca_train", 0) or 50000)
                        max_fit = max(1000, min(50000, max_fit))
                        if z_norm.shape[0] > max_fit:
                            rng = np.random.default_rng(int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42))
                            sel = rng.choice(z_norm.shape[0], size=max_fit, replace=False)
                            fit_arr = z_norm[sel]
                        else:
                            fit_arr = z_norm

                        kmeans = None
                        if MiniBatchKMeans is not None:
                            seed_k = int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42) + 77
                            kmeans = MiniBatchKMeans(n_clusters=int(n_states), random_state=int(seed_k))
                            kmeans.fit(np.asarray(fit_arr, dtype=np.float64))

                        if kmeans is not None:
                            labels_new = np.zeros_like(labels_out)
                            for key, idxs, stride_eff, T in key_meta:
                                z_key = z_blocks.pop(0)[:, :d_min].astype(np.float32, copy=False)
                                z_key_norm = ((z_key - mu) / std).astype(np.float32, copy=False)
                                lab_embed = np.asarray(
                                    kmeans.predict(np.asarray(z_key_norm, dtype=np.float64)), dtype=np.int64
                                ).reshape(-1)
                                if lab_embed.shape[0] != int(T):
                                    T = int(min(int(T), int(lab_embed.shape[0])))
                                for pos in idxs:
                                    raw_i = int(retrieval_mgr.frame_idx[pos])
                                    embed_i = raw_i // int(stride_eff)
                                    if embed_i < 0:
                                        embed_i = 0
                                    elif embed_i >= T:
                                        embed_i = T - 1
                                    labels_new[pos] = int(lab_embed[embed_i])

                            counts1 = np.bincount(labels_new.clip(0, n_states - 1), minlength=n_states)
                            total1 = int(counts1.sum())
                            occ1 = int(np.count_nonzero(counts1))
                            frac1 = float(counts1.max() / float(total1)) if total1 > 0 else 1.0
                            improved = force_kmeans or (occ1 >= 2 and frac1 <= 0.995)
                            if total1 > 0 and occ1 >= 2 and improved:
                                if force_kmeans:
                                    msg = f"[Retrieval Cache] Using linear-VAMP+kmeans labels ({tag})"
                                else:
                                    msg = f"[Retrieval Cache] VAMP labels collapsed ({tag}); using linear-VAMP+kmeans labels"
                                print(
                                    f"{msg}: occ {occ0}->{occ1}, max_frac {frac0:.3f}->{frac1:.3f}, pop={counts1.tolist()}",
                                    flush=True,
                                )
                                labels_out = labels_new
                                labels_mode_used = "linear_kmeans"
            except Exception:
                pass

    retrieval_mgr.labels = labels_out

    try:
        if idx_arr is not None:
            print(
                f"[Retrieval Cache] Updated cluster labels for {int(idx_arr.size)}/{len(labels_out)} frames "
                f"({tag}; mode={labels_mode_used})."
            )
            return
        counts = np.bincount(labels_out.clip(0, n_states - 1), minlength=n_states)
        cap = retrieval_cache_cap_fn()
        cfg_cap = int(getattr(config, "retrieval_cache_max_frames", 0) or 0)
        cfg_max_train = int(getattr(config, "max_pca_train", 0) or 0)
        empty_keys = 0
        src_counts: Dict[str, int] = {}
        try:
            for k in retrieval_mgr.keys:
                if not k:
                    empty_keys += 1
                b = key_source_bucket_fn(str(k))
                src_counts[b] = src_counts.get(b, 0) + 1
        except Exception:
            src_counts = {}
        miss_msg = f", missing_keys={missing}" if missing else ""
        src_msg = f", sources={src_counts}" if src_counts else ""
        cap_msg = (
            f", cap={cap if cap > 0 else 'none'}(retrieval_cache_max_frames={cfg_cap},max_pca_train={cfg_max_train})"
        )
        print(
            f"[Retrieval Cache] Refreshed cluster labels ({tag}; mode={labels_mode_used}): "
            f"frames={len(labels_out)}, keys={len(groups)}, pop={counts.tolist()}, empty_keys={empty_keys}{miss_msg}{cap_msg}{src_msg}"
        )
    except Exception:
        pass
