"""VAMP projector refresh for path-sampling loop.

Extracted from loop_generative_path.py. Handles:
- VAMP softmax + linear lobe training
- VAC-space k-means boundary fitting
- DGA/VAMP committor model fitting
- State permutation alignment
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from gespo.config import TrainingConfig
from gespo.projections.geom2vec import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector
from gespo.projections.geom2vec.vamp_trainer import train_vamp_lobe_from_cache
from gespo.training.dga_committor import (
    DGAKMeansCommittorModel,
    VampDiscreteCommittorModel,
    fit_dga_committor_kmeans,
    fit_vamp_discrete_committor,
)
from gespo.training.components.geom2vec_manager import align_states_by_overlap


def _refresh_vamp_projectors(
    reason: str,
    *,
    config,
    geom2vec_cache,
    geom2vec_projector,
    geom2vec_projector_linear,
    geom2vec_mgr,
    geom2vec_embed_stride,
    geom2vec_state_permutation_map,
    geom2vec_state_inverse_permutation_map,
    geom2vec_lagtime_embed,
    all_epoch_data,
    all_trajectory_lengths,
    all_epoch_traj_dcd_paths,
    all_epoch_vamp_exclude,
    endpoint_ca_ref_a,
    endpoint_ca_ref_b,
    endpoint_token_mask,
    committor_state,
    n_states,
    boundary_mode,
    boundary_plugin_fn,
    boundary_plugin_fit_keys,
    boundary_plugin_fit_nonce,
    _rmsd_to_endpoints,
    _boundary_masks_from_ca,
    _invalidate_loo_vamp_stats,
    _invalidate_vamp_bin_stats,
    _invalidate_msm_ig_stats,
    _invalidate_committor_state,
    _invalidate_boundary_vac_kmeans,
    _refresh_retrieval_cache_labels,
    _ensure_msm_ig_history_counts,
    _ensure_committor_tube_history_counts,
    _iter_history_trajs,
) -> Optional[Dict[str, Any]]:
    """Refresh VAMP (soft assignments) + VAC (linear) projectors from cached embeddings."""
    nonlocal boundary_plugin_fit_keys, boundary_plugin_fit_nonce

    if geom2vec_cache is None or geom2vec_projector is None or geom2vec_projector_linear is None:
        return
    if not all_epoch_data:
        print(f"[VAMP Refresh] Skipped ({reason}): no history frames available.")
        return
    if endpoint_ca_ref_a is None or endpoint_ca_ref_b is None:
        raise RuntimeError("[VAMP Refresh] Endpoint CA references are not available (check --pdb/--pdb-b).")

    lag_raw_i = int(getattr(config, "geom2vec_lagtime", 1) or 1)
    embed_stride_i = max(1, int(geom2vec_embed_stride))
    geom2vec_lagtime_embed = max(1, int(round(max(1, lag_raw_i) / embed_stride_i)))
    lag_embed = int(geom2vec_lagtime_embed)

    # Committor/DGA model is tied to the linear projector. We invalidate/refit
    # it only when committor reward is enabled, and only after a successful
    # projector refresh (see below).

    def _natural_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
        import re

        parts = re.split(r"(\d+)", str(text))
        key: List[Tuple[int, object]] = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append((1, int(part)))
            else:
                key.append((0, part.lower()))
        return tuple(key)

    # Collect cache keys across all loaded history epochs.
    all_keys: List[str] = []
    for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in _iter_history_trajs(len(all_epoch_data)):
        key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
        cached_key = geom2vec_mgr.ensure_cached(
            key=key,
            dcd_path=dcd_path,
            coords_nm=(None if dcd_path else coords_nm),
            overwrite=False,
            meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
        )
        if cached_key is not None:
            all_keys.append(str(cached_key))

    if not all_keys:
        print(f"[VAMP Refresh] Skipped ({reason}): no trajectories could be cached.")
        return

    keys_sorted = sorted(set(all_keys), key=_natural_sort_key)

    max_train_raw = int(getattr(config, "max_pca_train", 0) or 0)
    max_train_embed = int(math.ceil(max_train_raw / max(1, int(geom2vec_embed_stride)))) if max_train_raw > 0 else 0

    train_keys: List[str] = []
    total_embed_frames = 0
    for key in reversed(keys_sorted):
        try:
            rec = geom2vec_cache.get(key)
        except Exception:
            continue
        if int(rec.frames) <= lag_embed:
            continue
        if max_train_embed > 0 and total_embed_frames + int(rec.frames) > max_train_embed and train_keys:
            break
        train_keys.append(str(key))
        total_embed_frames += int(rec.frames)
    train_keys = list(reversed(train_keys))

    if not train_keys:
        print(f"[VAMP Refresh] Skipped ({reason}): no cached trajectories with frames > lag ({lag_embed}).")
        return

    # Make the (filtered, deterministic) key list available to boundary plugins.
    # Plugins can use this to fit custom boundary classifiers without duplicating
    # history iteration logic.
    try:
        boundary_plugin_fit_keys = [str(k) for k in train_keys]
        boundary_plugin_fit_nonce = int(boundary_plugin_fit_nonce) + 1
    except Exception:
        boundary_plugin_fit_keys = []
        boundary_plugin_fit_nonce = int(boundary_plugin_fit_nonce) + 1

    n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
    label_mode = str(getattr(config, "geom2vec_label_mode", "auto") or "auto").strip().lower()
    skip_vamp_lobe = label_mode in {"linear_kmeans", "vac_kmeans", "linear-kmeans", "kmeans"}

    # Save old hard labels for overlap alignment (canonicalized by current permutation map).
    old_labels_by_key: Optional[Dict[str, np.ndarray]] = None
    if (not skip_vamp_lobe) and (
        getattr(geom2vec_projector, "trained", None) is not None
        or bool(getattr(geom2vec_projector, "trained_ensemble", None))
    ):
        old_labels_by_key = {}
        for key in train_keys:
            try:
                probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                probs_canon = probs_raw[:, geom2vec_state_permutation_map]
                old_labels_by_key[key] = np.argmax(probs_canon, axis=1).astype(np.int64, copy=False)
            except Exception:
                continue

    seed_base = int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42)
    train_fraction = float(getattr(config, "geom2vec_vamp_train_fraction", 0.8) or 0.8)
    vamp_batch_size = int(getattr(config, "geom2vec_vamp_batch_size", 256) or 256)
    vamp_epochs = int(getattr(config, "geom2vec_vamp_epochs", 10) or 10)
    learning_rate = float(getattr(config, "geom2vec_vamp_learning_rate", 5e-4) or 5e-4)
    weight_decay = float(getattr(config, "geom2vec_vamp_weight_decay", 0.0) or 0.0)
    optimizer = str(getattr(config, "geom2vec_vamp_optimizer", "Adam") or "Adam")
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

    # Optional DataParallel for VAMP lobe training (single-process multi-GPU).
    dp_geom2vec = bool(getattr(config, "geom2vec_data_parallel", False))
    dp_geom2vec_devices: Optional[List[int]] = None
    if dp_geom2vec:
        dev_s = str(getattr(config, "geom2vec_data_parallel_devices", "") or "").strip()
        ids: List[int] = []
        for part in dev_s.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                ids.append(int(p))
            except Exception:
                continue
        dp_geom2vec_devices = ids if ids else None

    vamp_meta_for_save: Dict[str, Any] = {}
    vamp_collapsed = False

    # Train projectors for LOO rewards + diagnostics (standard VAMP on (t, t+lag)).
    # In path-mode, when labels are derived from linear-VAMP+kmeans, we can skip fitting
    # the dedicated VAMP lobe entirely (it often collapses and provides no useful labels).
    if skip_vamp_lobe:
        vamp_meta_for_save = {
            "fit_mode": "skipped",
            "reason": f"geom2vec_label_mode={label_mode}",
        }
        print(
            f"[VAMP Refresh] Skipping VAMP lobe fit (geom2vec_label_mode={label_mode}); "
            "using linear-VAMP+kmeans labels.",
            flush=True,
        )
    else:
        old_vamp_trained = (
            geom2vec_projector.trained
            if (
                getattr(geom2vec_projector, "trained", None) is not None
                or bool(getattr(geom2vec_projector, "trained_ensemble", None))
            )
            else None
        )
        trained_vamp = train_vamp_lobe_from_cache(
            geom2vec_cache,
            keys=train_keys,
            n_states=int(n_states),
            lag_time=int(lag_embed),
            pair_stride=int(getattr(config, "geom2vec_vamp_pair_stride", 1) or 1),
            train_fraction=float(train_fraction),
            shuffle_trajectories=True,
            train_shuffle_pairs=True,
            batch_size=int(vamp_batch_size),
            n_epochs=int(vamp_epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            optimizer=str(optimizer),
            device=torch.device(geom2vec_device_str),
            seed=int(seed_base),
            epsilon=float(epsilon),
            inverse_mode=str(inverse_mode),
            softmax_output=True,
            vamp_merger=bool(vamp_merger),
            vamp_merger_window=int(vamp_merger_window),
            double_lobe=bool(vamp_double_lobe),
            lobe_invariant=bool(lobe_invariant),
            data_parallel=bool(dp_geom2vec),
            data_parallel_devices=dp_geom2vec_devices,
            lobe_kwargs=lobe_kwargs,
        )
        vamp_meta_for_save = dict(trained_vamp.meta or {})
        vamp_meta_for_save["fit_mode"] = "vamp_soft_assignments"

        lobe_sm = trained_vamp.lobe
        if isinstance(lobe_sm, (tuple, list)):
            lobe_sm = lobe_sm[0]
        geom2vec_projector.set_trained(lobe_sm, meta=trained_vamp.meta)
        geom2vec_projector.last_fit_outcomes = [trained_vamp.meta]

        # Guardrail: detect trivial/collapsed label solutions and fall back to the previous
        # VAMP lobe (if any). When the lobe collapses, retrieval labels become
        # useless (all frames in one state) and LOO-VAMP deltas read as ~0.
        try:
            sample_keys = list(train_keys[: min(5, len(train_keys))])
            counts = np.zeros(int(n_states), dtype=np.int64)
            total = 0
            for k in sample_keys:
                probs_k = geom2vec_projector.state_probs_from_cache_key(str(k))
                labels_k = np.argmax(np.asarray(probs_k, dtype=np.float32), axis=1).astype(np.int64, copy=False)
                labels_k = labels_k[(labels_k >= 0) & (labels_k < int(n_states))]
                if labels_k.size:
                    counts += np.bincount(labels_k, minlength=int(n_states))
                    total += int(labels_k.size)
            if total > 0:
                occ = int(np.count_nonzero(counts))
                frac = float(counts.max() / float(total)) if total > 0 else 1.0
                if occ < 2 or frac > 0.995:
                    print(
                        f"[VAMP Refresh] Warning: VAMP labels appear collapsed "
                        f"(occupied={occ}/{int(n_states)}, max_frac={frac:.3f}).",
                        flush=True,
                    )
                    vamp_collapsed = True
                    vamp_meta_for_save["collapsed_labels"] = True
                    vamp_meta_for_save["collapsed_counts"] = counts.tolist()
                    vamp_meta_for_save["collapsed_max_frac"] = float(frac)

                    # Prefer an alternate VAMP fit over "keeping previous" when the previous
                    # lobe is absent (or may also be collapsed). This preserves retrieval
                    # balancing and avoids a degenerate discrete partition until the next
                    # refresh (when more history may be available).
                    used_fallback = False
                    try:
                        print("[VAMP Refresh] Attempting alternate VAMP fit after collapse...", flush=True)
                        trained_vamp = train_vamp_lobe_from_cache(
                            geom2vec_cache,
                            keys=train_keys,
                            n_states=int(n_states),
                            lag_time=int(lag_embed),
                            pair_stride=1,
                            train_fraction=float(train_fraction),
                            shuffle_trajectories=True,
                            train_shuffle_pairs=True,
                            batch_size=int(vamp_batch_size),
                            n_epochs=int(vamp_epochs),
                            learning_rate=float(learning_rate),
                            weight_decay=float(weight_decay),
                            optimizer=str(optimizer),
                            device=torch.device(geom2vec_device_str),
                            seed=int(seed_base) + 77,
                            epsilon=float(epsilon),
                            inverse_mode=str(inverse_mode),
                softmax_output=True,
                vamp_merger=bool(vamp_merger),
                vamp_merger_window=int(vamp_merger_window),
                double_lobe=bool(vamp_double_lobe),
                lobe_invariant=bool(lobe_invariant),
                data_parallel=bool(dp_geom2vec),
                data_parallel_devices=dp_geom2vec_devices,
                lobe_kwargs=lobe_kwargs,
            )
                        lobe_fallback = trained_vamp.lobe
                        if isinstance(lobe_fallback, (tuple, list)):
                            lobe_fallback = lobe_fallback[0]
                        geom2vec_projector.set_trained(lobe_fallback, meta=trained_vamp.meta)
                        geom2vec_projector.last_fit_outcomes = [trained_vamp.meta]
                        vamp_meta_for_save["fallback_mode"] = "alt_vamp"
                        vamp_meta_for_save["alt_vamp_fit_meta"] = dict(trained_vamp.meta or {})
                        used_fallback = True
                        print("[VAMP Refresh] Using alternate VAMP fit as fallback (until next refresh).", flush=True)
                    except Exception as exc_fallback:
                        print(
                            f"[VAMP Refresh] Alternate VAMP fit failed ({exc_fallback}); keeping previous if available.",
                            flush=True,
                        )
                        used_fallback = False

                    if (not used_fallback) and old_vamp_trained is not None:
                        geom2vec_projector.trained = old_vamp_trained
                        try:
                            geom2vec_projector.last_fit_outcomes = [dict(getattr(old_vamp_trained, "meta", {}) or {})]
                        except Exception:
                            geom2vec_projector.last_fit_outcomes = None
        except Exception:
            pass

    # Train "linear VAMP" for continuous coordinates used by VAMP-bin/MSM-IG/DGA.
    #
    # User preference: do NOT symmetrize the linear lobe. If the VAMP labels collapse,
    # still fit a linear lobe (used by VAMP-bin/MSM-IG/DGA).
    linear_meta_for_save: Dict[str, Any] = {}
    if vamp_collapsed:
        trained_lin = train_vamp_lobe_from_cache(
            geom2vec_cache,
            keys=train_keys,
            n_states=int(n_states),
            lag_time=int(lag_embed),
            pair_stride=1,
            train_fraction=float(train_fraction),
            shuffle_trajectories=True,
            train_shuffle_pairs=True,
            batch_size=int(vamp_batch_size),
            n_epochs=int(vamp_epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            optimizer=str(optimizer),
            device=torch.device(geom2vec_device_str),
            seed=int(seed_base) + 1000,
            epsilon=float(epsilon),
            inverse_mode=str(inverse_mode),
            softmax_output=False,
            vamp_merger=bool(vamp_merger),
            vamp_merger_window=int(vamp_merger_window),
            double_lobe=bool(vamp_double_lobe),
            lobe_invariant=bool(lobe_invariant),
            data_parallel=bool(dp_geom2vec),
            data_parallel_devices=dp_geom2vec_devices,
            lobe_kwargs=lobe_kwargs,
        )
        lobe_lin = trained_lin.lobe
        if isinstance(lobe_lin, (tuple, list)):
            lobe_lin = lobe_lin[0]
        geom2vec_projector_linear.set_trained(lobe_lin, meta=trained_lin.meta)
        geom2vec_projector_linear.last_fit_outcomes = [trained_lin.meta]
        linear_meta_for_save = dict(trained_lin.meta or {})
        linear_meta_for_save["fit_mode"] = "vamp_linear_fallback"
        print("[VAMP Refresh] Linear lobe fit after label collapse.", flush=True)
    else:
        trained_lin = train_vamp_lobe_from_cache(
            geom2vec_cache,
            keys=train_keys,
            n_states=int(n_states),
            lag_time=int(lag_embed),
            pair_stride=1,
            train_fraction=float(train_fraction),
            shuffle_trajectories=True,
            train_shuffle_pairs=True,
            batch_size=int(vamp_batch_size),
            n_epochs=int(vamp_epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            optimizer=str(optimizer),
            device=torch.device(geom2vec_device_str),
            seed=int(seed_base) + 1000,
            epsilon=float(epsilon),
            inverse_mode=str(inverse_mode),
            softmax_output=False,
            vamp_merger=bool(vamp_merger),
            vamp_merger_window=int(vamp_merger_window),
            double_lobe=bool(vamp_double_lobe),
            lobe_invariant=bool(lobe_invariant),
            data_parallel=bool(dp_geom2vec),
            data_parallel_devices=dp_geom2vec_devices,
            lobe_kwargs=lobe_kwargs,
        )
        lobe_lin = trained_lin.lobe
        if isinstance(lobe_lin, (tuple, list)):
            lobe_lin = lobe_lin[0]
        geom2vec_projector_linear.set_trained(lobe_lin, meta=trained_lin.meta)
        geom2vec_projector_linear.last_fit_outcomes = [trained_lin.meta]
        linear_meta_for_save = dict(trained_lin.meta or {})
        linear_meta_for_save["fit_mode"] = "vamp_linear"

    # Fit/update VAC-space k-means boundary model (optional).
    if boundary_mode in {"vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent"}:
        old_model = committor_state.boundary_vac_kmeans
        old_mu = committor_state.boundary_vac_mu
        old_std = committor_state.boundary_vac_std
        old_a = committor_state.boundary_vac_a_clusters
        old_b = committor_state.boundary_vac_b_clusters
        old_sig = committor_state.boundary_vac_signature
        try:
            from sklearn.cluster import MiniBatchKMeans  # type: ignore
        except Exception as exc:
            print(f"[Path Boundaries] Warning: sklearn unavailable ({exc}); VAC-kmeans boundaries disabled.", flush=True)
        else:
            try:
                Kb = int(getattr(config, "path_boundary_vac_kmeans_n_clusters", 50) or 50)
                dim_b = int(getattr(config, "path_boundary_vac_kmeans_dim", 3) or 3)
                max_samples = int(getattr(config, "path_boundary_vac_kmeans_max_fit_samples", 50000) or 50000)
                seed_b = int(getattr(config, "path_boundary_vac_kmeans_seed", 42) or 42)
                pct_a = float(getattr(config, "path_boundary_vac_kmeans_percent_a", 10.0) or 10.0)
                pct_b = float(getattr(config, "path_boundary_vac_kmeans_percent_b", pct_a) or pct_a)

                pct_a = max(0.0, min(100.0, pct_a))
                pct_b = max(0.0, min(100.0, pct_b))

                n_states_here = int(getattr(config, "geom2vec_dim", 2) or 2)
                dim_b = int(max(1, min(int(dim_b), int(n_states_here))))
                if int(n_states_here) < 1:
                    raise ValueError("geom2vec_dim must be >= 1")
                if int(n_states_here) < int(dim_b):
                    raise ValueError("Not enough linear dimensions for VAC-kmeans boundaries.")

                # Collect samples of (z[:dim_b], rmsd_a, rmsd_b) across training keys.
                z_samples: List[np.ndarray] = []
                ra_samples: List[np.ndarray] = []
                rb_samples: List[np.ndarray] = []
                n_collected = 0
                for key in train_keys:
                    if max_samples > 0 and n_collected >= int(max_samples):
                        break
                    try:
                        graph, ca = geom2vec_cache.load(str(key))
                    except Exception:
                        continue
                    try:
                        z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                        z = np.asarray(z_raw, dtype=np.float32)
                    except Exception:
                        continue
                    if z.ndim != 2 or int(z.shape[0]) <= 0 or int(z.shape[1]) < int(dim_b):
                        continue
                    try:
                        ca_np = ca.detach().cpu().numpy()
                        rmsd_a, rmsd_b = _rmsd_to_endpoints(ca_np)
                    except Exception:
                        continue

                    take = int(z.shape[0])
                    remaining = take
                    if max_samples > 0:
                        remaining = max(0, int(max_samples) - n_collected)
                    if max_samples > 0 and remaining <= 0:
                        break
                    if max_samples > 0 and take > remaining:
                        idx = np.linspace(0, take - 1, num=int(remaining), dtype=np.int64)
                        z_use = np.asarray(z[idx, :dim_b], dtype=np.float64)
                        ra_use = np.asarray(rmsd_a[idx], dtype=np.float64)
                        rb_use = np.asarray(rmsd_b[idx], dtype=np.float64)
                    else:
                        z_use = np.asarray(z[:, :dim_b], dtype=np.float64)
                        ra_use = np.asarray(rmsd_a, dtype=np.float64)
                        rb_use = np.asarray(rmsd_b, dtype=np.float64)
                    if z_use.size == 0:
                        continue
                    z_samples.append(z_use)
                    ra_samples.append(ra_use.reshape(-1))
                    rb_samples.append(rb_use.reshape(-1))
                    n_collected += int(z_use.shape[0])

                if not z_samples:
                    raise RuntimeError("No samples available for VAC-kmeans boundary fitting.")

                Z = np.concatenate(z_samples, axis=0)
                RA = np.concatenate(ra_samples, axis=0)
                RB = np.concatenate(rb_samples, axis=0)
                if int(Z.shape[0]) < int(max(2, Kb)):
                    raise RuntimeError(f"Insufficient frames for VAC-kmeans (need >= {int(Kb)}, have {int(Z.shape[0])}).")

                mu = Z.mean(axis=0)
                std = Z.std(axis=0)
                std = np.where(std > 1e-6, std, 1.0)
                Z_norm = (Z - mu[None, :]) / std[None, :]

                bs = int(min(2048, max(256, int(Z_norm.shape[0]))))
                kmeans = MiniBatchKMeans(
                    n_clusters=int(Kb),
                    random_state=int(seed_b),
                    batch_size=int(bs),
                    n_init=1,
                    max_iter=100,
                )
                kmeans.fit(Z_norm)
                labels = np.asarray(kmeans.predict(Z_norm), dtype=np.int64).reshape(-1)

                counts = np.bincount(labels, minlength=int(Kb)).astype(np.int64, copy=False)
                denom = np.maximum(counts, 1).astype(np.float64, copy=False)
                mean_ra = np.bincount(labels, weights=np.asarray(RA, dtype=np.float64), minlength=int(Kb)) / denom
                mean_rb = np.bincount(labels, weights=np.asarray(RB, dtype=np.float64), minlength=int(Kb)) / denom
                frac_closer_a = (
                    np.bincount(labels, weights=(np.asarray(RA) <= np.asarray(RB)).astype(np.float64), minlength=int(Kb))
                    / denom
                )
                side_a = (frac_closer_a >= 0.5) & (counts > 0)

                a_clusters = np.zeros((int(Kb),), dtype=bool)
                b_clusters = np.zeros((int(Kb),), dtype=bool)

                pop_idx = np.nonzero(counts > 0)[0]
                n_pop = int(pop_idx.size)
                if n_pop <= 0:
                    raise RuntimeError("VAC-kmeans produced no populated clusters.")

                if boundary_mode == "vac_kmeans_rmsd_cluster_percent":
                    # Select boundary sets by choosing the closest X% of populated clusters.
                    n_a = int(math.ceil((pct_a / 100.0) * float(n_pop))) if pct_a > 0 else 0
                    n_b = int(math.ceil((pct_b / 100.0) * float(n_pop))) if pct_b > 0 else 0
                    n_a = int(max(0, min(n_a, n_pop)))
                    n_b = int(max(0, min(n_b, n_pop)))

                    a_cand = np.nonzero(side_a)[0]
                    if int(a_cand.size) < int(n_a):
                        a_cand = pop_idx
                    else:
                        a_cand = a_cand[np.isin(a_cand, pop_idx)]
                    if a_cand.size == 0:
                        a_cand = pop_idx
                    a_order = a_cand[np.argsort(mean_ra[a_cand])]
                    for k in a_order[:n_a].tolist():
                        a_clusters[int(k)] = True

                    b_cand = np.nonzero((~side_a) & (counts > 0))[0]
                    if int(b_cand.size) < int(n_b):
                        b_cand = pop_idx
                    if b_cand.size == 0:
                        b_cand = pop_idx
                    b_order = b_cand[np.argsort(mean_rb[b_cand])]
                    for k in b_order[:n_b].tolist():
                        b_clusters[int(k)] = True

                    # Resolve overlaps by assigning each shared cluster to the closer endpoint.
                    overlap = np.nonzero(a_clusters & b_clusters)[0]
                    for k in overlap.tolist():
                        if float(mean_ra[int(k)]) <= float(mean_rb[int(k)]):
                            b_clusters[int(k)] = False
                        else:
                            a_clusters[int(k)] = False

                    # Ensure both boundaries are non-empty when requested.
                    if pct_a > 0 and not bool(np.any(a_clusters)):
                        cand = pop_idx[np.argsort(mean_ra[pop_idx])]
                        for k in cand.tolist():
                            if not b_clusters[int(k)]:
                                a_clusters[int(k)] = True
                                break
                    if pct_b > 0 and not bool(np.any(b_clusters)):
                        cand = pop_idx[np.argsort(mean_rb[pop_idx])]
                        for k in cand.tolist():
                            if not a_clusters[int(k)]:
                                b_clusters[int(k)] = True
                                break
                else:
                    # Legacy: select boundary sets by covering an approximate X% of frames.
                    target_a = int(math.ceil((pct_a / 100.0) * float(Z.shape[0]))) if pct_a > 0 else 0
                    target_b = int(math.ceil((pct_b / 100.0) * float(Z.shape[0]))) if pct_b > 0 else 0

                    a_cand = np.nonzero(side_a)[0]
                    if a_cand.size == 0:
                        a_cand = pop_idx
                    a_order = a_cand[np.argsort(mean_ra[a_cand])]
                    cum = 0
                    for k in a_order.tolist():
                        if target_a <= 0:
                            break
                        a_clusters[int(k)] = True
                        cum += int(counts[int(k)])
                        if cum >= target_a:
                            break

                    b_cand = np.nonzero((~side_a) & (counts > 0))[0]
                    if b_cand.size == 0:
                        b_cand = pop_idx
                    b_order = b_cand[np.argsort(mean_rb[b_cand])]
                    cum = 0
                    for k in b_order.tolist():
                        if target_b <= 0:
                            break
                        if a_clusters[int(k)]:
                            continue
                        b_clusters[int(k)] = True
                        cum += int(counts[int(k)])
                        if cum >= target_b:
                            break

                if not np.any(a_clusters) or not np.any(b_clusters):
                    raise RuntimeError("VAC-kmeans boundary selection produced empty A/B cluster sets.")

                committor_state.boundary_vac_kmeans = kmeans
                committor_state.boundary_vac_mu = mu.astype(np.float32, copy=False)
                committor_state.boundary_vac_std = std.astype(np.float32, copy=False)
                committor_state.boundary_vac_a_clusters = a_clusters
                committor_state.boundary_vac_b_clusters = b_clusters
                committor_state.boundary_vac_signature = (
                    str(boundary_mode),
                    int(Kb),
                    int(dim_b),
                    int(max_samples),
                    int(seed_b),
                    float(pct_a),
                    float(pct_b),
                )
                try:
                    nA = int(np.count_nonzero(a_clusters))
                    nB = int(np.count_nonzero(b_clusters))
                    n_frames = int(np.sum(counts))
                    frames_a = int(np.sum(counts[np.nonzero(a_clusters)[0]])) if n_frames > 0 else 0
                    frames_b = int(np.sum(counts[np.nonzero(b_clusters)[0]])) if n_frames > 0 else 0
                    pct_frames_a = (100.0 * float(frames_a) / float(n_frames)) if n_frames > 0 else 0.0
                    pct_frames_b = (100.0 * float(frames_b) / float(n_frames)) if n_frames > 0 else 0.0
                    pop = int(np.count_nonzero(counts))
                    print(
                        f"[Path Boundaries] Fitted VAC-kmeans: K={int(Kb)} dim={int(dim_b)} samples={int(Z.shape[0])} "
                        f"A_clusters={nA} B_clusters={nB} (mode={boundary_mode}, pct_a={pct_a:.2f} pct_b={pct_b:.2f}, "
                        f"populated={pop}, frame_pct_a={pct_frames_a:.2f} frame_pct_b={pct_frames_b:.2f})",
                        flush=True,
                    )
                except Exception:
                    pass
            except Exception as exc_fit:
                committor_state.boundary_vac_kmeans = old_model
                committor_state.boundary_vac_mu = old_mu
                committor_state.boundary_vac_std = old_std
                committor_state.boundary_vac_a_clusters = old_a
                committor_state.boundary_vac_b_clusters = old_b
                committor_state.boundary_vac_signature = old_sig
                print(f"[Path Boundaries] Warning: VAC-kmeans fit failed ({exc_fit}); keeping previous model.", flush=True)

    # Recompute stable state permutation based on overlap of hard-state assignments.
    if old_labels_by_key:
        old_all = []
        new_all = []
        for key, old_labels in old_labels_by_key.items():
            try:
                probs_new = geom2vec_projector.state_probs_from_cache_key(key)
                new_labels = np.argmax(np.asarray(probs_new, dtype=np.float32), axis=1).astype(np.int64, copy=False)
                old_all.append(old_labels)
                new_all.append(new_labels)
            except Exception:
                continue
        if old_all and new_all:
            old_flat = np.concatenate(old_all, axis=0)
            new_flat = np.concatenate(new_all, axis=0)
            try:
                perm = align_states_by_overlap(old_labels=old_flat, new_labels=new_flat, n_states=int(n_states))
                geom2vec_state_permutation_map = perm
                inv_perm = np.zeros_like(perm)
                inv_perm[perm] = np.arange(len(perm))
                geom2vec_state_inverse_permutation_map = inv_perm
                print(f"[VAMP Refresh] Applied confusion-matrix alignment: {perm}")
            except Exception as exc_align:
                print(f"[VAMP Refresh] Warning: overlap alignment failed: {exc_align}; using identity mapping.")
                identity = np.arange(int(n_states), dtype=int)
                geom2vec_state_permutation_map = identity
                geom2vec_state_inverse_permutation_map = identity
    else:
        identity = np.arange(int(n_states), dtype=int)
        geom2vec_state_permutation_map = identity
        geom2vec_state_inverse_permutation_map = identity

    # Auto-detect VAMP boundary states from endpoint PDB structures.
    # Project endpoint CA coords through geom2vec to get their VAMP state assignments.
    if boundary_mode == "vamp_state":
        committor_state.boundary_vamp_state_valid = False
        committor_state.boundary_vamp_a_state = None
        committor_state.boundary_vamp_b_state = None
        try:
            projector_ready = (
                geom2vec_projector is not None
                and (
                    getattr(geom2vec_projector, "trained", None) is not None
                    or bool(getattr(geom2vec_projector, "trained_ensemble", None))
                )
            )
            if (
                endpoint_ca_ref_a is not None
                and endpoint_ca_ref_b is not None
                and geom2vec_mgr is not None
                and projector_ready
            ):
                # Get required resources from the manager
                topology = geom2vec_mgr.topology_cache
                rep_model = geom2vec_mgr.projector.rep_model if geom2vec_mgr.projector is not None else None
                infer_device = geom2vec_mgr.device_str
                hidden_channels = geom2vec_mgr.hidden_channels_opt
                center_ca = geom2vec_mgr.center_ca
                reduction = geom2vec_mgr.reduction
                
                if topology is None or rep_model is None:
                    raise ValueError("topology cache or rep_model not available")
                
                pdb_a_path = str(getattr(config, "pdb_a", config.pdb) or config.pdb)
                pdb_b_path = str(getattr(config, "pdb_b", "") or "")
                
                if not pdb_a_path or not pdb_b_path:
                    raise ValueError("pdb_a or pdb_b path not set")
                
                # Load full protein coords from endpoint PDBs robustly
                # md.load(path, top=...) fails if atom counts mismatch (e.g. H vs no-H).
                # We load raw, select protein (and optionally heavy atoms), and check against topology.
                
                try:
                    tr_a = md.load(pdb_a_path)
                    tr_b = md.load(pdb_b_path)
                except Exception as exc_load:
                    print(f"[VAMP Boundaries] Failed to load endpoint PDBs: {exc_load}")
                    raise exc_load

                exclude_h = geom2vec_mgr.exclude_hydrogens if geom2vec_mgr is not None else True
                
                # Function to extract matching atoms
                def extract_matching_coords(traj, label):
                    # Strategy 1: Use shared inference logic (handles caps/non-standard residues)
                    try:
                        # infer_protein_atom_indices returns a list or array
                        prot_idx = infer_protein_atom_indices(traj.topology, include_caps=True, allow_empty=True)
                        
                        if exclude_h:
                            # Intersect with heavy atoms
                            heavy_idx = traj.topology.select("mass > 1.2")
                            # Use numpy intersect1d to intersect indices
                            inds = np.intersect1d(prot_idx, heavy_idx)
                            strategy_name = "infer_protein + heavy"
                        else:
                            inds = prot_idx
                            strategy_name = "infer_protein"
                            
                        if len(inds) == topology.num_atoms:
                            # Ensure sorted for consistency
                            inds = np.sort(inds)
                            return np.asarray(traj.xyz[:, inds, :], dtype=np.float32)
                    except Exception:
                        # Fallback if inference fails
                        pass

                    # Strategy 2: Simple selection string (fallback)
                    if exclude_h:
                        selection = "protein and (mass > 1.2)"
                    else:
                        selection = "protein"
                    
                    inds = traj.topology.select(selection)
                    if len(inds) == topology.num_atoms:
                        return np.asarray(traj.xyz[:, inds, :], dtype=np.float32)
                    
                    # Log mismatch for main strategies
                    print(
                        f"[VAMP Boundaries] Atom count mismatch for {label}: "
                        f"got {len(inds)} from selection '{selection}', expected {topology.num_atoms}. "
                        f"Traj has {traj.n_atoms} total atoms."
                    )
                    
                    # Strategy 3: Try all atoms (maybe PDB is already stripped)
                    if traj.n_atoms == topology.num_atoms:
                        print(f"[VAMP Boundaries] Using all atoms from {label} (count matches).")
                        return np.asarray(traj.xyz, dtype=np.float32)
                        
                    # Strategy 4: Try just 'protein' (maybe H exclusion logic differs or PDB lacks H)
                    inds_prot = traj.topology.select("protein")
                    if len(inds_prot) == topology.num_atoms:
                        print(f"[VAMP Boundaries] Using 'protein' atoms from {label} (count matches).")
                        return np.asarray(traj.xyz[:, inds_prot, :], dtype=np.float32)

                    raise ValueError(f"Could not extract {topology.num_atoms} atoms from {label} ({traj.n_atoms} total)")

                coords_a_nm = extract_matching_coords(tr_a, "A")
                coords_b_nm = extract_matching_coords(tr_b, "B")
                

                # Create a proxy topology for inference that maps 1:1 to the extracted coordinates.
                # This avoids index-out-of-bounds errors when the endpoint PDB's atom indices
                # (0..N_selected) don't match the original simulation topology indices (e.g. due to missing H).
                import dataclasses
                proxy_topology = dataclasses.replace(
                    topology,
                    atom_indices=np.arange(topology.num_atoms, dtype=np.int64)
                )

                # Compute embeddings using the proper inference function with the proxy topology
                graph_a, ca_a_tensor = infer_geom2vec_embeddings(
                    model=rep_model,
                    topology=proxy_topology,
                    coords=coords_a_nm,
                    coords_in_nm=True,
                    stride=1,
                    batch_size=1,
                    reduction=reduction,
                    hidden_channels=hidden_channels,
                    device=infer_device,
                    center_ca=center_ca,
                )
                graph_b, ca_b_tensor = infer_geom2vec_embeddings(
                    model=rep_model,
                    topology=proxy_topology,
                    coords=coords_b_nm,
                    coords_in_nm=True,
                    stride=1,
                    batch_size=1,
                    reduction=reduction,
                    hidden_channels=hidden_channels,
                    device=infer_device,
                    center_ca=center_ca,
                )
                
                # Convert to numpy for projector
                graph_a = np.asarray(graph_a.cpu().numpy(), dtype=np.float32)
                graph_b = np.asarray(graph_b.cpu().numpy(), dtype=np.float32)
                ca_a_np = np.asarray(ca_a_tensor.cpu().numpy(), dtype=np.float32)
                ca_b_np = np.asarray(ca_b_tensor.cpu().numpy(), dtype=np.float32)
                
                # Get VAMP state probabilities for each endpoint
                probs_a = geom2vec_projector.state_probs_from_embeddings(graph_a, ca_a_np)
                probs_b = geom2vec_projector.state_probs_from_embeddings(graph_b, ca_b_np)
                probs_a = np.asarray(probs_a, dtype=np.float32)
                probs_b = np.asarray(probs_b, dtype=np.float32)
                
                # Apply state permutation if available
                if geom2vec_state_permutation_map is not None:
                    try:
                        perm = np.asarray(geom2vec_state_permutation_map, dtype=np.int64)
                        if perm.shape[0] == probs_a.shape[1]:
                            probs_a = probs_a[:, perm]
                        if perm.shape[0] == probs_b.shape[1]:
                            probs_b = probs_b[:, perm]
                    except Exception:
                        pass
                
                # Take argmax to get the dominant state for each endpoint
                state_a = int(np.argmax(probs_a.reshape(-1)))
                state_b = int(np.argmax(probs_b.reshape(-1)))
                
                if state_a == state_b:
                    # Both endpoints map to the same state - this is problematic
                    # Try to find the second-best state for endpoint B
                    probs_b_copy = probs_b.reshape(-1).copy()
                    probs_b_copy[state_a] = -np.inf  # Mask out the A state
                    state_b_alt = int(np.argmax(probs_b_copy))
                    if np.isfinite(probs_b_copy[state_b_alt]) and probs_b_copy[state_b_alt] > 0.01:
                        print(
                            f"[VAMP Boundaries] Warning: both endpoints map to state {state_a}; "
                            f"using second-best state {state_b_alt} for B (prob={float(probs_b.reshape(-1)[state_b_alt]):.3f}).",
                            flush=True,
                        )
                        state_b = state_b_alt
                    else:
                        print(
                            f"[VAMP Boundaries] Warning: both endpoints map to state {state_a}; "
                            "VAMP boundary mode may not work well until VAMP learns to distinguish endpoints.",
                            flush=True,
                        )
                        # Still set the states - fallback to RMSD will kick in
                
                committor_state.boundary_vamp_a_state = state_a
                committor_state.boundary_vamp_b_state = state_b
                committor_state.boundary_vamp_state_valid = True
                
                # Log the detected states with probabilities
                prob_a_val = float(probs_a.reshape(-1)[state_a]) if state_a < probs_a.size else 0.0
                prob_b_val = float(probs_b.reshape(-1)[state_b]) if state_b < probs_b.size else 0.0
                print(
                    f"[VAMP Boundaries] Auto-detected endpoint states: "
                    f"A=state_{state_a} (prob={prob_a_val:.3f}), B=state_{state_b} (prob={prob_b_val:.3f})",
                    flush=True,
                )
            else:
                # Log why detection was skipped
                missing = []
                if endpoint_ca_ref_a is None:
                    missing.append("endpoint_ca_ref_a")
                if endpoint_ca_ref_b is None:
                    missing.append("endpoint_ca_ref_b")
                if geom2vec_mgr is None:
                    missing.append("geom2vec_mgr")
                if not projector_ready:
                    missing.append("geom2vec_projector (not trained)")
                print(
                    f"[VAMP Boundaries] Skipped endpoint state detection: missing {', '.join(missing)}; "
                    "will fall back to RMSD cutoffs.",
                    flush=True,
                )
        except Exception as exc_vamp_boundary:
            committor_state.boundary_vamp_state_valid = False
            committor_state.boundary_vamp_a_state = None
            committor_state.boundary_vamp_b_state = None
            print(
                f"[VAMP Boundaries] Warning: failed to detect endpoint VAMP states ({exc_vamp_boundary}); "
                "will fall back to RMSD cutoffs.",
                flush=True,
            )

    if config.snapshot_dir:
        out_dir = Path(config.snapshot_dir) / "geom2vec_vamp"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            cutoff_a_nm = float(getattr(config, "path_boundary_cutoff_a_nm", 0.20) or 0.20)
            cutoff_b_nm = float(getattr(config, "path_boundary_cutoff_b_nm", cutoff_a_nm) or cutoff_a_nm)
            if (not skip_vamp_lobe) and (
                getattr(geom2vec_projector, "trained", None) is not None
                or bool(getattr(geom2vec_projector, "trained_ensemble", None))
            ):
                torch.save(geom2vec_projector._require_lobe().state_dict(), out_dir / "vamp_lobe_state_dict.pt")
            torch.save(geom2vec_projector_linear._require_lobe().state_dict(), out_dir / "vac_lobe_state_dict.pt")
            torch.save(
                {
                    "train_keys": train_keys,
                    "lag_embed": int(lag_embed),
                    "embed_stride": int(geom2vec_embed_stride),
                    "state_permutation_map": np.asarray(geom2vec_state_permutation_map, dtype=np.int64),
                    "state_inverse_permutation_map": np.asarray(geom2vec_state_inverse_permutation_map, dtype=np.int64),
                    "vamp_fit_meta": vamp_meta_for_save,
                    "vac_fit_meta": linear_meta_for_save,
                    "boundary_cutoff_a_nm": float(cutoff_a_nm),
                    "boundary_cutoff_b_nm": float(cutoff_b_nm),
                },
                out_dir / "lobe_meta.pt",
            )
            print(f"[VAMP Refresh] Saved lobes/meta to {out_dir}")
        except Exception as exc_save:
            print(f"[VAMP Refresh] Warning: failed to save lobe weights/meta: {exc_save}")

    # Invalidate reward state derived from the projectors.
    _invalidate_loo_vamp_stats(tag=f"vamp refresh: {reason}")
    try:
        vamp_bin_ent_coef = float(getattr(config, "vamp_bin_entropy_coef", -1.0) or -1.0)
        if vamp_bin_ent_coef < 0:
            vamp_bin_ent_coef = float(getattr(config, "ent_coef", 0.0) or 0.0)
    except Exception:
        vamp_bin_ent_coef = 0.0
    if vamp_bin_ent_coef > 0.0:
        _invalidate_vamp_bin_stats(tag=f"vamp refresh: {reason}")

    try:
        msm_ig_coef = float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0)
    except Exception:
        msm_ig_coef = 0.0
    if msm_ig_coef > 0.0:
        _invalidate_msm_ig_stats(tag=f"vamp refresh: {reason}")
    try:
        _refresh_retrieval_cache_labels(tag=f"vamp refresh: {reason}")
    except Exception as exc_lbl:
        print(f"[Retrieval Cache] Warning: failed to refresh VAMP labels ({exc_lbl})")

    # Seed LOO-VAMP sufficient statistics under the newly trained VAMP lobe so
    # epoch-1 training starts with consistent (projector-aligned) reward state.
    try:
        _ensure_loo_vamp_history_stats(
            upto_epoch_exclusive=len(all_epoch_data),
            n_states=int(n_states),
            lag_embed=int(lag_embed),
            tag=f"vamp refresh: {reason}",
        )
    except Exception as exc_loo:
        print(f"[LOO VAMP] Warning: failed to seed stats after refresh ({exc_loo})", flush=True)

    # MSM-IG uses linear VAC coordinates; refit k-means (and reseed MSM counts) on refresh.
    try:
        if float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0) > 0.0:
            _ensure_msm_ig_history_counts(
                upto_epoch_exclusive=len(all_epoch_data),
                n_states=int(n_states),
                lag_embed=int(lag_embed),
                tag=f"vamp refresh: {reason}",
            )
    except Exception as exc_msm:
        print(f"[MSM IG] Warning: failed to refit/seed after refresh ({exc_msm})", flush=True)

    # Fit DGA committor model (k-means basis on VAC coordinates).
    # Do this after the VAMP/VAC refresh has been fully applied (and saved),
    # so logs clearly reflect the causal ordering.
    committor_weight = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
    committor_tube_weight = float(getattr(config, "committor_tube_diversity_reward_weight", 0.0) or 0.0)
    committor_edge_weight = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0)
    committor_conn_weight = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
    if (
        committor_weight > 0.0
        or committor_tube_weight > 0.0
        or committor_edge_weight > 0.0
        or committor_conn_weight > 0.0
    ):
        _invalidate_committor_state(tag=f"vamp refresh: {reason}")
        committor_state_mode = str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
        if committor_state_mode not in {"dga", "vamp"}:
            committor_state_mode = "dga"

        lag_saved_cfg = int(getattr(config, "committor_dga_lagtime", 0) or 0)
        if lag_saved_cfg <= 0:
            lag_comm_embed = int(lag_embed)
        else:
            lag_comm_embed = max(1, int(round(max(1, lag_saved_cfg) / embed_stride_i)))

        try:
            # IMPORTANT: committor fitting should use *all* cached trajectories, not just the
            # subset used to train VAMP/VAC (which may be capped by max_pca_train).
            committor_fit_keys: List[str] = []
            try:
                for key in keys_sorted:
                    try:
                        rec = geom2vec_cache.get(str(key))
                    except Exception:
                        continue
                    if int(getattr(rec, "frames", 0) or 0) <= int(lag_comm_embed):
                        continue
                    committor_fit_keys.append(str(key))
            except Exception:
                committor_fit_keys = [str(k) for k in keys_sorted]

            try:
                print(
                    f"[Committor][Fit] Using cached trajectories: n_fit={len(committor_fit_keys)} "
                    f"(train_keys={len(train_keys)}, total_cached={len(keys_sorted)})",
                    flush=True,
                )
            except Exception:
                pass

            z_trajs_fit: List[np.ndarray] = []
            labels_trajs_fit: List[np.ndarray] = []
            in_a_fit: List[np.ndarray] = []
            in_b_fit: List[np.ndarray] = []
            ra_fit: List[np.ndarray] = []
            rb_fit: List[np.ndarray] = []

            committor_dga_input_mode = str(getattr(config, "committor_dga_input_mode", "linear") or "linear").strip().lower()
            committor_dga_include_globals = bool(getattr(config, "committor_dga_include_global_tokens", False) or False)
            # Fixed VAC-kmeans bases expect the raw VAC dimension; concatenating globals breaks the basis.
            if boundary_mode in {"vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent"}:
                if committor_dga_include_globals:
                    print(
                        "[Committor][Fit] Warning: committor_dga_include_global_tokens is incompatible with VAC-kmeans boundary modes; disabling.",
                        flush=True,
                    )
                committor_dga_include_globals = False
            if global_dim <= 0:
                committor_dga_include_globals = False

            for key in committor_fit_keys:
                graph, ca = geom2vec_cache.load(str(key))
                z_lin_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                z_lin = np.asarray(z_lin_raw, dtype=np.float32)

                if committor_dga_input_mode in {"vamp_probs", "vamp_logits", "vamp"}:
                    if committor_dga_input_mode == "vamp_probs":
                        z_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)
                    else:
                        # vamp_logits or vamp -> raw output
                        z_raw = geom2vec_projector.state_output_from_embeddings(graph, ca)
                else:
                    z_raw = z_lin_raw

                z_arr = np.asarray(z_raw, dtype=np.float32)
                if str(committor_state_mode) != "vamp":
                    z_arr = _augment_committor_dga_features(z_arr, ca, use_globals=committor_dga_include_globals)
                ca_np = ca.detach().cpu().numpy()
                cutoff_a_nm = float(getattr(config, "path_boundary_cutoff_a_nm", 0.20) or 0.20)
                cutoff_b_nm = float(getattr(config, "path_boundary_cutoff_b_nm", cutoff_a_nm) or cutoff_a_nm)
                ia, ib, ra, rb = _boundary_masks_from_ca(
                    ca_np,
                    cutoff_a_nm=float(cutoff_a_nm),
                    cutoff_b_nm=float(cutoff_b_nm),
                    graph_features=graph,
                    z_lin=z_lin,  # Use linear for boundary if required
                    cache_key=str(key),
                )
                in_a_fit.append(np.asarray(ia, dtype=bool))
                in_b_fit.append(np.asarray(ib, dtype=bool))
                ra_fit.append(np.asarray(ra, dtype=np.float32))
                rb_fit.append(np.asarray(rb, dtype=np.float32))
                
                if str(committor_state_mode) == "vamp":
                    probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)
                    probs = np.asarray(probs_raw, dtype=np.float32)[:, geom2vec_state_permutation_map]
                    if probs.shape[1] != int(n_states):
                        if probs.shape[1] > int(n_states):
                            probs = probs[:, : int(n_states)]
                        else:
                            probs = np.pad(probs, ((0, 0), (0, int(n_states) - int(probs.shape[1]))), mode="constant")
                    labels_trajs_fit.append(np.argmax(probs, axis=1).astype(np.int64, copy=False))
                else:
                    z_trajs_fit.append(np.asarray(z_arr, dtype=np.float32))

            # Auto-loosening guard: if either boundary set is empty, pick the closest frames
            # to ensure a valid DGA/MSM solve.
            try:
                nA_ct = sum(np.count_nonzero(ia) for ia in in_a_fit)
                nB_ct = sum(np.count_nonzero(ib) for ib in in_b_fit)
                n_total_f = sum(len(ra) for ra in ra_fit)
                if n_total_f > 0:
                    if nA_ct == 0:
                        all_ra = np.concatenate(ra_fit)
                        n_aim = max(10, min(len(all_ra) // 2, 100))
                        pivot = np.partition(all_ra, min(len(all_ra) - 1, n_aim))[min(len(all_ra) - 1, n_aim)]
                        for idx_f in range(len(in_a_fit)):
                            in_a_fit[idx_f] = (ra_fit[idx_f] <= pivot)
                        print(f"[Committor] Auto-loosened A: nA=0 -> {n_aim} points (cutoff approx {pivot:.3f} nm)", flush=True)
                    if nB_ct == 0:
                        all_rb = np.concatenate(rb_fit)
                        n_aim = max(10, min(len(all_rb) // 2, 100))
                        pivot = np.partition(all_rb, min(len(all_rb) - 1, n_aim))[min(len(all_rb) - 1, n_aim)]
                        for idx_f in range(len(in_b_fit)):
                            in_b_fit[idx_f] = (rb_fit[idx_f] <= pivot)
                        print(f"[Committor] Auto-loosened B: nB=0 -> {n_aim} points (cutoff approx {pivot:.3f} nm)", flush=True)
            except Exception as exc_loos:
                print(f"[Committor] Warning: auto-loosening guard failed: {exc_loos}", flush=True)

            if str(committor_state_mode) == "vamp":
                committor_state.dga_model = None
                committor_state.vamp_model = fit_vamp_discrete_committor(
                    labels_trajs_fit,
                    in_a_trajs=in_a_fit,
                    in_b_trajs=in_b_fit,
                    n_states=int(n_states),
                    lag=int(lag_comm_embed),
                    prior_alpha=1e-3,
                )
                print(
                    f"[Committor] Fitted VAMP discrete committor: K={int(n_states)} lag={int(lag_comm_embed)} "
                    f"degenerate={int(bool(getattr(committor_state.vamp_model, 'degenerate', False)))}",
                    flush=True,
                )
            else:
                committor_state.vamp_model = None
                n_clusters = int(getattr(config, "committor_dga_n_clusters", 200) or 200)
                dim_cfg = int(getattr(config, "committor_dga_dim", 0) or 0)
                max_samples = int(getattr(config, "committor_dga_max_fit_samples", 50000) or 50000)
                seed_cfg = int(getattr(config, "committor_dga_seed", 42) or 42)
                delay_n_embed = int(max(0, int(getattr(config, "committor_dga_delay_n_embed", 0) or 0)))
                delay_delta_saved = int(getattr(config, "committor_dga_delay_delta", 0) or 0)
                if delay_n_embed > 0 and delay_delta_saved <= 0:
                    delay_delta_saved = int(lag_saved_cfg) if lag_saved_cfg > 0 else int(lag_raw_i)
                delay_delta_embed = (
                    max(1, int(round(max(1, int(delay_delta_saved)) / embed_stride_i))) if delay_n_embed > 0 else 1
                )
                fixed_kmeans = None
                fixed_mu = None
                fixed_std = None
                if boundary_mode in {"vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent"}:
                    if committor_state.boundary_vac_kmeans is not None and committor_state.boundary_vac_mu is not None and committor_state.boundary_vac_std is not None:
                        fixed_kmeans = committor_state.boundary_vac_kmeans
                        fixed_mu = committor_state.boundary_vac_mu
                        fixed_std = committor_state.boundary_vac_std
                        try:
                            n_clusters = int(getattr(committor_state.boundary_vac_kmeans, "n_clusters", n_clusters) or n_clusters)
                        except Exception:
                            pass
                        try:
                            dim_cfg = int(np.asarray(committor_state.boundary_vac_mu).reshape(-1).shape[0])
                        except Exception:
                            pass
                        if delay_n_embed > 0:
                            delay_n_embed = 0
                            delay_delta_embed = 1

                # If we augmented features with globals and dim is auto, use all dims.
                # (Otherwise the default dim heuristic would drop appended global-token dimensions.)
                if committor_dga_include_globals and fixed_kmeans is None and dim_cfg <= 0 and z_trajs_fit:
                    try:
                        dim_cfg = int(np.asarray(z_trajs_fit[0]).shape[1])
                    except Exception:
                        dim_cfg = int(dim_cfg)

                use_kmeans_msm_proxy = (fixed_kmeans is not None)
                if use_kmeans_msm_proxy:
                    # User request: use the kmeans cluster as the indicator basis for the DGA committor.
                    # We predict labels using the fixed k-means model and fit a discrete MSM committor.
                    labels_trajs_proxy: List[np.ndarray] = []
                    try:
                        f_mu = np.asarray(fixed_mu if fixed_mu is not None else 0.0, dtype=np.float32).reshape(1, -1)
                        f_std = np.asarray(fixed_std if fixed_std is not None else 1.0, dtype=np.float32).reshape(1, -1)
                        f_std = np.where(f_std > 1e-6, f_std, 1.0)
                        for z_p in z_trajs_fit:
                            z_norm = (z_p - f_mu) / f_std
                            lab_p = fixed_kmeans.predict(z_norm.astype(np.float64))
                            labels_trajs_proxy.append(np.asarray(lab_p, dtype=np.int64).reshape(-1))
                    except Exception as exc_proxy:
                        print(
                            f"[Committor] Warning: failed to predict k-means labels for MSM proxy ({exc_proxy}); "
                            "falling back to standard DGA.",
                            flush=True,
                        )
                        use_kmeans_msm_proxy = False

                if use_kmeans_msm_proxy:
                    kmsm_model = fit_vamp_discrete_committor(
                        labels_trajs_proxy,
                        in_a_trajs=in_a_fit,
                        in_b_trajs=in_b_fit,
                        n_states=int(getattr(fixed_kmeans, "n_clusters", n_clusters)),
                        lag=int(lag_comm_embed),
                        prior_alpha=1e-3,
                    )
                    # Wrap as DGAKMeansCommittorModel
                    committor_state.dga_model = DGAKMeansCommittorModel(
                        kmeans=fixed_kmeans,
                        nn_a=None,
                        nn_b=None,
                        coeffs=kmsm_model.q_state,
                        basis_scale=np.ones(kmsm_model.q_state.shape[0], dtype=np.float64),
                        dim=int(dim_cfg),
                        lag=int(kmsm_model.lag),
                        delay_n_embed=0,
                        delay_delta=1,
                        backend="vamp_discrete_proxy",
                        z_mu=np.asarray(fixed_mu, dtype=np.float64).reshape(-1) if fixed_mu is not None else None,
                        z_std=np.asarray(fixed_std, dtype=np.float64).reshape(-1) if fixed_std is not None else None,
                        degenerate=bool(kmsm_model.degenerate),
                        solve_method="vamp_discrete",
                    )
                else:
                    committor_state.dga_model = fit_dga_committor_kmeans(
                        z_trajs_fit,
                        in_a_trajs=in_a_fit,
                        in_b_trajs=in_b_fit,
                        lag=int(lag_comm_embed),
                        n_clusters=int(n_clusters),
                        dim=int(dim_cfg),
                        max_fit_samples=int(max_samples),
                        seed=int(seed_cfg),
                        backend=str(getattr(config, "committor_dga_backend", "auto") or "auto"),
                        delay_n_embed=int(delay_n_embed),
                        delay_delta=int(delay_delta_embed),
                        fixed_kmeans=fixed_kmeans,
                        fixed_z_mu=fixed_mu,
                        fixed_z_std=fixed_std,
                    )
                if committor_state.dga_model is None:
                    print(
                        "[Committor] Warning: DGA fit returned None; committor reward will be 0 until next refresh.",
                        flush=True,
                    )
                else:
                    # Tag with input mode
                    committor_state.dga_model.input_mode = committor_dga_input_mode
                    try:
                        committor_state.dga_model.include_global_tokens = bool(committor_dga_include_globals)  # type: ignore[attr-defined]
                        committor_state.dga_model.global_token_names = list(global_token_names)  # type: ignore[attr-defined]
                        committor_state.dga_model.feature_dim_total = (
                            int(np.asarray(z_trajs_fit[0]).shape[1]) if z_trajs_fit else None
                        )  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    print(
                        f"[Committor] Fitted DGA committor: K={int(getattr(committor_state.dga_model, 'coeffs', np.zeros((0,))).shape[0])} "
                        f"lag={int(getattr(committor_state.dga_model, 'lag', 0) or 0)} "
                        f"degenerate={int(bool(getattr(committor_state.dga_model, 'degenerate', False)))} "
                        f"input={getattr(committor_state.dga_model, 'input_mode', 'linear')} "
                        f"globals={int(bool(getattr(committor_state.dga_model, 'include_global_tokens', False)))}",
                        flush=True,
                    )
        except Exception as exc_comm:
            committor_state.dga_model = None
            committor_state.vamp_model = None
            print(f"[Committor] Warning: committor fit failed ({exc_comm}); committor reward disabled.", flush=True)

    """
    if committor_weight > 0.0 or committor_tube_weight > 0.0 or committor_edge_weight > 0.0:
        try:
            _invalidate_committor_state(tag=f"vamp refresh: {reason}")
            committor_state_mode = str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
            if committor_state_mode not in {"dga", "vamp"}:
                committor_state_mode = "dga"
                lag_saved_cfg = int(getattr(config, "committor_dga_lagtime", 0) or 0)
                if lag_saved_cfg <= 0:
                    lag_comm_embed = int(lag_embed)
                else:
                lag_comm_embed = max(1, int(round(max(1, lag_saved_cfg) / embed_stride_i)))

            n_clusters = int(getattr(config, "committor_dga_n_clusters", 200) or 200)
            dim_cfg = int(getattr(config, "committor_dga_dim", 0) or 0)
            max_samples = int(getattr(config, "committor_dga_max_fit_samples", 50000) or 50000)
            seed_cfg = int(getattr(config, "committor_dga_seed", 42) or 42)
            delay_n_embed = int(getattr(config, "committor_dga_delay_n_embed", 0) or 0)
            delay_n_embed = int(max(0, delay_n_embed))
            delay_delta_saved = int(getattr(config, "committor_dga_delay_delta", 0) or 0)
            if delay_n_embed > 0 and delay_delta_saved <= 0:
                delay_delta_saved = int(lag_saved_cfg) if int(lag_saved_cfg) > 0 else int(lag_raw_i)
            delay_delta_embed = (
                max(1, int(round(max(1, int(delay_delta_saved)) / embed_stride_i))) if delay_n_embed > 0 else 1
            )

            # If boundaries are defined in VAC-kmeans microstates, reuse the same k-means basis for DGA.
            fixed_kmeans = None
            fixed_mu = None
            fixed_std = None
            if boundary_mode in {"vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent"}:
                if committor_state.boundary_vac_kmeans is not None and committor_state.boundary_vac_mu is not None and committor_state.boundary_vac_std is not None:
                    fixed_kmeans = committor_state.boundary_vac_kmeans
                    fixed_mu = committor_state.boundary_vac_mu
                    fixed_std = committor_state.boundary_vac_std
                    try:
                        n_clusters = int(getattr(committor_state.boundary_vac_kmeans, "n_clusters", n_clusters) or n_clusters)
                    except Exception:
                        pass
                    try:
                        dim_cfg = int(np.asarray(committor_state.boundary_vac_mu).reshape(-1).shape[0])
                    except Exception:
                        pass
                    if delay_n_embed > 0:
                        print(
                            f"[Committor][DGA] Note: disabling delay embedding (delay_n={int(delay_n_embed)}) when using "
                            f"VAC-kmeans microstates (boundary_mode={boundary_mode}).",
                            flush=True,
                        )
                        delay_n_embed = 0
                        delay_delta_embed = 1

                z_trajs_fit: List[np.ndarray] = []
                labels_trajs_fit: List[np.ndarray] = []
                in_a_fit: List[np.ndarray] = []
                in_b_fit: List[np.ndarray] = []
                metric_a_fit: List[np.ndarray] = []
                metric_b_fit: List[np.ndarray] = []
            fit_diag = {
                "n_trajs": 0,
                "n_frames": 0,
                "n_in_a": 0,
                "n_in_b": 0,
                "min_rmsd_a": [],
                "min_rmsd_b": [],
                "mean_in_a": [],
                "mean_in_b": [],
            }
                for key in train_keys:
                    try:
                        graph, ca = geom2vec_cache.load(str(key))
                        z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                        z_arr = np.asarray(z_raw, dtype=np.float32)
                            labels_this = None
                            if str(committor_state_mode) == "vamp":
                                probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)
                                probs = np.asarray(probs_raw, dtype=np.float32)[:, geom2vec_state_permutation_map]
                                if probs.shape[1] != int(n_states):
                                    if probs.shape[1] > int(n_states):
                                        probs = probs[:, : int(n_states)]
                                    else:
                                        probs = np.pad(
                                            probs,
                                            ((0, 0), (0, int(n_states) - int(probs.shape[1]))),
                                            mode="constant",
                                        )
                                labels_this = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                        ca_np = ca.detach().cpu().numpy()
                        cutoff_a_nm = float(getattr(config, "path_boundary_cutoff_a_nm", 0.20) or 0.20)
                        cutoff_b_nm = float(getattr(config, "path_boundary_cutoff_b_nm", cutoff_a_nm) or cutoff_a_nm)
                        ia, ib, ma, mb = _boundary_masks_from_ca(
                        ca_np,
                        cutoff_a_nm=float(cutoff_a_nm),
                        cutoff_b_nm=float(cutoff_b_nm),
                        graph_features=graph,
                        z_lin=z_arr,
                        cache_key=str(key),
                    )
                            if str(committor_state_mode) == "vamp":
                                if labels_this is not None:
                                    labels_trajs_fit.append(np.asarray(labels_this, dtype=np.int64))
                            else:
                                z_trajs_fit.append(z_arr)
                        in_a_fit.append(np.asarray(ia, dtype=bool))
                        in_b_fit.append(np.asarray(ib, dtype=bool))
                        metric_a_fit.append(np.asarray(ma, dtype=np.float64))
                        metric_b_fit.append(np.asarray(mb, dtype=np.float64))
                    try:
                        ia_b = np.asarray(ia, dtype=bool).reshape(-1)
                        ib_b = np.asarray(ib, dtype=bool).reshape(-1)
                        fit_diag["n_trajs"] += 1
                        fit_diag["n_frames"] += int(ia_b.size)
                        fit_diag["n_in_a"] += int(np.count_nonzero(ia_b))
                        fit_diag["n_in_b"] += int(np.count_nonzero(ib_b))
                        fit_diag["mean_in_a"].append(float(np.mean(ia_b.astype(np.float32))) if ia_b.size else 0.0)
                        fit_diag["mean_in_b"].append(float(np.mean(ib_b.astype(np.float32))) if ib_b.size else 0.0)
                        try:
                            ra, rb = _rmsd_to_endpoints(ca_np)
                            ra = np.asarray(ra, dtype=np.float64).reshape(-1)
                            rb = np.asarray(rb, dtype=np.float64).reshape(-1)
                            if ra.size:
                                fit_diag["min_rmsd_a"].append(float(np.min(ra)))
                            if rb.size:
                                fit_diag["min_rmsd_b"].append(float(np.min(rb)))
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    continue

            def _seed_empty_boundary(
                *,
                label: str,
                masks: List[np.ndarray],
                metrics: List[np.ndarray],
                avoid: Optional[Tuple[int, int]] = None,
            ) -> Optional[Tuple[int, int, float]]:
                best: Optional[Tuple[int, int, float]] = None
                for traj_i, metric_i in enumerate(metrics):
                    try:
                        m = np.asarray(metric_i, dtype=np.float64).reshape(-1)
                    except Exception:
                        continue
                    if m.size == 0:
                        continue
                    finite = np.isfinite(m)
                    if avoid is not None and int(avoid[0]) == int(traj_i):
                        j = int(avoid[1])
                        if 0 <= j < int(finite.size):
                            finite[j] = False
                    if not bool(np.any(finite)):
                        continue
                    j_best = int(np.argmin(np.where(finite, m, np.inf)))
                    v_best = float(m[j_best])
                    if not np.isfinite(v_best):
                        continue
                    if best is None or float(v_best) < float(best[2]):
                        best = (int(traj_i), int(j_best), float(v_best))
                if best is None:
                    return None
                traj_i, j_best, _ = best
                try:
                    masks[int(traj_i)][int(j_best)] = True
                except Exception:
                    return None
                print(
                    f"[Committor][Fit] Note: seeded empty {label} boundary with closest-metric frame "
                    f"(traj={int(traj_i)} frame={int(j_best)} metric={float(best[2]):.4g}).",
                    flush=True,
                )
                return best

            need_a = int(fit_diag["n_in_a"]) == 0
            need_b = int(fit_diag["n_in_b"]) == 0
            if need_a or need_b:
                seeded_a = _seed_empty_boundary(label="A", masks=in_a_fit, metrics=metric_a_fit) if need_a else None
                avoid = (int(seeded_a[0]), int(seeded_a[1])) if (need_a and need_b and seeded_a is not None) else None
                if need_b:
                    _seed_empty_boundary(label="B", masks=in_b_fit, metrics=metric_b_fit, avoid=avoid)
                try:
                    fit_diag["n_in_a"] = int(sum(int(np.count_nonzero(x)) for x in in_a_fit))
                    fit_diag["n_in_b"] = int(sum(int(np.count_nonzero(x)) for x in in_b_fit))
                except Exception:
                    pass

            try:
                if int(fit_diag["n_trajs"]) > 0:
                    def _m(xs):
                        return float(np.mean(xs)) if xs else 0.0
                    print(
                        f"[Committor][Fit] boundaries: trajs={int(fit_diag['n_trajs'])} "
                        f"frames={int(fit_diag['n_frames'])} inA={int(fit_diag['n_in_a'])} inB={int(fit_diag['n_in_b'])} "
                        f"mean_inA={_m(fit_diag['mean_in_a']):.4f} mean_inB={_m(fit_diag['mean_in_b']):.4f} "
                        f"min_rmsdA={_m(fit_diag['min_rmsd_a']):.3f}nm min_rmsdB={_m(fit_diag['min_rmsd_b']):.3f}nm "
                        f"cutA={float(cutoff_a_nm):.3f}nm cutB={float(cutoff_b_nm):.3f}nm",
                        flush=True,
                    )
                    if int(fit_diag["n_in_a"]) == 0 or int(fit_diag["n_in_b"]) == 0:
                        print(
                            "[Committor][Fit] Warning: empty A or B boundary set under current cutoffs; "
                            "DGA committor (and Q reward) will be unreliable until boundaries are hit.",
                            flush=True,
                        )
            except Exception:
                pass

                if str(committor_state_mode) == "vamp":
                    committor_state.dga_model = None
                    committor_state.vamp_model = fit_vamp_discrete_committor(
                        labels_trajs_fit,
                        in_a_trajs=in_a_fit,
                        in_b_trajs=in_b_fit,
                        n_states=int(n_states),
                        lag=int(lag_comm_embed),
                        prior_alpha=1e-3,
                    )
                    try:
                        nA = int(np.count_nonzero(getattr(committor_state.vamp_model, "a_states", np.zeros((0,), dtype=bool))))
                        nB = int(np.count_nonzero(getattr(committor_state.vamp_model, "b_states", np.zeros((0,), dtype=bool))))
                        deg = bool(getattr(committor_state.vamp_model, "degenerate", False))
                        print(
                            f"[Committor] Fitted VAMP discrete committor: K={int(n_states)} lag={int(lag_comm_embed)} "
                            f"A_states={nA} B_states={nB} degenerate={int(deg)}",
                            flush=True,
                        )
                    except Exception:
                        print(
                            f"[Committor] Fitted VAMP discrete committor: K={int(n_states)} lag={int(lag_comm_embed)}",
                            flush=True,
                        )
                else:
                    committor_state.vamp_model = None
                    committor_state.dga_model = fit_dga_committor_kmeans(
                        z_trajs_fit,
                        in_a_trajs=in_a_fit,
                        in_b_trajs=in_b_fit,
                        lag=int(lag_comm_embed),
                        n_clusters=int(n_clusters),
                        dim=int(dim_cfg),
                        max_fit_samples=int(max_samples),
                        seed=int(seed_cfg),
                        delay_n_embed=int(delay_n_embed),
                        delay_delta=int(delay_delta_embed),
                        fixed_kmeans=fixed_kmeans,
                        fixed_z_mu=fixed_mu,
                        fixed_z_std=fixed_std,
                    )
                    if committor_state.dga_model is None:
                        print(
                            "[Committor] Warning: DGA fit returned None (insufficient data); "
                            "committor-progress reward will be 0 until the next refresh."
                        )
                    else:
                        K_fit = int(getattr(committor_state.dga_model, "coeffs", np.zeros((0,))).shape[0])
                        solve_method = str(getattr(committor_state.dga_model, "solve_method", "") or "")
                        degenerate = bool(getattr(committor_state.dga_model, "degenerate", False))
                        q_dom_std = getattr(committor_state.dga_model, "q_domain_std", float("nan"))
                        q_dom_std_s = f"{float(q_dom_std):.3g}" if np.isfinite(float(q_dom_std)) else "nan"
                        print(
                            f"[Committor] Fitted DGA committor: clusters_req={int(n_clusters)} clusters_fit={int(K_fit)} "
                            f"dim={int(committor_state.dga_model.dim)} lag={int(committor_state.dga_model.lag)} "
                            f"delay_n={int(getattr(committor_state.dga_model, 'delay_n_embed', 0) or 0)} "
                            f"delay_delta={int(getattr(committor_state.dga_model, 'delay_delta', 1) or 1)} "
                            f"backend={getattr(committor_state.dga_model, 'backend', 'unknown')} "
                            f"solve={solve_method or 'unknown'} degenerate={int(degenerate)} q_dom_std={q_dom_std_s}"
                        )
        except Exception as exc_dga:
            committor_state.dga_model = None
            committor_state.vamp_model = None
            print(f"[Committor] Warning: DGA committor fit failed ({exc_dga}); disabling committor reward.")
    """
    if getattr(config, "reset_policy_on_projector_refresh", False):
        try:
            if isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy):
                with torch.no_grad():
                    if getattr(agent, "logits", None) is not None:
                        agent.logits.zero_()
                    if getattr(agent_old, "logits", None) is not None:
                        agent_old.logits.zero_()
                for opt in policy_opts:
                    opt.state.clear()
                print("[VAMP Refresh] Reset StateLogitPolicy logits to uniform after refresh.")
        except Exception as exc_reset:
            print(f"[VAMP Refresh] Warning: failed to reset policy after refresh: {exc_reset}")

    return {
        "perm": geom2vec_state_permutation_map,
        "inv_perm": geom2vec_state_inverse_permutation_map,
        "lagtime_embed": geom2vec_lagtime_embed,
        "boundary_plugin_fit_keys": boundary_plugin_fit_keys,
        "boundary_plugin_fit_nonce": boundary_plugin_fit_nonce,
    }
