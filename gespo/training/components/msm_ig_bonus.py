"""MSM information-gain exploration bonus manager.

Maintains MSM transition count statistics and provides a Bayesian
information-gain reward based on Dirichlet-posterior KL divergence.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy.special import digamma, gammaln  # type: ignore
except Exception:  # pragma: no cover
    digamma = None  # type: ignore
    gammaln = None  # type: ignore


class MSMIGBonus:
    """Manages MSM information-gain exploration bonus state and computation."""

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.total_pairs: int = 0
        self.counted_keys: Set[str] = set()
        self.signature: Optional[Tuple[Any, ...]] = None
        self.kmeans: Optional[Any] = None
        self.norm_ema_m1: Optional[float] = None
        self.norm_ema_m2: Optional[float] = None
        self.norm_ema_scale: Optional[float] = None

    def invalidate(self, tag: str) -> None:
        """Reset all MSM-IG statistics."""
        self.mu = None
        self.std = None
        self.counts = None
        self.total_pairs = 0
        self.counted_keys = set()
        self.signature = None
        self.kmeans = None
        self.norm_ema_m1 = None
        self.norm_ema_m2 = None
        self.norm_ema_scale = None
        print(f"[MSM IG] Invalidated MSM IG stats ({tag}).")

    @staticmethod
    def reward_from_counts(
        *,
        base_counts: np.ndarray,
        delta_counts: np.ndarray,
        prior_alpha: float,
    ) -> float:
        """Compute Dirichlet-posterior KL divergence (information gain) from transition counts."""
        base = np.asarray(base_counts, dtype=np.float64)
        delta = np.asarray(delta_counts, dtype=np.float64)
        if base.shape != delta.shape or base.ndim != 2 or base.shape[0] != base.shape[1]:
            raise ValueError("MSM count matrices must be square and have the same shape.")
        if float(prior_alpha) <= 0.0:
            raise ValueError("prior_alpha must be > 0.")
        K = int(base.shape[0])
        if K <= 0:
            return 0.0
        if float(np.sum(delta)) <= 0.0:
            return 0.0

        alpha_q = base + float(prior_alpha)
        alpha_p = alpha_q + delta
        sum_p = np.sum(alpha_p, axis=1)
        sum_q = np.sum(alpha_q, axis=1)
        term1 = gammaln(sum_p) - gammaln(sum_q)
        term2 = -np.sum(gammaln(alpha_p) - gammaln(alpha_q), axis=1)
        term3 = np.sum(delta * (digamma(alpha_p) - digamma(sum_p)[:, None]), axis=1)
        kl_rows = term1 + term2 + term3
        kl = float(np.sum(kl_rows))
        if not np.isfinite(kl):
            return 0.0
        return float(max(0.0, kl))

    @staticmethod
    def transition_counts(labels: np.ndarray, *, n_states: int, lag: int) -> Tuple[np.ndarray, int]:
        """Compute transition count matrix from a sequence of state labels."""
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        K = int(n_states)
        lag_i = int(lag)
        if labels.size <= lag_i or K <= 0 or lag_i <= 0:
            return np.zeros((K, K), dtype=np.int64), 0
        a = labels[: int(labels.size - lag_i)]
        b = labels[int(lag_i):]
        idx = a * int(K) + b
        flat = np.bincount(idx, minlength=int(K) * int(K))
        return flat.reshape((int(K), int(K))), int(a.shape[0])

    def ensure_history_counts(
        self,
        *,
        upto_epoch_exclusive: int,
        n_states: int,
        lag_embed: int,
        tag: str,
        config: Any,
        geom2vec_cache: Any,
        geom2vec_projector: Any,
        geom2vec_projector_linear: Any,
        geom2vec_mgr: Any,
        geom2vec_state_permutation_map: np.ndarray,
        iter_history_trajs: Callable[..., Iterator],
    ) -> None:
        """Populate/extend MSM transition counts from history.

        Modes:
        - kmeans: k-means microstates in linear VAMP/VAC coordinates.
        - vamp:   argmax over softmax VAMP probabilities ("softmax clusters").
        """
        coef = float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0)
        if coef <= 0.0:
            return
        state_mode = str(getattr(config, "msm_ig_state_mode", "kmeans") or "kmeans").strip().lower()
        if state_mode not in {"kmeans", "vamp"}:
            state_mode = "kmeans"

        n_states_i = int(n_states)
        lag_i = int(lag_embed)
        if lag_i <= 0:
            return

        if state_mode == "vamp":
            self._ensure_vamp_mode(
                upto_epoch_exclusive=upto_epoch_exclusive,
                n_states_i=n_states_i, lag_i=lag_i, tag=tag,
                geom2vec_cache=geom2vec_cache,
                geom2vec_projector=geom2vec_projector,
                geom2vec_mgr=geom2vec_mgr,
                geom2vec_state_permutation_map=geom2vec_state_permutation_map,
                iter_history_trajs=iter_history_trajs,
            )
            return

        self._ensure_kmeans_mode(
            upto_epoch_exclusive=upto_epoch_exclusive,
            n_states_i=n_states_i, lag_i=lag_i, tag=tag, config=config,
            geom2vec_cache=geom2vec_cache,
            geom2vec_projector_linear=geom2vec_projector_linear,
            geom2vec_mgr=geom2vec_mgr,
            iter_history_trajs=iter_history_trajs,
        )

    def _ensure_vamp_mode(
        self, *, upto_epoch_exclusive, n_states_i, lag_i, tag,
        geom2vec_cache, geom2vec_projector, geom2vec_mgr,
        geom2vec_state_permutation_map, iter_history_trajs,
    ) -> None:
        if (
            geom2vec_cache is None
            or geom2vec_projector is None
            or (
                getattr(geom2vec_projector, "trained", None) is None
                and not bool(getattr(geom2vec_projector, "trained_ensemble", None))
            )
        ):
            return
        K = int(n_states_i)
        if K <= 1:
            return
        sig = ("vamp", int(K), int(lag_i), int(n_states_i))
        if self.counts is None or self.signature != sig:
            self.mu = None
            self.std = None
            self.counts = np.zeros((int(K), int(K)), dtype=np.float64)
            self.total_pairs = 0
            self.counted_keys = set()
            self.signature = sig
            self.kmeans = None

        if self.counts is None:
            return

        added = 0
        skipped_missing = 0
        skipped_short = 0
        for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in iter_history_trajs(upto_epoch_exclusive):
            key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
            key_s = str(key)
            if key_s in self.counted_keys:
                continue

            cached_key = key if geom2vec_mgr.has_cached_key(key) else geom2vec_mgr.ensure_cached(
                key=key, dcd_path=dcd_path, coords_nm=coords_nm, overwrite=False,
                meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
            )
            if cached_key is None:
                skipped_missing += 1
                continue
            cached_key_s = str(cached_key)
            if cached_key_s in self.counted_keys:
                self.counted_keys.add(key_s)
                continue

            try:
                rec = geom2vec_cache.get(cached_key)
                if int(rec.frames) <= int(lag_i):
                    self.counted_keys.add(key_s)
                    self.counted_keys.add(cached_key_s)
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
                if probs.ndim != 2:
                    raise ValueError("probs must have shape (T, K)")
                probs = probs[:, geom2vec_state_permutation_map]
                if probs.shape[1] != int(K):
                    if probs.shape[1] > int(K):
                        probs = probs[:, :int(K)]
                    else:
                        probs = np.pad(probs, ((0, 0), (0, int(K) - int(probs.shape[1]))), mode="constant")
                labels = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                counts, n_pairs = self.transition_counts(labels, n_states=int(K), lag=int(lag_i))
            except Exception:
                skipped_missing += 1
                continue

            if n_pairs > 0:
                self.counts += np.asarray(counts, dtype=np.float64)
                self.total_pairs += int(n_pairs)

            self.counted_keys.add(key_s)
            self.counted_keys.add(cached_key_s)
            added += 1

        if added:
            try:
                nonzero = int(np.count_nonzero(self.counts))
                print(
                    f"[MSM IG] Seeded history ({tag}): mode=vamp added_trajs={added}, skipped_short={skipped_short}, "
                    f"skipped_missing={skipped_missing}, nonzero_edges={nonzero}, total_pairs={int(self.total_pairs)}",
                    flush=True,
                )
            except Exception:
                print(f"[MSM IG] Seeded history ({tag}): mode=vamp added_trajs={added}", flush=True)

    def _ensure_kmeans_mode(
        self, *, upto_epoch_exclusive, n_states_i, lag_i, tag, config,
        geom2vec_cache, geom2vec_projector_linear, geom2vec_mgr, iter_history_trajs,
    ) -> None:
        if (
            geom2vec_cache is None
            or geom2vec_projector_linear is None
            or getattr(geom2vec_projector_linear, "trained", None) is None
        ):
            return

        K = int(getattr(config, "msm_ig_n_states", 50) or 50)
        dim = int(getattr(config, "msm_ig_dim", 2) or 2)
        max_samples = int(getattr(config, "msm_ig_max_fit_samples", 20000) or 20000)
        seed = int(getattr(config, "msm_ig_seed", 42) or 42)

        if K <= 1 or dim <= 0 or n_states_i < dim:
            return
        max_samples = max(int(max_samples), int(K))

        sig = ("kmeans", int(K), int(dim), int(lag_i), int(max_samples), int(seed), int(n_states_i))
        if self.counts is None or self.signature != sig:
            self.mu = None
            self.std = None
            self.counts = np.zeros((int(K), int(K)), dtype=np.float64)
            self.total_pairs = 0
            self.counted_keys = set()
            self.signature = sig
            self.kmeans = None

        if self.counts is None:
            return

        if self.mu is None or self.std is None or self.kmeans is None:
            samples: list[np.ndarray] = []
            n_collected = 0
            for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in iter_history_trajs(upto_epoch_exclusive):
                if n_collected >= max_samples:
                    break
                key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
                cached_key = geom2vec_mgr.ensure_cached(
                    key=key, dcd_path=dcd_path, coords_nm=coords_nm, overwrite=False,
                    meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                )
                if cached_key is None:
                    continue
                try:
                    graph, ca = geom2vec_cache.load(cached_key)
                except Exception:
                    continue
                try:
                    z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                    z = np.asarray(z_raw, dtype=np.float32)
                    z_d = z[:, :int(dim)]
                except Exception:
                    continue
                if z_d.size == 0:
                    continue
                remaining = max_samples - n_collected
                if int(z_d.shape[0]) > int(remaining) and remaining > 0:
                    try:
                        idx = np.linspace(0, int(z_d.shape[0]) - 1, num=int(remaining), dtype=np.int64)
                        z_d = z_d[idx]
                    except Exception:
                        z_d = z_d[:int(remaining)]
                samples.append(np.asarray(z_d, dtype=np.float64))
                n_collected += int(z_d.shape[0])

            if not samples:
                self.mu = np.zeros((int(dim),), dtype=np.float32)
                self.std = np.ones((int(dim),), dtype=np.float32)
                self.kmeans = None
                return

            arr = np.concatenate(samples, axis=0)
            mu = arr.mean(axis=0)
            std = arr.std(axis=0)
            std = np.where(std > 1e-6, std, 1.0)
            self.mu = mu.astype(np.float32, copy=False)
            self.std = std.astype(np.float32, copy=False)

            if int(arr.shape[0]) < int(K):
                self.kmeans = None
                print(
                    f"[MSM IG] Insufficient frames for k-means (need >= {int(K)}, have {int(arr.shape[0])}); "
                    "skipping MSM-IG for now.",
                    flush=True,
                )
                return

            try:
                from sklearn.cluster import MiniBatchKMeans
            except Exception as exc:
                self.kmeans = None
                print(f"[MSM IG] Warning: sklearn unavailable ({exc}); MSM-IG disabled.", flush=True)
                return

            arr_norm = (arr - mu[None, :]) / std[None, :]
            bs = int(min(2048, max(256, int(arr_norm.shape[0]))))
            try:
                kmeans = MiniBatchKMeans(
                    n_clusters=int(K), random_state=int(seed), batch_size=int(bs), n_init=1, max_iter=100,
                )
                kmeans.fit(arr_norm)
                self.kmeans = kmeans
                print(
                    f"[MSM IG] Fitted k-means: K={int(K)} dim={int(dim)} samples={int(arr_norm.shape[0])} "
                    f"lag_embed={int(lag_i)}",
                    flush=True,
                )
            except Exception as exc:
                self.kmeans = None
                print(f"[MSM IG] Warning: k-means fit failed ({exc}); MSM-IG disabled.", flush=True)
                return

        if self.kmeans is None:
            return

        added = 0
        skipped_missing = 0
        skipped_short = 0
        for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in iter_history_trajs(upto_epoch_exclusive):
            key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
            key_s = str(key)
            if key_s in self.counted_keys:
                continue

            cached_key = key if geom2vec_mgr.has_cached_key(key) else geom2vec_mgr.ensure_cached(
                key=key, dcd_path=dcd_path, coords_nm=coords_nm, overwrite=False,
                meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
            )
            if cached_key is None:
                skipped_missing += 1
                continue
            cached_key_s = str(cached_key)
            if cached_key_s in self.counted_keys:
                self.counted_keys.add(key_s)
                continue

            try:
                rec = geom2vec_cache.get(cached_key)
                if int(rec.frames) <= int(lag_i):
                    self.counted_keys.add(key_s)
                    self.counted_keys.add(cached_key_s)
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
                z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                z = np.asarray(z_raw, dtype=np.float32)
                z_d = z[:, :int(dim)]
                z_norm = (z_d - np.asarray(self.mu, dtype=np.float32)[None, :]) / np.asarray(self.std, dtype=np.float32)[None, :]
                labels = self.kmeans.predict(np.asarray(z_norm, dtype=np.float64))
                counts, n_pairs = self.transition_counts(labels, n_states=int(K), lag=int(lag_i))
            except Exception:
                skipped_missing += 1
                continue

            if n_pairs > 0:
                self.counts += np.asarray(counts, dtype=np.float64)
                self.total_pairs += int(n_pairs)

            self.counted_keys.add(key_s)
            self.counted_keys.add(cached_key_s)
            added += 1

        if added:
            try:
                nonzero = int(np.count_nonzero(self.counts))
                print(
                    f"[MSM IG] Seeded history ({tag}): added_trajs={added}, skipped_short={skipped_short}, "
                    f"skipped_missing={skipped_missing}, nonzero_edges={nonzero}, total_pairs={int(self.total_pairs)}",
                    flush=True,
                )
            except Exception:
                print(f"[MSM IG] Seeded history ({tag}): added_trajs={added}", flush=True)
