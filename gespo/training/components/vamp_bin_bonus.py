"""VAMP-bin exploration bonus manager.

Maintains 2D bin-count statistics in the first two linear VAMP coordinates
and provides an entropy-based exploration bonus.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Set, Tuple

import numpy as np


class VampBinBonus:
    """Manages VAMP-bin exploration bonus state and computation."""

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None  # shape (2,)
        self.std: Optional[np.ndarray] = None  # shape (2,)
        self.counts: Optional[np.ndarray] = None  # shape (bins, bins)
        self.total: int = 0
        self.counted_keys: Set[str] = set()
        self.signature: Optional[Tuple[int, float, int, int]] = None
        self.norm_ema_m1: Optional[float] = None
        self.norm_ema_m2: Optional[float] = None
        self.norm_ema_scale: Optional[float] = None

    def invalidate(self, tag: str) -> None:
        """Reset all bin-count statistics."""
        self.mu = None
        self.std = None
        self.counts = None
        self.total = 0
        self.counted_keys = set()
        self.signature = None
        self.norm_ema_m1 = None
        self.norm_ema_m2 = None
        self.norm_ema_scale = None
        print(f"[VAMP Bin] Invalidated bin stats ({tag}).")

    @staticmethod
    def bin_indices(z2_standardized: np.ndarray, *, bins: int, clip: float) -> Tuple[np.ndarray, np.ndarray]:
        """Map standardized 2D coordinates to integer bin indices."""
        z = np.asarray(z2_standardized, dtype=np.float32).reshape(-1, 2)
        b = int(max(1, int(bins)))
        c = float(clip)
        if not np.isfinite(c) or c <= 0:
            c = 1.0
        zc = np.clip(z, -c, c)
        u = (zc + c) / (2.0 * c)
        idx = np.floor(u * float(b)).astype(np.int64)
        idx = np.clip(idx, 0, b - 1)
        return idx[:, 0], idx[:, 1]

    def ensure_history_counts(
        self,
        *,
        upto_epoch_exclusive: int,
        n_states: int,
        tag: str,
        config: Any,
        geom2vec_cache: Any,
        geom2vec_projector_linear: Any,
        geom2vec_mgr: Any,
        iter_history_trajs: Callable[..., Iterator],
    ) -> None:
        """Ensure VAMP-bin statistics are seeded from history cache.

        Parameters
        ----------
        iter_history_trajs : callable
            A callable ``iter_history_trajs(upto_epoch_exclusive)`` that yields
            ``(epoch_idx, traj_idx, L_i, dcd_path, coords_nm)`` tuples.
        """
        coef = float(getattr(config, "vamp_bin_entropy_coef", -1.0) or -1.0)
        if coef < 0:
            coef = float(getattr(config, "ent_coef", 0.0) or 0.0)
        if coef <= 0.0:
            return

        if (
            geom2vec_cache is None
            or geom2vec_projector_linear is None
            or getattr(geom2vec_projector_linear, "trained", None) is None
            or geom2vec_mgr is None
        ):
            return

        bins = int(max(1, int(getattr(config, "vamp_bin_bins", 50) or 50)))
        clip = float(getattr(config, "vamp_bin_clip", 3.0) or 3.0)
        max_samples = int(max(256, int(getattr(config, "vamp_bin_max_samples", 20000) or 20000)))

        sig = (int(bins), float(clip), int(max_samples), int(n_states))
        if self.counts is None or self.signature != sig:
            self.mu = None
            self.std = None
            self.counts = np.zeros((int(bins), int(bins)), dtype=np.int64)
            self.total = 0
            self.counted_keys = set()
            self.signature = sig

        if self.counts is None:
            return

        n_states_i = int(n_states)
        if n_states_i < 2:
            return

        # 1) Estimate mean/std in the first two linear VAMP coordinates.
        if self.mu is None or self.std is None:
            samples: List[np.ndarray] = []
            n_collected = 0
            for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in iter_history_trajs(upto_epoch_exclusive):
                if n_collected >= max_samples:
                    break
                key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
                cached_key = geom2vec_mgr.ensure_cached(
                    key=key,
                    dcd_path=dcd_path,
                    coords_nm=coords_nm,
                    overwrite=False,
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
                    z2 = np.asarray(z[:, :2], dtype=np.float64)
                except Exception:
                    continue
                if z2.size == 0:
                    continue
                remaining = max_samples - n_collected
                if int(z2.shape[0]) > int(remaining) and remaining > 0:
                    try:
                        idx = np.linspace(0, int(z2.shape[0]) - 1, num=int(remaining), dtype=np.int64)
                        z2 = z2[idx]
                    except Exception:
                        z2 = z2[: int(remaining)]
                samples.append(z2)
                n_collected += int(z2.shape[0])

            if not samples:
                self.mu = np.zeros((2,), dtype=np.float32)
                self.std = np.ones((2,), dtype=np.float32)
            else:
                arr = np.concatenate(samples, axis=0)
                mu = arr.mean(axis=0)
                std = arr.std(axis=0)
                std = np.where(std > 1e-6, std, 1.0)
                self.mu = mu.astype(np.float32, copy=False)
                self.std = std.astype(np.float32, copy=False)
                print(
                    f"[VAMP Bin] Estimated mean/std ({tag}): samples={int(arr.shape[0])} bins={int(bins)} clip={float(clip):.2f}",
                    flush=True,
                )

        # 2) Seed bin counts from history, once per cached trajectory key.
        mu2 = np.asarray(self.mu, dtype=np.float32).reshape(2)
        std2 = np.asarray(self.std, dtype=np.float32).reshape(2)
        std2 = np.where(std2 > 1e-8, std2, 1.0).astype(np.float32, copy=False)

        added = 0
        skipped_missing = 0
        for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in iter_history_trajs(upto_epoch_exclusive):
            key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
            key_s = str(key)
            if key_s in self.counted_keys:
                continue

            cached_key = key if geom2vec_mgr.has_cached_key(key) else geom2vec_mgr.ensure_cached(
                key=key,
                dcd_path=dcd_path,
                coords_nm=coords_nm,
                overwrite=False,
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
                graph, ca = geom2vec_cache.load(cached_key)
            except Exception:
                skipped_missing += 1
                continue

            try:
                z_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                z = np.asarray(z_raw, dtype=np.float32)
                z2 = np.asarray(z[:, :2], dtype=np.float32)
                if z2.size == 0:
                    self.counted_keys.add(key_s)
                    self.counted_keys.add(cached_key_s)
                    continue
                z2s = (z2 - mu2[None, :]) / std2[None, :]
                ix, iy = self.bin_indices(z2s, bins=bins, clip=clip)
                np.add.at(self.counts, (ix, iy), 1)
                self.total += int(ix.shape[0])
            except Exception:
                skipped_missing += 1
                continue

            self.counted_keys.add(key_s)
            self.counted_keys.add(cached_key_s)
            added += 1

        if added:
            try:
                visited = int(np.count_nonzero(self.counts))
                nbins = int(bins) * int(bins)
                print(
                    f"[VAMP Bin] Seeded history ({tag}): added_trajs={added}, skipped_missing={skipped_missing}, "
                    f"visited={visited}/{nbins}, total_frames={int(self.total)}",
                    flush=True,
                )
            except Exception:
                print(f"[VAMP Bin] Seeded history ({tag}): added_trajs={added}", flush=True)
