"""Unified reward combination with EMA normalization.

Replaces ~400-730 lines of duplicated reward assembly in each training loop
with a single configurable combiner that handles:
- LOO-VAMP base reward
- Start-cluster entropy bonus
- VAMP-bin exploration bonus
- MSM-IG information-gain bonus
- Committor progress / tube / edge / connectivity rewards
- Off-action discount
- Per-term EMA z-score normalization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class RewardTerms:
    """Container for all raw reward components computed per epoch."""
    loo_vamp: Optional[np.ndarray] = None
    start_cluster: Optional[np.ndarray] = None
    start_basin: Optional[np.ndarray] = None
    vamp_bin: Optional[np.ndarray] = None
    msm_ig: Optional[np.ndarray] = None
    committor_progress: Optional[np.ndarray] = None
    committor_bin_entropy: Optional[np.ndarray] = None
    committor_tube: Optional[np.ndarray] = None
    committor_edge: Optional[np.ndarray] = None
    committor_conn: Optional[np.ndarray] = None
    off_action_mae: Optional[np.ndarray] = None


class EMANormalizer:
    """Exponential moving average z-score normalizer for a reward term.

    Tracks running mean and second moment via EMA, then z-scores raw values
    and rescales to match a target scale (typically LOO-VAMP std).
    """

    def __init__(self, beta: float = 0.9):
        self.beta = float(max(0.0, min(1.0, beta)))
        self.ema_m1: Optional[float] = None
        self.ema_m2: Optional[float] = None

    def normalize(
        self,
        raw: np.ndarray,
        target_scale: float = 1.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Z-score normalize *raw* and rescale to *target_scale*."""
        raw = np.asarray(raw, dtype=np.float32).ravel()
        raw = np.where(np.isfinite(raw), raw, 0.0)
        if raw.size == 0:
            return raw

        cur_m1 = float(raw.mean())
        cur_m2 = float((raw * raw).mean())

        if self.beta > 0:
            if self.ema_m1 is None:
                self.ema_m1, self.ema_m2 = cur_m1, cur_m2
            else:
                b = self.beta
                self.ema_m1 = b * self.ema_m1 + (1 - b) * cur_m1
                self.ema_m2 = b * self.ema_m2 + (1 - b) * cur_m2
            mean = self.ema_m1
            std = float(np.sqrt(max(self.ema_m2 - mean * mean, 0.0)))
        else:
            mean = cur_m1
            std = float(raw.std())

        if np.isfinite(std) and std > eps:
            z = (raw - mean) / std
        else:
            z = raw * 0.0

        scale = target_scale if (np.isfinite(target_scale) and target_scale > eps) else 1.0
        return (z * scale).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"beta": self.beta, "ema_m1": self.ema_m1, "ema_m2": self.ema_m2}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.beta = d.get("beta", self.beta)
        self.ema_m1 = d.get("ema_m1")
        self.ema_m2 = d.get("ema_m2")


class RewardCombiner:
    """Combines raw reward terms into a single per-walker reward tensor.

    Usage::

        combiner = RewardCombiner(config, device)
        terms = RewardTerms(loo_vamp=..., vamp_bin=..., ...)
        combined, metrics = combiner.combine(terms, launch_rewards, active_mask)
    """

    def __init__(self, config: Any, device: torch.device):
        self.device = device

        # Read coefficients from config
        self.loo_weight = float(getattr(config, "loo_vamp_reward_weight", 1.0) or 1.0)
        self.start_cluster_coef = float(getattr(config, "start_cluster_entropy_coef", 0.0) or 0.0)
        self.vamp_bin_coef = float(getattr(config, "vamp_bin_entropy_coef", 0.0) or 0.0)
        self.msm_ig_coef = float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0)
        self.off_action_coef = float(getattr(config, "off_action_penalty_coef", 0.0) or 0.0)

        # Committor coefficients
        self.comm_progress_coef = float(getattr(config, "committor_progress_reward_weight", 0.0) or 0.0)
        self.comm_bin_ent_coef = float(getattr(config, "committor_bin_entropy_reward_weight", 0.0) or 0.0)
        self.comm_tube_coef = float(getattr(config, "committor_tube_diversity_reward_weight", 0.0) or 0.0)
        self.comm_edge_coef = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0)
        self.comm_conn_coef = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0)
        self.comm_start_basin_coef = float(getattr(config, "committor_start_basin_entropy_reward_weight", 0.0) or 0.0)

        # Tie negative coefficients to ent_coef
        ent_coef = float(getattr(config, "ent_coef", 1e-3) or 1e-3)
        if self.start_cluster_coef < 0:
            self.start_cluster_coef = ent_coef
        if self.vamp_bin_coef < 0:
            self.vamp_bin_coef = ent_coef

        # EMA normalizers (per-term)
        vamp_bin_beta = float(getattr(config, "vamp_bin_norm_ema_beta", 0.9) or 0.9)
        msm_ig_beta = float(getattr(config, "msm_ig_norm_ema_beta", 0.9) or 0.9)
        comm_beta = float(getattr(config, "committor_norm_ema_beta", 0.9) or 0.9)

        self._norm_vamp_bin = EMANormalizer(vamp_bin_beta)
        self._norm_msm_ig = EMANormalizer(msm_ig_beta)
        self._norm_comm_progress = EMANormalizer(comm_beta)
        self._norm_comm_bin_ent = EMANormalizer(comm_beta)
        self._norm_comm_tube = EMANormalizer(comm_beta)
        self._norm_comm_edge = EMANormalizer(comm_beta)
        self._norm_comm_conn = EMANormalizer(comm_beta)

    def combine(
        self,
        terms: RewardTerms,
        launch_rewards: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Combine reward terms into a single tensor.

        Returns (combined_rewards, metrics_dict).
        """
        n = launch_rewards.shape[0]
        dev = self.device
        metrics: Dict[str, float] = {}
        combined = torch.zeros(n, device=dev, dtype=torch.float32)

        if active_mask is None:
            active_mask = torch.ones(n, dtype=torch.bool, device=dev)
        active_mask = active_mask.to(dev).bool()

        # LOO-VAMP base
        if terms.loo_vamp is not None and np.any(np.abs(terms.loo_vamp) > 1e-8):
            loo_t = torch.tensor(terms.loo_vamp, device=dev, dtype=torch.float32)
            combined = self.loo_weight * loo_t
            metrics["loo_vamp_mean"] = float(loo_t.mean())

        # Compute target scale for normalizing other terms to LOO-VAMP magnitude
        eps = 1e-8
        try:
            loo_std = float(np.std(np.asarray(terms.loo_vamp or [0], dtype=np.float32)))
            target_scale = self.loo_weight * loo_std if np.isfinite(loo_std) and loo_std > eps else 0.0
        except Exception:
            target_scale = 0.0
        if not (np.isfinite(target_scale) and target_scale > eps):
            try:
                target_scale = float(combined[active_mask].std(unbiased=False).item())
            except Exception:
                target_scale = 1.0

        # Simple additive terms (no normalization needed)
        combined = self._add_simple(combined, terms.start_cluster,
                                    self.start_cluster_coef, "start_cluster", metrics)
        combined = self._add_simple(combined, terms.start_basin,
                                    self.comm_start_basin_coef, "start_basin", metrics)

        # EMA-normalized terms
        combined = self._add_normalized(combined, terms.vamp_bin, self.vamp_bin_coef,
                                        self._norm_vamp_bin, target_scale, "vamp_bin", metrics)
        combined = self._add_normalized(combined, terms.msm_ig, self.msm_ig_coef,
                                        self._norm_msm_ig, target_scale, "msm_ig", metrics)
        combined = self._add_normalized(combined, terms.committor_progress, self.comm_progress_coef,
                                        self._norm_comm_progress, target_scale, "comm_progress", metrics)
        combined = self._add_normalized(combined, terms.committor_bin_entropy, self.comm_bin_ent_coef,
                                        self._norm_comm_bin_ent, target_scale, "comm_bin_ent", metrics)
        combined = self._add_normalized(combined, terms.committor_tube, self.comm_tube_coef,
                                        self._norm_comm_tube, target_scale, "comm_tube", metrics)
        combined = self._add_normalized(combined, terms.committor_edge, self.comm_edge_coef,
                                        self._norm_comm_edge, target_scale, "comm_edge", metrics)
        combined = self._add_normalized(combined, terms.committor_conn, self.comm_conn_coef,
                                        self._norm_comm_conn, target_scale, "comm_conn", metrics)

        # Off-action discount
        if self.off_action_coef > 0 and terms.off_action_mae is not None:
            mae = np.asarray(terms.off_action_mae, dtype=np.float32).ravel()
            mae = np.where(np.isfinite(mae), mae, 0.0)
            discount = np.exp(-self.off_action_coef * mae).astype(np.float32)
            discount_t = torch.tensor(discount, device=dev, dtype=torch.float32)
            combined = combined * discount_t
            metrics["off_action_discount_mean"] = float(discount.mean())

        metrics["combined_reward_mean"] = float(combined[active_mask].mean().item())
        metrics["combined_reward_std"] = float(combined[active_mask].std(unbiased=False).item())

        return combined, metrics

    def _add_simple(
        self,
        combined: torch.Tensor,
        raw: Optional[np.ndarray],
        coef: float,
        name: str,
        metrics: Dict[str, float],
    ) -> torch.Tensor:
        if coef <= 0 or raw is None:
            return combined
        t = torch.tensor(
            np.asarray(raw, dtype=np.float32).ravel(),
            device=self.device, dtype=torch.float32,
        )
        combined = combined + coef * t
        metrics[f"{name}_mean"] = float(t.mean())
        return combined

    def _add_normalized(
        self,
        combined: torch.Tensor,
        raw: Optional[np.ndarray],
        coef: float,
        normalizer: EMANormalizer,
        target_scale: float,
        name: str,
        metrics: Dict[str, float],
    ) -> torch.Tensor:
        if coef == 0 or raw is None:
            return combined
        normed = normalizer.normalize(raw, target_scale=target_scale)
        t = torch.tensor(normed, device=self.device, dtype=torch.float32)
        combined = combined + coef * t
        metrics[f"{name}_raw_mean"] = float(np.mean(np.asarray(raw, dtype=np.float32)))
        metrics[f"{name}_normed_mean"] = float(t.mean())
        return combined

    def state_dict(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k).state_dict()
            for k in dir(self)
            if k.startswith("_norm_") and isinstance(getattr(self, k), EMANormalizer)
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            norm = getattr(self, k, None)
            if isinstance(norm, EMANormalizer) and isinstance(v, dict):
                norm.load_state_dict(v)
