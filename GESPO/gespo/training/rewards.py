from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

from ..config import TrainingConfig


class RewardCalculator:
    """
    Computes scalar launch/trajectory rewards used by the sampler for logging.

    Note: Legacy projector reward paths have been removed; the active
    GESPO training loop uses geom2vec_vamp for discrete starts + LOO-VAMP
    rewards and keeps only lightweight PCA-grid coverage diagnostics here.
    """

    def __init__(
        self,
        config: TrainingConfig,
        pca_scorer,
        feature_builder,
        vendi_launch_pca,
        vendi_hill_pca,
        vendi_launch_angle,
        vendi_hill_angle,
        ref_positions_nm=None,
        ca_indices=None,
    ):
        if config is None:
            raise ValueError("config cannot be None")
        if pca_scorer is None:
            raise ValueError("pca_scorer cannot be None")
        if feature_builder is None:
            raise ValueError("feature_builder cannot be None")

        self.config = config
        self.pca_scorer = pca_scorer
        self.feature_builder = feature_builder

        # Novelty estimators kept for API compatibility; the streamlined loop
        # does not currently use Vendi rewards.
        self.vendi_launch_pca = vendi_launch_pca
        self.vendi_hill_pca = vendi_hill_pca
        self.vendi_launch_angle = vendi_launch_angle
        self.vendi_hill_angle = vendi_hill_angle

        self.ref_positions_nm = ref_positions_nm
        self.ca_indices = ca_indices

        # Weights
        self.w_launch = float(getattr(config, "novelty_launch_weight", 0.0) or 0.0)
        self.w_traj = float(getattr(config, "novelty_traj_weight", 0.0) or 0.0)
        self.w_cov = float(getattr(config, "pca_reward_weight", 0.0) or 0.0)
        self.w_trans = float(getattr(config, "geom2vec_start_transition_weight", 0.0) or 0.0)
        self.w_balance = float(getattr(config, "anchor_balance_weight", 0.0) or 0.0)

        # Reward normalization (EMA)
        self.normalize_rewards = bool(getattr(config, "normalize_trajectory_rewards", True))
        self.ema_alpha = float(getattr(config, "reward_ema_alpha", 0.1) or 0.1)
        self.pca_ema_std = 1.0

    def _to_pca_2d(self, cv_vec: np.ndarray, coords_nm: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if self.pca_scorer is None or not getattr(self.pca_scorer, "is_ready", False):
            return None
        try:
            feat = self.feature_builder.build_projection_features(cv_vec, coords_nm).reshape(1, -1)
            p = np.asarray(self.pca_scorer.pca.transform(feat)[0], dtype=np.float32)
            if p.shape[0] < 2 or not np.all(np.isfinite(p[:2])):
                return None
            return p[:2].copy()
        except Exception:
            return None

    def compute_launch_reward(self, env) -> Tuple[float, Optional[np.ndarray]]:
        # Launch rewards are disabled in the streamlined geom2vec_vamp loop.
        return 0.0, None

    def compute_trajectory_reward(
        self,
        trajectory_data: List,
        env,
        start_cv: np.ndarray,
        start_macro: Optional[int],
        end_macro: Optional[int],
        walker_idx: int,
    ) -> Tuple[float, float, float, float]:
        if not trajectory_data:
            return 0.0, 0.0, 0.0, 0.0

        # Coverage reward: average PCA-grid novelty over frames.
        pca_val = 0.0
        if getattr(self.pca_scorer, "is_ready", False) and self.w_cov != 0.0:
            vals: List[float] = []
            for frame in trajectory_data:
                try:
                    frame_cv = np.asarray(frame[0], dtype=np.float32)
                    frame_coords = np.asarray(frame[1], dtype=np.float32) if frame[1] is not None else None
                    feat_vec = self.feature_builder.build_projection_features(frame_cv, frame_coords)
                    vals.append(float(self.pca_scorer(None, feature_point=feat_vec)))
                except Exception:
                    continue
            if vals:
                pca_val = float(self.w_cov) * float(np.mean(vals))

        # Optional dual-anchor helpers (kept as no-op unless env exposes the metadata).
        dual_anchor = bool(getattr(env, "secondary_ref_ca_nm", None))

        total_reward = 0.0
        if dual_anchor and self.w_balance != 0.0:
            z0 = self._to_pca_2d(np.asarray(start_cv, dtype=np.float32), None)
            if z0 is not None:
                total_reward += float(self.w_balance) * float(np.exp(-float(np.sum(z0 * z0))))

        if dual_anchor and self.w_trans != 0.0:
            z_start = self._to_pca_2d(
                np.asarray(trajectory_data[0][0], dtype=np.float32),
                np.asarray(trajectory_data[0][1], dtype=np.float32),
            )
            z_end = self._to_pca_2d(
                np.asarray(trajectory_data[-1][0], dtype=np.float32),
                np.asarray(trajectory_data[-1][1], dtype=np.float32),
            )
            if z_start is not None and z_end is not None:
                total_reward += float(self.w_trans) * float(np.linalg.norm(z_end - z_start))

        # Normalize PCA coverage to a roughly stable scale.
        pca_val_norm = pca_val
        if self.normalize_rewards and pca_val != 0.0:
            pca_dev = abs(pca_val)
            self.pca_ema_std = (1.0 - self.ema_alpha) * self.pca_ema_std + self.ema_alpha * pca_dev
            pca_std_safe = max(float(self.pca_ema_std), 1e-6)
            pca_val_norm = float(pca_val) / pca_std_safe

        total_reward += float(pca_val_norm)
        return float(total_reward), 0.0, float(pca_val_norm), 0.0

    def project_to_pca_2d(self, cv_vec: np.ndarray, coords_nm: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self._to_pca_2d(cv_vec, coords_nm)
