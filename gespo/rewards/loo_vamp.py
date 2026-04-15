"""
LOO-VAMP Rewards Manager - Manages leave-one-out VAMP-2 reward calculations.

This module extracts LOO-VAMP reward computation logic from the training loops,
providing a clean interface for:
- Global sufficient statistics tracking
- History seeding from cached trajectories
- Per-trajectory LOO reward computation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gespo.projections.vamp_stats import SufficientStatistics


class LOOVAMPRewards:
    """
    Manages LOO-VAMP reward calculation using sufficient statistics.
    
    Maintains global sufficient statistics for fast leave-one-out VAMP-2
    scoring, enabling efficient exploration bonuses based on trajectory novelty.
    """
    
    def __init__(self, n_models: int = 1, n_states: int = 2, lag_embed: int = 1):
        """
        Initialize LOOVAMPRewards manager.
        
        Args:
            n_models: Number of ensemble models
            n_states: Number of discrete states
            lag_embed: Lag time in embedding frames
        """
        self.n_models = n_models
        self.n_states = n_states
        self.lag_embed = lag_embed
        
        # Global sufficient statistics (one per ensemble model)
        self.stats_by_model: Optional[List[SufficientStatistics]] = None
        self.stats_keys: Set[str] = set()
        self.stats_signature: Optional[Tuple[int, int, int]] = None
        
        self._reset_stats()
    
    def _reset_stats(self) -> None:
        """Reset statistics to match current configuration."""
        sig = (self.n_models, self.n_states, self.lag_embed)
        if self.stats_signature != sig:
            self.stats_by_model = [
                SufficientStatistics(self.n_states) for _ in range(self.n_models)
            ]
            self.stats_keys = set()
            self.stats_signature = sig
    
    def invalidate(self, tag: str = "") -> None:
        """
        Invalidate global statistics (e.g., after projector refresh).
        
        Args:
            tag: Description of why invalidation occurred
        """
        self.stats_by_model = None
        self.stats_keys = set()
        self.stats_signature = None
        print(f"[LOO VAMP] Invalidated global stats ({tag}).")
    
    def update_config(self, n_models: int, n_states: int, lag_embed: int) -> None:
        """
        Update configuration and reset statistics if needed.
        
        Args:
            n_models: Number of ensemble models
            n_states: Number of discrete states
            lag_embed: Lag time in embedding frames
        """
        self.n_models = n_models
        self.n_states = n_states
        self.lag_embed = lag_embed
        self._reset_stats()
    
    def seed_from_trajectories(
        self,
        trajectories: List[Tuple[str, np.ndarray]],
        perm_maps: List[np.ndarray],
        tag: str = "",
    ) -> Tuple[int, int, int]:
        """
        Seed global statistics from a list of trajectories.
        
        Args:
            trajectories: List of (cache_key, state_probs) tuples
                         state_probs shape: [T, n_states] for each trajectory
            perm_maps: Permutation maps for each model [n_models, n_states]
            tag: Description for logging
            
        Returns:
            Tuple of (added_count, skipped_short_count, skipped_missing_count)
        """
        self._reset_stats()
        
        added = 0
        skipped_short = 0
        skipped_missing = 0
        
        for cache_key, probs_raw in trajectories:
            if cache_key in self.stats_keys:
                continue
            
            if probs_raw is None:
                skipped_missing += 1
                continue
            
            # Check trajectory length
            T = int(probs_raw.shape[0])
            if T <= self.lag_embed:
                self.stats_keys.add(str(cache_key))
                skipped_short += 1
                continue
            
            # Add to each model's statistics with appropriate permutation
            for m in range(self.n_models):
                try:
                    # Apply permutation
                    probs = probs_raw[:, perm_maps[m]]
                    
                    # Ensure correct dimension
                    if probs.shape[1] != self.n_states:
                        if probs.shape[1] > self.n_states:
                            probs = probs[:, :self.n_states]
                        else:
                            probs = np.pad(
                                probs,
                                ((0, 0), (0, self.n_states - probs.shape[1])),
                                mode="constant"
                            )
                    
                    # Add trajectory pair to statistics
                    self.stats_by_model[m].add_trajectory(
                        probs[:-self.lag_embed],
                        probs[self.lag_embed:]
                    )
                except Exception:
                    continue
            
            self.stats_keys.add(str(cache_key))
            added += 1
        
        if added and tag:
            print(
                f"[LOO VAMP] Seeded global stats ({tag}): "
                f"added_trajs={added}, skipped_short={skipped_short}, "
                f"skipped_missing={skipped_missing}"
            )
        
        return added, skipped_short, skipped_missing
    
    def compute_loo_reward(
        self,
        cache_key: str,
        probs: np.ndarray,
        perm_map: np.ndarray,
        model_idx: int = 0,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Compute leave-one-out VAMP-2 reward for a trajectory.
        
        Args:
            cache_key: Unique identifier for this trajectory
            probs: State probabilities [T, n_states]
            perm_map: Permutation map for this model [n_states]
            model_idx: Which ensemble model to use
            epsilon: Regularization for VAMP-2 calculation
            
        Returns:
            LOO VAMP-2 score (exploration bonus)
        """
        if self.stats_by_model is None:
            return 0.0
        
        if model_idx >= len(self.stats_by_model):
            return 0.0
        
        # Apply permutation
        probs_perm = probs[:, perm_map]
        
        # Ensure correct dimension
        if probs_perm.shape[1] != self.n_states:
            if probs_perm.shape[1] > self.n_states:
                probs_perm = probs_perm[:, :self.n_states]
            else:
                probs_perm = np.pad(
                    probs_perm,
                    ((0, 0), (0, self.n_states - probs_perm.shape[1])),
                    mode="constant"
                )
        
        T = int(probs_perm.shape[0])
        if T <= self.lag_embed:
            return 0.0
        
        stats = self.stats_by_model[model_idx]
        
        # Compute baseline score (with trajectory)
        try:
            stats.add_trajectory(
                probs_perm[:-self.lag_embed],
                probs_perm[self.lag_embed:]
            )
            score_with = stats.compute_vamp2_score(epsilon=epsilon)
            
            # Remove trajectory and compute LOO score
            stats.remove_trajectory(
                probs_perm[:-self.lag_embed],
                probs_perm[self.lag_embed:]
            )
            score_without = stats.compute_vamp2_score(epsilon=epsilon)
            
            # LOO reward is the improvement
            reward = float(score_with - score_without)
            
            # Re-add trajectory to maintain global state
            stats.add_trajectory(
                probs_perm[:-self.lag_embed],
                probs_perm[self.lag_embed:]
            )
            
            return reward
        except Exception:
            return 0.0
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of current statistics state."""
        return {
            "n_models": self.n_models,
            "n_states": self.n_states,
            "lag_embed": self.lag_embed,
            "n_trajectories": len(self.stats_keys),
            "signature": self.stats_signature,
            "initialized": self.stats_by_model is not None,
        }
