"""Base training loop with template method pattern.

Provides the shared epoch structure (refresh → sample → run → reward → update → log)
that all GESPO training loops follow. Subclasses override the abstract hooks for
start sampling and reward computation.
"""
from __future__ import annotations

import gc
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from gespo.rewards.combiner import RewardCombiner, RewardTerms


class BaseTrainingLoop(ABC):
    """Template method base class for GESPO training loops.

    Subclasses must implement:
    - ``_setup_policy``: Initialize the start policy and optimizer.
    - ``_setup_loop_specific``: Any loop-specific initialization.
    - ``_sample_starts``: Generate start configurations for an epoch.
    - ``_compute_raw_rewards``: Compute per-walker reward terms.
    - ``_pack_update_data``: Package data for the optimizer.

    The base class handles:
    - Device setup and environment initialization.
    - Geom2vec + VAMP projector management.
    - History accumulation and retrieval cache.
    - Checkpoint save/restore.
    - Epoch timing and stats logging.
    """

    def __init__(self, args):
        self.args = args
        # These are populated during _setup()
        self.config = None
        self.torch_device = None
        self.gen_device = None
        self.envs = None
        self.env_ref = None
        self.agent = None
        self.agent_old = None
        self.optimizer = None
        self.sampler = None
        self.feature_builder = None
        self.geom2vec_mgr = None
        self.geom2vec_cache = None
        self.geom2vec_projector = None
        self.geom2vec_projector_linear = None
        self.reward_combiner = None
        self.ckpt_manager = None

        # History state (accumulated across epochs)
        self.all_epoch_data: List[List[Any]] = []
        self.all_trajectory_lengths: List[List[int]] = []
        self.all_epoch_traj_dcd_paths: List[List[Optional[str]]] = []
        self.all_epoch_traj_dcd_frame_indices: List[List[Optional[np.ndarray]]] = []
        self.all_epoch_vamp_exclude: List[bool] = []

        # Projector state
        self.geom2vec_state_permutation_map: Optional[np.ndarray] = None
        self.geom2vec_state_inverse_permutation_map: Optional[np.ndarray] = None
        self.geom2vec_lagtime_embed: int = 0

        self.ep_start = 0

    # ── Public entry point ──────────────────────────────────────────

    def train(self):
        """Run the full training loop."""
        self._setup()

        for ep in range(self.ep_start, self.config.episodes):
            self._run_epoch(ep)

    # ── Epoch structure (template method) ───────────────────────────

    def _run_epoch(self, ep: int):
        """Single epoch — calls phase hooks in order."""
        epoch_start_time = time.time()

        # Phase A: Refresh projector if needed
        self._maybe_refresh_projector(ep)
        self._pre_sample_seed_stats(ep)

        # Phase B: Sample start configurations (abstract)
        starts, start_meta = self._sample_starts(ep)

        # Phase C: Run MD trajectories
        epoch_data, metrics = self._run_trajectories(ep, starts, start_meta)

        # Phase D: Compute raw rewards (abstract)
        raw_terms = self._compute_raw_rewards(ep, epoch_data, metrics)

        # Phase E: Combine rewards + policy update
        self._combine_and_update(ep, raw_terms, starts, start_meta, metrics)

        # Phase F: Logging, checkpointing, plotting
        self._log_epoch(ep, metrics, epoch_start_time)

    # ── Setup phases (shared) ───────────────────────────────────────

    def _setup(self):
        """Full pre-loop initialization. Calls abstract hooks at the right points."""
        self._setup_config()
        self._setup_devices()
        self._setup_environments()
        self._setup_policy()          # abstract
        self._setup_geom2vec()
        self._setup_history()
        self._setup_loop_specific()   # abstract
        self._setup_checkpoint()
        self._setup_sampler()
        self._setup_reward_combiner()
        self._setup_anchoring()

    def _setup_config(self):
        """Parse config from args."""
        from gespo.config import TrainingConfig
        self.config = TrainingConfig.from_args(self.args)

    def _setup_devices(self):
        """Resolve torch and generator devices."""
        gen_device_str = str(self.config.gen_device)
        self.gen_device = gen_device_str
        if torch.cuda.is_available() and gen_device_str.startswith("cuda"):
            self.torch_device = torch.device(gen_device_str)
        else:
            self.torch_device = torch.device("cpu")

    def _setup_environments(self):
        """Initialize OpenMM simulation environments."""
        # Subclasses can override for custom env setup
        pass

    def _setup_geom2vec(self):
        """Initialize geom2vec backbone and VAMP projectors."""
        # Subclasses populate self.geom2vec_mgr, self.geom2vec_cache, etc.
        pass

    def _setup_history(self):
        """Initialize history data structures."""
        self.all_epoch_data = []
        self.all_trajectory_lengths = []
        self.all_epoch_traj_dcd_paths = []
        self.all_epoch_traj_dcd_frame_indices = []
        self.all_epoch_vamp_exclude = []

    def _setup_checkpoint(self):
        """Restore from checkpoint if configured."""
        pass

    def _setup_sampler(self):
        """Initialize the trajectory sampler."""
        pass

    def _setup_reward_combiner(self):
        """Initialize the reward combiner."""
        if self.config is not None:
            self.reward_combiner = RewardCombiner(self.config, self.torch_device)

    def _setup_anchoring(self):
        """Run optional anchoring round to seed history."""
        pass

    # ── Abstract hooks (subclass-specific) ──────────────────────────

    @abstractmethod
    def _setup_policy(self):
        """Initialize the start policy, optimizer, and any policy-specific state."""
        ...

    @abstractmethod
    def _setup_loop_specific(self):
        """Any loop-specific initialization after common setup."""
        ...

    @abstractmethod
    def _sample_starts(self, ep: int) -> Tuple[Any, Dict[str, Any]]:
        """Generate start configurations for the epoch.

        Returns:
            starts: Start CV values or coordinates for each walker.
            start_meta: Dict with extra metadata (log_probs, global_values,
                        replay_coords, start_cluster_entropy, etc.)
        """
        ...

    @abstractmethod
    def _compute_raw_rewards(
        self, ep: int, epoch_data: List[Any], metrics: Dict[str, Any],
    ) -> RewardTerms:
        """Compute per-walker raw reward terms.

        Returns a RewardTerms dataclass with LOO-VAMP, exploration bonuses, etc.
        """
        ...

    @abstractmethod
    def _pack_update_data(
        self,
        ep: int,
        combined_rewards: torch.Tensor,
        starts: Any,
        start_meta: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Package data for the policy optimizer update.

        Returns the `start_data` dict expected by GRPOOptimizer.update().
        """
        ...

    # ── Shared epoch phases ─────────────────────────────────────────

    def _maybe_refresh_projector(self, ep: int):
        """Phase A: Refresh geom2vec VAMP projector if due."""
        # Subclasses override with actual refresh logic
        pass

    def _pre_sample_seed_stats(self, ep: int):
        """Phase A2: Seed LOO/VAMP/MSM stats from history before sampling."""
        pass

    def _run_trajectories(
        self, ep: int, starts: Any, start_meta: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Phase C: Run MD trajectories via sampler.

        Appends results to history. Returns (epoch_data, metrics).
        """
        # Default implementation delegates to self.sampler
        # Subclasses populate this with actual sampler calls
        return [], {}

    def _combine_and_update(
        self,
        ep: int,
        raw_terms: RewardTerms,
        starts: Any,
        start_meta: Dict[str, Any],
        metrics: Dict[str, Any],
    ):
        """Phase E: Combine rewards and run policy gradient update."""
        if self.reward_combiner is None:
            return

        launch_rewards = torch.zeros(
            self.config.grp_size, device=self.torch_device, dtype=torch.float32,
        )
        active_mask = metrics.get("active_mask")
        if active_mask is not None:
            active_mask = torch.as_tensor(active_mask, device=self.torch_device).bool()

        combined, reward_metrics = self.reward_combiner.combine(
            raw_terms, launch_rewards, active_mask,
        )
        metrics.update(reward_metrics)

        # Pack and update
        start_data = self._pack_update_data(ep, combined, starts, start_meta, metrics)
        if self.optimizer is not None and start_data:
            update_stats = self.optimizer.update(start_data)
            metrics.update(update_stats)

            # Sync old policy periodically
            sync = getattr(self.config, "sync_interval", 5)
            if sync > 0 and (ep + 1) % sync == 0 and self.agent_old is not None:
                self.agent_old.load_state_dict(self.agent.state_dict())

    def _log_epoch(self, ep: int, metrics: Dict[str, Any], epoch_start_time: float):
        """Phase F: Stats logging, checkpointing, plotting."""
        elapsed = time.time() - epoch_start_time
        print(f"Epoch {ep+1}/{self.config.episodes} done ({elapsed:.1f}s)", flush=True)
        gc.collect()

    # ── History management ──────────────────────────────────────────

    def _append_to_history(
        self,
        epoch_data: List[Any],
        traj_lengths: List[int],
        dcd_paths: Optional[List[Optional[str]]] = None,
        frame_indices: Optional[List[Optional[np.ndarray]]] = None,
        vamp_exclude: bool = False,
    ):
        """Append an epoch's data to the running history."""
        self.all_epoch_data.append(epoch_data)
        self.all_trajectory_lengths.append(traj_lengths)
        self.all_epoch_traj_dcd_paths.append(dcd_paths or [None] * len(epoch_data))
        self.all_epoch_traj_dcd_frame_indices.append(
            frame_indices or [None] * len(epoch_data),
        )
        self.all_epoch_vamp_exclude.append(vamp_exclude)
