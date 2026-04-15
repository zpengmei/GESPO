"""
Sampling Orchestrator - Manages trajectory sampling and rollout execution.

Handles:
- Walker initialization
- Parallel sampling execution
- Trajectory processing
- Reward computation
- History updates
"""

from __future__ import annotations

import time
import numpy as np
import torch
from typing import List, Optional, Tuple, Any, Dict

from ..sampler import Sampler

class SamplingOrchestrator:
    """
    Orchestrates the sampling phase of training.
    
    Wraps the low-level Sampler to provide a high-level interface for:
    - Executing epoch rollouts
    - Managing walker states
    - Computing rewards (using LOOVAMPRewards)
    - Collecting and processing trajectories
    """
    
    def __init__(
        self,
        config: Any,
        sampler: Sampler,
        loo_vamp_rewards: Any,
        device: str = "cuda",
    ):
        """
        Initialize SamplingOrchestrator.
        
        Args:
            config: Training configuration
            sampler: Initialized Sampler instance
            loo_vamp_rewards: LOOVAMPRewards component
            device: Torch device string
        """
        self.config = config
        self.sampler = sampler
        self.loo_vamp = loo_vamp_rewards
        self.device = device
        
        # State tracking
        self.total_steps = 0
        self.total_trajectories = 0
        
    def execute_epoch(
        self, 
        epoch: int,
        agent: Any,
        desc: str = ""
    ) -> Tuple[List[Any], Dict[str, float]]:
        """
        Execute sampling for a single epoch.
        
        Args:
            epoch: Current epoch index
            agent: Policy/Agent to use for sampling
            desc: Description for progress bars
            
        Returns:
            Tuple of (trajectories, metrics)
        """
        t0 = time.time()
        
        # Run sampling using the underlying Sampler
        # The Sampler handles the actual simulation distribution
        trajectories, metrics = self.sampler.sample_epoch(
            agent=agent,
            epoch=epoch,
            max_steps=self.config.steps_per_epoch,
            desc=desc
        )
        
        duration = time.time() - t0
        
        # Augment metrics
        metrics["sampling_time"] = duration
        if trajectories:
            metrics["fps"] = sum(len(t) for t in trajectories) / duration
            
        self.total_trajectories += len(trajectories)
        
        return trajectories, metrics
        
    def compute_rewards(
        self,
        trajectories: List[Any],
        geom2vec_projector: Any,
        retrieval_cache: Optional[Any] = None
    ) -> List[float]:
        """
        Compute rewards for improved trajectories using LOO-VAMP.
        
        Args:
            trajectories: List of raw trajectories
            geom2vec_projector: Projector for state assignment
            retrieval_cache: Optional retrieval cache for bonus rewards
            
        Returns:
            List of reward values, one per trajectory
        """
        if not trajectories:
            return []
            
        # Extract features/states if needed
        # This calls into LOOVAMPRewards to compute the actual scores
        return self.loo_vamp.compute_rewards(
            trajectories, 
            projector=geom2vec_projector
        )

    def update_history(self, trajectories: List[Any], rewards: List[float]):
        """
        Update replay buffer/history with new trajectories.
        
        Args:
            trajectories: New trajectories
            rewards: Computed rewards
        """
        # (Placeholder for history interaction - typically passed to dataset/loader)
        pass
