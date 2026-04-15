"""
Epoch Coordinator - Orchestrates the main training epoch loop.

Manages:
- Epoch iteration and timing
- Refresh triggers (Geom2Vec, Retrieval)
- Checkpoint saving logic
- History bootstrapping
"""

from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..config import TrainingConfig
    from .checkpointing import CheckpointManager

@dataclass
class EpochState:
    """State of the current epoch."""
    epoch_idx: int
    step: int
    start_time: float
    episodes: int
    walkers: int

class EpochCoordinator:
    """
    Coordinates the main training loop execution.
    
    Handles the high-level logic for:
    - Determining when to refresh projectors (Geom2Vec)
    - Determining when to save checkpoints
    - Managing epoch timing and progress reporting
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        checkpoint_manager: CheckpointManager,
        geom2vec_manager: Any,  # Typed as Any to avoid circular imports
        retrieval_mgr: Any = None,
    ):
        """
        Initialize EpochCoordinator.
        
        Args:
            config: Training configuration
            checkpoint_manager: Checkpoint manager instance
            geom2vec_manager: Geom2Vec manager instance
            retrieval_mgr: Optional Retrieval manager
        """
        self.config = config
        self.ckpt_manager = checkpoint_manager
        self.geom2vec_mgr = geom2vec_manager
        self.retrieval_mgr = retrieval_mgr
        
        self.start_epoch = 0
        self.n_epochs = config.episodes
        
        # Refresh configuration
        self.geom2vec_refresh_rate = getattr(config, "geom2vec_refresh_rate", None) or getattr(config, "tica_refresh_rate", 10)
        self.retrieval_refresh_rate = getattr(config, "retrieval_refresh_rate", 5)
        
    def should_refresh_geom2vec(self, epoch: int) -> bool:
        """Check if Geom2Vec should be refreshed at this epoch."""
        if epoch == 0:
            return False  # Usually initialized before loop
        return (epoch % self.geom2vec_refresh_rate) == 0

    def should_refresh_retrieval(self, epoch: int) -> bool:
        """Check if Retrieval Cache should be refreshed/rebalanced."""
        if not self.retrieval_mgr:
            return False
        return (epoch % self.retrieval_refresh_rate) == 0
        
    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved."""
        freq = self.config.ckpt_interval
        return freq > 0 and (epoch % freq) == 0

    def save_checkpoint(self, epoch: int, step: int, agent: Any, **kwargs) -> Optional[str]:
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch index
            step: Current step index
            agent: The agent/policy to save
            **kwargs: Additional state to save (e.g. optimizers)
            
        Returns:
            Path to saved checkpoint or None
        """
        state = {
            "epoch": epoch,
            "step": step,
            "agent_state": agent.state_dict(),
            "config": self.config.to_dict(),
        }
        state.update(kwargs)
        
        return self.ckpt_manager.save(state, epoch, is_best=False)
