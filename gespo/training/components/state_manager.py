"""
State Manager - Manages state permutations and alignment for VAMP projectors.

This module extracts state tracking logic from training loops, providing:
- State permutation maps (for stable state indexing across refreshes)
- Previous state centers tracking
- State alignment utilities
"""

from __future__ import annotations

from typing import Optional
import numpy as np


class StateManager:
    """
    Manages state permutations and alignment for VAMP-based projectors.
    
    Tracks state permutation maps to maintain stable state indices across
    projector refreshes, and stores previous state centers for alignment.
    """
    
    def __init__(self, n_states: int):
        """
        Initialize StateManager.
        
        Args:
            n_states: Number of discrete states
        """
        self.n_states = n_states
        
        # Identity permutation initially
        self.permutation_map = np.arange(n_states, dtype=int)
        self.inverse_permutation_map = np.arange(n_states, dtype=int)
        
        # Previous state centers (for alignment)
        self.prev_centers: Optional[np.ndarray] = None
    
    def update_permutation(self, new_permutation: np.ndarray) -> None:
        """
        Update state permutation map.
        
        Args:
            new_permutation: New permutation array [n_states]
        """
        self.permutation_map = np.asarray(new_permutation, dtype=int)
        
        # Compute inverse permutation
        inv = np.zeros_like(self.permutation_map)
        inv[self.permutation_map] = np.arange(len(self.permutation_map))
        self.inverse_permutation_map = inv
    
    def reset_permutation(self) -> None:
        """Reset to identity permutation."""
        self.permutation_map = np.arange(self.n_states, dtype=int)
        self.inverse_permutation_map = np.arange(self.n_states, dtype=int)
    
    def update_prev_centers(self, centers: Optional[np.ndarray]) -> None:
        """
        Update previous state centers.
        
        Args:
            centers: State centers array or None
        """
        if centers is not None:
            self.prev_centers = np.asarray(centers, dtype=np.float32).copy()
        else:
            self.prev_centers = None
    
    def get_permuted_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Apply permutation to state labels.
        
        Args:
            labels: Original state labels
            
        Returns:
            Permuted state labels
        """
        return self.permutation_map[labels]
    
    def get_inverse_permuted_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Apply inverse permutation to state labels.
        
        Args:
            labels: Permuted state labels
            
        Returns:
            Original state labels
        """
        return self.inverse_permutation_map[labels]
    
    def get_stats(self) -> dict:
        """Get state manager statistics."""
        return {
            "n_states": self.n_states,
            "has_prev_centers": self.prev_centers is not None,
            "is_identity": np.array_equal(
                self.permutation_map,
                np.arange(self.n_states, dtype=int)
            ),
        }
