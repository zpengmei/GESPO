"""
Retrieval Manager - Manages torsion retrieval cache for TorsionAR policy.

This module extracts retrieval-related operations from loop_generative.py,
providing a clean interface for:
- Torsion φ/ψ cache management
- Candidate frame retrieval
- Cache rebalancing & rebuilding
- Label refreshing after projector updates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class RetrievalManager:
    """
    Manages retrieval cache for TorsionAR policy nearest-neighbor starts.
    
    Maintains a cache of historical torsion conformations with their coordinates,
    CV values, and labels for efficient candidate sampling during training.
    """
    
    def __init__(
        self,
        max_cache_size: int = 10000,
        global_dim: int = 0,
        seed: int = 0,
    ):
        """
        Initialize RetrievalManager.
        
        Args:
            max_cache_size: Maximum number of frames to cache
            global_dim: Dimension of global features
            seed: Random seed for sampling
        """
        self.max_cache_size = max_cache_size
        self.global_dim = global_dim
        
        # Cache arrays
        self.cv: List[np.ndarray] = []
        self.coords: List[np.ndarray] = []
        self.dcd_paths: List[Optional[str]] = []
        self.dcd_frame_idx: List[int] = []
        self.globals: List[np.ndarray] = []
        self.keys: List[str] = []
        self.frame_idx: List[int] = []
        self.labels: Optional[np.ndarray] = None
        
        # RNG for sampling
        self.rng = np.random.default_rng(seed)
    
    def __len__(self) -> int:
        """Return number of cached frames."""
        return len(self.cv)
    
    def append(
        self,
        new_cv: List[np.ndarray],
        new_coords: List[np.ndarray],
        new_keys: List[str],
        new_frame_idx: List[int],
        new_dcd_paths: List[Optional[str]],
        new_dcd_frame_idx: List[int],
        new_globals: Optional[List[np.ndarray]] = None,
        tag: str = "",
    ) -> int:
        """
        Append new frames to the retrieval cache.
        
        Args:
            new_cv: List of CV arrays
            new_coords: List of coordinate arrays
            new_keys: List of cache keys
            new_frame_idx: List of frame indices
            new_dcd_paths: List of DCD paths
            new_dcd_frame_idx: List of DCD frame indices
            new_globals: Optional list of global feature arrays
            tag: Description for logging
            
        Returns:
            Number of frames appended
        """
        if not new_cv:
            return 0
        
        old_len = len(self.cv)
        
        self.cv.extend(new_cv)
        self.coords.extend(new_coords)
        
        # Handle globals
        if new_globals is not None and len(new_globals) > 0:
            self.globals.extend(list(new_globals))
        else:
            # Pad with zeros if no globals provided
            self.globals.extend([
                np.zeros((max(0, self.global_dim),), dtype=np.float32)
                for _ in range(len(new_coords))
            ])
        
        self.keys.extend(new_keys)
        self.frame_idx.extend(new_frame_idx)
        self.dcd_paths.extend(new_dcd_paths)
        self.dcd_frame_idx.extend(new_dcd_frame_idx)
        
        # Update labels array
        if len(new_cv) > 0:
            if self.labels is not None and self.labels.shape[0] == old_len:
                pad = np.zeros((len(new_cv),), dtype=np.int64)
                self.labels = np.concatenate([self.labels, pad], axis=0)
            else:
                self.labels = None
        
        if tag:
            print(f"[Retrieval Cache] Appended {len(new_cv)} frames ({tag}); cache_frames={len(self.cv)}")
        
        return len(new_cv)
    
    def apply_keep_indices(self, keep_idx: np.ndarray, tag: str = "") -> None:
        """
        Keep only specified indices in the cache.
        
        Args:
            keep_idx: Indices to keep
            tag: Description for logging
        """
        old_len = len(self.cv)
        
        self.cv = [self.cv[i] for i in keep_idx if i < len(self.cv)]
        self.coords = [self.coords[i] for i in keep_idx if i < len(self.coords)]
        self.dcd_paths = [self.dcd_paths[i] for i in keep_idx if i < len(self.dcd_paths)]
        self.dcd_frame_idx = [self.dcd_frame_idx[i] for i in keep_idx if i < len(self.dcd_frame_idx)]
        self.globals = [self.globals[i] for i in keep_idx if i < len(self.globals)]
        self.keys = [self.keys[i] for i in keep_idx if i < len(self.keys)]
        self.frame_idx = [self.frame_idx[i] for i in keep_idx if i < len(self.frame_idx)]
        
        if self.labels is not None and old_len == self.labels.shape[0]:
            self.labels = self.labels[keep_idx]
        else:
            self.labels = None
        
        if tag:
            print(
                f"[Retrieval Cache] Applied keep_indices ({tag}): "
                f"kept={len(keep_idx)}/{old_len}, cache_frames={len(self.cv)}"
            )
    
    def rebalance(self, tag: str = "") -> None:
        """
        Rebalance cache to keep it under max size, preserving diversity.
        
        Args:
            tag: Description for logging
        """
        n = len(self.cv)
        cap = self.max_cache_size
        
        if n <= cap:
            return
        
        # Group frames by source
        source_buckets: Dict[str, List[int]] = {}
        for idx, key in enumerate(self.keys):
            bucket = self._key_source_bucket(key)
            source_buckets.setdefault(bucket, []).append(idx)
        
        # Sample uniformly from each bucket
        keep_indices = []
        buckets = list(source_buckets.keys())
        per_bucket = max(1, cap // max(1, len(buckets)))
        
        for bucket in buckets:
            bucket_indices = source_buckets[bucket]
            if len(bucket_indices) <= per_bucket:
                keep_indices.extend(bucket_indices)
            else:
                sampled = self.rng.choice(bucket_indices, size=per_bucket, replace=False)
                keep_indices.extend(sampled)
        
        # If we're still over cap, sample uniformly
        if len(keep_indices) > cap:
            keep_indices = list(self.rng.choice(keep_indices, size=cap, replace=False))
        
        keep_indices.sort()
        keep_arr = np.array(keep_indices, dtype=np.int64)
        
        self.apply_keep_indices(keep_arr, tag=f"rebalance after {tag}")
    
    def clear(self, tag: str = "") -> None:
        """Clear all cached data."""
        self.cv = []
        self.coords = []
        self.dcd_paths = []
        self.dcd_frame_idx = []
        self.globals = []
        self.keys = []
        self.frame_idx = []
        self.labels = None
        
        if tag:
            print(f"[Retrieval Cache] Cleared cache ({tag})")
    
    def update_labels(
        self,
        new_labels: np.ndarray,
        only_indices: Optional[np.ndarray] = None,
        tag: str = "",
    ) -> None:
        """
        Update state labels for cached frames.
        
        Args:
            new_labels: New label array
            only_indices: Optional indices to update
            tag: Description for logging
        """
        if only_indices is not None:
            if self.labels is None:
                self.labels = np.zeros((len(self.cv),), dtype=np.int64)
            self.labels[only_indices] = new_labels
        else:
            self.labels = new_labels.copy()
        
        if tag:
            print(f"[Retrieval Cache] Updated labels ({tag})")
    
    def _key_source_bucket(self, key: str) -> str:
        """Extract source bucket from cache key."""
        parts = key.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0] if parts else "unknown"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "n_frames": len(self.cv),
            "max_size": self.max_cache_size,
            "has_labels": self.labels is not None,
            "global_dim": self.global_dim,
        }
