"""Committor state container for path-sampling loop.

Holds all mutable state related to committor estimation, tube/edge/connectivity
bookkeeping, and boundary models. Replaces ~30 nonlocal variables.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class CommittorState:
    """Mutable state container for committor-related path-sampling bookkeeping."""

    def __init__(self) -> None:
        # Committor model state.
        self.dga_model: Optional[Any] = None
        self.vamp_model: Optional[Any] = None
        self.norm_ema_m1: Optional[float] = None
        self.norm_ema_m2: Optional[float] = None

        # Tube-diversity stats.
        self.tube_counts: Optional[np.ndarray] = None
        self.tube_total_weight: float = 0.0
        self.tube_counted_keys: Set[str] = set()
        self.tube_signature: Optional[Tuple] = None
        self.tube_norm_ema_m1: Optional[float] = None
        self.tube_norm_ema_m2: Optional[float] = None

        # Edge-novelty stats.
        self.edge_counts: Optional[Any] = None
        self.edge_total_count: float = 0.0
        self.edge_counted_keys: Set[str] = set()
        self.edge_signature: Optional[Tuple] = None
        self.edge_norm_ema_m1: Optional[float] = None
        self.edge_norm_ema_m2: Optional[float] = None

        # Connectivity graph state.
        self.conn_adj: Optional[List[Set[int]]] = None
        self.conn_edges: Set[int] = set()
        self.conn_a_cells: Set[int] = set()
        self.conn_b_cells: Set[int] = set()
        self.conn_counted_keys: Set[str] = set()
        self.conn_signature: Optional[Tuple] = None
        self.conn_norm_ema_m1: Optional[float] = None
        self.conn_norm_ema_m2: Optional[float] = None

        # Bin-entropy normalization.
        self.bin_entropy_norm_ema_m1: Optional[float] = None
        self.bin_entropy_norm_ema_m2: Optional[float] = None

        # Start-basin counts.
        self.start_basin_counts_a: int = 0
        self.start_basin_counts_b: int = 0

        # Boundary models (VAC k-means).
        self.boundary_vac_kmeans: Optional[Any] = None
        self.boundary_vac_mu: Optional[np.ndarray] = None
        self.boundary_vac_std: Optional[np.ndarray] = None
        self.boundary_vac_a_clusters: Optional[np.ndarray] = None
        self.boundary_vac_b_clusters: Optional[np.ndarray] = None
        self.boundary_vac_signature: Optional[Tuple] = None

        # VAMP state-based boundary.
        self.boundary_vamp_a_state: Optional[int] = None
        self.boundary_vamp_b_state: Optional[int] = None
        self.boundary_vamp_state_valid: bool = False

    def invalidate_model(self, tag: str) -> None:
        """Invalidate committor model + normalization."""
        self.dga_model = None
        self.vamp_model = None
        self.norm_ema_m1 = None
        self.norm_ema_m2 = None
        self.invalidate_tube_stats(tag=f"committor invalidated: {tag}")
        print(f"[Committor] Invalidated committor model/EMA ({tag}).")

    def invalidate_tube_stats(self, tag: str) -> None:
        """Invalidate tube-diversity, edge-novelty, and connectivity stats."""
        self.tube_counts = None
        self.tube_total_weight = 0.0
        self.tube_counted_keys = set()
        self.tube_signature = None
        self.tube_norm_ema_m1 = None
        self.tube_norm_ema_m2 = None
        self.edge_counts = None
        self.edge_total_count = 0.0
        self.edge_counted_keys = set()
        self.edge_signature = None
        self.edge_norm_ema_m1 = None
        self.edge_norm_ema_m2 = None
        self.conn_adj = None
        self.conn_edges = set()
        self.conn_a_cells = set()
        self.conn_b_cells = set()
        self.conn_counted_keys = set()
        self.conn_signature = None
        self.conn_norm_ema_m1 = None
        self.conn_norm_ema_m2 = None
        print(f"[Committor Tube] Invalidated tube stats ({tag}).")

    def invalidate_boundary_vac(self, tag: str) -> None:
        """Invalidate VAC-kmeans boundary model."""
        self.boundary_vac_kmeans = None
        self.boundary_vac_mu = None
        self.boundary_vac_std = None
        self.boundary_vac_a_clusters = None
        self.boundary_vac_b_clusters = None
        self.boundary_vac_signature = None
        print(f"[Path Boundaries] Invalidated VAC-kmeans boundary model ({tag}).")
