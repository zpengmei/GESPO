from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


FrameId = Tuple[int, int]  # (cluster_id, local_index)


@dataclass(frozen=True)
class FrameRef:
    """
    Pointer to a coordinate frame stored on disk.

    This avoids storing full explicit-solvent coordinates in RAM.
    """

    dcd_path: str
    frame_idx: int


class ClusterReservoirReplayBuffer:
    """
    Fixed-size replay buffer that subsamples frames using DeepTICA/VAMP clusters.

    Design:
    - Total capacity N_max is split across K clusters (~N_max/K each).
    - Within each cluster bucket, we use reservoir sampling so that items are
      approximately uniform over *all* frames ever observed in that cluster.
    """

    def __init__(
        self,
        n_clusters: int,
        capacity: int,
        emb_dim: int,
        seed: int = 0,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {emb_dim}")

        self.n_clusters = int(n_clusters)
        self.capacity = int(capacity)
        self.emb_dim = int(emb_dim)
        self.rng = np.random.default_rng(int(seed))

        base = self.capacity // self.n_clusters
        rem = self.capacity % self.n_clusters
        self._caps = np.array(
            [base + (1 if i < rem else 0) for i in range(self.n_clusters)],
            dtype=np.int64,
        )

        self._seen = np.zeros(self.n_clusters, dtype=np.int64)
        self._size = np.zeros(self.n_clusters, dtype=np.int64)

        self._emb: List[np.ndarray] = []
        self._ref: List[List[Optional[FrameRef]]] = []
        for k in range(self.n_clusters):
            cap_k = int(self._caps[k])
            self._emb.append(np.zeros((cap_k, self.emb_dim), dtype=np.float32))
            self._ref.append([None] * cap_k)

    def __len__(self) -> int:
        return int(self._size.sum())

    def cluster_sizes(self) -> np.ndarray:
        return self._size.copy()

    def add(
        self,
        cluster_id: int,
        emb: np.ndarray,
        ref: FrameRef,
    ) -> Optional[FrameId]:
        """
        Add a frame into the cluster bucket using reservoir sampling.

        Returns:
            FrameId if the item was inserted/replaced, else None.
        """
        k = int(cluster_id)
        if k < 0 or k >= self.n_clusters:
            raise ValueError(f"cluster_id out of range: {k} not in [0, {self.n_clusters})")

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.emb_dim:
            raise ValueError(f"emb_dim mismatch: expected {self.emb_dim}, got {emb.shape[0]}")

        self._seen[k] += 1
        cap_k = int(self._caps[k])
        size_k = int(self._size[k])

        if cap_k == 0:
            return None

        if size_k < cap_k:
            idx = size_k
            self._emb[k][idx] = emb
            self._ref[k][idx] = ref
            self._size[k] += 1
            return (k, idx)

        # Reservoir sampling replacement in cluster k
        seen_k = int(self._seen[k])
        j = int(self.rng.integers(0, seen_k))
        if j < cap_k:
            self._emb[k][j] = emb
            self._ref[k][j] = ref
            return (k, j)
        return None

    def get_ref(self, frame_id: FrameId) -> FrameRef:
        k, i = int(frame_id[0]), int(frame_id[1])
        if k < 0 or k >= self.n_clusters:
            raise KeyError(f"Invalid cluster_id: {k}")
        if i < 0 or i >= int(self._size[k]):
            raise KeyError(f"Invalid index {i} for cluster {k} (size={int(self._size[k])})")
        ref = self._ref[k][i]
        if ref is None:
            raise KeyError(f"Missing ref at {frame_id}")
        return ref

    def get_emb(self, frame_id: FrameId) -> np.ndarray:
        k, i = int(frame_id[0]), int(frame_id[1])
        if k < 0 or k >= self.n_clusters:
            raise KeyError(f"Invalid cluster_id: {k}")
        if i < 0 or i >= int(self._size[k]):
            raise KeyError(f"Invalid index {i} for cluster {k} (size={int(self._size[k])})")
        return self._emb[k][i]

    def sample_candidates(
        self,
        m: int,
        stratified: bool = True,
        replace_if_needed: bool = False,
        backfill: bool = True,
    ) -> Tuple[List[FrameId], np.ndarray]:
        """
        Sample a candidate pool and return (ids, embeddings).

        Args:
            m: total number of candidates requested.
            stratified: if True, attempt ~m/K per cluster then fill remainder.
            replace_if_needed: if True, sample with replacement when a cluster
                has fewer than its quota; otherwise take all available and fill
                remaining from the global pool.
        """
        m = int(m)
        if m <= 0:
            return [], np.zeros((0, self.emb_dim), dtype=np.float32)

        total = len(self)
        if total == 0:
            return [], np.zeros((0, self.emb_dim), dtype=np.float32)

        m = min(m, total) if not replace_if_needed else m

        ids: List[FrameId] = []
        if stratified:
            quota = max(1, m // self.n_clusters)
            for k in range(self.n_clusters):
                size_k = int(self._size[k])
                if size_k <= 0:
                    continue
                want = quota
                if replace_if_needed:
                    picks = self.rng.integers(0, size_k, size=want)
                else:
                    want = min(want, size_k)
                    picks = self.rng.choice(size_k, size=want, replace=False)
                ids.extend([(k, int(i)) for i in np.asarray(picks).reshape(-1)])

        # Fill remainder uniformly from the global pool unless disabled.
        if backfill and len(ids) < m:
            all_ids: List[FrameId] = []
            for k in range(self.n_clusters):
                size_k = int(self._size[k])
                all_ids.extend([(k, i) for i in range(size_k)])
            remaining = m - len(ids)
            if remaining > 0:
                picks = self.rng.choice(len(all_ids), size=remaining, replace=False)
                ids.extend([all_ids[int(i)] for i in np.asarray(picks).reshape(-1)])

        emb = np.stack([self.get_emb(fid) for fid in ids], axis=0).astype(np.float32)
        return ids, emb
