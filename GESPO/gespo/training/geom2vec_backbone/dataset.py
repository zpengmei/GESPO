from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


TrajectoryEntry = Union[torch.Tensor, dict, Tuple[torch.Tensor, Optional[torch.Tensor]]]


@dataclass(frozen=True)
class PairIndex:
    traj_idx: int
    t: int
    t_lag: int


class TimeLaggedIndexDataset(Dataset):
    """Index-based time-lagged dataset without pair materialization.

    This mirrors the intent of `geom2vec`'s `_TimeLaggedDataset` but avoids
    `index_select` + `torch.cat` across all pairs.

    Each item is returned as `(inst, lag)` where `inst`/`lag` are dicts:
      - `graph_features`: frame tensor
      - `ca_coords`: optional
      - `index`: optional `PairIndex` (when `return_index=True`)
    """

    def __init__(
        self,
        trajectories: Sequence[TrajectoryEntry],
        *,
        lag_time: int,
        pair_stride: int = 1,
        return_index: bool = False,
    ) -> None:
        if lag_time <= 0:
            raise ValueError("lag_time must be > 0")
        if pair_stride <= 0:
            raise ValueError("pair_stride must be > 0")

        self._lag = int(lag_time)
        self._stride = int(pair_stride)
        self._return_index = bool(return_index)

        graphs: List[torch.Tensor] = []
        coords: List[Optional[torch.Tensor]] = []
        pair_counts: List[int] = []
        feature_shape: Optional[torch.Size] = None

        for i, traj in enumerate(trajectories):
            graph, ca = self._normalize_trajectory(traj)
            if graph.ndim < 2:
                raise ValueError(f"Trajectory {i} must have shape (frames, ...).")
            frames = int(graph.shape[0])
            if frames <= self._lag:
                continue

            if feature_shape is None:
                feature_shape = graph.shape[1:]
            elif graph.shape[1:] != feature_shape:
                raise ValueError("All trajectories must share the same feature shape.")

            graphs.append(graph)
            coords.append(ca)

            n_pairs = (frames - self._lag + self._stride - 1) // self._stride
            pair_counts.append(int(n_pairs))

        if not graphs:
            raise ValueError("At least one trajectory with frames > lag_time is required.")

        self._graphs = graphs
        self._coords = coords

        # Build prefix sums so we can map a global index -> (traj_idx, local_pair_idx)
        self._pair_counts = np.asarray(pair_counts, dtype=np.int64)
        self._prefix = np.cumsum(np.concatenate([[0], self._pair_counts]))

    @staticmethod
    def _normalize_trajectory(traj: TrajectoryEntry) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(traj, dict):
            if "graph_features" not in traj:
                raise ValueError("Trajectory dict must contain 'graph_features'.")
            graph = traj["graph_features"]
            ca = traj.get("ca_coords")
        elif isinstance(traj, (tuple, list)):
            if len(traj) != 2:
                raise ValueError("Tuple/list trajectories must be (graph_features, ca_coords).")
            graph, ca = traj
        else:
            graph = traj
            ca = None

        graph_t = graph if isinstance(graph, torch.Tensor) else torch.as_tensor(graph)
        ca_t: Optional[torch.Tensor]
        if ca is None:
            ca_t = None
        else:
            ca_t = ca if isinstance(ca, torch.Tensor) else torch.as_tensor(ca)
            if ca_t.ndim != 3:
                raise ValueError("ca_coords must have shape (frames, tokens, 3).")
            if ca_t.shape[0] != graph_t.shape[0]:
                raise ValueError("ca_coords must have the same number of frames as graph_features.")
        return graph_t, ca_t

    def __len__(self) -> int:  # type: ignore[override]
        return int(self._prefix[-1])

    def _pair_index(self, index: int) -> PairIndex:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        traj_idx = int(np.searchsorted(self._prefix[1:], index, side="right"))
        local_pair = int(index - self._prefix[traj_idx])
        t = int(local_pair * self._stride)
        return PairIndex(traj_idx=traj_idx, t=t, t_lag=t + self._lag)

    def __getitem__(self, index: int):  # type: ignore[override]
        pi = self._pair_index(int(index))
        graph = self._graphs[pi.traj_idx]
        inst: dict = {"graph_features": graph[pi.t]}
        lag: dict = {"graph_features": graph[pi.t_lag]}

        ca = self._coords[pi.traj_idx]
        if ca is not None:
            inst["ca_coords"] = ca[pi.t]
            lag["ca_coords"] = ca[pi.t_lag]

        if self._return_index:
            inst["index"] = pi
            lag["index"] = pi

        return inst, lag

