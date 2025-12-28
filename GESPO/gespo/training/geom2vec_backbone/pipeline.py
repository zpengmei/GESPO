from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .cache import EmbeddingCache, EmbeddingRecord
from .inference import infer_geom2vec_embeddings
from .topology import Geom2VecTopologyCache


def infer_and_cache(
    cache: EmbeddingCache,
    *,
    key: Union[str, Path],
    model,
    topology: Geom2VecTopologyCache,
    coords: np.ndarray,
    coords_in_nm: bool = True,
    stride: int = 1,
    batch_size: int = 32,
    reduction: str = "sum",
    hidden_channels: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    center_ca: bool = True,
    overwrite: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> EmbeddingRecord:
    """Compute embeddings for a trajectory and persist them to the float32 cache."""

    if not overwrite:
        try:
            existing = cache.get(key)
            if Path(existing.embedding_path).exists() and Path(existing.ca_path).exists():
                return existing
        except KeyError:
            pass

    graph_cpu, ca_cpu = infer_geom2vec_embeddings(
        model=model,
        topology=topology,
        coords=coords,
        coords_in_nm=coords_in_nm,
        stride=stride,
        batch_size=batch_size,
        reduction=reduction,
        hidden_channels=hidden_channels,
        device=device,
        center_ca=center_ca,
    )

    meta_out = dict(meta or {})
    meta_out.setdefault("coords_in_nm", bool(coords_in_nm))
    meta_out.setdefault("stride", int(stride))
    meta_out.setdefault("batch_size", int(batch_size))
    meta_out.setdefault("reduction", str(reduction))
    meta_out.setdefault("center_ca", bool(center_ca))

    return cache.store(
        key,
        graph_features=graph_cpu,
        ca_coords=ca_cpu,
        meta=meta_out,
        overwrite=overwrite,
    )

