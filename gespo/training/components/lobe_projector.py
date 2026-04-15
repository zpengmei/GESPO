"""Minimal projector wrapper around a trained VAMP/VAC lobe for cached embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..geom2vec_backbone import EmbeddingCache


def state_probs_from_output_np(output: np.ndarray) -> np.ndarray:
    """Convert raw lobe outputs to state probabilities (softmax if needed)."""
    out = np.asarray(output, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError("Expected (T, K) outputs.")
    if np.all(out >= 0) and np.all(out <= 1.0):
        row_sums = out.sum(axis=1)
        if np.allclose(row_sums, np.ones_like(row_sums), atol=1e-3):
            return out
    m = np.max(out, axis=1, keepdims=True)
    ex = np.exp(out - m)
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return (ex / denom).astype(np.float32, copy=False)


@dataclass
class TrainedLobe:
    lobe: torch.nn.Module
    meta: Dict[str, Any]


@dataclass(frozen=True)
class VampPairRecord:
    x: np.ndarray
    y: np.ndarray
    ind_stop: Optional[np.ndarray] = None


class LobeProjector:
    """Minimal projector wrapper around a trained lobe (for cached embeddings)."""

    def __init__(
        self,
        *,
        cache: EmbeddingCache,
        device: torch.device,
    ) -> None:
        self.cache = cache
        self.device = torch.device(device)
        self.trained: Optional[TrainedLobe] = None
        self.last_fit_outcomes: Optional[List[Dict[str, Any]]] = None

    def is_trained(self) -> bool:
        return self.trained is not None and isinstance(self.trained.lobe, torch.nn.Module)

    def set_trained(self, lobe: torch.nn.Module, *, meta: Optional[Dict[str, Any]] = None) -> None:
        self.trained = TrainedLobe(lobe=lobe, meta=dict(meta or {}))

    def _require_lobe(self) -> torch.nn.Module:
        if not self.is_trained():
            raise RuntimeError("Projector is not trained yet.")
        assert self.trained is not None
        return self.trained.lobe

    @torch.no_grad()
    def state_output_from_embeddings(self, graph_features: np.ndarray, ca_coords: np.ndarray) -> np.ndarray:
        lobe = self._require_lobe().to(self.device).eval()
        graph = torch.as_tensor(graph_features, dtype=torch.float32, device=self.device)
        ca = torch.as_tensor(ca_coords, dtype=torch.float32, device=self.device)
        out = lobe({"graph_features": graph, "ca_coords": ca})
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def state_output_from_cache_key(self, key: str) -> np.ndarray:
        graph, ca = self.cache.load(key)
        return self.state_output_from_embeddings(graph, ca)

    @torch.no_grad()
    def state_probs_from_embeddings(self, graph_features: np.ndarray, ca_coords: np.ndarray) -> np.ndarray:
        out = self.state_output_from_embeddings(graph_features, ca_coords)
        return state_probs_from_output_np(out)

    @torch.no_grad()
    def state_probs_from_cache_key(self, key: str) -> np.ndarray:
        graph, ca = self.cache.load(key)
        return self.state_probs_from_embeddings(graph, ca)

    def assign_states(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError("probs must have shape (T, K)")
        return np.argmax(probs, axis=1).astype(np.int64, copy=False)
