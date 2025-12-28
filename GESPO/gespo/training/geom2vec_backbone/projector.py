from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .cache import EmbeddingCache
from .inference import infer_geom2vec_embeddings
from .topology import Geom2VecTopologyCache
from .vamp_trainer import TrainedVAMPLobe, train_vamp_lobe_from_cache


PathLike = Union[str, Path]


def _state_probs_from_output(output: torch.Tensor) -> torch.Tensor:
    """Convert arbitrary model outputs to probabilities."""
    if output.ndim != 2:
        raise ValueError("Expected (batch, K) output.")
    if torch.all(output >= 0) and torch.all(output <= 1.0):
        row_sums = output.sum(dim=1)
        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            return output
    return torch.softmax(output, dim=1)


@dataclass
class Geom2VecVAMPProjector:
    """High-level projector adapter: coords -> geom2vec embeddings -> VAMP lobe outputs."""

    rep_model: Any
    topology: Geom2VecTopologyCache
    cache: EmbeddingCache
    device: torch.device = torch.device("cpu")
    hidden_channels: Optional[int] = None

    trained: Optional[TrainedVAMPLobe] = None
    trained_ensemble: Optional[List[TrainedVAMPLobe]] = None
    best_model_idx: int = 0
    last_fit_outcomes: Optional[List[Dict[str, Any]]] = None

    def fit_from_cache(
        self,
        *,
        keys: Sequence[str],
        n_states: int,
        lag_time: int,
        seed: Optional[int] = None,
        pair_stride: int = 1,
        train_fraction: float = 0.8,
        shuffle_trajectories: bool = True,
        train_shuffle_pairs: bool = True,
        batch_size: int = 256,
        n_epochs: int = 10,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        optimizer: str = "Adam",
        epsilon: float = 1e-6,
        inverse_mode: str = "regularize",
        softmax_output: bool = True,
        lobe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TrainedVAMPLobe:
        trained = train_vamp_lobe_from_cache(
            self.cache,
            keys=keys,
            n_states=n_states,
            lag_time=lag_time,
            pair_stride=pair_stride,
            train_fraction=train_fraction,
            shuffle_trajectories=shuffle_trajectories,
            train_shuffle_pairs=train_shuffle_pairs,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            device=self.device,
            seed=None if seed is None else int(seed),
            softmax_output=softmax_output,
            epsilon=epsilon,
            inverse_mode=inverse_mode,
            lobe_kwargs=lobe_kwargs,
        )
        self.trained = trained
        self.trained_ensemble = [trained]
        self.best_model_idx = 0
        self.last_fit_outcomes = [
            {
                "model_idx": 0,
                "val_score": trained.meta.get("best_val_score"),
                "train_score": trained.meta.get("best_train_score"),
                "epsilon": trained.meta.get("epsilon"),
                "inverse_mode": trained.meta.get("inverse_mode"),
                "softmax_output": trained.meta.get("softmax_output"),
                "seed": trained.meta.get("seed"),
            }
        ]
        return trained

    def fit_ensemble_from_cache(
        self,
        *,
        keys: Sequence[str],
        n_states: int,
        lag_time: int,
        n_models: int = 3,
        seed_base: int = 42,
        pair_stride: int = 1,
        train_fraction: float = 0.8,
        shuffle_trajectories: bool = True,
        train_shuffle_pairs: bool = True,
        batch_size: int = 256,
        n_epochs: int = 10,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        optimizer: str = "Adam",
        epsilon: float = 1e-6,
        inverse_mode: str = "regularize",
        softmax_output: bool = True,
        lobe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[TrainedVAMPLobe]:
        n_models_i = max(1, int(n_models))
        seed_base_i = int(seed_base)

        trained_list: List[TrainedVAMPLobe] = []
        outcomes: List[Dict[str, Any]] = []
        for i in range(n_models_i):
            seed_i = seed_base_i + i * 1000
            try:
                trained = train_vamp_lobe_from_cache(
                    self.cache,
                    keys=keys,
                    n_states=n_states,
                    lag_time=lag_time,
                    pair_stride=pair_stride,
                    train_fraction=train_fraction,
                    shuffle_trajectories=shuffle_trajectories,
                    train_shuffle_pairs=train_shuffle_pairs,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    optimizer=optimizer,
                    device=self.device,
                    seed=seed_i,
                    epsilon=epsilon,
                    inverse_mode=inverse_mode,
                    softmax_output=softmax_output,
                    lobe_kwargs=lobe_kwargs,
                )
                trained_list.append(trained)
                outcomes.append(
                    {
                        "model_idx": int(i),
                        "ok": True,
                        "val_score": trained.meta.get("best_val_score"),
                        "train_score": trained.meta.get("best_train_score"),
                        "epsilon": trained.meta.get("epsilon"),
                        "inverse_mode": trained.meta.get("inverse_mode"),
                        "seed": trained.meta.get("seed"),
                    }
                )
            except Exception as exc:
                outcomes.append(
                    {
                        "model_idx": int(i),
                        "ok": False,
                        "error": str(exc),
                        "seed": int(seed_i),
                        "epsilon": float(epsilon),
                        "inverse_mode": str(inverse_mode),
                    }
                )
                continue

        # Pick best by highest validation score if available.
        best_idx = 0
        try:
            ok_outcomes = [(i, rec) for i, rec in enumerate(outcomes) if rec.get("ok")]
            scored_v = [(i, rec.get("val_score")) for i, rec in ok_outcomes if rec.get("val_score") is not None]
            if scored_v:
                best_idx = int(max(scored_v, key=lambda t: float(t[1]))[0])
            else:
                scored_t = [(i, rec.get("train_score")) for i, rec in ok_outcomes if rec.get("train_score") is not None]
                if scored_t:
                    best_idx = int(max(scored_t, key=lambda t: float(t[1]))[0])
        except Exception:
            best_idx = 0

        if not trained_list:
            self.trained_ensemble = None
            self.trained = None
            self.best_model_idx = 0
            self.last_fit_outcomes = outcomes
            raise RuntimeError("All geom2vec VAMP ensemble fits failed; see fit_outcomes for errors.")

        self.trained_ensemble = trained_list
        # best_idx indexes outcomes; map to successful-model index
        ok_indices = [i for i, rec in enumerate(outcomes) if rec.get("ok")]
        best_ok_pos = 0
        if ok_indices:
            try:
                best_ok_pos = ok_indices.index(best_idx)
            except ValueError:
                best_ok_pos = 0
        self.best_model_idx = int(best_ok_pos)
        self.trained = trained_list[self.best_model_idx]
        self.last_fit_outcomes = outcomes
        return trained_list

    def _require_lobe(self, member_idx: Optional[int] = None) -> torch.nn.Module:
        trained = self.trained
        if member_idx is not None:
            if self.trained_ensemble is None:
                raise RuntimeError("Projector ensemble is not trained yet. Call fit_ensemble_from_cache(...) first.")
            idx = int(member_idx)
            if idx < 0 or idx >= len(self.trained_ensemble):
                raise IndexError(f"member_idx out of range: {idx}")
            trained = self.trained_ensemble[idx]
        if trained is None:
            raise RuntimeError("Projector is not trained yet. Call fit_from_cache(...) first.")
        lobe = trained.lobe
        if isinstance(lobe, (tuple, list)):
            lobe = lobe[0]
        if not isinstance(lobe, torch.nn.Module):
            raise TypeError("Unexpected lobe type.")
        return lobe

    @torch.no_grad()
    def state_output_from_embeddings(
        self,
        graph_features: torch.Tensor,
        ca_coords: torch.Tensor,
        *,
        member_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return raw VAMP lobe outputs (no softmax conversion)."""
        # Default: use best model when ensemble exists.
        if member_idx is None and self.trained_ensemble is not None:
            member_idx = int(self.best_model_idx)
        lobe = self._require_lobe(member_idx=member_idx).to(self.device).eval()
        graph = graph_features.to(device=self.device, dtype=torch.float32)
        ca = ca_coords.to(device=self.device, dtype=torch.float32)
        out = lobe({"graph_features": graph, "ca_coords": ca})
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out)
        return out.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def state_output_from_cache_key(
        self, key: Union[str, Path], *, member_idx: Optional[int] = None
    ) -> np.ndarray:
        graph, ca = self.cache.load(key)
        return self.state_output_from_embeddings(graph, ca, member_idx=member_idx)

    @torch.no_grad()
    def state_output_from_coords(
        self,
        coords_nm: np.ndarray,
        *,
        embed_stride: int = 1,
        embed_batch_size: int = 32,
        reduction: str = "sum",
        center_ca: bool = True,
        member_idx: Optional[int] = None,
    ) -> np.ndarray:
        graph, ca = infer_geom2vec_embeddings(
            model=self.rep_model,
            topology=self.topology,
            coords=coords_nm,
            coords_in_nm=True,
            stride=embed_stride,
            batch_size=embed_batch_size,
            reduction=reduction,
            hidden_channels=self.hidden_channels,
            device=self.device,
            center_ca=center_ca,
        )
        return self.state_output_from_embeddings(graph, ca, member_idx=member_idx)

    @torch.no_grad()
    def state_probs_from_embeddings(
        self,
        graph_features: torch.Tensor,
        ca_coords: torch.Tensor,
        *,
        member_idx: Optional[int] = None,
    ) -> np.ndarray:
        # Default: use best model when ensemble exists.
        if member_idx is None and self.trained_ensemble is not None:
            member_idx = int(self.best_model_idx)
        lobe = self._require_lobe(member_idx=member_idx).to(self.device).eval()
        graph = graph_features.to(device=self.device, dtype=torch.float32)
        ca = ca_coords.to(device=self.device, dtype=torch.float32)
        out = lobe({"graph_features": graph, "ca_coords": ca})
        probs = _state_probs_from_output(out).detach().cpu().numpy().astype(np.float32)
        return probs

    @torch.no_grad()
    def state_probs_from_cache_key(self, key: Union[str, Path], *, member_idx: Optional[int] = None) -> np.ndarray:
        graph, ca = self.cache.load(key)
        return self.state_probs_from_embeddings(graph, ca, member_idx=member_idx)

    @torch.no_grad()
    def state_probs_from_coords(
        self,
        coords_nm: np.ndarray,
        *,
        embed_stride: int = 1,
        embed_batch_size: int = 32,
        reduction: str = "sum",
        center_ca: bool = True,
    ) -> np.ndarray:
        graph, ca = infer_geom2vec_embeddings(
            model=self.rep_model,
            topology=self.topology,
            coords=coords_nm,
            coords_in_nm=True,
            stride=embed_stride,
            batch_size=embed_batch_size,
            reduction=reduction,
            hidden_channels=self.hidden_channels,
            device=self.device,
            center_ca=center_ca,
        )
        return self.state_probs_from_embeddings(graph, ca)

    def assign_states(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError("probs must have shape (N, K)")
        return np.argmax(probs, axis=1).astype(np.int64)
