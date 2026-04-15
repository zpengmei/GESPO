from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import EmbeddingCache
from .dataset import TimeLaggedIndexDataset


def _ensure_geom2vec_importable() -> None:
    """Ensure `geom2vec` is importable, trying repo-local path as a fallback."""
    try:
        from geom2vec.models.downstream.vamp.vampnet import VAMPNet  # noqa: F401

        return
    except Exception:
        pass

    import importlib
    import glob
    import os
    import sys

    def _try_src(candidate: Path) -> bool:
        if not (candidate / "geom2vec").exists():
            return False
        sys.path.insert(0, str(candidate))
        # If a namespace package shadowed the real geom2vec, purge it before retrying.
        for name in list(sys.modules.keys()):
            if name == "geom2vec" or name.startswith("geom2vec."):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()
        try:
            from geom2vec.models.downstream.vamp.vampnet import VAMPNet  # noqa: F401
        except Exception:
            return False
        return True

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "apps" / "geom2vec" / "src"
        if _try_src(candidate):
            return

    # Non-repo fallback: common Dinner-group shared checkout locations.
    extra_candidates: List[Path] = []
    env_src = str(os.environ.get("GEOM2VEC_SRC", "") or "").strip()
    if env_src:
        extra_candidates.append(Path(env_src))
    extra_candidates.append(Path("/project/dinner/zpengmei/apps/geom2vec/src"))
    try:
        extra_candidates.extend(Path(p) for p in glob.glob("/project/dinner/*/apps/geom2vec/src"))
    except Exception:
        pass

    for candidate in extra_candidates:
        if _try_src(candidate):
            return

    raise ImportError(
        "Could not import `geom2vec` downstream modules. Install geom2vec (e.g. `pip install -e apps/geom2vec`) "
        "or ensure `apps/geom2vec/src` is on PYTHONPATH (or set GEOM2VEC_SRC)."
    )


def _natural_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
    import re

    parts = re.split(r"(\d+)", text)
    key: List[Tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part.lower()))
    return tuple(key)


@dataclass(frozen=True)
class TrainedVAMPLobe:
    lobe: torch.nn.Module
    vampnet: Any
    train_keys: List[str]
    val_keys: List[str]
    meta: Dict[str, Any]


def train_vamp_lobe_from_cache(
    cache: EmbeddingCache,
    *,
    keys: Sequence[str],
    n_states: int,
    lag_time: int,
    pair_stride: int = 1,
    train_fraction: float = 0.8,
    shuffle_trajectories: bool = True,
    train_shuffle_pairs: bool = True,
    batch_size: int = 256,
    n_epochs: int = 10,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.0,
    optimizer: str = "Adam",
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
    epsilon: float = 1e-6,
    inverse_mode: str = "regularize",
    softmax_output: bool = True,
    vamp_merger: bool = False,
    vamp_merger_window: int = 6,
    double_lobe: bool = False,
    lobe_invariant: bool = False,
    data_parallel: bool = False,
    data_parallel_devices: Optional[Sequence[int]] = None,
    lobe_kwargs: Optional[Dict[str, Any]] = None,
) -> TrainedVAMPLobe:
    """Train a geom2vec `Lobe` with a VAMP objective using cached embeddings.

    Uses :class:`TimeLaggedIndexDataset` to avoid materializing all lag pairs.
    """

    if n_states <= 1:
        raise ValueError("n_states must be > 1")
    if lag_time <= 0:
        raise ValueError("lag_time must be > 0")
    if pair_stride <= 0:
        raise ValueError("pair_stride must be > 0")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be > 0")

    _ensure_geom2vec_importable()

    seed_i: Optional[int] = None
    if seed is not None:
        seed_i = int(seed)
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed_i)
            except Exception:
                pass

    def _is_retryable_linalg_error(exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "linalgerror" in name:
            return True
        if "linalg.eigh" in msg or "failed to converge" in msg:
            return True
        if "svd" in msg and "converge" in msg:
            return True
        if "cholesky" in msg and ("not positive" in msg or "not positive definite" in msg or "singular" in msg):
            return True
        if "singular" in msg:
            return True
        return False

    def _is_oom_error(exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg

    # Import deps only when training is requested. The geom2vec `Lobe` requires
    # torch_geometric; fall back to a simple pooled MLP lobe when unavailable.
    try:
        from geom2vec.models.downstream.vamp.vampnet import VAMPNet, VAMPNetConfig  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Failed to import geom2vec VAMPNet modules.") from exc

    Geom2VecLobe = None
    lobe_import_exc: Optional[BaseException] = None
    try:
        from geom2vec.models.downstream.lobe import Lobe as Geom2VecLobe  # type: ignore
    except Exception as exc:  # pragma: no cover
        Geom2VecLobe = None
        lobe_import_exc = exc

    class _PooledMLPLobe(torch.nn.Module):
        def __init__(
            self,
            *,
            output_channels: int,
            input_channels: int,
            hidden_channels: Optional[int] = None,
            softmax_output: bool = True,
        ) -> None:
            super().__init__()
            hidden = int(hidden_channels) if hidden_channels is not None else int(input_channels)
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(int(input_channels)),
                torch.nn.Linear(int(input_channels), hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, int(output_channels)),
            )
            self.out_act = torch.nn.Softmax(dim=-1) if softmax_output else None

        def forward(self, batch):  # type: ignore[override]
            graph = batch
            if isinstance(batch, dict):
                graph = batch.get("graph_features")
            if graph is None:
                raise ValueError("Expected 'graph_features' in batch.")
            if not isinstance(graph, torch.Tensor):
                graph = torch.as_tensor(graph)
            if graph.ndim < 2:
                raise ValueError("graph_features must have shape (batch, ..., hidden_dim).")
            if graph.ndim == 2:
                pooled = graph
            else:
                reduce_dims = tuple(range(1, graph.ndim - 1))
                pooled = graph.mean(dim=reduce_dims) if reduce_dims else graph
            out = self.net(pooled)
            if self.out_act is not None:
                out = self.out_act(out)
            return out

    keys_sorted = sorted([str(k) for k in keys], key=_natural_sort_key)
    if not keys_sorted:
        raise ValueError("keys must be non-empty")

    trajectories: List[dict] = []
    for key in keys_sorted:
        graph, ca = cache.load(key)
        if graph.ndim != 4:
            raise ValueError(f"Expected graph_features with shape (T, tokens, 4, H), got {tuple(graph.shape)} for {key}")
        if ca.ndim != 3:
            raise ValueError(f"Expected ca_coords with shape (T, tokens, 3), got {tuple(ca.shape)} for {key}")
        if graph.shape[0] != ca.shape[0]:
            raise ValueError(f"Frame mismatch between graph and ca for {key}: {graph.shape[0]} vs {ca.shape[0]}")
        trajectories.append({"graph_features": graph, "ca_coords": ca})

    num_tokens = int(trajectories[0]["graph_features"].shape[1])
    hidden_dim = int(trajectories[0]["graph_features"].shape[-1])
    for i, tr in enumerate(trajectories):
        g = tr["graph_features"]
        if int(g.shape[1]) != num_tokens or int(g.shape[-1]) != hidden_dim:
            raise ValueError("All cached trajectories must have the same num_tokens and hidden_dim.")

    # Trajectory-level train/val split
    indices = list(range(len(trajectories)))
    if shuffle_trajectories:
        rng = np.random.default_rng(seed_i)
        rng.shuffle(indices)

    n_train = int(round(train_fraction * len(indices)))
    n_train = max(1, min(n_train, len(indices) - 1)) if len(indices) > 1 else len(indices)
    train_idx = set(indices[:n_train])
    train_trajs = [trajectories[i] for i in range(len(trajectories)) if i in train_idx]
    val_trajs = [trajectories[i] for i in range(len(trajectories)) if i not in train_idx] or train_trajs

    train_keys = [keys_sorted[i] for i in range(len(keys_sorted)) if i in train_idx]
    val_keys = [keys_sorted[i] for i in range(len(keys_sorted)) if i not in train_idx] or list(train_keys)

    # Build datasets (index-based)
    train_ds = TimeLaggedIndexDataset(train_trajs, lag_time=lag_time, pair_stride=pair_stride)
    val_ds = TimeLaggedIndexDataset(val_trajs, lag_time=lag_time, pair_stride=pair_stride)

    # geom2vec's VAMP covariance estimator uses a factor 1/(batch_size-1) and will
    # crash on batches of size 1. Ensure we have enough pairs and never feed a
    # 1-sample batch (e.g., the trailing remainder when drop_last=False).
    try:
        n_pairs_train = int(len(train_ds))
    except Exception:
        n_pairs_train = -1
    try:
        n_pairs_val = int(len(val_ds))
    except Exception:
        n_pairs_val = -1
    if n_pairs_train >= 0 and n_pairs_train < 2:
        raise RuntimeError(
            f"Not enough time-lagged pairs to train VAMP (train_pairs={n_pairs_train}, lag_time={lag_time}, pair_stride={pair_stride})."
        )
    if n_pairs_val >= 0 and n_pairs_val < 2:
        raise RuntimeError(
            f"Not enough time-lagged pairs to validate VAMP (val_pairs={n_pairs_val}, lag_time={lag_time}, pair_stride={pair_stride})."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)
    dp_enabled = bool(data_parallel) and device_t.type == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 1
    dp_device_ids: Optional[List[int]] = None
    if dp_enabled:
        n_dev = int(torch.cuda.device_count())
        primary = int(device_t.index or 0)
        ids: List[int] = []
        if data_parallel_devices is not None:
            for d in list(data_parallel_devices):
                try:
                    di = int(d)
                except Exception:
                    continue
                if 0 <= di < n_dev and di not in ids:
                    ids.append(di)
        if not ids:
            ids = list(range(n_dev))
        if primary not in ids:
            ids = [primary] + ids
        ids = [primary] + [d for d in ids if int(d) != primary]
        dp_device_ids = ids if len(ids) >= 2 else None
        dp_enabled = dp_device_ids is not None

    base_epsilon = float(epsilon) if float(epsilon) > 0 else 1e-6
    # Retry with increasingly stronger regularization for ill-conditioned
    # covariance inverses in the VAMP estimator.
    epsilon_candidates = [min(base_epsilon * (10**k), 1e-1) for k in range(0, 6)]
    epsilon_used: Optional[float] = None
    last_exc: Optional[BaseException] = None
    vampnet: Optional[Any] = None
    lobe: Optional[torch.nn.Module] = None
    batch_size_used: Optional[int] = None
    used_lobe_kind: Optional[str] = None

    out_activation = torch.nn.Softmax(dim=-1) if softmax_output else None
    lobe_kwargs_in = dict(lobe_kwargs or {})

    batch_candidates: List[int] = []
    bs = max(2, int(batch_size))
    while bs >= 2:
        batch_candidates.append(bs)
        if bs <= 32:
            break
        bs = max(2, bs // 2)

    warned_fallback = False

    for bs in batch_candidates:
        # Ensure batch_size <= len(ds) so drop_last doesn't drop *all* batches for tiny datasets.
        if n_pairs_train > 0:
            train_bs = max(2, min(int(bs), int(n_pairs_train)))
        else:
            train_bs = int(bs)
        if n_pairs_val > 0:
            val_bs = max(2, min(int(bs), int(n_pairs_val)))
        else:
            val_bs = int(bs)

        # drop_last=True prevents a final remainder batch of size 1 (which crashes geom2vec).
        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=train_shuffle_pairs, drop_last=True)
        # Shuffle validation pairs so the VAMP validation estimate is not biased by ordering.
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=True, drop_last=True)

        for eps_try in epsilon_candidates:
            try:
                lobe_kwargs_try = dict(lobe_kwargs_in)
                hidden_channels_try = int(lobe_kwargs_try.pop("hidden_channels", hidden_dim))
                lobe_kwargs_try.pop("num_tokens", None)

                # Add merger options to lobe kwargs if enabled
                if vamp_merger:
                    lobe_kwargs_try["merger"] = True
                    lobe_kwargs_try["merger_window"] = int(vamp_merger_window)

                if Geom2VecLobe is not None:
                    used_lobe_kind = "geom2vec_lobe"
                    lobe = Geom2VecLobe(
                        output_channels=int(n_states),
                        input_channels=int(hidden_dim),
                        hidden_channels=int(hidden_channels_try),
                        num_tokens=int(num_tokens),
                        mlp_out_activation=out_activation,
                        device=device_t,
                        equi_rep=(not bool(lobe_invariant)),
                        **lobe_kwargs_try,
                    )
                    # NOTE: geom2vec's Lobe currently does not move parameters/buffers based on the
                    # `device=` kwarg; VAMPNet will move batches to `config.device`, so we must
                    # explicitly move the lobe to avoid CPU/CUDA mismatches.
                    lobe = lobe.to(device_t)
                else:
                    used_lobe_kind = "pooled_mlp_fallback"
                    if not warned_fallback:
                        reason = (
                            f"{type(lobe_import_exc).__name__}: {lobe_import_exc}"
                            if lobe_import_exc is not None
                            else "unknown"
                        )
                        print(
                            "[Geom2Vec VAMP] Warning: geom2vec Lobe unavailable; using pooled-MLP fallback "
                            f"(install torch_geometric to enable the full lobe). Reason: {reason}"
                        )
                        warned_fallback = True
                    lobe = _PooledMLPLobe(
                        output_channels=int(n_states),
                        input_channels=int(hidden_dim),
                        hidden_channels=int(hidden_channels_try),
                        softmax_output=bool(softmax_output),
                    ).to(device_t)

                if dp_enabled and dp_device_ids is not None and lobe is not None:
                    lobe = torch.nn.DataParallel(lobe, device_ids=dp_device_ids, output_device=int(device_t.index or 0))
                    used_lobe_kind = f"{used_lobe_kind}_dp"

                lobe_lagged = None
                if bool(double_lobe):
                    if Geom2VecLobe is not None:
                        lobe_lagged = Geom2VecLobe(
                            output_channels=int(n_states),
                            input_channels=int(hidden_dim),
                            hidden_channels=int(hidden_channels_try),
                            num_tokens=int(num_tokens),
                            mlp_out_activation=out_activation,
                            device=device_t,
                            equi_rep=(not bool(lobe_invariant)),
                            **lobe_kwargs_try,
                        ).to(device_t)
                    else:
                        lobe_lagged = _PooledMLPLobe(
                            output_channels=int(n_states),
                            input_channels=int(hidden_dim),
                            hidden_channels=int(hidden_channels_try),
                            softmax_output=bool(softmax_output),
                        ).to(device_t)
                    if dp_enabled and dp_device_ids is not None and lobe_lagged is not None:
                        lobe_lagged = torch.nn.DataParallel(
                            lobe_lagged, device_ids=dp_device_ids, output_device=int(device_t.index or 0)
                        )

                vamp_cfg = VAMPNetConfig(
                    optimizer=str(optimizer),
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    epsilon=float(eps_try),
                    mode=str(inverse_mode),
                    dtype=torch.float32,
                    device=device_t,
                    score_method="vamp-2",
                )
                vampnet = VAMPNet(lobe=lobe, lobe_lagged=lobe_lagged, config=vamp_cfg)
                vampnet.fit(
                    train_loader,
                    n_epochs=int(n_epochs),
                    validation_loader=val_loader,
                    progress=lambda x, **_: x,
                )
                epsilon_used = float(eps_try)
                batch_size_used = int(bs)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if _is_oom_error(exc):
                    vampnet = None
                    lobe = None
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    break
                if _is_retryable_linalg_error(exc):
                    vampnet = None
                    lobe = None
                    continue
                raise

        if vampnet is not None and lobe is not None:
            break

    if vampnet is None or lobe is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to train VAMP lobe (unknown error).")

    best_val_score = None
    try:
        val_scores = getattr(vampnet, "validation_scores", None)
        if val_scores is not None:
            arr = np.asarray(val_scores)
            if arr.size > 0 and np.any(np.isfinite(arr)):
                best_val_score = float(np.nanmax(arr))
    except Exception:
        best_val_score = None

    best_train_score = None
    try:
        tr_scores = getattr(vampnet, "training_scores", None)
        if tr_scores is not None:
            arr = np.asarray(tr_scores)
            if arr.size > 0 and np.any(np.isfinite(arr)):
                best_train_score = float(np.nanmax(arr))
    except Exception:
        best_train_score = None

    return TrainedVAMPLobe(
        lobe=vampnet.fetch_lobe() if hasattr(vampnet, "fetch_lobe") else lobe,
        vampnet=vampnet,
        train_keys=train_keys,
        val_keys=val_keys,
        meta={
            "n_states": int(n_states),
            "lag_time": int(lag_time),
            "pair_stride": int(pair_stride),
            "num_tokens": int(num_tokens),
            "hidden_dim": int(hidden_dim),
            "train_fraction": float(train_fraction),
            "optimizer": str(optimizer),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "epsilon": float(epsilon_used) if epsilon_used is not None else float(base_epsilon),
            "inverse_mode": str(inverse_mode),
            "softmax_output": bool(softmax_output),
            "device": str(device_t),
            "seed": None if seed is None else int(seed),
            "batch_size": int(batch_size_used) if batch_size_used is not None else int(batch_size),
            "lobe_kind": str(used_lobe_kind or ("geom2vec_lobe" if Geom2VecLobe is not None else "pooled_mlp_fallback")),
            "best_val_score": best_val_score,
            "best_train_score": best_train_score,
        },
    )


def train_stopvamp_lobe_from_cache(
    cache: EmbeddingCache,
    *,
    keys: Sequence[str],
    in_a: Sequence[np.ndarray],
    in_b: Sequence[np.ndarray],
    n_states: int,
    lag_time: int,
    stopping_mode: str = "mask",
    symmetrized: bool = False,
    train_fraction: float = 0.8,
    shuffle_trajectories: bool = True,
    batch_size: int = 256,
    n_epochs: int = 10,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.0,
    optimizer: str = "AdamAtan2",
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
    epsilon: float = 1e-6,
    inverse_mode: str = "regularize",
    softmax_output: bool = True,
    vamp_merger: bool = False,
    vamp_merger_window: int = 6,
    double_lobe: bool = False,
    lobe_invariant: bool = False,
    lobe_kwargs: Optional[Dict[str, Any]] = None,
) -> TrainedVAMPLobe:
    """Train a geom2vec lobe with the stopping-time VAMP objective using cached embeddings.

    This uses geom2vec's :class:`StopVAMPWorkflow` to route stopping indicators derived
    from boundary masks ``in_a``/``in_b`` into the covariance estimator.
    """

    if n_states <= 1:
        raise ValueError("n_states must be > 1")
    if lag_time <= 0:
        raise ValueError("lag_time must be > 0")
    stopping_mode_s = str(stopping_mode or "mask").strip().lower()
    if stopping_mode_s not in {"mask", "stopped_time"}:
        raise ValueError("stopping_mode must be one of {'mask', 'stopped_time'}")
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in the interval (0, 1].")
    if batch_size <= 1:
        raise ValueError("batch_size must be > 1 (stopped covariance uses 1/(n-1)).")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be > 0")

    keys_sorted = sorted([str(k) for k in keys], key=_natural_sort_key)
    if not keys_sorted:
        raise ValueError("keys must be non-empty")
    if len(in_a) != len(keys_sorted) or len(in_b) != len(keys_sorted):
        raise ValueError("in_a/in_b must have the same length as keys.")

    _ensure_geom2vec_importable()

    # Import deps only when training is requested. The geom2vec `Lobe` requires
    # torch_geometric; fall back to a simple pooled MLP lobe when unavailable.
    try:
        from geom2vec.models.downstream.vamp.workflow import StopVAMPWorkflow  # type: ignore
        from geom2vec.models.downstream.vamp.vampnet import VAMPNetConfig  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Failed to import geom2vec StopVAMP workflow modules.") from exc

    Geom2VecLobe = None
    lobe_import_exc: Optional[BaseException] = None
    try:
        from geom2vec.models.downstream.lobe import Lobe as Geom2VecLobe  # type: ignore
    except Exception as exc:  # pragma: no cover
        Geom2VecLobe = None
        lobe_import_exc = exc

    class _PooledMLPLobe(torch.nn.Module):
        def __init__(
            self,
            *,
            output_channels: int,
            input_channels: int,
            hidden_channels: Optional[int] = None,
            softmax_output: bool = True,
        ) -> None:
            super().__init__()
            hidden = int(hidden_channels) if hidden_channels is not None else int(input_channels)
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(int(input_channels)),
                torch.nn.Linear(int(input_channels), hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, int(output_channels)),
            )
            self.out_act = torch.nn.Softmax(dim=-1) if softmax_output else None

        def forward(self, batch):  # type: ignore[override]
            graph = batch
            if isinstance(batch, dict):
                graph = batch.get("graph_features")
            if graph is None:
                raise ValueError("Expected 'graph_features' in batch.")
            if not isinstance(graph, torch.Tensor):
                graph = torch.as_tensor(graph)
            if graph.ndim < 2:
                raise ValueError("graph_features must have shape (batch, ..., hidden_dim).")
            if graph.ndim == 2:
                pooled = graph
            else:
                reduce_dims = tuple(range(1, graph.ndim - 1))
                pooled = graph.mean(dim=reduce_dims) if reduce_dims else graph
            out = self.net(pooled)
            if self.out_act is not None:
                out = self.out_act(out)
            return out

    seed_i: Optional[int] = None
    if seed is not None:
        seed_i = int(seed)
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed_i)
            except Exception:
                pass

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    trajectories: List[dict] = []
    in_a_list: List[np.ndarray] = []
    in_b_list: List[np.ndarray] = []
    for key, ia, ib in zip(keys_sorted, in_a, in_b):
        graph, ca = cache.load(key)
        if graph.ndim != 4:
            raise ValueError(f"Expected graph_features with shape (T, tokens, 4, H), got {tuple(graph.shape)} for {key}")
        if ca.ndim != 3:
            raise ValueError(f"Expected ca_coords with shape (T, tokens, 3), got {tuple(ca.shape)} for {key}")
        if graph.shape[0] != ca.shape[0]:
            raise ValueError(f"Frame mismatch between graph and ca for {key}: {graph.shape[0]} vs {ca.shape[0]}")
        T = int(graph.shape[0])
        ia_b = np.asarray(ia, dtype=bool).reshape(-1)
        ib_b = np.asarray(ib, dtype=bool).reshape(-1)
        if ia_b.shape[0] != T or ib_b.shape[0] != T:
            raise ValueError(f"Boundary mask length mismatch for {key}: T={T}, in_a={ia_b.shape[0]}, in_b={ib_b.shape[0]}")
        if T <= int(lag_time):
            # Too short for any lag pairs; skip.
            continue
        trajectories.append({"graph_features": graph, "ca_coords": ca})
        in_a_list.append(ia_b)
        in_b_list.append(ib_b)

    if not trajectories:
        raise RuntimeError("No cached trajectories with frames > lag_time available for StopVAMP training.")

    # Ensure we have enough total lagged pairs for the stopped covariance estimator.
    total_pairs = int(sum(max(0, int(tr["graph_features"].shape[0]) - int(lag_time)) for tr in trajectories))
    if total_pairs < 2:
        raise RuntimeError(
            f"Not enough time-lagged pairs to train StopVAMP (pairs={total_pairs}, lag_time={lag_time})."
        )

    num_tokens = int(trajectories[0]["graph_features"].shape[1])
    hidden_dim = int(trajectories[0]["graph_features"].shape[-1])
    for tr in trajectories:
        g = tr["graph_features"]
        if int(g.shape[1]) != num_tokens or int(g.shape[-1]) != hidden_dim:
            raise ValueError("All cached trajectories must have the same num_tokens and hidden_dim.")

    # IMPORTANT: StopVAMPWorkflow already treats the lobe output as a feature map for
    # VAMP covariances, and in practice may apply its own softmax/state mapping.
    # Applying a softmax inside the lobe here can lead to "double-softmax" behavior
    # and collapse the discrete labels to a near-uniform/one-state assignment.
    #
    # We always train the lobe to output logits/features; callers can convert to
    # probabilities via a softmax when they need discrete state memberships.
    out_activation = None
    lobe_kwargs_in = dict(lobe_kwargs or {})
    hidden_channels_try = int(lobe_kwargs_in.pop("hidden_channels", hidden_dim))
    lobe_kwargs_in.pop("num_tokens", None)

    # Add merger options to lobe kwargs if enabled.
    if vamp_merger:
        lobe_kwargs_in["merger"] = True
        lobe_kwargs_in["merger_window"] = int(vamp_merger_window)

    warned_fallback = False
    if Geom2VecLobe is not None:
        lobe_kind = "geom2vec_lobe"
        lobe = Geom2VecLobe(
            output_channels=int(n_states),
            input_channels=int(hidden_dim),
            hidden_channels=int(hidden_channels_try),
            num_tokens=int(num_tokens),
            mlp_out_activation=out_activation,
            device=device_t,
            equi_rep=(not bool(lobe_invariant)),
            **lobe_kwargs_in,
        ).to(device_t)
    else:
        lobe_kind = "pooled_mlp_fallback"
        if not warned_fallback:
            reason = f"{type(lobe_import_exc).__name__}: {lobe_import_exc}" if lobe_import_exc is not None else "unknown"
            print(
                "[Geom2Vec StopVAMP] Warning: geom2vec Lobe unavailable; using pooled-MLP fallback "
                f"(install torch_geometric to enable the full lobe). Reason: {reason}"
            )
            warned_fallback = True
        lobe = _PooledMLPLobe(
            output_channels=int(n_states),
            input_channels=int(hidden_dim),
            hidden_channels=int(hidden_channels_try),
            softmax_output=False,
        ).to(device_t)

    base_epsilon = float(epsilon) if float(epsilon) > 0 else 1e-6
    epsilon_candidates = [min(base_epsilon * (10**k), 1e-1) for k in range(0, 6)]
    epsilon_used: Optional[float] = None

    trainer: Optional[Any] = None
    last_exc: Optional[BaseException] = None

    for eps_try in epsilon_candidates:
        try:
            remove_mean = stopping_mode_s == "stopped_time"
            vamp_cfg = VAMPNetConfig(
                optimizer=str(optimizer),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                epsilon=float(eps_try),
                mode=str(inverse_mode),
                symmetrized=bool(symmetrized),
                dtype=torch.float32,
                device=device_t,
                score_method="vamp-2",
                remove_mean=bool(remove_mean),
            )
            lobe_lagged = None
            if bool(double_lobe):
                if Geom2VecLobe is not None:
                    lobe_lagged = Geom2VecLobe(
                        output_channels=int(n_states),
                        input_channels=int(hidden_dim),
                        hidden_channels=int(hidden_channels_try),
                        num_tokens=int(num_tokens),
                        mlp_out_activation=out_activation,
                        device=device_t,
                        equi_rep=(not bool(lobe_invariant)),
                        **lobe_kwargs_in,
                    ).to(device_t)
                else:
                    lobe_lagged = _PooledMLPLobe(
                        output_channels=int(n_states),
                        input_channels=int(hidden_dim),
                        hidden_channels=int(hidden_channels_try),
                        softmax_output=False,
                    ).to(device_t)

            workflow = StopVAMPWorkflow(
                lobe=lobe,
                trajectories=trajectories,
                in_a=in_a_list,
                in_b=in_b_list,
                lag_time=int(lag_time),
                stopping_mode=stopping_mode_s,
                lobe_lagged=lobe_lagged,
                config=vamp_cfg,
                train_fraction=float(train_fraction),
                enable_validation=bool(train_fraction < 1.0),
                shuffle_trajectories=bool(shuffle_trajectories),
                seed=seed_i,
                batch_size=int(batch_size),
                dtype=torch.float32,
                drop_last=True,
            )
            trainer = workflow.fit(n_epochs=int(n_epochs))
            epsilon_used = float(eps_try)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            # Retry with stronger epsilon on ill-conditioned covariances.
            name = type(exc).__name__.lower()
            msg = str(exc).lower()
            if "linalg" in name or "failed to converge" in msg or "singular" in msg:
                continue
            raise

    if trainer is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to train StopVAMP lobe (unknown error).")

    best_val_score = None
    best_train_score = None
    try:
        val_scores = getattr(trainer, "validation_scores", None)
        if callable(val_scores):
            arr = np.asarray(val_scores())
        else:
            arr = np.asarray(val_scores) if val_scores is not None else np.asarray([])
        if arr.size > 0 and np.any(np.isfinite(arr)):
            best_val_score = float(np.nanmax(arr))
    except Exception:
        best_val_score = None
    try:
        tr_scores = getattr(trainer, "training_scores", None)
        if callable(tr_scores):
            arr = np.asarray(tr_scores())
        else:
            arr = np.asarray(tr_scores) if tr_scores is not None else np.asarray([])
        if arr.size > 0 and np.any(np.isfinite(arr)):
            best_train_score = float(np.nanmax(arr))
    except Exception:
        best_train_score = None

    # StopVAMPNet exposes fetch_lobe(); fall back to the lobe instance.
    trained_lobe = trainer.fetch_lobe() if hasattr(trainer, "fetch_lobe") else lobe

    return TrainedVAMPLobe(
        lobe=trained_lobe,
        vampnet=trainer,
        train_keys=list(keys_sorted),
        val_keys=list(keys_sorted),
        meta={
            "n_states": int(n_states),
            "lag_time": int(lag_time),
            "stopping_mode": str(stopping_mode_s),
            "symmetrized": bool(symmetrized),
            "num_tokens": int(num_tokens),
            "hidden_dim": int(hidden_dim),
            "train_fraction": float(train_fraction),
            "optimizer": str(optimizer),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "epsilon": float(epsilon_used) if epsilon_used is not None else float(base_epsilon),
            "inverse_mode": str(inverse_mode),
            # "softmax_output" is kept for backward compatibility: it indicates
            # that this lobe is intended to represent discrete states, but the
            # returned module emits logits/features (no softmax applied).
            "softmax_output": bool(softmax_output),
            "lobe_applies_softmax": False,
            "device": str(device_t),
            "seed": None if seed is None else int(seed),
            "batch_size": int(batch_size),
            "lobe_kind": str(lobe_kind),
            "best_val_score": best_val_score,
            "best_train_score": best_train_score,
        },
    )
