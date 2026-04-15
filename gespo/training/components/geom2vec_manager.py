"""
Geom2Vec Manager - Centralized management of geom2vec cache, projector, and state alignment.

This module extracts all geom2vec-related operations from the training loops,
providing a clean interface for:
- Cache initialization and management
- Embedding computation and caching
- VAMP projector training/refresh
- State alignment across projector updates
"""

from __future__ import annotations

import hashlib
import os
import importlib
import sys
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ...utils.selection import infer_protein_atom_indices

from ..geom2vec_backbone import (
    EmbeddingCache,
    Geom2VecTopologyCache,
    Geom2VecVAMPProjector,
    infer_and_cache,
)


def _parse_cuda_device_list(spec: str, *, max_devices: int) -> List[int]:
    spec_s = str(spec or "").strip()
    if not spec_s:
        return list(range(int(max_devices)))
    out: List[int] = []
    for part in spec_s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            idx = int(p)
        except Exception:
            continue
        if 0 <= idx < int(max_devices) and idx not in out:
            out.append(idx)
    return out


def _primary_cuda_index(device_str: str) -> Optional[int]:
    ds = str(device_str or "").strip()
    if not ds:
        return None
    if ds == "cuda":
        return 0
    if ds.startswith("cuda:"):
        try:
            return int(ds.split(":", 1)[1])
        except Exception:
            return None
    return None


def align_states_by_overlap(
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Compute a permutation that best aligns new states to previous ones
    based on a confusion matrix of frame-wise assignments.

    Only aligns states that are actually populated in the data.
    Unpopulated states use identity mapping.

    Args:
        old_labels: [N] - state labels under the previous projector
        new_labels: [N] - raw state labels under the refreshed projector
        n_states:   int - number of discrete states (assumed consistent)

    Returns:
        permutation: [n_states] - mapping old_label -> new_label
    """
    old_labels = np.asarray(old_labels, dtype=int)
    new_labels = np.asarray(new_labels, dtype=int)
    if old_labels.shape[0] != new_labels.shape[0]:
        raise ValueError(
            f"old_labels and new_labels must have same length, "
            f"got {old_labels.shape[0]} vs {new_labels.shape[0]}"
        )

    # Identify which states are actually populated in old and new
    old_populated = np.unique(old_labels)
    new_populated = np.unique(new_labels)
    old_populated = old_populated[(old_populated >= 0) & (old_populated < n_states)]
    new_populated = new_populated[(new_populated >= 0) & (new_populated < n_states)]

    if len(old_populated) == 0 or len(new_populated) == 0:
        # No valid states to align
        return np.arange(n_states, dtype=int)

    # Build confusion matrix ONLY for populated states
    # overlap[i, j] = count of frames where old=old_populated[i] and new=new_populated[j]
    overlap = np.zeros((len(old_populated), len(new_populated)), dtype=np.int64)
    old_to_idx = {state: idx for idx, state in enumerate(old_populated)}
    new_to_idx = {state: idx for idx, state in enumerate(new_populated)}

    for o, n in zip(old_labels, new_labels):
        if o in old_to_idx and n in new_to_idx:
            overlap[old_to_idx[o], new_to_idx[n]] += 1

    if overlap.sum() == 0:
        # No usable overlap information; fall back to identity mapping.
        return np.arange(n_states, dtype=int)

    # Hungarian solver minimizes cost, so use negative overlap as cost
    # to maximize agreement between old and new labels.
    cost_matrix = -overlap.astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build permutation: start with identity for all states
    permutation = np.arange(n_states, dtype=int)

    # Only update permutation for populated states that were matched
    for old_idx, new_idx in zip(row_ind, col_ind):
        old_state = int(old_populated[int(old_idx)])
        new_state = int(new_populated[int(new_idx)])
        permutation[old_state] = new_state

    # Ensure permutation is a true bijection (no collisions). After the
    # Hungarian assignment, some old states may remain unmapped and
    # some new states may be unused. We fill the remaining old states
    # so that every new state appears exactly once in the image.
    used_new = set(permutation[old_populated])
    all_new = set(range(n_states))
    unused_new = [s for s in sorted(all_new) if s not in used_new]

    for old_state in range(n_states):
        if old_state in old_populated:
            continue
        if old_state not in used_new:
            permutation[old_state] = old_state
            used_new.add(old_state)
        elif unused_new:
            new_state = unused_new.pop(0)
            permutation[old_state] = new_state
            used_new.add(new_state)

    print(
        f"[State Alignment] Matched {len(row_ind)} populated states: "
        f"{len(old_populated)} old → {len(new_populated)} new"
    )

    return permutation


class Geom2VecManager:
    """
    Manages geom2vec cache, projector, and state alignment operations.

    This class centralizes all geom2vec-related functionality that was
    previously scattered throughout the training loops.
    """

    def __init__(
        self,
        config: Any,
        env_ref: Any,
        torch_device: str = "cpu",
    ):
        """
        Initialize Geom2VecManager.

        Args:
            config: TrainingConfig instance
            env_ref: Reference environment (for topology)
            torch_device: Device string for torch operations
        """
        self.config = config
        self.env_ref = env_ref
        self.torch_device = torch_device

        # Cache and projector state
        self.cache: Optional[EmbeddingCache] = None
        self.cache_root_path: Optional[Path] = None
        self.topology_cache: Optional[Geom2VecTopologyCache] = None
        self.projector: Optional[Geom2VecVAMPProjector] = None
        self.projector_linear: Optional[Geom2VecVAMPProjector] = None
        self.md_top_protein: Optional[Any] = None

        # Configuration
        self.device_str = str(getattr(config, "geom2vec_device", "") or "").strip() or torch_device
        self.hidden_channels = int(getattr(config, "geom2vec_hidden_channels", 0) or 0)
        self.hidden_channels_opt = self.hidden_channels if self.hidden_channels > 0 else None
        self.embed_stride = int(getattr(config, "geom2vec_embed_stride", 1) or 1)
        self.embed_batch_size = int(getattr(config, "geom2vec_embed_batch_size", 32) or 32)
        self.reduction = str(getattr(config, "geom2vec_reduction", "sum") or "sum")
        self.center_ca = bool(getattr(config, "geom2vec_center_ca", False))
        self.exclude_hydrogens = bool(getattr(config, "geom2vec_exclude_hydrogens", True))
        self.data_parallel = bool(getattr(config, "geom2vec_data_parallel", False))
        self.data_parallel_devices = str(getattr(config, "geom2vec_data_parallel_devices", "") or "").strip()

        # Initialize if geom2vec mode is enabled
        proj_mode = str(getattr(config, "projection", "")).lower()
        if proj_mode == "geom2vec_vamp":
            self._initialize_geom2vec()

    def _import_geom2vec_module(self):
        """Import and return the geom2vec module."""
        try:
            import geom2vec  # type: ignore

            if hasattr(geom2vec, "create_model_from_checkpoint"):
                return geom2vec
        except ImportError:
            pass

        def _try_src(candidate: Path):
            if not (candidate / "geom2vec").exists():
                return None
            sys.path.insert(0, str(candidate))
            # If a namespace package shadowed the real geom2vec, purge it before retrying.
            for name in list(sys.modules.keys()):
                if name == "geom2vec" or name.startswith("geom2vec."):
                    sys.modules.pop(name, None)
            importlib.invalidate_caches()
            import geom2vec  # type: ignore

            if hasattr(geom2vec, "create_model_from_checkpoint"):
                return geom2vec
            return None

        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "apps" / "geom2vec" / "src"
            mod = _try_src(candidate)
            if mod is not None:
                return mod

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
            mod = _try_src(candidate)
            if mod is not None:
                return mod

        raise ImportError(
            "Could not import `geom2vec`. Install it (e.g. `pip install -e apps/geom2vec`) "
            "or ensure `apps/geom2vec/src` is on PYTHONPATH (or set GEOM2VEC_SRC)."
        )

    def _initialize_geom2vec(self) -> None:
        """Initialize geom2vec cache, topology, and projector."""
        # Setup cache
        cache_root = str(getattr(self.config, "geom2vec_cache_root", "") or "").strip() or str(
            getattr(self.config, "snapshot_dir", "") or "."
        )
        manifest_name = str(
            getattr(self.config, "geom2vec_manifest_name", "geom2vec_manifest.jsonl") or "geom2vec_manifest.jsonl"
        )
        self.cache = EmbeddingCache(cache_root, manifest_name=manifest_name)
        self.cache_root_path = Path(cache_root).resolve()

        # Import geom2vec and load model
        geom2vec = self._import_geom2vec_module()
        rep_ckpt = str(getattr(self.config, "geom2vec_checkpoint", "") or "").strip()
        rep_model = geom2vec.create_model_from_checkpoint(rep_ckpt, device=self.device_str)

        # Optional DataParallel: single-process multi-GPU inference/training for the geom2vec GNN.
        # This is most useful when geom2vec is the bottleneck and you can reserve GPUs for it.
        try:
            if (
                bool(self.data_parallel)
                and torch.cuda.is_available()
                and int(torch.cuda.device_count()) > 1
                and str(self.device_str).startswith("cuda")
            ):
                n_dev = int(torch.cuda.device_count())
                device_ids = _parse_cuda_device_list(self.data_parallel_devices, max_devices=n_dev)
                primary = _primary_cuda_index(self.device_str)
                if primary is None:
                    primary = int(device_ids[0]) if device_ids else 0
                if primary not in device_ids:
                    device_ids = [int(primary)] + list(device_ids)
                device_ids = [int(primary)] + [d for d in device_ids if int(d) != int(primary)]
                if len(device_ids) >= 2:
                    rep_model = torch.nn.DataParallel(rep_model, device_ids=device_ids, output_device=int(primary))
                    self.device_str = f"cuda:{int(primary)}"
                    print(
                        f"[Geom2Vec] Enabled DataParallel on devices={device_ids} (primary=cuda:{int(primary)}).",
                        flush=True,
                    )
        except Exception as exc_dp:
            print(f"[Geom2Vec] Warning: failed to enable DataParallel ({exc_dp}); continuing without DP.", flush=True)

        # Build protein-only topology
        try:
            md_top_full = self.env_ref.md_topology
            protein_indices = md_top_full.select("protein")
            if protein_indices is None or len(protein_indices) == 0:
                protein_indices = infer_protein_atom_indices(md_top_full, include_caps=True, allow_empty=True)
            self.md_top_protein = md_top_full.subset(protein_indices)
        except Exception:
            self.md_top_protein = None

        if self.md_top_protein is None:
            raise RuntimeError("[Geom2Vec] Failed to build protein-only mdtraj topology for embedding inference.")

        self.topology_cache = Geom2VecTopologyCache.from_mdtraj(
            self.md_top_protein,
            exclude_hydrogens=self.exclude_hydrogens,
            meta={"selection": "protein"},
        )

        # Create projectors (softmax and linear lobes)
        self.projector = Geom2VecVAMPProjector(
            rep_model=rep_model,
            topology=self.topology_cache,
            cache=self.cache,
            device=torch.device(self.device_str),
            hidden_channels=self.hidden_channels_opt,
        )
        self.projector_linear = Geom2VecVAMPProjector(
            rep_model=rep_model,
            topology=self.topology_cache,
            cache=self.cache,
            device=torch.device(self.device_str),
            hidden_channels=self.hidden_channels_opt,
        )

    def is_trained(self) -> bool:
        """Check if the projector has been trained."""
        return self.projector is not None and getattr(self.projector, "trained", None) is not None

    def cache_key_for_history_traj(
        self,
        epoch_idx: int,
        traj_idx: int,
        dcd_path: Optional[str],
    ) -> str:
        """
        Generate a unique cache key for a history trajectory.

        Args:
            epoch_idx: Epoch index
            traj_idx: Trajectory index within epoch
            dcd_path: Optional path to DCD file

        Returns:
            Unique cache key string
        """
        if dcd_path:
            p = Path(str(dcd_path))
            if self.cache_root_path is not None:
                try:
                    rel = p.relative_to(self.cache_root_path)
                    return rel.with_suffix("").as_posix()
                except Exception:
                    pass
            # Fallback: place under an "external" bucket to avoid absolute-path keys
            try:
                abs_path = str(p.resolve())
                payload = abs_path.encode("utf-8")
                try:
                    h = hashlib.md5(payload, usedforsecurity=False)
                except TypeError:
                    h = hashlib.md5(payload)
                digest = h.hexdigest()[:10]
                return (Path("external") / f"{digest}_{p.stem}").as_posix()
            except Exception:
                return (Path("external") / p.name).with_suffix("").as_posix()
        return (Path("history_bootstrap") / f"epoch_{epoch_idx:04d}" / f"traj_{traj_idx:03d}").as_posix()

    def has_cached_key(self, key: str) -> bool:
        """Check if a cache key exists and has valid embeddings."""
        if self.cache is None:
            return False
        try:
            rec = self.cache.get(key)
        except KeyError:
            return False
        try:
            # VAMP needs at least 2 frames (even for lag=1). Treat <2-frame caches as invalid
            # so we can overwrite them once the trajectory/DCD is fully written.
            if int(getattr(rec, "frames", 0)) < 2:
                return False
            if int(getattr(rec, "tokens", 0)) <= 0:
                return False
        except Exception:
            return False
        try:
            return Path(rec.embedding_path).exists() and Path(rec.ca_path).exists()
        except Exception:
            return False

    def ensure_cached(
        self,
        *,
        key: str,
        dcd_path: Optional[str],
        coords_nm: Optional[np.ndarray],
        overwrite: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Ensure embeddings are cached for the given trajectory.

        Args:
            key: Cache key
            dcd_path: Optional DCD file path
            coords_nm: Optional coordinates array [T, N, 3] in nm
            overwrite: Force recomputation
            meta: Optional metadata dict

        Returns:
            Cache key if successful, None otherwise
        """
        if self.cache is None or self.projector is None or self.topology_cache is None:
            return None

        effective_stride = int(self.embed_stride)

        # Check for stale cache
        if not overwrite and self.has_cached_key(key):
            # TODO: Add staleness checking logic here if needed
            return key

        # Load coordinates if not provided
        if coords_nm is None:
            if not dcd_path or self.md_top_protein is None:
                return None
            try:
                traj = md.load(str(dcd_path), top=self.md_top_protein, stride=effective_stride)
                coords_nm = np.asarray(traj.xyz, dtype=np.float32)
                infer_stride = 1  # Already strided during load
            except Exception:
                # Fallback: load with full topology and slice to protein
                try:
                    md_top_full = self.env_ref.md_topology
                    protein_indices = md_top_full.select("protein")
                    if protein_indices is None or len(protein_indices) == 0:
                        protein_indices = infer_protein_atom_indices(md_top_full, include_caps=True, allow_empty=True)
                    traj_full = md.load(str(dcd_path), top=md_top_full, stride=effective_stride)
                    traj = traj_full.atom_slice(protein_indices)
                    coords_nm = np.asarray(traj.xyz, dtype=np.float32)
                    infer_stride = 1
                except Exception as exc2:
                    print(f"[Geom2Vec Cache] Failed to load DCD {dcd_path}: {exc2}")
                    return None
        else:
            infer_stride = effective_stride

        # Guard: avoid caching empty trajectories (can happen if a DCD is not fully flushed yet).
        try:
            n_frames_i = int(np.asarray(coords_nm).shape[0]) if coords_nm is not None else 0
            if n_frames_i <= 0:
                if dcd_path:
                    print(
                        f"[Geom2Vec Cache] Warning: no frames available for key={key} from dcd={dcd_path}; "
                        "skipping embedding cache.",
                        flush=True,
                    )
                return None
            if n_frames_i < 2:
                if dcd_path:
                    print(
                        f"[Geom2Vec Cache] Warning: only {n_frames_i} frame available for key={key} from dcd={dcd_path}; "
                        "skipping embedding cache until more frames are available.",
                        flush=True,
                    )
                return None
        except Exception:
            return None

        # Build metadata
        meta_extra: Dict[str, Any] = {"stride": effective_stride, "source_dcd": str(dcd_path) if dcd_path else ""}
        if dcd_path:
            try:
                st = Path(str(dcd_path)).stat()
                meta_extra["source_dcd_mtime_ns"] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                meta_extra["source_dcd_size"] = int(st.st_size)
            except Exception:
                pass

        # Compute and cache embeddings
        try:
            rec = infer_and_cache(
                self.cache,
                key=key,
                model=self.projector.rep_model,
                topology=self.topology_cache,
                coords=coords_nm,
                coords_in_nm=True,
                stride=infer_stride,
                batch_size=self.embed_batch_size,
                reduction=self.reduction,
                hidden_channels=self.hidden_channels_opt,
                device=torch.device(self.device_str),
                center_ca=self.center_ca,
                overwrite=overwrite,
                meta={
                    **(meta or {}),
                    **meta_extra,
                },
            )
            return rec.key
        except Exception as exc:
            print(f"[Geom2Vec Cache] Failed to compute/store embeddings for key={key}: {exc}")
            return None
