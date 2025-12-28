from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .cache import EmbeddingCache, EmbeddingRecord
from .pipeline import infer_and_cache
from .topology import Geom2VecTopologyCache


PathLike = Union[str, Path]


def _natural_sort_key(text: str) -> List[object]:
    """Natural sort helper (e.g. epoch_2 before epoch_10)."""
    import re

    parts = re.split(r"(\d+)", text)
    key: List[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def scan_dcd_files(
    base_dir: PathLike,
    *,
    file_postfix: str = ".dcd",
    sorting: bool = True,
) -> List[Path]:
    base = Path(base_dir)
    paths: List[Path] = []
    for root, _, files in os.walk(base):
        for name in files:
            if name.endswith(file_postfix):
                paths.append(Path(root) / name)
    if sorting:
        paths.sort(key=lambda p: _natural_sort_key(p.as_posix()))
    return paths


def _maybe_atom_slice(traj, selection: Optional[str]):
    if selection is None or selection == "all":
        return traj
    if selection in {"protein", "backbone"}:
        idx = traj.top.select(selection)
        return traj.atom_slice(idx)
    if selection == "heavy":
        idx = traj.top.select("not water and not hydrogen")
        return traj.atom_slice(idx)
    raise ValueError(f"Unsupported selection '{selection}'. Use one of: all, protein, backbone, heavy.")


def bootstrap_cache_from_dcd_dir(
    *,
    model,
    topology_file: PathLike,
    trajectory_folder: PathLike,
    cache_root: PathLike,
    stride: int = 1,
    selection: Optional[str] = "protein",
    file_postfix: str = ".dcd",
    sorting: bool = True,
    hidden_channels: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    embed_batch_size: int = 32,
    reduction: str = "sum",
    exclude_hydrogens: bool = True,
    center_ca: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Scan a DCD directory tree and populate an :class:`EmbeddingCache`.

    This mirrors the notebook inference setup (stride applied at trajectory-load
    time) and writes ``*.pt`` / ``*_ca.pt`` alongside the DCDs when
    ``cache_root == trajectory_folder`` (recommended).

    Returns a summary dict with computed/skipped keys and records.
    """

    if stride <= 0:
        raise ValueError("stride must be > 0")
    if embed_batch_size <= 0:
        raise ValueError("embed_batch_size must be > 0")

    try:
        import mdtraj as md  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mdtraj is required to bootstrap embeddings from DCD files.") from exc

    trajectory_folder = Path(trajectory_folder)
    cache = EmbeddingCache(cache_root)
    dcd_paths = scan_dcd_files(trajectory_folder, file_postfix=file_postfix, sorting=sorting)
    if not dcd_paths:
        raise FileNotFoundError(f"No '{file_postfix}' files found under {trajectory_folder}")

    computed: List[EmbeddingRecord] = []
    skipped: List[str] = []
    failed: List[Dict[str, str]] = []

    for dcd_path in dcd_paths:
        rel = dcd_path.relative_to(trajectory_folder)
        key = rel.with_suffix("").as_posix()

        if not overwrite:
            try:
                existing = cache.get(key)
            except KeyError:
                existing = None
            if existing is not None and Path(existing.embedding_path).exists() and Path(existing.ca_path).exists():
                skipped.append(key)
                continue

        try:
            traj = md.load(str(dcd_path), top=str(topology_file), stride=int(stride))
            traj = _maybe_atom_slice(traj, selection)
            coords_nm = np.asarray(traj.xyz, dtype=np.float32)  # mdtraj stores xyz in nm
            topo_cache = Geom2VecTopologyCache.from_mdtraj(
                traj.topology,
                exclude_hydrogens=exclude_hydrogens,
                meta={
                    "source_topology_file": str(topology_file),
                    "selection": selection,
                },
            )

            rec = infer_and_cache(
                cache,
                key=key,
                model=model,
                topology=topo_cache,
                coords=coords_nm,
                coords_in_nm=True,
                stride=1,  # we already subsampled via mdtraj stride
                batch_size=embed_batch_size,
                reduction=reduction,
                hidden_channels=hidden_channels,
                device=device,
                center_ca=center_ca,
                overwrite=overwrite,
                meta={
                    "stride": int(stride),  # effective trajectory stride (not the internal inference stride)
                    "source_dcd": str(dcd_path),
                },
            )
            computed.append(rec)
        except Exception as exc:
            failed.append({"key": key, "dcd_path": str(dcd_path), "error": str(exc)})

    return {
        "computed": computed,
        "skipped": skipped,
        "failed": failed,
        "total_dcd": len(dcd_paths),
        "cache_root": str(Path(cache_root)),
    }

