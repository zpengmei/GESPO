from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import torch

PathLike = Union[str, Path]


def _normalize_key(key: Union[str, Path]) -> Path:
    """Normalize a cache key to a relative path without suffixes.

    Accepts keys like:
    - "epoch_0001/traj_03"
    - "epoch_0001/traj_03.pt"
    - "epoch_0001/traj_03_ca.pt"  (will be normalized to ".../traj_03")
    """

    key_path = Path(key)
    if key_path.is_absolute():
        raise ValueError(f"cache key must be a relative path, got: {key_path}")

    if key_path.suffix == ".pt":
        key_path = key_path.with_suffix("")
    if key_path.name.endswith("_ca"):
        key_path = key_path.with_name(key_path.name[: -len("_ca")])
    return key_path


@dataclass(frozen=True)
class EmbeddingRecord:
    """Manifest record for a cached geom2vec embedding trajectory."""

    key: str
    embedding_path: str
    ca_path: str
    frames: int
    tokens: int
    dtype: str = "float32"
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "embedding_path": self.embedding_path,
                "ca_path": self.ca_path,
                "frames": int(self.frames),
                "tokens": int(self.tokens),
                "dtype": self.dtype,
                "meta": self.meta,
            },
            sort_keys=True,
        )


class EmbeddingCache:
    """File-backed cache for per-trajectory geom2vec embeddings.

    Design goals (per the GEOM2VEC_BACKBONE_PLAN):
    - Store CPU tensors to disk once, reuse across refreshes.
    - Always store float32 (no float16).
    - Keep an append-only JSONL manifest for robust indexing.
    """

    def __init__(
        self,
        root: PathLike,
        *,
        manifest_name: str = "manifest.jsonl",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / manifest_name
        self._index: dict[str, EmbeddingRecord] = {}
        if self.manifest_path.exists():
            self._load_manifest()

    def _load_manifest(self) -> None:
        for line in self.manifest_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = EmbeddingRecord(
                key=str(obj["key"]),
                embedding_path=str(obj["embedding_path"]),
                ca_path=str(obj["ca_path"]),
                frames=int(obj["frames"]),
                tokens=int(obj["tokens"]),
                dtype=str(obj.get("dtype", "float32")),
                meta=dict(obj.get("meta", {})),
            )
            # Last-write-wins on repeated keys (append-only manifest).
            self._index[record.key] = record

    def _paths_for_key(self, key: Union[str, Path]) -> tuple[Path, Path]:
        key_path = _normalize_key(key)
        base = self.root / key_path
        embedding_path = base.with_suffix(".pt")
        ca_path = (base.parent / f"{base.name}_ca").with_suffix(".pt")
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        return embedding_path, ca_path

    @staticmethod
    def _to_float32_cpu(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()

    def store(
        self,
        key: Union[str, Path],
        *,
        graph_features: torch.Tensor,
        ca_coords: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> EmbeddingRecord:
        """Store a single embedding trajectory to disk (float32)."""

        embedding_path, ca_path = self._paths_for_key(key)
        if not overwrite and (embedding_path.exists() or ca_path.exists()):
            raise FileExistsError(f"Cache entry already exists for key='{key}' at {embedding_path}")

        graph_cpu = self._to_float32_cpu(graph_features)
        ca_cpu = self._to_float32_cpu(ca_coords)

        if graph_cpu.ndim < 1:
            raise ValueError("graph_features must have a leading frame dimension.")
        if ca_cpu.ndim != 3:
            raise ValueError("ca_coords must have shape (frames, tokens, 3).")
        if ca_cpu.shape[0] != graph_cpu.shape[0]:
            raise ValueError(
                f"Frame mismatch between graph_features ({graph_cpu.shape[0]}) "
                f"and ca_coords ({ca_cpu.shape[0]})."
            )

        torch.save(graph_cpu, embedding_path)
        torch.save(ca_cpu, ca_path)

        record = EmbeddingRecord(
            key=_normalize_key(key).as_posix(),
            embedding_path=str(embedding_path),
            ca_path=str(ca_path),
            frames=int(graph_cpu.shape[0]),
            tokens=int(ca_cpu.shape[1]),
            dtype="float32",
            meta=dict(meta or {}),
        )

        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(record.as_json() + "\n")

        self._index[record.key] = record
        return record

    def get(self, key: Union[str, Path]) -> EmbeddingRecord:
        norm = _normalize_key(key).as_posix()
        if norm not in self._index:
            raise KeyError(f"Unknown cache key: {norm}")
        return self._index[norm]

    def keys(self) -> Iterable[str]:
        return self._index.keys()

    def load(
        self,
        key: Union[str, Path],
        *,
        map_location: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load (graph_features, ca_coords) for a cache key."""

        rec = self.get(key)
        # Prefer `weights_only=True` when supported to avoid pickle execution for
        # non-tensor objects; these cache files are expected to contain tensors only.
        try:
            graph = torch.load(rec.embedding_path, map_location=map_location, weights_only=True)
            ca = torch.load(rec.ca_path, map_location=map_location, weights_only=True)
        except TypeError:  # older PyTorch without `weights_only`
            graph = torch.load(rec.embedding_path, map_location=map_location)
            ca = torch.load(rec.ca_path, map_location=map_location)
        if not isinstance(graph, torch.Tensor) or not isinstance(ca, torch.Tensor):
            raise TypeError("Expected tensors from torch.load for cached embeddings.")
        # Enforce float32 in memory too.
        return graph.to(dtype=torch.float32), ca.to(dtype=torch.float32)
