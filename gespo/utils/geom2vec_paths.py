"""Locate the geom2vec backbone checkpoint automatically.

geom2vec is a separate package that ships its pretrained backbones in a
``checkpoints/`` folder. Rather than force every run to spell out an absolute
``--geom2vec-ckpt`` path, GESPO auto-discovers a checkpoint from the geom2vec
install (mirroring how ``projections/geom2vec/vamp_trainer.py`` resolves the
package ``src/``).

Dependency-light on purpose (only ``os``/``glob``/``pathlib`` + ``find_spec``,
which does NOT import geom2vec/torch), so the CLI preflight can call it without
pulling the scientific stack.

Resolution order for an empty request:
  1. ``$GEOM2VEC_CKPT`` (a file)
  2. ``<prefer>.pth`` in any known checkpoints dir
  3. any ``tensornet_*.pth`` then any ``*.pth`` in a known checkpoints dir

Known checkpoints dirs: ``$GEOM2VEC_SRC/../checkpoints``, the pip-installed
geom2vec package's ``checkpoints/``, an ``apps/geom2vec/checkpoints`` checkout near
the GESPO repo, and ``/project/dinner/*/apps/geom2vec/checkpoints``.
"""
from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import List, Optional

# The backbone the bundled examples use; preferred when auto-detecting.
DEFAULT_CHECKPOINT_NAME = "tensornet_l3_h64_rbf32_r5.pth"


def candidate_checkpoint_dirs() -> List[Path]:
    """Ordered, de-duplicated list of directories that may hold geom2vec ``.pth``."""
    dirs: List[Path] = []

    env_src = str(os.environ.get("GEOM2VEC_SRC", "") or "").strip()
    if env_src:
        dirs.append(Path(env_src).parent / "checkpoints")

    # Locate a pip-installed geom2vec WITHOUT importing it (find_spec won't exec it).
    try:
        spec = importlib.util.find_spec("geom2vec")
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        pkg_init = Path(spec.origin).resolve()  # .../[src/]geom2vec/__init__.py
        for up in pkg_init.parents:
            cand = up / "checkpoints"
            dirs.append(cand)
            if up.name in ("src", "geom2vec"):
                continue
            break  # stop one level past the package root

    here = Path(__file__).resolve()
    for parent in here.parents:
        dirs.append(parent / "apps" / "geom2vec" / "checkpoints")

    dirs.append(Path("/project/dinner/zpengmei/apps/geom2vec/checkpoints"))
    try:
        dirs.extend(Path(p) for p in glob.glob("/project/dinner/*/apps/geom2vec/checkpoints"))
    except Exception:
        pass

    seen, out = set(), []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def find_geom2vec_checkpoint(prefer: str = DEFAULT_CHECKPOINT_NAME) -> Optional[str]:
    """Auto-discover a geom2vec checkpoint path, or ``None`` if none is found."""
    env_ckpt = str(os.environ.get("GEOM2VEC_CKPT", "") or "").strip()
    if env_ckpt and Path(env_ckpt).is_file():
        return env_ckpt

    dirs = candidate_checkpoint_dirs()
    for d in dirs:
        cand = d / prefer
        if cand.is_file():
            return str(cand)
    for pattern in ("tensornet_*.pth", "*.pth"):
        for d in dirs:
            if not d.is_dir():
                continue
            hits = sorted(glob.glob(str(d / pattern)))
            if hits:
                return hits[0]
    return None


def resolve_geom2vec_checkpoint(explicit: str = "", prefer: str = DEFAULT_CHECKPOINT_NAME) -> str:
    """Return a usable geom2vec checkpoint path.

    - An ``explicit`` path that exists is returned unchanged.
    - An ``explicit`` bare filename / missing path is looked up by basename in the
      known checkpoints dirs (so ``--geom2vec-ckpt tensornet_l3_h64_rbf32_r5.pth``
      works without a directory); if still not found it is returned unchanged so
      the caller surfaces a clear error.
    - An empty request auto-discovers (env + known dirs).

    Returns ``""`` when nothing is found and nothing was requested.
    """
    explicit = str(explicit or "").strip()
    if explicit:
        if Path(explicit).is_file():
            return explicit
        base = Path(explicit).name
        for d in candidate_checkpoint_dirs():
            cand = d / base
            if cand.is_file():
                return str(cand)
        return explicit
    return find_geom2vec_checkpoint(prefer=prefer) or ""
