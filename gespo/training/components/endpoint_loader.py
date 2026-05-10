"""Endpoint PDB loading and boundary plugin utilities for path sampling."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import mdtraj as md
import numpy as np

from gespo.projections.geom2vec import Geom2VecTopologyCache


def load_endpoint_ca_ref(
    pdb_path: str,
    *,
    label: str,
    topology_cache: Optional[Geom2VecTopologyCache],
    center_ca: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load CA coordinates from an endpoint PDB aligned to geom2vec token order.

    Parameters
    ----------
    pdb_path : str
        Path to the endpoint PDB file.
    label : str
        Human-readable label for log messages (e.g. "A" or "B").
    topology_cache : Geom2VecTopologyCache or None
        Geom2vec topology cache for residue-to-token mapping.
    center_ca : bool
        If True, center the CA coordinates (subtract mean of valid tokens).

    Returns
    -------
    ca_coords : ndarray or None
        CA coordinates in nm, shape ``(n_tokens, 3)``.
    valid_mask : ndarray or None
        Boolean mask of tokens with valid CA positions.
    """
    if topology_cache is None:
        return None, None
    pdb_path = str(pdb_path or "").strip()
    if not pdb_path:
        return None, None
    try:
        tr = md.load(pdb_path)
        if int(tr.n_frames) <= 0:
            raise ValueError("no frames in PDB")
        top = tr.topology
        n_tok = int(topology_cache.num_tokens)
        token_res_idx = np.asarray(topology_cache.token_residue_indices, dtype=np.int64).reshape(-1)
        if int(token_res_idx.shape[0]) != n_tok:
            raise ValueError("token_residue_indices length mismatch")

        ca_out = np.zeros((n_tok, 3), dtype=np.float32)
        valid = np.zeros((n_tok,), dtype=bool)
        xyz = np.asarray(tr.xyz[0], dtype=np.float32)  # nm

        # Build residue->CA atom index map in the endpoint topology.
        ca_by_res: Dict[int, int] = {}
        for res in top.residues:
            ai = None
            for atom in res.atoms:
                if str(atom.name) == "CA":
                    ai = int(atom.index)
                    break
            if ai is not None:
                ca_by_res[int(res.index)] = int(ai)

        for t in range(n_tok):
            r = int(token_res_idx[t])
            ai = ca_by_res.get(r, None)
            if ai is None:
                continue
            if ai < 0 or ai >= int(xyz.shape[0]):
                continue
            ca_out[t] = xyz[int(ai)]
            valid[t] = True

        if center_ca and bool(np.any(valid)):
            ca_out = ca_out - ca_out[valid].mean(axis=0, keepdims=True)

        if int(top.n_residues) != int(np.max(token_res_idx) + 1):
            print(
                f"[Geom2Vec] Note: endpoint {label} residue count {int(top.n_residues)} "
                f"differs from geom2vec token residue max {int(np.max(token_res_idx) + 1)}; "
                "CA matching uses residue index order and may drop tokens.",
                flush=True,
            )
        return ca_out, valid
    except Exception as exc:
        print(f"[Geom2Vec] Warning: failed to build endpoint CA ref for {label} from {pdb_path}: {exc}", flush=True)
        return None, None


def load_boundary_plugin(spec: str) -> Callable[..., Any]:
    """Dynamically load a user-provided boundary plugin callable.

    Parameters
    ----------
    spec : str
        Plugin specification in the form ``"/path/to/file.py:callable_name"``.

    Returns
    -------
    callable
        The loaded boundary function.
    """
    spec = str(spec or "").strip()
    if not spec:
        raise ValueError("Empty boundary plugin spec.")
    if ":" not in spec:
        raise ValueError("Boundary plugin spec must have the form '/path/file.py:callable_name'.")
    path_str, name = spec.rsplit(":", 1)
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Boundary plugin file not found: {path}")
    mod_name = f"_gespo_path_boundary_{hashlib.md5(str(path).encode('utf-8')).hexdigest()}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Could not import boundary plugin module from: {path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = module
    spec_obj.loader.exec_module(module)
    fn = getattr(module, name, None)
    if fn is None:
        raise AttributeError(f"Boundary plugin callable '{name}' not found in {path}.")
    if not callable(fn):
        raise TypeError(f"Boundary plugin '{name}' is not callable.")
    return fn
