"""Endpoint (A/B) PDB validation utilities.

These helpers enforce that endpoint A and B describe the same solute topology so
reference-based metrics and solute-only history are unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from .selection import residue_is_solvent_like


@dataclass(frozen=True)
class EndpointValidationResult:
    pdb_a: str
    pdb_b: str
    solute_atom_indices_a: np.ndarray
    solute_atom_indices_b: np.ndarray
    solute_residue_keys_a: Tuple[Tuple[int, str], ...]
    solute_residue_keys_b: Tuple[Tuple[int, str], ...]
    solute_atom_keys_a: Tuple[Tuple[int, str, str, str], ...]
    solute_atom_keys_b: Tuple[Tuple[int, str, str, str], ...]

    @property
    def n_solute_atoms(self) -> int:
        return int(self.solute_atom_indices_a.size)

    @property
    def n_solute_residues(self) -> int:
        return int(len(self.solute_residue_keys_a))


def _ensure_file(path: str, label: str) -> str:
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{label} PDB not found: {p}")
    return str(p)


def _iter_solute_residues(topology) -> Iterable:
    for res in getattr(topology, "residues", []):
        if not residue_is_solvent_like(res):
            yield res


def _solute_atom_indices(topology) -> np.ndarray:
    idx: list[int] = []
    for atom in getattr(topology, "atoms", []):
        try:
            if residue_is_solvent_like(getattr(atom, "residue", None)):
                continue
            idx.append(int(getattr(atom, "index", -1)))
        except Exception:
            continue
    return np.asarray([i for i in idx if i >= 0], dtype=np.int64)


def _residue_key(res) -> Tuple[int, str]:
    try:
        ridx = int(getattr(res, "index", -1))
    except Exception:
        ridx = -1
    try:
        name = str(getattr(res, "name", "") or "").strip().upper()
    except Exception:
        name = ""
    return ridx, name


def _atom_key(atom) -> Tuple[int, str, str, str]:
    # (solute residue index, residue name, atom name, element symbol)
    res = getattr(atom, "residue", None)
    ridx, rname = _residue_key(res)
    try:
        aname = str(getattr(atom, "name", "") or "").strip().upper()
    except Exception:
        aname = ""
    sym = ""
    try:
        elem = getattr(atom, "element", None)
        sym = str(getattr(elem, "symbol", "") or "").strip().upper()
    except Exception:
        sym = ""
    return ridx, rname, aname, sym


def validate_endpoint_pdbs(
    pdb_a: str,
    pdb_b: str,
    *,
    require_solute_only: bool = False,
) -> EndpointValidationResult:
    """Validate that endpoint A and B PDBs describe the same solute topology.

    By default, this is robust to solvent/ions in the PDBs:
    - Solvent/ions/lipids are detected and ignored (require_solute_only=False).
    - All solute residues and solute atoms (including hydrogens) must match in
      the same order, so the A↔B mapping is the identity.
    """

    pdb_a = _ensure_file(pdb_a, "Endpoint A")
    pdb_b = _ensure_file(pdb_b, "Endpoint B")

    try:
        import mdtraj as md
    except Exception as exc:
        raise RuntimeError("mdtraj is required for endpoint validation.") from exc

    tr_a = md.load(pdb_a)
    tr_b = md.load(pdb_b)

    top_a = tr_a.topology
    top_b = tr_b.topology

    sol_a = _solute_atom_indices(top_a)
    sol_b = _solute_atom_indices(top_b)

    if int(sol_a.size) != int(getattr(top_a, "n_atoms", 0) or 0):
        if require_solute_only:
             raise ValueError("Endpoint A PDB contains solvent; provide a solute-only PDB.")
        print(f"[Warning] Endpoint A contains solvent/ion/lipid residues; these will be ignored for validation.")

    if int(sol_b.size) != int(getattr(top_b, "n_atoms", 0) or 0):
        if require_solute_only:
             raise ValueError("Endpoint B PDB contains solvent; provide a solute-only PDB.")
        print(f"[Warning] Endpoint B contains solvent/ion/lipid residues; these will be ignored for validation.")

    res_keys_a = tuple(_residue_key(r) for r in _iter_solute_residues(top_a))
    res_keys_b = tuple(_residue_key(r) for r in _iter_solute_residues(top_b))
    if res_keys_a != res_keys_b:
        # Find first mismatch for a concise error.
        n = min(len(res_keys_a), len(res_keys_b))
        mismatch = None
        for i in range(n):
            if res_keys_a[i] != res_keys_b[i]:
                mismatch = (i, res_keys_a[i], res_keys_b[i])
                break
        if mismatch is None:
            mismatch_msg = f"n_solute_residues A={len(res_keys_a)} B={len(res_keys_b)}"
        else:
            i, a_k, b_k = mismatch
            mismatch_msg = f"first mismatch at solute residue {i}: A={a_k} B={b_k}"
        raise ValueError(f"Endpoint residue sequences differ ({mismatch_msg}).")

    # Atom keys in solute atom order (must match exactly).
    try:
        atoms_a = [top_a.atom(int(i)) for i in sol_a]
        atoms_b = [top_b.atom(int(i)) for i in sol_b]
    except Exception:
        atoms_a = [a for a in getattr(top_a, "atoms", []) if not residue_is_solvent_like(getattr(a, "residue", None))]
        atoms_b = [a for a in getattr(top_b, "atoms", []) if not residue_is_solvent_like(getattr(a, "residue", None))]

    atom_keys_a = tuple(_atom_key(a) for a in atoms_a)
    atom_keys_b = tuple(_atom_key(a) for a in atoms_b)
    if atom_keys_a != atom_keys_b:
        n = min(len(atom_keys_a), len(atom_keys_b))
        mismatch = None
        for i in range(n):
            if atom_keys_a[i] != atom_keys_b[i]:
                mismatch = (i, atom_keys_a[i], atom_keys_b[i])
                break
        if mismatch is None:
            mismatch_msg = f"n_solute_atoms A={len(atom_keys_a)} B={len(atom_keys_b)}"
        else:
            i, a_k, b_k = mismatch
            mismatch_msg = f"first mismatch at solute atom {i}: A={a_k} B={b_k}"
        raise ValueError(f"Endpoint atom lists differ ({mismatch_msg}).")

    return EndpointValidationResult(
        pdb_a=pdb_a,
        pdb_b=pdb_b,
        solute_atom_indices_a=np.asarray(sol_a, dtype=np.int64),
        solute_atom_indices_b=np.asarray(sol_b, dtype=np.int64),
        solute_residue_keys_a=res_keys_a,
        solute_residue_keys_b=res_keys_b,
        solute_atom_keys_a=atom_keys_a,
        solute_atom_keys_b=atom_keys_b,
    )
