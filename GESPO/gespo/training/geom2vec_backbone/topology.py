from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch


def _is_hydrogen(atomic_number: Optional[int], element_symbol: Optional[str]) -> bool:
    if atomic_number == 1:
        return True
    if element_symbol is None:
        return False
    return str(element_symbol).upper() == "H"


@dataclass(frozen=True)
class Geom2VecTopologyCache:
    """Topology-derived metadata needed for geom2vec embedding inference.

    This cache is designed around the exact atom ordering of the coordinate arrays you
    will pass at runtime (e.g., the protein-only atom order used in GESPO rollouts or
    DCD snapshots).

    Fields
    ------
    atom_indices :
        Indices into the original coordinate atom axis selecting the atoms used for
        representation inference (typically heavy atoms only).
    atomic_numbers :
        Atomic numbers for the selected atoms (aligned with ``atom_indices``).
    cg_map :
        Token/residue index per selected atom (length = num_atoms_selected).
    cg_counts :
        Number of selected atoms per token (length = num_tokens).
    ca_indices :
        Index of the Cα atom within the selected-atom list for each token (or -1 if
        missing), length = num_tokens.
    token_residue_indices :
        Original residue indices (from the input topology) for each token.
    meta :
        Optional metadata (e.g., selection strings) for diagnostics.
    """

    atom_indices: np.ndarray
    atomic_numbers: np.ndarray
    cg_map: np.ndarray
    cg_counts: np.ndarray
    ca_indices: np.ndarray
    token_residue_indices: np.ndarray
    meta: Dict[str, Any]

    @property
    def num_atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.cg_counts.shape[0])

    def select_atoms(self, coords: np.ndarray) -> np.ndarray:
        """Slice a coordinate array to the selected atom set.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates with shape ``(frames, atoms, 3)`` matching the topology used to
            build this cache.
        """

        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape (frames, atoms, 3).")
        return coords[:, self.atom_indices]

    def extract_ca_coords(
        self,
        selected_coords: np.ndarray,
        *,
        center: bool = True,
        fill_value: float = 0.0,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """Build a per-token Cα coordinate array from selected-atom coordinates."""

        if selected_coords.ndim != 3 or selected_coords.shape[-1] != 3:
            raise ValueError("selected_coords must have shape (frames, selected_atoms, 3).")
        if selected_coords.shape[1] != self.num_atoms:
            raise ValueError(
                f"Atom dimension mismatch: selected_coords has {selected_coords.shape[1]} atoms "
                f"but cache expects {self.num_atoms}."
            )

        frames = int(selected_coords.shape[0])
        out = np.full((frames, self.num_tokens, 3), fill_value, dtype=dtype)
        valid = self.ca_indices >= 0
        if np.any(valid):
            out[:, valid, :] = selected_coords[:, self.ca_indices[valid], :]
        if center and out.size > 0:
            out = out - out.mean(axis=1, keepdims=True)
        return out

    def to_torch(self, device: torch.device) -> "TorchGeom2VecTopologyCache":
        """Materialize device-resident tensors used during inference."""

        atomic = torch.as_tensor(self.atomic_numbers, dtype=torch.long, device=device)
        cg_map = torch.as_tensor(self.cg_map, dtype=torch.long, device=device)
        cg_counts = torch.as_tensor(self.cg_counts, dtype=torch.float32, device=device)
        ca_idx = torch.as_tensor(self.ca_indices, dtype=torch.long, device=device)
        return TorchGeom2VecTopologyCache(
            atomic_numbers=atomic,
            cg_map=cg_map,
            cg_counts=cg_counts,
            ca_indices=ca_idx,
            num_tokens=self.num_tokens,
        )

    @staticmethod
    def from_mdtraj(
        topology,
        *,
        exclude_hydrogens: bool = True,
        ca_atom_name: str = "CA",
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Geom2VecTopologyCache":
        """Build a cache from an ``mdtraj.Topology``.

        Notes
        -----
        - This method does not perform any atom selection besides optional hydrogen
          exclusion. Pass a protein-only topology (e.g., a DCD's topology) if needed.
        - Tokenization is residue-based: one token per residue that contributes at least
          one selected atom.
        """

        try:
            import mdtraj as md  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("mdtraj is required to build a Geom2VecTopologyCache from topology.") from exc

        if not isinstance(topology, md.Topology):  # type: ignore[attr-defined]
            raise TypeError("topology must be an mdtraj.Topology")

        atom_indices: list[int] = []
        atomic_numbers: list[int] = []
        residue_indices: list[int] = []
        atom_names: list[str] = []

        for atom in topology.atoms:
            element = getattr(atom, "element", None)
            atomic_number = getattr(element, "atomic_number", None)
            element_symbol = getattr(element, "symbol", None)
            if atomic_number is None:
                raise ValueError(
                    "Unable to determine atomic number for atom "
                    f"{getattr(atom, 'name', '?')} in residue {getattr(atom.residue, 'name', '?')} "
                    f"(index={getattr(atom, 'index', '?')})."
                )
            if exclude_hydrogens and _is_hydrogen(int(atomic_number), element_symbol):
                continue
            atom_indices.append(int(atom.index))
            atomic_numbers.append(int(atomic_number))
            residue_indices.append(int(atom.residue.index))
            atom_names.append(str(atom.name))

        if not atom_indices:
            raise ValueError("No atoms selected; cannot build topology cache.")

        # Token list: residues that contributed at least one selected atom, in topology order.
        selected_residues = set(residue_indices)
        token_residue_indices: list[int] = [
            int(r.index) for r in topology.residues if int(r.index) in selected_residues
        ]
        res_to_token = {res_idx: i for i, res_idx in enumerate(token_residue_indices)}

        cg_map = np.asarray([res_to_token[int(ridx)] for ridx in residue_indices], dtype=np.int64)
        num_tokens = int(len(token_residue_indices))
        cg_counts = np.bincount(cg_map, minlength=num_tokens).astype(np.int64)
        if int(cg_counts.sum()) != int(len(atom_indices)):
            raise RuntimeError("Internal error: cg_counts does not sum to selected atom count.")

        # CA atom per residue token (index into selected-atom list).
        orig_to_selected_pos = {orig_idx: pos for pos, orig_idx in enumerate(atom_indices)}
        ca_indices = np.full((num_tokens,), -1, dtype=np.int64)
        ca_name = str(ca_atom_name)
        residue_by_index = {int(r.index): r for r in topology.residues}
        for res_idx in token_residue_indices:
            token = res_to_token[res_idx]
            # Search for CA in this residue in the original topology, then map to selected pos.
            residue = residue_by_index.get(int(res_idx))
            if residue is None:
                continue
            ca_atom = next((a for a in residue.atoms if str(a.name) == ca_name), None)
            if ca_atom is None:
                continue
            sel_pos = orig_to_selected_pos.get(int(ca_atom.index))
            if sel_pos is not None:
                ca_indices[token] = int(sel_pos)

        return Geom2VecTopologyCache(
            atom_indices=np.asarray(atom_indices, dtype=np.int64),
            atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
            cg_map=cg_map,
            cg_counts=cg_counts,
            ca_indices=ca_indices,
            token_residue_indices=np.asarray(token_residue_indices, dtype=np.int64),
            meta=dict(meta or {}),
        )


@dataclass(frozen=True)
class TorchGeom2VecTopologyCache:
    """Torch tensors derived from :class:`Geom2VecTopologyCache`."""

    atomic_numbers: torch.Tensor  # (atoms,) long
    cg_map: torch.Tensor  # (atoms,) long
    cg_counts: torch.Tensor  # (tokens,) float32
    ca_indices: torch.Tensor  # (tokens,) long
    num_tokens: int
