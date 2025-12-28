from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .topology import Geom2VecTopologyCache


def _resolve_hidden_channels(model, hidden_channels: Optional[int]) -> int:
    if hidden_channels is not None:
        return int(hidden_channels)

    visited: set[int] = set()

    def _walk(obj) -> int:
        obj_id = id(obj)
        if obj_id in visited:
            raise ValueError("Unable to infer hidden_channels from model.")
        visited.add(obj_id)

        for attr in ("hidden_channels", "embedding_dimension"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return int(value)

        module = getattr(obj, "module", None)
        if module is not None and module is not obj:
            try:
                return _walk(module)
            except ValueError:
                pass

        children = getattr(obj, "children", None)
        if callable(children):
            for child in children():
                try:
                    return _walk(child)
                except ValueError:
                    continue

        raise ValueError("Unable to infer hidden_channels from model.")

    return _walk(model)


def _resolve_device(model, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        param = next(model.parameters())
    except Exception:
        module = getattr(model, "module", None)
        if module is not None and module is not model:
            return _resolve_device(module, None)
        return torch.device("cpu")
    return param.device


def _coarse_grain(
    atom_rep: torch.Tensor,
    *,
    cg_map: torch.Tensor,
    cg_counts: torch.Tensor,
    num_tokens: int,
    reduction: str,
) -> torch.Tensor:
    """Residue/token aggregation along the atom dimension."""

    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'.")
    if cg_map.ndim != 1:
        raise ValueError("cg_map must be a 1D tensor.")
    if cg_map.shape[0] != atom_rep.shape[1]:
        raise ValueError("cg_map length must match atom dimension of atom_rep.")
    if cg_counts.ndim != 1 or int(cg_counts.shape[0]) != int(num_tokens):
        raise ValueError("cg_counts must have shape (num_tokens,).")

    out = torch.zeros(
        (atom_rep.shape[0], num_tokens, atom_rep.shape[2], atom_rep.shape[3]),
        device=atom_rep.device,
        dtype=atom_rep.dtype,
    )
    out.index_add_(1, cg_map, atom_rep)
    if reduction == "mean":
        denom = cg_counts.view(1, num_tokens, 1, 1).clamp_min(1.0)
        out = out / denom
    return out


@torch.no_grad()
def infer_geom2vec_embeddings(
    model,
    topology: Geom2VecTopologyCache,
    coords: np.ndarray,
    *,
    coords_in_nm: bool = True,
    stride: int = 1,
    batch_size: int = 32,
    reduction: str = "sum",
    hidden_channels: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    center_ca: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer geom2vec token embeddings + CA coords for a single trajectory.

    Returns
    -------
    (graph_features, ca_coords)
        - ``graph_features``: float32 CPU tensor with shape ``(T, num_tokens, 4, H)``
        - ``ca_coords``: float32 CPU tensor with shape ``(T, num_tokens, 3)``
    """

    if stride <= 0:
        raise ValueError("stride must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    resolved_hidden = _resolve_hidden_channels(model, hidden_channels)
    resolved_device = _resolve_device(model, device)
    model = model.to(device=resolved_device).eval()

    coords_np = np.asarray(coords, dtype=np.float32)
    if coords_np.ndim != 3 or coords_np.shape[-1] != 3:
        raise ValueError("coords must have shape (frames, atoms, 3).")

    coords_np = coords_np[:: int(stride)]
    coords_sel = topology.select_atoms(coords_np)
    if coords_in_nm:
        coords_sel = coords_sel * 10.0  # nm -> Å

    ca_np = topology.extract_ca_coords(coords_sel, center=center_ca, dtype=np.float32)
    ca_cpu = torch.from_numpy(ca_np).to(dtype=torch.float32, device="cpu").contiguous()

    traj_tensor = torch.from_numpy(coords_sel).to(device=resolved_device, dtype=torch.float32)
    torch_top = topology.to_torch(resolved_device)

    n_frames = int(traj_tensor.shape[0])
    n_atoms = int(traj_tensor.shape[1])
    num_tokens = int(topology.num_tokens)

    outputs = []
    n_batches = max(1, int(math.ceil(n_frames / batch_size)))
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_frames)
        pos_batch = traj_tensor[start:end]
        n_samples = int(pos_batch.shape[0])

        z_batch = torch_top.atomic_numbers.expand(n_samples, -1).reshape(-1)
        batch_batch = (
            torch.arange(n_samples, device=resolved_device)
            .unsqueeze(1)
            .expand(-1, n_atoms)
            .reshape(-1)
        )

        out = model(
            z=z_batch,
            pos=pos_batch.reshape(-1, 3).contiguous(),
            batch=batch_batch,
        )
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise TypeError("Expected model(z, pos, batch) to return at least (x_rep, v_rep, ...).")
        x_rep, v_rep = out[0], out[1]

        x_rep = x_rep.reshape(-1, n_atoms, 1, resolved_hidden)
        v_rep = v_rep.reshape(-1, n_atoms, 3, resolved_hidden)
        atom_rep = torch.cat([x_rep, v_rep], dim=-2)

        cg_rep = _coarse_grain(
            atom_rep,
            cg_map=torch_top.cg_map,
            cg_counts=torch_top.cg_counts,
            num_tokens=num_tokens,
            reduction=reduction,
        )
        outputs.append(cg_rep.detach().to(device="cpu", dtype=torch.float32))

    graph_cpu = torch.cat(outputs, dim=0).contiguous()
    return graph_cpu, ca_cpu

