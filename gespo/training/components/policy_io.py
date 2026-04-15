"""Policy checkpoint I/O for TorsionAR policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def save_torsion_ar_policy(
    ep_idx: int,
    *,
    checkpoint_dir: str,
    agent: torch.nn.Module,
    agent_old: torch.nn.Module,
    ar_cfg: Optional[Dict[str, Any]],
    torsion_ckpt: Optional[str],
    env_ref: Any,
    residue_types_1d: torch.Tensor,
    global_token_names: Optional[List[str]],
) -> None:
    """Save TorsionAR policy checkpoints for the current and reference agents.

    Parameters
    ----------
    ep_idx : int
        1-based epoch index.
    checkpoint_dir : str
        Directory to write checkpoint files. If empty, this is a no-op.
    agent, agent_old : torch.nn.Module
        Current and reference TorsionAR policy networks.
    ar_cfg : dict or None
        Architecture config loaded from a prior checkpoint (if resuming).
    torsion_ckpt : str or None
        Path to the checkpoint used to initialize the policy (if any).
    env_ref : Any
        Reference environment (used for ``n_residues``).
    residue_types_1d : torch.Tensor
        Residue-type integer codes.
    global_token_names : list[str] or None
        Names of global tokens appended to the torsion sequence.
    """
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_dir is None:
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {}
    try:
        if torsion_ckpt:
            cfg = dict(ar_cfg) if isinstance(ar_cfg, dict) else {}
    except Exception:
        cfg = {}
    if not cfg:
        cfg = {
            "num_residues": int(getattr(agent, "num_residues", env_ref.n_residues)),
            "residue_vocab_size": int(
                getattr(agent, "residue_vocab_size", int(residue_types_1d.max()) + 1)
            ),
            "d_model": int(getattr(agent, "d_model", 256)),
            "n_layers": int(getattr(agent, "num_layers", 4)),
            "n_heads": int(getattr(agent, "nhead", 8)),
            "kmix": int(getattr(agent, "kmix", 8)),
            "n_angles": int(getattr(agent, "n_angles", 3) or 3),
            "angle_names": (
                list(getattr(agent, "angle_names", []))
                if getattr(agent, "angle_names", None) is not None
                else None
            ),
            "n_global_tokens": int(getattr(agent, "n_global_tokens", 0) or 0),
        }
    try:
        if global_token_names:
            cfg["global_token_names"] = list(global_token_names)
    except Exception:
        pass

    agent_path = ckpt_dir / f"torsion_ar_ep_{ep_idx:04d}.pt"
    agent_old_path = ckpt_dir / f"torsion_ar_old_ep_{ep_idx:04d}.pt"
    torch.save(
        {"state_dict": agent.state_dict(), "config": cfg, "epoch": ep_idx},
        agent_path,
    )
    torch.save(
        {"state_dict": agent_old.state_dict(), "config": cfg, "epoch": ep_idx},
        agent_old_path,
    )
    print(f"[TorsionAR] Saved policy checkpoints: {agent_path.name}, {agent_old_path.name}")
