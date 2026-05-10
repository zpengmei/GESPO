"""VAMP/TICA projection skill: project trajectories through geom2vec + VAMP lobe."""
from __future__ import annotations

from typing import Any, Dict

from .base import Skill, ToolResult


class VAMPProjectionSkill(Skill):
    """Project trajectory coordinates through a geom2vec backbone + VAMP lobe."""

    name = "vamp_project"
    description = (
        "Project an MD trajectory through a trained geom2vec backbone and VAMP lobe "
        "to obtain slow-mode coordinates and discrete state assignments. "
        "Requires a geom2vec checkpoint and a VAMP lobe checkpoint."
    )
    schema = {
        "topology": "string (required) — PDB file for topology",
        "trajectory": "string (required) — trajectory file (.dcd, .xtc)",
        "geom2vec_checkpoint": "string (required) — path to geom2vec representation checkpoint",
        "vamp_checkpoint": "string (required) — path to trained VAMP lobe checkpoint (.pt)",
        "output": "string (optional) — save projected coordinates to .npy file",
        "stride": "int (optional, default 1) — frame stride",
        "device": "string (optional, default 'cpu') — torch device",
        "max_frames": "int (optional, default 0) — max frames (0=all)",
    }

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            import numpy as np
            import torch
        except ImportError as e:
            return ToolResult.error(self.name, f"Missing dependency: {e}")

        topology = str(params.get("topology", ""))
        trajectory = str(params.get("trajectory", ""))
        g2v_ckpt = str(params.get("geom2vec_checkpoint", ""))
        vamp_ckpt = str(params.get("vamp_checkpoint", ""))
        if not all([topology, trajectory, g2v_ckpt, vamp_ckpt]):
            return ToolResult.error(
                self.name,
                "All of 'topology', 'trajectory', 'geom2vec_checkpoint', "
                "and 'vamp_checkpoint' are required.",
            )

        output = str(params.get("output", "") or "")
        stride = int(params.get("stride", 1) or 1)
        device = str(params.get("device", "cpu") or "cpu")
        max_frames = int(params.get("max_frames", 0) or 0)

        try:
            import mdtraj as md

            # Load trajectory
            traj = md.load(trajectory, top=topology, stride=stride)
            if max_frames > 0 and len(traj) > max_frames:
                traj = traj[:max_frames]
            n_frames = len(traj)

            # Try to use the GESPO geom2vec backbone infrastructure
            try:
                from gespo.projections.geom2vec.projector import Geom2VecVAMPProjector
                projector = Geom2VecVAMPProjector.from_checkpoints(
                    geom2vec_ckpt=g2v_ckpt,
                    vamp_ckpt=vamp_ckpt,
                    topology_pdb=topology,
                    device=device,
                )
                coords_nm = traj.xyz  # (n_frames, n_atoms, 3)
                projected = projector.project(coords_nm)
                states = projector.assign_states(coords_nm)

            except (ImportError, AttributeError):
                # Fallback: load checkpoints manually
                # Load geom2vec model
                g2v_state = torch.load(g2v_ckpt, map_location=device, weights_only=False)
                vamp_state = torch.load(vamp_ckpt, map_location=device, weights_only=False)

                # If the VAMP checkpoint contains projected data, use it
                if isinstance(vamp_state, dict) and "projected" in vamp_state:
                    projected = vamp_state["projected"]
                    states = np.argmax(projected, axis=1) if projected.ndim == 2 else None
                else:
                    return ToolResult.error(
                        self.name,
                        "Could not load geom2vec projector. Ensure GESPO training modules "
                        "are importable or provide pre-projected data.",
                    )

            n_dim = projected.shape[1] if projected.ndim == 2 else 0

            # Save if requested
            if output:
                np.save(output, projected)

            # Build summary
            lines = [
                f"Trajectory: {trajectory}",
                f"Frames: {n_frames} → Projected: {projected.shape}",
                f"VAMP dimensions: {n_dim}",
                "",
            ]

            if states is not None:
                unique, counts = np.unique(states, return_counts=True)
                lines.append(f"State assignments ({len(unique)} states):")
                for s, c in zip(unique, counts):
                    pct = 100 * c / len(states)
                    lines.append(f"  State {s}: {c:6d} frames ({pct:5.1f}%)")

            if projected.ndim == 2:
                lines.append("")
                lines.append(f"{'Dim':<6s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
                lines.append("-" * 50)
                for d in range(min(n_dim, 10)):
                    col = projected[:, d]
                    lines.append(
                        f"{'VC' + str(d):<6s} {col.mean():10.4f} {col.std():10.4f} "
                        f"{col.min():10.4f} {col.max():10.4f}"
                    )

            if output:
                lines.append(f"\nProjected coordinates saved to {output}")

            return ToolResult.ok(self.name, "\n".join(lines))

        except Exception as e:
            return ToolResult.error(self.name, f"{type(e).__name__}: {e}")
