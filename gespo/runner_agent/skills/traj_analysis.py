"""Trajectory analysis skill: RMSD, end-to-end distance, radius of gyration."""
from __future__ import annotations

from typing import Any, Dict

from .base import Skill, ToolResult


class TrajectoryAnalysisSkill(Skill):
    """Compute structural metrics from MD trajectory files."""

    name = "traj_analyze"
    description = (
        "Analyze an MD trajectory: compute RMSD (to reference), end-to-end distance, "
        "and radius of gyration. Returns per-frame and summary statistics."
    )
    schema = {
        "topology": "string (required) — PDB file for topology",
        "trajectory": "string (required) — trajectory file (.dcd, .xtc, .trr)",
        "reference": "string (optional) — reference PDB for RMSD (default: first frame)",
        "stride": "int (optional, default 1) — frame stride",
        "atom_selection": "string (optional, default 'name CA') — MDTraj atom selection",
        "max_frames": "int (optional, default 0) — max frames to load (0=all)",
    }

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            import mdtraj as md
            import numpy as np
        except ImportError as e:
            return ToolResult.error(self.name, f"Missing dependency: {e}")

        topology = str(params.get("topology", ""))
        trajectory = str(params.get("trajectory", ""))
        if not topology or not trajectory:
            return ToolResult.error(self.name, "Both 'topology' and 'trajectory' are required.")

        reference = params.get("reference", "")
        stride = int(params.get("stride", 1) or 1)
        atom_sel = str(params.get("atom_selection", "name CA") or "name CA")
        max_frames = int(params.get("max_frames", 0) or 0)

        try:
            # Load trajectory
            traj = md.load(trajectory, top=topology, stride=stride)
            if max_frames > 0 and len(traj) > max_frames:
                traj = traj[:max_frames]

            n_frames = len(traj)
            n_atoms = traj.n_atoms

            # Atom selection for RMSD
            sel = traj.topology.select(atom_sel)
            if len(sel) == 0:
                return ToolResult.error(self.name, f"Atom selection '{atom_sel}' matched 0 atoms.")

            # Reference structure
            if reference:
                ref = md.load(reference)
            else:
                ref = traj[0]

            # RMSD (nm)
            rmsd = md.rmsd(traj, ref, atom_indices=sel)

            # End-to-end distance
            ca_indices = traj.topology.select("name CA")
            if len(ca_indices) >= 2:
                first_ca, last_ca = ca_indices[0], ca_indices[-1]
                e2e = np.linalg.norm(
                    traj.xyz[:, last_ca, :] - traj.xyz[:, first_ca, :], axis=1
                )
            else:
                e2e = np.zeros(n_frames)

            # Radius of gyration (nm)
            rg = md.compute_rg(traj)

            # Summary
            lines = [
                f"Trajectory: {trajectory}",
                f"Frames: {n_frames}  Atoms: {n_atoms}  Selection: '{atom_sel}' ({len(sel)} atoms)",
                "",
                f"{'Metric':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}",
                "-" * 52,
                f"{'RMSD (nm)':<20s} {rmsd.mean():8.4f} {rmsd.std():8.4f} {rmsd.min():8.4f} {rmsd.max():8.4f}",
                f"{'E2E dist (nm)':<20s} {e2e.mean():8.4f} {e2e.std():8.4f} {e2e.min():8.4f} {e2e.max():8.4f}",
                f"{'Rg (nm)':<20s} {rg.mean():8.4f} {rg.std():8.4f} {rg.min():8.4f} {rg.max():8.4f}",
            ]

            return ToolResult.ok(self.name, "\n".join(lines))

        except Exception as e:
            return ToolResult.error(self.name, f"{type(e).__name__}: {e}")
