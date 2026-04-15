"""Dihedral angle computation skill: backbone φ/ψ/ω and sidechain χ."""
from __future__ import annotations

from typing import Any, Dict

from .base import Skill, ToolResult


class DihedralSkill(Skill):
    """Compute backbone and sidechain dihedral angles from trajectories."""

    name = "compute_dihedrals"
    description = (
        "Compute backbone (phi, psi, omega) and optionally sidechain (chi) dihedral "
        "angles from an MD trajectory. Returns per-residue circular statistics."
    )
    schema = {
        "topology": "string (required) — PDB file for topology",
        "trajectory": "string (required) — trajectory file (.dcd, .xtc)",
        "stride": "int (optional, default 1) — frame stride",
        "sidechain": "bool (optional, default false) — include sidechain chi angles",
        "residues": "string (optional) — comma-separated residue indices (1-based) to include",
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

        stride = int(params.get("stride", 1) or 1)
        include_sidechain = bool(params.get("sidechain", False))
        max_frames = int(params.get("max_frames", 0) or 0)
        residues_str = str(params.get("residues", "") or "")

        try:
            traj = md.load(trajectory, top=topology, stride=stride)
            if max_frames > 0 and len(traj) > max_frames:
                traj = traj[:max_frames]

            n_frames = len(traj)

            # Compute phi/psi
            _, phi = md.compute_phi(traj)
            _, psi = md.compute_psi(traj)

            phi_deg = np.degrees(phi)
            psi_deg = np.degrees(psi)

            # Residue filtering
            phi_res = list(range(1, phi_deg.shape[1] + 1))  # phi starts at residue 1
            psi_res = list(range(0, psi_deg.shape[1]))  # psi starts at residue 0

            if residues_str:
                try:
                    selected = [int(r.strip()) for r in residues_str.split(",")]
                except ValueError:
                    return ToolResult.error(self.name, f"Invalid residues format: {residues_str}")
            else:
                selected = None

            # Build report
            lines = [
                f"Trajectory: {trajectory}",
                f"Frames: {n_frames}",
                "",
                f"{'Residue':<10s} {'Angle':<6s} {'Mean (°)':>10s} {'Std (°)':>10s} {'Circ.Var':>10s}",
                "-" * 50,
            ]

            def _circ_stats(angles_rad):
                """Circular mean, std, and variance."""
                c = np.cos(angles_rad)
                s = np.sin(angles_rad)
                mean_c, mean_s = c.mean(), s.mean()
                R = np.sqrt(mean_c**2 + mean_s**2)
                mean_ang = np.degrees(np.arctan2(mean_s, mean_c))
                circ_var = 1 - R
                circ_std = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-10))))
                return mean_ang, circ_std, circ_var

            for i, res_id in enumerate(phi_res):
                if selected and res_id not in selected:
                    continue
                m, s, v = _circ_stats(phi[:, i])
                lines.append(f"{'Res ' + str(res_id):<10s} {'φ':<6s} {m:10.1f} {s:10.1f} {v:10.3f}")

            for i, res_id in enumerate(psi_res):
                if selected and res_id not in selected:
                    continue
                m, s, v = _circ_stats(psi[:, i])
                lines.append(f"{'Res ' + str(res_id):<10s} {'ψ':<6s} {m:10.1f} {s:10.1f} {v:10.3f}")

            if include_sidechain:
                try:
                    _, chi1 = md.compute_chi1(traj)
                    if chi1.shape[1] > 0:
                        lines.append("")
                        lines.append("Sidechain χ1:")
                        for i in range(min(chi1.shape[1], 30)):
                            m, s, v = _circ_stats(chi1[:, i])
                            lines.append(f"{'  χ1_' + str(i):<10s} {'χ1':<6s} {m:10.1f} {s:10.1f} {v:10.3f}")
                except Exception:
                    lines.append("(sidechain chi1 computation failed)")

            return ToolResult.ok(self.name, "\n".join(lines))

        except Exception as e:
            return ToolResult.error(self.name, f"{type(e).__name__}: {e}")
