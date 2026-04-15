"""DGA committor analysis skill: compute forward committor probabilities."""
from __future__ import annotations

from typing import Any, Dict

from .base import Skill, ToolResult


class CommittorAnalysisSkill(Skill):
    """Compute DGA committor probabilities from projected coordinates."""

    name = "committor_analyze"
    description = (
        "Compute forward committor probabilities q+(x) from projected coordinates "
        "using the DGA (Diffusion-map-based Galerkin Approximation) method. "
        "Requires projected trajectory data and endpoint definitions."
    )
    schema = {
        "topology": "string (required) — PDB file for topology",
        "trajectory": "string (required) — trajectory file (.dcd, .xtc)",
        "pdb_a": "string (required) — endpoint A PDB file",
        "pdb_b": "string (required) — endpoint B PDB file",
        "cutoff_a_nm": "float (optional, default 0.2) — RMSD cutoff for state A (nm)",
        "cutoff_b_nm": "float (optional, default 0.2) — RMSD cutoff for state B (nm)",
        "n_clusters": "int (optional, default 100) — k-means clusters for DGA basis",
        "lagtime": "int (optional, default 10) — lag time in frames",
        "stride": "int (optional, default 1) — frame stride",
        "output": "string (optional) — save committor values to .npy file",
        "max_frames": "int (optional, default 0) — max frames (0=all)",
    }

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            import numpy as np
            import mdtraj as md
        except ImportError as e:
            return ToolResult.error(self.name, f"Missing dependency: {e}")

        topology = str(params.get("topology", ""))
        trajectory = str(params.get("trajectory", ""))
        pdb_a = str(params.get("pdb_a", ""))
        pdb_b = str(params.get("pdb_b", ""))
        if not all([topology, trajectory, pdb_a, pdb_b]):
            return ToolResult.error(
                self.name,
                "'topology', 'trajectory', 'pdb_a', and 'pdb_b' are all required.",
            )

        cutoff_a = float(params.get("cutoff_a_nm", 0.2) or 0.2)
        cutoff_b = float(params.get("cutoff_b_nm", 0.2) or 0.2)
        n_clusters = int(params.get("n_clusters", 100) or 100)
        lagtime = int(params.get("lagtime", 10) or 10)
        stride = int(params.get("stride", 1) or 1)
        output = str(params.get("output", "") or "")
        max_frames = int(params.get("max_frames", 0) or 0)

        try:
            # Load trajectory and endpoints
            traj = md.load(trajectory, top=topology, stride=stride)
            if max_frames > 0 and len(traj) > max_frames:
                traj = traj[:max_frames]

            ref_a = md.load(pdb_a)
            ref_b = md.load(pdb_b)

            ca = traj.topology.select("name CA")
            ca_a = ref_a.topology.select("name CA")
            ca_b = ref_b.topology.select("name CA")

            if len(ca) == 0:
                return ToolResult.error(self.name, "No CA atoms found in trajectory.")

            # Compute RMSD to both endpoints
            rmsd_a = md.rmsd(traj, ref_a, atom_indices=ca)
            rmsd_b = md.rmsd(traj, ref_b, atom_indices=ca)

            # Classify frames
            in_a = rmsd_a < cutoff_a
            in_b = rmsd_b < cutoff_b
            n_a = int(in_a.sum())
            n_b = int(in_b.sum())
            n_transition = int((~in_a & ~in_b).sum())

            # Try the GESPO DGA committor code
            try:
                from gespo.training.dga_committor import fit_dga_committor_kmeans

                # Use CA coordinates as features for DGA
                ca_coords = traj.xyz[:, ca, :].reshape(len(traj), -1)

                model = fit_dga_committor_kmeans(
                    ca_coords,
                    state_a_mask=in_a,
                    state_b_mask=in_b,
                    n_clusters=n_clusters,
                    lagtime=lagtime,
                )
                q_plus = model.predict(ca_coords)
                method = "DGA (k-means basis)"

            except (ImportError, Exception) as dga_err:
                # Fallback: RMSD-based proxy committor
                q_plus = _rmsd_proxy_committor(rmsd_a, rmsd_b, cutoff_a, cutoff_b)
                method = f"RMSD proxy (DGA unavailable: {dga_err})"

            # Save if requested
            if output:
                np.save(output, q_plus)

            # Bin statistics
            q_bins = np.linspace(0, 1, 11)
            hist, _ = np.histogram(q_plus[~in_a & ~in_b], bins=q_bins)

            # Build report
            lines = [
                f"Committor Analysis ({method})",
                f"Trajectory: {trajectory}",
                f"Frames: {len(traj)}, Lagtime: {lagtime}",
                "",
                "Endpoint classification:",
                f"  State A (RMSD < {cutoff_a:.2f} nm): {n_a} frames ({100*n_a/len(traj):.1f}%)",
                f"  State B (RMSD < {cutoff_b:.2f} nm): {n_b} frames ({100*n_b/len(traj):.1f}%)",
                f"  Transition region:                  {n_transition} frames ({100*n_transition/len(traj):.1f}%)",
                "",
                f"{'q+ range':<15s} {'Count':>8s} {'Fraction':>10s}",
                "-" * 35,
            ]
            for i in range(len(q_bins) - 1):
                lo, hi = q_bins[i], q_bins[i + 1]
                c = hist[i]
                frac = c / max(n_transition, 1)
                lines.append(f"{lo:.1f} – {hi:.1f}       {c:8d} {frac:10.3f}")

            lines.append("")
            lines.append(
                f"Mean committor (transition region): {q_plus[~in_a & ~in_b].mean():.3f}"
            )

            if output:
                lines.append(f"\nCommittor values saved to {output}")

            return ToolResult.ok(self.name, "\n".join(lines))

        except Exception as e:
            return ToolResult.error(self.name, f"{type(e).__name__}: {e}")


def _rmsd_proxy_committor(
    rmsd_a: "np.ndarray",
    rmsd_b: "np.ndarray",
    cutoff_a: float,
    cutoff_b: float,
) -> "np.ndarray":
    """Simple RMSD-based proxy committor: q+ ≈ rmsd_a / (rmsd_a + rmsd_b)."""
    import numpy as np
    denom = rmsd_a + rmsd_b
    denom = np.where(denom > 0, denom, 1.0)
    q = rmsd_a / denom
    q = np.where(rmsd_a < cutoff_a, 0.0, q)
    q = np.where(rmsd_b < cutoff_b, 1.0, q)
    return q.astype(np.float32)
