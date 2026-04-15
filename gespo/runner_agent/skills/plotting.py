"""Visualization skills: Ramachandran plots, free energy surfaces, trajectory summaries."""
from __future__ import annotations

from typing import Any, Dict

from .base import Skill, ToolResult


class RamachandranPlotSkill(Skill):
    """Generate per-residue Ramachandran plots from a trajectory."""

    name = "plot_ramachandran"
    description = (
        "Generate per-residue Ramachandran (phi/psi) scatter plots from an MD trajectory "
        "and save to an image file."
    )
    schema = {
        "topology": "string (required) — PDB file for topology",
        "trajectory": "string (required) — trajectory file (.dcd, .xtc)",
        "output": "string (required) — output image path (.png, .pdf)",
        "stride": "int (optional, default 1) — frame stride",
        "residues": "string (optional) — comma-separated residue indices (1-based), default: auto-select up to 10",
        "max_frames": "int (optional, default 0) — max frames (0=all)",
    }

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            import mdtraj as md
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            return ToolResult.error(self.name, f"Missing dependency: {e}")

        topology = str(params.get("topology", ""))
        trajectory = str(params.get("trajectory", ""))
        output = str(params.get("output", ""))
        if not topology or not trajectory or not output:
            return ToolResult.error(self.name, "'topology', 'trajectory', and 'output' are required.")

        stride = int(params.get("stride", 1) or 1)
        max_frames = int(params.get("max_frames", 0) or 0)
        residues_str = str(params.get("residues", "") or "")

        try:
            traj = md.load(trajectory, top=topology, stride=stride)
            if max_frames > 0 and len(traj) > max_frames:
                traj = traj[:max_frames]

            _, phi = md.compute_phi(traj)
            _, psi = md.compute_psi(traj)
            phi_deg = np.degrees(phi)
            psi_deg = np.degrees(psi)

            # Determine residues to plot
            n_phi = phi_deg.shape[1]
            n_psi = psi_deg.shape[1]
            n_common = min(n_phi, n_psi)

            if residues_str:
                indices = [int(r.strip()) - 1 for r in residues_str.split(",")]
                indices = [i for i in indices if 0 <= i < n_common]
            else:
                # Auto-select up to 10 evenly spaced
                if n_common <= 10:
                    indices = list(range(n_common))
                else:
                    indices = np.linspace(0, n_common - 1, 10, dtype=int).tolist()

            n_plots = len(indices)
            if n_plots == 0:
                return ToolResult.error(self.name, "No valid residues to plot.")

            ncols = min(n_plots, 5)
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

            for ax_idx, res_idx in enumerate(indices):
                row, col = divmod(ax_idx, ncols)
                ax = axes[row][col]
                ax.scatter(phi_deg[:, res_idx], psi_deg[:, res_idx], s=1, alpha=0.3, c="steelblue")
                ax.set_xlim(-180, 180)
                ax.set_ylim(-180, 180)
                ax.set_xlabel("φ (°)")
                ax.set_ylabel("ψ (°)")
                ax.set_title(f"Residue {res_idx + 1}")
                ax.set_aspect("equal")

            # Hide unused axes
            for ax_idx in range(n_plots, nrows * ncols):
                row, col = divmod(ax_idx, ncols)
                axes[row][col].set_visible(False)

            fig.suptitle("Ramachandran Plots", fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ToolResult.ok(
                self.name,
                f"Ramachandran plot saved to {output}\n"
                f"  Frames: {len(traj)}, Residues plotted: {[i+1 for i in indices]}",
            )
        except Exception as e:
            return ToolResult.error(self.name, f"{type(e).__name__}: {e}")


class FreeEnergySurfaceSkill(Skill):
    """Plot a 2D free energy surface from coordinate data."""

    name = "plot_fes"
    description = (
        "Compute and plot a 2D free energy surface (PMF) from trajectory data. "
        "Can use RMSD vs E2E, phi vs psi, or custom CVs from a NumPy file."
    )
    schema = {
        "topology": "string (optional) — PDB for topology (required if using trajectory)",
        "trajectory": "string (optional) — trajectory file (auto-computes RMSD vs E2E)",
        "data": "string (optional) — .npy/.npz file with 2D CV data (N,2)",
        "cv_x": "string (optional) — column name in npz file for x-axis",
        "cv_y": "string (optional) — column name in npz file for y-axis",
        "output": "string (required) — output image path (.png, .pdf)",
        "bins": "int (optional, default 50) — number of histogram bins per axis",
        "temperature": "float (optional, default 300) — temperature in Kelvin",
        "stride": "int (optional, default 1) — frame stride for trajectory",
        "max_frames": "int (optional, default 0) — max frames (0=all)",
    }

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            return ToolResult.error(self.name, f"Missing dependency: {e}")

        output = str(params.get("output", ""))
        if not output:
            return ToolResult.error(self.name, "'output' path is required.")

        bins = int(params.get("bins", 50) or 50)
        temperature = float(params.get("temperature", 300.0) or 300.0)
        kbt = 8.314e-3 * temperature  # kJ/mol

        try:
            data_file = str(params.get("data", "") or "")
            traj_file = str(params.get("trajectory", "") or "")

            if data_file:
                # Load from numpy file
                if data_file.endswith(".npz"):
                    npz = np.load(data_file)
                    cv_x = str(params.get("cv_x", "") or "")
                    cv_y = str(params.get("cv_y", "") or "")
                    keys = list(npz.keys())
                    if cv_x and cv_y:
                        x_data = npz[cv_x]
                        y_data = npz[cv_y]
                    elif len(keys) >= 2:
                        x_data = npz[keys[0]]
                        y_data = npz[keys[1]]
                    else:
                        arr = npz[keys[0]]
                        x_data, y_data = arr[:, 0], arr[:, 1]
                    xlabel = cv_x or keys[0] if len(keys) > 0 else "CV1"
                    ylabel = cv_y or keys[1] if len(keys) > 1 else "CV2"
                else:
                    arr = np.load(data_file)
                    x_data, y_data = arr[:, 0], arr[:, 1]
                    xlabel, ylabel = "CV1", "CV2"
            elif traj_file:
                # Compute RMSD vs E2E from trajectory
                import mdtraj as md
                topo = str(params.get("topology", ""))
                if not topo:
                    return ToolResult.error(self.name, "'topology' required with 'trajectory'.")
                stride = int(params.get("stride", 1) or 1)
                max_frames = int(params.get("max_frames", 0) or 0)
                traj = md.load(traj_file, top=topo, stride=stride)
                if max_frames > 0 and len(traj) > max_frames:
                    traj = traj[:max_frames]
                ca = traj.topology.select("name CA")
                x_data = md.rmsd(traj, traj[0], atom_indices=ca)
                ca_idx = traj.topology.select("name CA")
                y_data = np.linalg.norm(
                    traj.xyz[:, ca_idx[-1], :] - traj.xyz[:, ca_idx[0], :], axis=1
                )
                xlabel, ylabel = "RMSD (nm)", "E2E dist (nm)"
            else:
                return ToolResult.error(self.name, "Either 'data' or 'trajectory' is required.")

            # Compute 2D histogram → free energy
            H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
            H = H.T
            H = np.where(H > 0, H, np.nan)
            fes = -kbt * np.log(H / np.nanmax(H))
            fes -= np.nanmin(fes)

            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(
                fes, origin="lower", extent=extent, aspect="auto",
                cmap="viridis", vmin=0,
            )
            cb = fig.colorbar(im, ax=ax, label="ΔG (kJ/mol)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Free Energy Surface (T={temperature:.0f} K)")
            fig.tight_layout()
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ToolResult.ok(
                self.name,
                f"Free energy surface saved to {output}\n"
                f"  Frames: {len(x_data)}, Bins: {bins}×{bins}, T={temperature:.0f} K\n"
                f"  FES range: 0 – {np.nanmax(fes):.1f} kJ/mol",
            )
        except Exception as e:
            return ToolResult.error(self.name, f"{type(e).__name__}: {e}")
