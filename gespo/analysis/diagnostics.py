"""Diagnostic plotting extracted from training loops.

All functions are optional — they silently no-op on ImportError (matplotlib)
or any plotting failure, so training is never interrupted.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def plot_committor_scatter(
    z1: np.ndarray,
    z2: np.ndarray,
    q_vals: np.ndarray,
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    epoch: int,
    out_path: str,
    mode: str = "dga",
    is_proxy_fallback: bool = False,
) -> None:
    """Scatter plot of committor values on linear VAMP coordinates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    try:
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(1, 1, 1)
        sc = ax.scatter(z1, z2, c=q_vals, s=4, cmap="viridis",
                        vmin=0.0, vmax=1.0, alpha=0.7, linewidths=0)
        if mask_a is not None and np.any(mask_a):
            ax.scatter(z1[mask_a], z2[mask_a], s=12, c="tab:blue",
                       alpha=0.8, marker="x", linewidths=0.8, label="in A")
        if mask_b is not None and np.any(mask_b):
            ax.scatter(z1[mask_b], z2[mask_b], s=12, c="tab:red",
                       alpha=0.8, marker="x", linewidths=0.8, label="in B")

        title = f"Committor q on linear VAMP space (epoch {epoch})"
        if is_proxy_fallback:
            title += " [proxy fallback]"
        ax.set_title(title)
        ax.set_xlabel("z1 (linear)")
        ax.set_ylabel("z2 (linear)")
        cb = fig.colorbar(sc, ax=ax)
        cb_label = f"q (committor; mode={mode})"
        if is_proxy_fallback:
            cb_label += " [proxy fallback]"
        cb.set_label(cb_label)
        if (mask_a is not None and np.any(mask_a)) or (mask_b is not None and np.any(mask_b)):
            ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[Committor] Saved VAC committor plot: {out_path}", flush=True)
    except Exception as exc:
        print(f"[Committor] Note: skipped plot ({exc})", flush=True)


def plot_adk_feature_pairs(
    theta_nmp_deg: np.ndarray,
    theta_lid_deg: np.ndarray,
    dist_145_52: np.ndarray,
    dist_169_55: np.ndarray,
    endpoint_a: Optional[np.ndarray],
    endpoint_b: Optional[np.ndarray],
    epoch: int,
    out_path: str,
    *,
    theta_nmp_hist: Optional[np.ndarray] = None,
    theta_lid_hist: Optional[np.ndarray] = None,
    dist_145_52_hist: Optional[np.ndarray] = None,
    dist_169_55_hist: Optional[np.ndarray] = None,
    overlay_history: bool = False,
) -> None:
    """ADK domain angle + distance hexbin plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return

    try:
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
        ax0, ax1 = axs[0], axs[1]

        # Angles subplot
        if theta_nmp_hist is not None and theta_lid_hist is not None and theta_nmp_hist.size:
            ax0.hexbin(theta_nmp_hist, theta_lid_hist, gridsize=140,
                       mincnt=1, cmap="Greys", norm=LogNorm(), alpha=0.25)
        hb0 = ax0.hexbin(theta_nmp_deg, theta_lid_deg, gridsize=140,
                         mincnt=1, cmap="magma", norm=LogNorm())
        if endpoint_a is not None:
            ax0.scatter([endpoint_a[0]], [endpoint_a[1]], c="tab:blue", s=50,
                        marker="o", edgecolors="k", linewidths=0.5, label="A")
        if endpoint_b is not None:
            ax0.scatter([endpoint_b[0]], [endpoint_b[1]], c="tab:red", s=50,
                        marker="o", edgecolors="k", linewidths=0.5, label="B")
        ax0.set_xlabel("theta_nmp_deg")
        ax0.set_ylabel("theta_lid_deg")
        suffix = "; +history" if overlay_history else ""
        ax0.set_title(f"ADK domain angles (epoch {epoch}{suffix})")
        fig.colorbar(hb0, ax=ax0).set_label("count (log)")
        ax0.legend(loc="best", frameon=False)

        # Distances subplot
        if dist_145_52_hist is not None and dist_169_55_hist is not None and dist_145_52_hist.size:
            ax1.hexbin(dist_145_52_hist, dist_169_55_hist, gridsize=140,
                       mincnt=1, cmap="Greys", norm=LogNorm(), alpha=0.25)
        hb1 = ax1.hexbin(dist_145_52, dist_169_55, gridsize=140,
                         mincnt=1, cmap="magma", norm=LogNorm())
        if endpoint_a is not None and len(endpoint_a) > 3:
            ax1.scatter([endpoint_a[2]], [endpoint_a[3]], c="tab:blue", s=50,
                        marker="o", edgecolors="k", linewidths=0.5, label="A")
        if endpoint_b is not None and len(endpoint_b) > 3:
            ax1.scatter([endpoint_b[2]], [endpoint_b[3]], c="tab:red", s=50,
                        marker="o", edgecolors="k", linewidths=0.5, label="B")
        ax1.set_xlabel("dist_145_52_A")
        ax1.set_ylabel("dist_169_55_A")
        ax1.set_title(f"ADK distances (epoch {epoch}{suffix})")
        fig.colorbar(hb1, ax=ax1).set_label("count (log)")
        ax1.legend(loc="best", frameon=False)

        title = "ADK feature pairs"
        if overlay_history:
            title += " (epoch in color; history in gray)"
        else:
            title += " (from cached epoch trajectories)"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[ADK Pairs] Saved feature-pair plot: {out_path}", flush=True)
    except Exception as exc:
        msg = str(exc)
        if msg and msg != "disabled":
            print(f"[ADK Pairs] Note: skipped ({exc})", flush=True)
