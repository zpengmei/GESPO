import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple
from gespo.utils.logging import log_print

import numpy as np
import torch

from ..policies.rl import TorsionARPolicy
from .pretrain_torsion_ar import _build_torsion_dataset_from_dcd, _load_npz_dataset


def _circular_mean_std(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan"), float("nan")
    s = np.sin(x).mean()
    c = np.cos(x).mean()
    mean = float(np.arctan2(s, c))
    R = float(np.sqrt(s * s + c * c))
    # Circular std (approx) in radians; clamp to avoid log(0).
    R = max(min(R, 1.0), 1e-12)
    std = float(np.sqrt(-2.0 * np.log(R)))
    return mean, std


def _hist_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def _load_dataset(
    source_path: str,
    pdb_path: Optional[str],
    stride: int,
    max_frames: int,
    max_files: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if os.path.isdir(source_path):
        if not pdb_path:
            raise ValueError("--pdb is required when --data is a directory of .dcd files")
        torsions_np, angle_mask_np, residue_types_np, global_values_np = _build_torsion_dataset_from_dcd(
            base_dir=source_path,
            pdb_path=pdb_path,
            stride=stride,
        )
    else:
        torsions_np, angle_mask_np, residue_types_np, global_values_np = _load_npz_dataset(source_path)

    if angle_mask_np is None:
        angle_mask_np = np.ones_like(torsions_np, dtype=bool)
    if residue_types_np is None:
        residue_types_np = np.zeros(torsions_np.shape[:2], dtype=np.int64)

    # Optional subsampling/capping to keep evaluation bounded.
    if max_frames > 0 and torsions_np.shape[0] > max_frames:
        torsions_np = torsions_np[:max_frames]
        angle_mask_np = angle_mask_np[:max_frames]
        residue_types_np = residue_types_np[:max_frames]
        if global_values_np is not None:
            global_values_np = global_values_np[:max_frames]

    global_values_out = None
    if global_values_np is not None:
        global_values_out = np.asarray(global_values_np, dtype=np.float32)
    return torsions_np, angle_mask_np.astype(bool), residue_types_np.astype(np.int64), global_values_out


def main():
    ap = argparse.ArgumentParser(description="Evaluate a pretrained TorsionARPolicy on torsion datasets.")
    ap.add_argument("--checkpoint", required=True, help="Path to pretrained TorsionAR checkpoint (.pt payload).")
    ap.add_argument(
        "--data",
        required=True,
        help="Path to torsion dataset: either a .npz file or a directory containing .dcd files.",
    )
    ap.add_argument("--pdb", default="", help="Reference PDB (required if --data is a DCD directory).")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride when loading DCDs.")
    ap.add_argument("--max-frames", type=int, default=0, help="Cap dataset frames (0 = no cap).")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation.")
    ap.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction (0..1).")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for train/val split.")
    ap.add_argument("--device", default="", help="Device override (e.g. cuda, cuda:0, cpu).")
    ap.add_argument("--n-samples", type=int, default=4096, help="How many samples to draw from the model for comparison.")
    ap.add_argument("--bins", type=int, default=72, help="Histogram bins on [-pi, pi].")
    ap.add_argument("--out-dir", default=".", help="Output directory for summary + optional plots.")
    ap.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots.")
    ap.add_argument("--plot-points", type=int, default=20000, help="Max points in each Ramachandran scatter.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(args.checkpoint, map_location=device)
    cfg = payload.get("config", {}) or {}
    num_residues = int(cfg.get("num_residues", 0) or 0)
    residue_vocab_size = int(cfg.get("residue_vocab_size", 21))
    d_model = int(cfg.get("d_model", 256))
    n_layers = int(cfg.get("n_layers", 4))
    n_heads = int(cfg.get("n_heads", 8))
    kmix = int(cfg.get("kmix", 8))
    n_global_tokens = int(cfg.get("n_global_tokens", 0) or 0)

    torsions_np, angle_mask_np, residue_types_np, global_values_np = _load_dataset(
        source_path=args.data,
        pdb_path=args.pdb or None,
        stride=max(1, int(args.stride)),
        max_frames=max(0, int(args.max_frames)),
        max_files=0,
    )

    N, L, C = torsions_np.shape
    if C != 3:
        raise RuntimeError(f"Expected torsions with last dim 3, got {torsions_np.shape}")
    if num_residues and L != num_residues:
        raise RuntimeError(f"Checkpoint expects L={num_residues}, but dataset has L={L}")

    model = TorsionARPolicy(
        num_residues=L,
        residue_vocab_size=residue_vocab_size,
        d_model=d_model,
        nhead=n_heads,
        num_layers=n_layers,
        dropout=0.0,
        kmix=kmix,
        n_global_tokens=n_global_tokens,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    rng = np.random.default_rng(int(args.seed))
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(float(args.val_fraction) * N))
    n_val = max(0, min(N, n_val))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    torsions = torch.from_numpy(torsions_np).float()
    angle_mask = torch.from_numpy(angle_mask_np.astype(np.bool_))
    residue_types = torch.from_numpy(residue_types_np.astype(np.int64))
    global_values = torch.from_numpy(global_values_np) if global_values_np is not None else None

    def _eval_split(split_idx: np.ndarray, name: str) -> dict:
        if split_idx.size == 0:
            return {
                "n": 0,
                "nll_seq_mean": float("nan"),
                "nll_per_angle_mean": float("nan"),
                "n_valid_angles_mean": float("nan"),
            }
        bs = max(1, int(args.batch_size))
        logps = []
        n_valid = []
        with torch.inference_mode():
            for i0 in range(0, split_idx.size, bs):
                sl = split_idx[i0 : i0 + bs]
                t_b = torsions[sl].to(device)
                m_b = angle_mask[sl].to(device)
                r_b = residue_types[sl].to(device)
                if n_global_tokens > 0:
                    if global_values is None:
                        gv_b = torch.zeros((t_b.shape[0], n_global_tokens), dtype=torch.float32, device=device)
                    else:
                        gv_b = global_values[sl].to(device)
                    lp = model.log_prob(t_b, residue_types=r_b, angle_mask=m_b, global_values=gv_b).detach().cpu().numpy()
                    nv = (m_b.sum(dim=(1, 2)) + n_global_tokens).detach().cpu().numpy().astype(np.float64)
                else:
                    lp = model.log_prob(t_b, residue_types=r_b, angle_mask=m_b).detach().cpu().numpy()
                    nv = m_b.sum(dim=(1, 2)).detach().cpu().numpy().astype(np.float64)
                logps.append(lp)
                n_valid.append(nv)

        logps = np.concatenate(logps, axis=0)
        n_valid = np.concatenate(n_valid, axis=0)
        nll_seq = -logps
        nll_per_angle = nll_seq / np.maximum(n_valid, 1.0)

        return {
            "n": int(split_idx.size),
            "nll_seq_mean": float(np.mean(nll_seq)),
            "nll_seq_std": float(np.std(nll_seq)),
            "nll_per_angle_mean": float(np.mean(nll_per_angle)),
            "nll_per_angle_std": float(np.std(nll_per_angle)),
            "n_valid_angles_mean": float(np.mean(n_valid)),
        }

    train_stats = _eval_split(tr_idx, "train")
    val_stats = _eval_split(val_idx, "val")

    baseline_nll_per_angle = float(math.log(2.0 * math.pi))
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "pdb": os.path.abspath(args.pdb) if args.pdb else "",
        "device": str(device),
        "dataset": {"N": int(N), "L": int(L)},
        "train": train_stats,
        "val": val_stats,
        "baseline_uniform": {"nll_per_angle": baseline_nll_per_angle},
        "improvement_vs_uniform": {
            "train": float(baseline_nll_per_angle - train_stats["nll_per_angle_mean"])
            if np.isfinite(train_stats["nll_per_angle_mean"])
            else float("nan"),
            "val": float(baseline_nll_per_angle - val_stats["nll_per_angle_mean"])
            if np.isfinite(val_stats["nll_per_angle_mean"])
            else float("nan"),
        },
        "model_config": cfg,
    }

    # Compare empirical φ/ψ distributions vs model samples (pooled over residues).
    bins = int(args.bins)
    edges = np.linspace(-math.pi, math.pi, bins + 1, dtype=np.float64)

    # Data pooled, masked
    phi_data = torsions_np[..., 0][angle_mask_np[..., 0]]
    psi_data = torsions_np[..., 1][angle_mask_np[..., 1]]

    # Samples pooled, masked (use residue types + mask from the first frame; typically constant).
    n_samp = max(0, int(args.n_samples))
    if n_samp > 0:
        res0 = residue_types[0:1].to(device)
        mask0 = angle_mask[0:1].to(device)
        res_s = res0.expand(n_samp, -1)
        mask_s = mask0.expand(n_samp, -1, -1)
        with torch.inference_mode():
            if n_global_tokens > 0:
                samp, _ = model.sample_with_globals(n_samp, residue_types=res_s, angle_mask=mask_s)
                samp = samp.detach().cpu().numpy()
            else:
                samp = model.sample(n_samp, residue_types=res_s, angle_mask=mask_s).detach().cpu().numpy()
        phi_samp = samp[..., 0][mask_s.detach().cpu().numpy()[..., 0]]
        psi_samp = samp[..., 1][mask_s.detach().cpu().numpy()[..., 1]]
    else:
        phi_samp = np.zeros((0,), dtype=np.float32)
        psi_samp = np.zeros((0,), dtype=np.float32)

    # Histograms for KL
    phi_h_data, _ = np.histogram(phi_data, bins=edges)
    phi_h_samp, _ = np.histogram(phi_samp, bins=edges)
    psi_h_data, _ = np.histogram(psi_data, bins=edges)
    psi_h_samp, _ = np.histogram(psi_samp, bins=edges)

    summary["torsion_stats"] = {
        "phi_data_circ_mean": _circular_mean_std(phi_data)[0],
        "phi_data_circ_std": _circular_mean_std(phi_data)[1],
        "psi_data_circ_mean": _circular_mean_std(psi_data)[0],
        "psi_data_circ_std": _circular_mean_std(psi_data)[1],
        "phi_samp_circ_mean": _circular_mean_std(phi_samp)[0],
        "phi_samp_circ_std": _circular_mean_std(phi_samp)[1],
        "psi_samp_circ_mean": _circular_mean_std(psi_samp)[0],
        "psi_samp_circ_std": _circular_mean_std(psi_samp)[1],
        "phi_kl_data_to_sample": _hist_kl(phi_h_data, phi_h_samp),
        "psi_kl_data_to_sample": _hist_kl(psi_h_data, psi_h_samp),
    }

    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "eval_hists.npz",
        edges=edges,
        phi_hist_data=phi_h_data,
        phi_hist_sample=phi_h_samp,
        psi_hist_data=psi_h_data,
        psi_hist_sample=psi_h_samp,
    )

    log_print("[TorsionAR Eval] Dataset:", f"N={N}, L={L}")
    log_print("[TorsionAR Eval] Train NLL/angle:", f"{train_stats['nll_per_angle_mean']:.4f}",
          "(baseline:", f"{baseline_nll_per_angle:.4f})")
    if val_stats["n"] > 0:
        log_print("[TorsionAR Eval] Val   NLL/angle:", f"{val_stats['nll_per_angle_mean']:.4f}",
              "(baseline:", f"{baseline_nll_per_angle:.4f})")
    log_print("[TorsionAR Eval] KL(data||sample):",
          f"phi={summary['torsion_stats']['phi_kl_data_to_sample']:.4f},",
          f"psi={summary['torsion_stats']['psi_kl_data_to_sample']:.4f}")
    log_print("[TorsionAR Eval] Wrote:", str(out_dir / "eval_summary.json"))

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            log_print(f"[TorsionAR Eval] matplotlib unavailable; skipping plots ({exc})")
            return

        # Ramachandran scatter
        n_plot = int(args.plot_points)
        if n_plot > 0:
            n_plot_data = min(n_plot, phi_data.size, psi_data.size)
            n_plot_samp = min(n_plot, phi_samp.size, psi_samp.size)
            if n_plot_data > 0 and n_plot_samp > 0:
                sel_d = rng.choice(phi_data.size, size=n_plot_data, replace=False)
                sel_s = rng.choice(phi_samp.size, size=n_plot_samp, replace=False)
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.scatter(phi_data[sel_d], psi_data[sel_d], s=1, alpha=0.3, rasterized=True)
                plt.title("Dataset φ/ψ")
                plt.xlabel("φ (rad)")
                plt.ylabel("ψ (rad)")
                plt.xlim(-math.pi, math.pi)
                plt.ylim(-math.pi, math.pi)
                plt.subplot(1, 2, 2)
                plt.scatter(phi_samp[sel_s], psi_samp[sel_s], s=1, alpha=0.3, rasterized=True)
                plt.title("Model samples φ/ψ")
                plt.xlabel("φ (rad)")
                plt.ylabel("ψ (rad)")
                plt.xlim(-math.pi, math.pi)
                plt.ylim(-math.pi, math.pi)
                plt.tight_layout()
                plt.savefig(out_dir / "ramachandran_compare.png", dpi=200)
                plt.close()

        # 1D hist overlays
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(centers, phi_h_data / max(phi_h_data.sum(), 1.0), label="data")
        plt.plot(centers, phi_h_samp / max(phi_h_samp.sum(), 1.0), label="sample")
        plt.title("φ histogram (pooled)")
        plt.xlim(-math.pi, math.pi)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(centers, psi_h_data / max(psi_h_data.sum(), 1.0), label="data")
        plt.plot(centers, psi_h_samp / max(psi_h_samp.sum(), 1.0), label="sample")
        plt.title("ψ histogram (pooled)")
        plt.xlim(-math.pi, math.pi)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "hist_compare.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
