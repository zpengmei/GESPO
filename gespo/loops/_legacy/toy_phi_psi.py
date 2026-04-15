"""
Toy training loop: optimize a simple (phi, psi) start distribution using GRPO.

This is intended for minimal systems like Ala2 (ACE-ALA-NME) where we want:
- Start policy: independent Von Mises factors for (phi, psi)
- Reset: replay from reference coordinates + OpenMM targeted torsion steering
- Objective: same wiring style as the generative loop:
    ΔLOO-VAMP2  +  (optional) VAMP-bin bonus  +  (optional) MSM-IG bonus
- Per-epoch Ramachandran plot.

Unlike the full generative loop, this toy loop does not require geom2vec; it
computes VAMP/bonuses directly from the sampled torsion time series.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import TrainingConfig
from ..sampling import PeptideSampler
from ..utils.openmm_utils import resolve_device
from ..utils.plotting import plot_ramachandran_by_residue
from .optimization import GRPOOptimizer
from .rl import VonMisesStartPolicy
from .vamp_stats import SufficientStatistics

try:
    from scipy.special import digamma, gammaln  # type: ignore
except Exception:  # pragma: no cover
    digamma = None
    gammaln = None


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def _wrapped_abs_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = _wrap_pi(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return np.abs(d)


def _vamp_bin_indices(z2_standardized: np.ndarray, *, bins: int, clip: float) -> Tuple[np.ndarray, np.ndarray]:
    z2 = np.asarray(z2_standardized, dtype=np.float64).reshape(-1, 2)
    b = int(max(1, int(bins)))
    clip_v = float(max(1e-12, float(clip)))
    zc = np.clip(z2, -clip_v, clip_v)
    u = (zc + clip_v) / (2.0 * clip_v)  # [0,1]
    idx = np.floor(u * float(b)).astype(np.int64)
    idx = np.clip(idx, 0, b - 1)
    return idx[:, 0], idx[:, 1]


def _msm_transition_counts(labels: np.ndarray, *, n_states: int, lag: int) -> Tuple[np.ndarray, int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    k = int(n_states)
    lag_i = int(lag)
    if labels.size <= lag_i or k <= 0 or lag_i <= 0:
        return np.zeros((k, k), dtype=np.int64), 0
    a = labels[: int(labels.size - lag_i)]
    b = labels[int(lag_i) :]
    idx = a * int(k) + b
    flat = np.bincount(idx, minlength=int(k) * int(k))
    return flat.reshape((int(k), int(k))), int(a.shape[0])


def _msm_ig_reward_from_counts(*, base_counts: np.ndarray, delta_counts: np.ndarray, prior_alpha: float) -> float:
    if digamma is None or gammaln is None:
        return 0.0
    base = np.asarray(base_counts, dtype=np.float64)
    delta = np.asarray(delta_counts, dtype=np.float64)
    if base.shape != delta.shape or base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("MSM count matrices must be square and have the same shape.")
    if float(prior_alpha) <= 0.0:
        raise ValueError("prior_alpha must be > 0.")
    k = int(base.shape[0])
    if k <= 0:
        return 0.0
    if float(np.sum(delta)) <= 0.0:
        return 0.0

    alpha_q = base + float(prior_alpha)
    alpha_p = alpha_q + delta
    sum_p = np.sum(alpha_p, axis=1)
    sum_q = np.sum(alpha_q, axis=1)
    term1 = gammaln(sum_p) - gammaln(sum_q)
    term2 = -np.sum(gammaln(alpha_p) - gammaln(alpha_q), axis=1)
    term3 = np.sum(delta * (digamma(alpha_p) - digamma(sum_p)[:, None]), axis=1)
    kl_rows = term1 + term2 + term3
    kl = float(np.sum(kl_rows))
    if not np.isfinite(kl):
        return 0.0
    return float(max(0.0, kl))


@dataclass
class _ToyKMeans:
    centers: np.ndarray  # [K, D]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        c = np.asarray(self.centers, dtype=np.float64)
        if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
            raise ValueError("Shape mismatch for k-means prediction.")
        d2 = np.sum((x[:, None, :] - c[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)


def _fit_kmeans(z: np.ndarray, *, k: int, seed: int) -> Optional[Any]:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < int(k) or int(k) <= 1:
        return None

    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore

        bs = int(min(2048, max(256, int(z.shape[0]))))
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=int(seed),
            batch_size=int(bs),
            n_init=1,
            max_iter=100,
        )
        model.fit(z)
        return model
    except Exception:
        pass

    try:
        from scipy.cluster.vq import kmeans2  # type: ignore

        centers, _ = kmeans2(z, int(k), minit="points", iter=25, seed=int(seed))
        return _ToyKMeans(centers=np.asarray(centers, dtype=np.float64))
    except Exception:
        return None


def train(args: argparse.Namespace) -> None:
    config = TrainingConfig.from_args(args)
    if not getattr(config, "toy_phi_psi_loop", False):
        raise ValueError("loop_toy_phi_psi.train called but toy_phi_psi_loop is not enabled.")

    torch_device = "cuda" if torch.cuda.is_available() and config.platform != "CPU" else "cpu"
    gen_device = resolve_device(config.gen_device)

    snapshot_dir = Path(config.snapshot_dir or ".")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dcd_dir = snapshot_dir / "run_snaps"
    dcd_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = snapshot_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("GESPO Toy Loop (phi/psi) with GRPO + ΔLOO-VAMP2 + VAMP-bin + MSM-IG")
    print(f"PDB: {config.pdb}")
    print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")

    print("Initializing environments...")
    envs: List[PeptideSampler] = []
    for env_idx in range(int(config.grp_size)):
        envs.append(PeptideSampler(args, generator_device=gen_device))

    env_ref = envs[0]
    cv_names = list(getattr(env_ref, "cv_names", []))
    if len(cv_names) < 2:
        raise RuntimeError(
            f"Toy loop requires at least 2 CVs (phi, psi). Found {len(cv_names)} CVs: {cv_names}.\n"
            "If using capped peptides (ACE/NME), ensure find_all_backbone_dihedrals detects caps correctly."
        )

    # Pick a single residue that has both phi and psi.
    residue_ids = sorted(
        {int(n.split("_")[1]) for n in cv_names if n.startswith("phi_")}
        & {int(n.split("_")[1]) for n in cv_names if n.startswith("psi_")}
    )
    if not residue_ids:
        raise RuntimeError(f"Could not infer a residue with both phi and psi from CV names: {cv_names}")
    residue_id = int(residue_ids[0])
    phi_name = f"phi_{residue_id}"
    psi_name = f"psi_{residue_id}"
    try:
        phi_idx = int(cv_names.index(phi_name))
        psi_idx = int(cv_names.index(psi_name))
    except ValueError as exc:
        raise RuntimeError(f"Missing required CVs {phi_name}/{psi_name} in {cv_names}") from exc

    # Start policy: independent Von Mises for (phi, psi).
    agent = VonMisesStartPolicy(
        action_dim=2,
        kappa_floor=float(getattr(config, "toy_start_kappa_floor", 1e-3) or 1e-3),
        log_kappa_init=float(getattr(config, "toy_start_log_kappa_init", 0.0) or 0.0),
        train_kappa=bool(getattr(config, "toy_start_train_kappa", True)),
        kappa_max=(
            None
            if float(getattr(config, "toy_start_kappa_max", 0.0) or 0.0) <= 0.0
            else float(getattr(config, "toy_start_kappa_max", 0.0) or 0.0)
        ),
    ).to(torch_device)
    agent_old = VonMisesStartPolicy(
        action_dim=2,
        kappa_floor=float(getattr(config, "toy_start_kappa_floor", 1e-3) or 1e-3),
        log_kappa_init=float(getattr(config, "toy_start_log_kappa_init", 0.0) or 0.0),
        train_kappa=bool(getattr(config, "toy_start_train_kappa", True)),
        kappa_max=(
            None
            if float(getattr(config, "toy_start_kappa_max", 0.0) or 0.0) <= 0.0
            else float(getattr(config, "toy_start_kappa_max", 0.0) or 0.0)
        ),
    ).to(torch_device)
    with torch.no_grad():
        try:
            base_cv = np.asarray(env_ref.get_cvs(), dtype=np.float64).reshape(-1)
            if base_cv.size > max(phi_idx, psi_idx):
                agent.mu.copy_(torch.tensor([float(base_cv[phi_idx]), float(base_cv[psi_idx])], device=torch_device))
                agent_old.mu.copy_(agent.mu)
        except Exception:
            pass
    agent_old.load_state_dict(agent.state_dict())

    policy_opt = torch.optim.Adam(agent.parameters(), lr=float(config.lr))
    optimizer = GRPOOptimizer(config, agent, agent_old, [policy_opt])

    loo_stats = SufficientStatistics(dim=4)
    loo_keys: set[str] = set()

    vamp_bin_mu: Optional[np.ndarray] = None
    vamp_bin_std: Optional[np.ndarray] = None
    vamp_bin_counts: Optional[np.ndarray] = None
    vamp_bin_total: int = 0
    vamp_bin_counted_keys: set[str] = set()
    vamp_bin_signature: Optional[Tuple[int, float, int]] = None  # (bins, clip, max_samples)
    vamp_bin_norm_ema_m1: Optional[float] = None
    vamp_bin_norm_ema_m2: Optional[float] = None
    vamp_bin_norm_ema_scale: Optional[float] = None

    msm_ig_mu: Optional[np.ndarray] = None
    msm_ig_std: Optional[np.ndarray] = None
    msm_ig_counts: Optional[np.ndarray] = None
    msm_ig_total_pairs: int = 0
    msm_ig_counted_keys: set[str] = set()
    msm_ig_signature: Optional[Tuple[int, int, int, int]] = None  # (K, dim, lag, max_samples)
    msm_ig_kmeans: Optional[Any] = None
    msm_ig_norm_ema_m1: Optional[float] = None
    msm_ig_norm_ema_m2: Optional[float] = None
    msm_ig_norm_ema_scale: Optional[float] = None

    # Cache torsion time series for history statistics.
    history_phi_psi: Dict[str, np.ndarray] = {}
    history_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    lag = int(getattr(config, "toy_lag", 1) or 1)
    lag = max(1, lag)

    # Reward coefficients: match generative-loop semantics.
    loo_weight = float(getattr(config, "loo_vamp_reward_weight", 1.0) or 1.0)
    vamp_bin_coef = float(getattr(config, "vamp_bin_entropy_coef", -1.0) or -1.0)
    if vamp_bin_coef < 0:
        vamp_bin_coef = float(getattr(config, "ent_coef", 0.0) or 0.0)
    msm_ig_coef = float(getattr(config, "msm_ig_reward_coef", 0.0) or 0.0)

    # Reference coords (nm) used for replay resets.
    ref_coords_nm = getattr(env_ref, "_ref_positions_nm_full", None)
    if ref_coords_nm is not None:
        ref_coords_nm = np.asarray(ref_coords_nm, dtype=np.float32).reshape(-1, 3)

    stats_path = snapshot_dir / "toy_phi_psi_stats.csv"
    if not stats_path.exists():
        stats_path.write_text(
            "epoch,avg_loo,std_loo,avg_vamp_bin,avg_msm_ig,avg_reward,avg_reset_mae,valid\n"
        )

    for ep in range(int(config.episodes)):
        t0 = time.perf_counter()

        # Sample start actions from the old policy.
        with torch.no_grad():
            dist_old = agent_old.start_distribution(int(config.grp_size))
            actions = dist_old.sample()  # [B,2]
        actions_np = actions.detach().cpu().numpy().astype(np.float64, copy=False)
        actions_np = _wrap_pi(actions_np)

        # Execute rollouts.
        epoch_cvs: List[np.ndarray] = []
        epoch_phi_psi: List[Optional[np.ndarray]] = []
        epoch_phi_psi_by_key: Dict[str, np.ndarray] = {}
        epoch_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        epoch_keys: List[str] = []
        valid_mask = np.zeros((int(config.grp_size),), dtype=bool)
        reset_mae = np.zeros((int(config.grp_size),), dtype=np.float32)

        for w_idx, env in enumerate(envs):
            key = f"epoch_{ep+1:04d}_walker_{w_idx:02d}"
            epoch_keys.append(key)

            phi_tgt = float(actions_np[w_idx, 0])
            psi_tgt = float(actions_np[w_idx, 1])

            # Build CV target vector (keep other CVs at their current/reference values).
            cv_targets = np.zeros((len(cv_names),), dtype=np.float64)
            cv_targets[phi_idx] = phi_tgt
            cv_targets[psi_idx] = psi_tgt

            # Trigger torsion steering using replay reset (explicit_coords_nm) with a dummy global target.
            # Recommend setting steer_k_rmsd=0 steer_k_e2e=0 steer_global_steps=0 in the launcher.
            try:
                env.reset(
                    cv_targets.tolist(),
                    explicit_coords_nm=(ref_coords_nm.copy() if ref_coords_nm is not None else None),
                    force_pdb_coords=False,
                    target_globals_nm=np.asarray([0.0], dtype=np.float32),
                    target_global_names=["rmsd_nm"],
                )
                realized = np.asarray(env.cv_values, dtype=np.float64).reshape(-1)
                mae = float(
                    np.mean(
                        _wrapped_abs_err(
                            realized[[phi_idx, psi_idx]],
                            np.asarray([phi_tgt, psi_tgt], dtype=np.float64),
                        )
                    )
                )
                reset_mae[w_idx] = float(mae)
            except Exception as exc_reset:
                print(f"[Epoch {ep+1}] Walker {w_idx}: reset failed ({exc_reset}); skipping.", flush=True)
                epoch_phi_psi.append(None)
                epoch_pairs.append(None)
                continue

            try:
                traj_data, is_stable, _ = env.step(
                    height_scales=[0.0] * int(config.num_mixtures),
                    sigmas=[[float(max(config.sigma_min, 1e-3))] * int(env.n_cvs) for _ in range(int(config.num_mixtures))],
                    record_frequency=int(config.record_stride),
                    output_dir=dcd_dir,
                    epoch_index=ep + 1,
                    walker_idx=w_idx,
                    equilibration_steps=int(config.equilibration_steps),
                )
            except Exception as exc_step:
                print(f"[Epoch {ep+1}] Walker {w_idx}: rollout failed ({exc_step}); skipping.", flush=True)
                epoch_phi_psi.append(None)
                epoch_pairs.append(None)
                continue

            if not is_stable or not traj_data:
                epoch_phi_psi.append(None)
                epoch_pairs.append(None)
                continue

            cvs = np.asarray([fr[0] for fr in traj_data], dtype=np.float64)
            epoch_cvs.append(cvs.astype(np.float32, copy=False))
            valid_mask[w_idx] = True

            phi = _wrap_pi(cvs[:, phi_idx])
            psi = _wrap_pi(cvs[:, psi_idx])
            phi_psi = np.stack([phi, psi], axis=1).astype(np.float32, copy=False)
            epoch_phi_psi.append(phi_psi)
            epoch_phi_psi_by_key[key] = phi_psi

            x = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=1).astype(np.float64, copy=False)
            if x.shape[0] <= lag:
                epoch_pairs.append(None)
            else:
                epoch_pairs.append((x[:-lag], x[lag:]))

        # Per-epoch Ramachandran plot.
        try:
            if epoch_cvs:
                all_cvs_ep = np.concatenate(epoch_cvs, axis=0)
                plot_ramachandran_by_residue(
                    all_cvs_ep,
                    cv_names,
                    str(plot_dir / f"ramachandran_epoch_{ep+1:04d}.png"),
                    residue_ids=[residue_id],
                )
        except Exception as exc_plot:
            print(f"[Epoch {ep+1}] Warning: failed to plot Ramachandran ({exc_plot})")

        # ------------------------------------------------------------------
        # LOO-VAMP2 reward (same contribution computation pattern as generative loop).
        # ------------------------------------------------------------------
        for key, pair in zip(epoch_keys, epoch_pairs):
            if pair is None or key in loo_keys:
                continue
            loo_stats.add_trajectory(pair[0], pair[1])
            loo_keys.add(key)
            # Store for history seeding (used only for plotting/diagnostics).
            history_pairs[key] = pair

        vamp_baseline = float(loo_stats.compute_vamp2_score())
        loo_rewards = np.zeros((int(config.grp_size),), dtype=np.float32)
        for i, (key, pair) in enumerate(zip(epoch_keys, epoch_pairs)):
            if not valid_mask[i] or pair is None:
                continue
            try:
                loo_stats.remove_trajectory(pair[0], pair[1])
                vamp_wo = float(loo_stats.compute_vamp2_score())
                loo_rewards[i] = float(vamp_baseline - vamp_wo)
            finally:
                loo_stats.add_trajectory(pair[0], pair[1])

        # ------------------------------------------------------------------
        # VAMP-bin bonus (phi/psi space): same formula + normalization wiring.
        # ------------------------------------------------------------------
        vamp_bin_bonus_raw = None
        if vamp_bin_coef > 0.0:
            bins_v = int(getattr(config, "vamp_bin_bins", 50) or 50)
            clip_v = float(getattr(config, "vamp_bin_clip", 3.0) or 3.0)
            max_samples = int(getattr(config, "vamp_bin_max_samples", 20000) or 20000)
            sig = (int(bins_v), float(clip_v), int(max_samples))

            if vamp_bin_signature != sig or vamp_bin_counts is None:
                vamp_bin_signature = sig
                vamp_bin_mu = None
                vamp_bin_std = None
                vamp_bin_counts = np.zeros((int(bins_v), int(bins_v)), dtype=np.int64)
                vamp_bin_total = 0
                vamp_bin_counted_keys = set()
                vamp_bin_norm_ema_m1 = None
                vamp_bin_norm_ema_m2 = None
                vamp_bin_norm_ema_scale = None

            if vamp_bin_mu is None or vamp_bin_std is None:
                # Estimate mean/std from history only (exclude current epoch).
                samples: List[np.ndarray] = []
                n = 0
                for arr in history_phi_psi.values():
                    if arr is None or arr.size == 0:
                        continue
                    remaining = max_samples - n
                    if remaining <= 0:
                        break
                    take = arr[:remaining]
                    samples.append(np.asarray(take, dtype=np.float64))
                    n += int(take.shape[0])
                if samples:
                    cat = np.concatenate(samples, axis=0)
                    mu = cat.mean(axis=0)
                    std = cat.std(axis=0)
                    std = np.where(std > 1e-6, std, 1.0)
                    vamp_bin_mu = mu.astype(np.float32, copy=False)
                    vamp_bin_std = std.astype(np.float32, copy=False)
                else:
                    vamp_bin_mu = np.zeros((2,), dtype=np.float32)
                    vamp_bin_std = np.ones((2,), dtype=np.float32)

            bonuses = np.zeros((int(config.grp_size),), dtype=np.float32)
            if vamp_bin_counts is not None and vamp_bin_mu is not None and vamp_bin_std is not None:
                mu2 = np.asarray(vamp_bin_mu, dtype=np.float32).reshape(2)
                std2 = np.asarray(vamp_bin_std, dtype=np.float32).reshape(2)
                std2 = np.where(std2 > 1e-8, std2, 1.0).astype(np.float32, copy=False)

                # Ensure history bin counts are seeded exactly once (for signature init/reset).
                for key_h, arr_h in history_phi_psi.items():
                    if key_h in vamp_bin_counted_keys:
                        continue
                    if arr_h is None or arr_h.size == 0:
                        vamp_bin_counted_keys.add(key_h)
                        continue
                    z2 = np.asarray(arr_h, dtype=np.float32)
                    z2s = (z2 - mu2[None, :]) / std2[None, :]
                    ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                    np.add.at(vamp_bin_counts, (ix, iy), 1)
                    vamp_bin_total += int(ix.shape[0])
                    vamp_bin_counted_keys.add(key_h)

                # Bonus computed against history counts (before updating with current epoch).
                for i, arr in enumerate(epoch_phi_psi):
                    if not valid_mask[i] or arr is None or arr.size == 0:
                        continue
                    z2 = np.asarray(arr, dtype=np.float32)
                    z2s = (z2 - mu2[None, :]) / std2[None, :]
                    ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                    c = vamp_bin_counts[ix, iy].astype(np.float64, copy=False)
                    bonuses[i] = float((1.0 / np.sqrt(c + 1.0)).mean())

                vamp_bin_bonus_raw = bonuses.copy()

                # Update counts once per key.
                for i, (key, arr) in enumerate(zip(epoch_keys, epoch_phi_psi)):
                    if not valid_mask[i] or arr is None or arr.size == 0:
                        continue
                    if key in vamp_bin_counted_keys:
                        continue
                    z2 = np.asarray(arr, dtype=np.float32)
                    z2s = (z2 - mu2[None, :]) / std2[None, :]
                    ix, iy = _vamp_bin_indices(z2s, bins=bins_v, clip=clip_v)
                    np.add.at(vamp_bin_counts, (ix, iy), 1)
                    vamp_bin_total += int(ix.shape[0])
                    vamp_bin_counted_keys.add(key)

        # ------------------------------------------------------------------
        # MSM-IG bonus (k-means in standardized phi/psi space): same formula + normalization.
        # ------------------------------------------------------------------
        msm_ig_rewards = None
        if msm_ig_coef > 0.0:
            if digamma is None or gammaln is None:
                print("[MSM IG] scipy.special unavailable; disabling MSM-IG for this run.", flush=True)
                msm_ig_coef = 0.0
            else:
                k_msm = int(getattr(config, "msm_ig_n_states", 50) or 50)
                dim_msm = int(getattr(config, "msm_ig_dim", 2) or 2)
                dim_msm = max(1, min(dim_msm, 2))
                max_samples = int(getattr(config, "msm_ig_max_fit_samples", 20000) or 20000)
                prior_alpha = float(getattr(config, "msm_ig_prior_alpha", 1.0) or 1.0)
                seed = int(getattr(config, "msm_ig_seed", 42) or 42)

                sig = (int(k_msm), int(dim_msm), int(lag), int(max_samples))
                if msm_ig_signature != sig or msm_ig_counts is None:
                    msm_ig_signature = sig
                    msm_ig_mu = None
                    msm_ig_std = None
                    msm_ig_counts = np.zeros((int(k_msm), int(k_msm)), dtype=np.float64)
                    msm_ig_total_pairs = 0
                    msm_ig_counted_keys = set()
                    msm_ig_kmeans = None
                    msm_ig_norm_ema_m1 = None
                    msm_ig_norm_ema_m2 = None
                    msm_ig_norm_ema_scale = None

                # Fit mu/std and k-means from history if needed.
                if msm_ig_mu is None or msm_ig_std is None or msm_ig_kmeans is None:
                    samples: List[np.ndarray] = []
                    n = 0
                    for arr in history_phi_psi.values():
                        if arr is None or arr.size == 0:
                            continue
                        remaining = max_samples - n
                        if remaining <= 0:
                            break
                        take = np.asarray(arr[:remaining, :dim_msm], dtype=np.float64)
                        samples.append(take)
                        n += int(take.shape[0])
                    if samples:
                        cat = np.concatenate(samples, axis=0)
                        mu = cat.mean(axis=0)
                        std = cat.std(axis=0)
                        std = np.where(std > 1e-6, std, 1.0)
                        msm_ig_mu = mu.astype(np.float32, copy=False)
                        msm_ig_std = std.astype(np.float32, copy=False)
                        cat_norm = (cat - mu[None, :]) / std[None, :]
                        msm_ig_kmeans = _fit_kmeans(cat_norm, k=int(k_msm), seed=int(seed))

                if msm_ig_counts is not None and msm_ig_mu is not None and msm_ig_std is not None and msm_ig_kmeans is not None:
                    mu_msm = np.asarray(msm_ig_mu, dtype=np.float32).reshape(-1)[:dim_msm]
                    std_msm = np.asarray(msm_ig_std, dtype=np.float32).reshape(-1)[:dim_msm]
                    std_msm = np.where(std_msm > 1e-8, std_msm, 1.0).astype(np.float32, copy=False)

                    # Seed global MSM counts from history once per key.
                    for key_h, arr_h in history_phi_psi.items():
                        if key_h in msm_ig_counted_keys:
                            continue
                        if arr_h is None or arr_h.size == 0 or int(arr_h.shape[0]) <= int(lag):
                            msm_ig_counted_keys.add(key_h)
                            continue
                        z_d = np.asarray(arr_h[:, :dim_msm], dtype=np.float32)
                        z_norm = (z_d - mu_msm[None, :]) / std_msm[None, :]
                        try:
                            labels = msm_ig_kmeans.predict(np.asarray(z_norm, dtype=np.float64))
                        except Exception:
                            msm_ig_counted_keys.add(key_h)
                            continue
                        counts_h, n_pairs_h = _msm_transition_counts(labels, n_states=int(k_msm), lag=int(lag))
                        if int(n_pairs_h) > 0:
                            msm_ig_counts += np.asarray(counts_h, dtype=np.float64)
                            msm_ig_total_pairs += int(n_pairs_h)
                        msm_ig_counted_keys.add(key_h)

                    rewards_msm = np.zeros((int(config.grp_size),), dtype=np.float32)
                    delta_counts_epoch: List[Optional[np.ndarray]] = [None] * int(config.grp_size)
                    delta_pairs_epoch = np.zeros((int(config.grp_size),), dtype=np.int64)

                    for i, arr in enumerate(epoch_phi_psi):
                        if not valid_mask[i] or arr is None or arr.size == 0 or int(arr.shape[0]) <= int(lag):
                            continue
                        z_d = np.asarray(arr[:, :dim_msm], dtype=np.float32)
                        z_norm = (z_d - mu_msm[None, :]) / std_msm[None, :]
                        try:
                            labels = msm_ig_kmeans.predict(np.asarray(z_norm, dtype=np.float64))
                        except Exception:
                            continue
                        counts_i, n_pairs_i = _msm_transition_counts(labels, n_states=int(k_msm), lag=int(lag))
                        if int(n_pairs_i) <= 0:
                            continue
                        delta_counts_epoch[i] = counts_i
                        delta_pairs_epoch[i] = int(n_pairs_i)
                        try:
                            rewards_msm[i] = float(
                                _msm_ig_reward_from_counts(
                                    base_counts=msm_ig_counts,
                                    delta_counts=counts_i,
                                    prior_alpha=float(prior_alpha),
                                )
                            )
                        except Exception:
                            rewards_msm[i] = 0.0

                    # Update global MSM counts once per key.
                    for i, (key, delta_i) in enumerate(zip(epoch_keys, delta_counts_epoch)):
                        if not valid_mask[i] or delta_i is None:
                            continue
                        if key in msm_ig_counted_keys:
                            continue
                        msm_ig_counts += np.asarray(delta_i, dtype=np.float64)
                        msm_ig_total_pairs += int(delta_pairs_epoch[i])
                        msm_ig_counted_keys.add(key)

                    msm_ig_rewards = rewards_msm

        # ------------------------------------------------------------------
        # Combine rewards with generative-loop normalization style.
        # ------------------------------------------------------------------
        combined = (loo_weight * loo_rewards).astype(np.float32, copy=False)

        if vamp_bin_coef > 0.0 and vamp_bin_bonus_raw is not None:
            bonus_raw = np.asarray(vamp_bin_bonus_raw, dtype=np.float32)
            bonus_raw_mean = float(np.mean(bonus_raw))
            bonus_raw_std = float(np.std(bonus_raw))

            eps = 1e-8
            ema_beta = float(getattr(config, "vamp_bin_norm_ema_beta", 0.9) or 0.9)
            ema_beta = max(0.0, min(1.0, ema_beta))
            use_ema = ema_beta > 0.0

            cur_m1 = bonus_raw_mean
            cur_m2 = float(np.mean(bonus_raw * bonus_raw))
            if not np.isfinite(cur_m2):
                cur_m2 = float(cur_m1 * cur_m1 + bonus_raw_std * bonus_raw_std)

            if use_ema:
                if vamp_bin_norm_ema_m1 is None or vamp_bin_norm_ema_m2 is None:
                    vamp_bin_norm_ema_m1 = float(cur_m1)
                    vamp_bin_norm_ema_m2 = float(cur_m2)
                else:
                    vamp_bin_norm_ema_m1 = float(ema_beta) * float(vamp_bin_norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                    vamp_bin_norm_ema_m2 = float(ema_beta) * float(vamp_bin_norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)
                mean_use = float(vamp_bin_norm_ema_m1)
                var_use = float(max(float(vamp_bin_norm_ema_m2) - mean_use * mean_use, 0.0))
                std_use = float(np.sqrt(var_use))
            else:
                mean_use = float(cur_m1)
                std_use = float(bonus_raw_std)

            if np.isfinite(std_use) and std_use > eps:
                bonus_z = (bonus_raw - mean_use) / std_use
            else:
                bonus_z = bonus_raw * 0.0

            loo_std = float(np.std(np.asarray(loo_rewards, dtype=np.float32)))
            target_scale_cur = float(loo_weight) * loo_std if (np.isfinite(loo_std) and loo_std > eps) else 1.0
            if use_ema:
                if vamp_bin_norm_ema_scale is None:
                    vamp_bin_norm_ema_scale = float(target_scale_cur)
                else:
                    vamp_bin_norm_ema_scale = float(ema_beta) * float(vamp_bin_norm_ema_scale) + float(1.0 - ema_beta) * float(target_scale_cur)
                target_scale = float(vamp_bin_norm_ema_scale)
            else:
                target_scale = float(target_scale_cur)

            bonus_norm = bonus_z * float(target_scale)
            combined = combined + float(vamp_bin_coef) * bonus_norm.astype(np.float32, copy=False)

        if msm_ig_coef > 0.0 and msm_ig_rewards is not None:
            msm_raw = np.asarray(msm_ig_rewards, dtype=np.float32)
            msm_raw = np.where(np.isfinite(msm_raw), msm_raw, 0.0).astype(np.float32, copy=False)
            msm_raw_active = msm_raw[valid_mask] if msm_raw.size == int(config.grp_size) else msm_raw
            msm_raw_mean = float(np.mean(msm_raw_active)) if msm_raw_active.size else 0.0
            msm_raw_std = float(np.std(msm_raw_active)) if msm_raw_active.size else 0.0

            eps = 1e-8
            ema_beta = float(getattr(config, "msm_ig_norm_ema_beta", 0.9) or 0.9)
            ema_beta = max(0.0, min(1.0, ema_beta))
            use_ema = ema_beta > 0.0

            cur_m1 = float(msm_raw_mean)
            try:
                cur_m2 = float(np.mean(msm_raw_active.astype(np.float64) ** 2)) if msm_raw_active.size else 0.0
            except Exception:
                cur_m2 = float(cur_m1 * cur_m1 + msm_raw_std * msm_raw_std)

            if use_ema:
                if msm_ig_norm_ema_m1 is None or msm_ig_norm_ema_m2 is None:
                    msm_ig_norm_ema_m1 = float(cur_m1)
                    msm_ig_norm_ema_m2 = float(cur_m2)
                else:
                    msm_ig_norm_ema_m1 = float(ema_beta) * float(msm_ig_norm_ema_m1) + float(1.0 - ema_beta) * float(cur_m1)
                    msm_ig_norm_ema_m2 = float(ema_beta) * float(msm_ig_norm_ema_m2) + float(1.0 - ema_beta) * float(cur_m2)
                mean_use = float(msm_ig_norm_ema_m1)
                var_use = float(max(float(msm_ig_norm_ema_m2) - mean_use * mean_use, 0.0))
                std_use = float(np.sqrt(var_use))
            else:
                mean_use = float(cur_m1)
                std_use = float(msm_raw_std)

            if np.isfinite(std_use) and std_use > eps:
                msm_z = (msm_raw - float(mean_use)) / float(std_use)
            else:
                msm_z = msm_raw * 0.0

            loo_std = float(np.std(np.asarray(loo_rewards, dtype=np.float32)))
            target_scale_cur = float(loo_weight) * loo_std if (np.isfinite(loo_std) and loo_std > eps) else 0.0
            if not (np.isfinite(target_scale_cur) and target_scale_cur > eps):
                target_scale_cur = 1.0

            if use_ema:
                if msm_ig_norm_ema_scale is None:
                    msm_ig_norm_ema_scale = float(target_scale_cur)
                else:
                    msm_ig_norm_ema_scale = float(ema_beta) * float(msm_ig_norm_ema_scale) + float(1.0 - ema_beta) * float(target_scale_cur)
                target_scale = float(msm_ig_norm_ema_scale)
            else:
                target_scale = float(target_scale_cur)

            msm_norm = msm_z * float(target_scale)
            msm_norm = np.where(valid_mask, msm_norm, 0.0).astype(np.float32, copy=False)
            combined = combined + float(msm_ig_coef) * msm_norm

        # Optional off-action discount: reward *= exp(-coef * reset_cv_mae)
        off_coef = float(getattr(config, "off_action_penalty_coef", 0.0) or 0.0)
        if off_coef > 0.0:
            mae = np.asarray(reset_mae, dtype=np.float32)
            mae = np.where(np.isfinite(mae), mae, 0.0)
            mae = np.maximum(mae, 0.0)
            discount = np.exp((-float(off_coef)) * mae).astype(np.float32, copy=False)
            combined = combined * discount

        # ------------------------------------------------------------------
        # GRPO update (continuous start policy).
        # ------------------------------------------------------------------
        rewards_t = torch.as_tensor(combined, device=torch_device, dtype=torch.float32)
        valid_t = torch.as_tensor(valid_mask, device=torch_device, dtype=torch.bool)
        feats_t = torch.zeros((int(config.grp_size), 1), device=torch_device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions_np, device=torch_device, dtype=torch.float32)

        start_data = {
            "features": feats_t,
            "actions": actions_t,
            "rewards": rewards_t,
            "valid_mask": valid_t,
        }
        update_stats = optimizer.update(start_data, hill_data=None)

        if int(config.sync_interval) > 0 and (ep + 1) % int(config.sync_interval) == 0:
            agent_old.load_state_dict(agent.state_dict())

        avg_loo = float(np.mean(loo_rewards[valid_mask])) if np.any(valid_mask) else 0.0
        std_loo = float(np.std(loo_rewards[valid_mask])) if np.any(valid_mask) else 0.0
        avg_vamp_bin = float(np.mean(vamp_bin_bonus_raw[valid_mask])) if (vamp_bin_bonus_raw is not None and np.any(valid_mask)) else 0.0
        avg_msm = float(np.mean(msm_ig_rewards[valid_mask])) if (msm_ig_rewards is not None and np.any(valid_mask)) else 0.0
        avg_reward = float(np.mean(combined[valid_mask])) if np.any(valid_mask) else 0.0
        avg_reset = float(np.mean(reset_mae[valid_mask])) if np.any(valid_mask) else 0.0
        n_valid = int(np.count_nonzero(valid_mask))

        dt = time.perf_counter() - t0
        print(
            f"[Epoch {ep+1:04d}] valid={n_valid}/{int(config.grp_size)} "
            f"avg_loo={avg_loo:+.4f} avg_vampbin={avg_vamp_bin:+.4f} avg_msm={avg_msm:+.4f} "
            f"avg_reward={avg_reward:+.4f} reset_mae={avg_reset:.4f} "
            f"pg_s={update_stats.get('pg_s', 0.0):+.4f} kl_s={update_stats.get('kl_s', 0.0):.4f} "
            f"ent={update_stats.get('entropy', 0.0):.4f} dt={dt:.2f}s",
            flush=True,
        )

        with stats_path.open("a") as handle:
            handle.write(
                f"{ep+1},{avg_loo:.6f},{std_loo:.6f},{avg_vamp_bin:.6f},{avg_msm:.6f},{avg_reward:.6f},{avg_reset:.6f},{n_valid}\n"
            )

        # Commit this epoch's torsion time series to history (for future epochs).
        for key, arr in epoch_phi_psi_by_key.items():
            if key not in history_phi_psi:
                history_phi_psi[key] = np.asarray(arr, dtype=np.float32)
