"""
Enhanced GESPO Training Loop with LOO VAMP Rewards (single-model)

This implementation features:
- Geom2Vec backbone + online VAMP lobe
- Leave-one-out VAMP-2 score calculation using sufficient statistics
- Discrete state starts (categorical policy over VAMP states)
- State relabeling using energy/RMSD/population matching
- Anchoring rounds for trajectory seeding
- Agent training with GRPO optimizer
- Comprehensive plotting and analysis
"""

import os
import copy
import gc
import time
import math
import json
import argparse
from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict, Set
import numpy as np
import torch
import mdtraj as md
import openmm
from openmm import unit
from openmm.app import Modeller, Simulation
from scipy.optimize import linear_sum_assignment

from ..config import TrainingConfig
from ..sampling import PeptideSampler
from ..utils.openmm_utils import resolve_device
from .setup import initialize_policy
from .rl import StateLogitPolicy
from .features import FeatureBuilder
from .sampler import Sampler
from .checkpointing import CheckpointManager
from .optimization import GRPOOptimizer

from ..utils.plotting import plot_ramachandran_by_residue, plot_analysis_summary
from ..utils.geometry import _dihedral, find_all_backbone_dihedrals, calculate_rmsd
from .geom2vec_backbone import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector, infer_and_cache


from .vamp_stats import SufficientStatistics  # noqa: E402 — canonical LOO VAMP-2 stats


class EnsembleDeepTICA:
    """
    Ensemble of 3 independent DeepTICA models for uncertainty quantification.
    Includes state discretization functionality.
    """

    def __init__(
        self,
        n_components: int,
        lagtime: int,
        device: str,
        n_models: int = 3,
        n_states: Optional[int] = None,
        output: str = "softmax",
    ):
        self.n_models = n_models
        # Keep discrete state count aligned with projection dimension
        self.n_states = n_components if n_components is not None else (n_states or 0)
        self.models = [
            DeepTICAProjector(
                n_components=n_components,
                lagtime=lagtime,
                device=device,
                output=output,
            )
            for _ in range(n_models)
        ]
        self.is_trained = False
        self.state_centers = None  # [n_states, n_components] - cluster centers in projection space
        self.state_populations = None  # [n_states] - population of each state
        self.state_energies = None  # [n_states] - average energy per state
        self.best_model_idx = 0  # index of model used for state assignment

    def fit(self, features: np.ndarray, times: Optional[np.ndarray] = None,
            energies: Optional[np.ndarray] = None,
            trajectory_lengths: Optional[List[int]] = None):
        """
        Train all models independently on the same data with different random seeds.
        Also performs state discretization on the mean projection.
        
        Args:
            features: [N, feature_dim] - input features
            times: [N] - rescaled time for each frame
            energies: [N] - energy for each frame (optional, for state characterization)
        """
        print(f"[EnsembleDeepTICA] Training {self.n_models} independent models...")
        
        # Optionally reshape flat frames into per-trajectory lists for DeepTICA
        train_features = features
        train_times: Optional[List[np.ndarray]] = None
        if trajectory_lengths:
            total_len = sum(trajectory_lengths)
            if total_len != len(features):
                print(
                    f"[EnsembleDeepTICA] Warning: trajectory_lengths sum ({total_len}) "
                    f"!= feature frames ({len(features)}); using pooled frames as single trajectory."
                )
            else:
                train_features_list: List[np.ndarray] = []
                train_times_list: List[np.ndarray] = [] if times is not None else None
                cursor = 0
                for L in trajectory_lengths:
                    L_i = int(L)
                    if L_i <= 0:
                        continue
                    seg_end = cursor + L_i
                    train_features_list.append(features[cursor:seg_end])
                    if times is not None:
                        train_times_list.append(times[cursor:seg_end])
                    cursor = seg_end
                train_features = train_features_list
                train_times = train_times_list

        # Save global RNG state to prevent resetting exploration randomness
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()
        else:
            cuda_rng_state = None

        fit_outcomes: List[Dict[str, Any]] = []
        for i, model in enumerate(self.models):
            print(f"  Training model {i+1}/{self.n_models}...")
            # Each model gets different random initialization
            torch.manual_seed(42 + i * 1000)
            np.random.seed(42 + i * 1000)
            model.fit(train_features, times=train_times)
            fit_outcomes.append(
                {
                    "model_idx": int(i),
                    "val_loss": getattr(model, "last_val_loss", None),
                    "val_vamp2": getattr(model, "last_val_vamp2", None),
                    "best_step": getattr(model, "last_best_step", None),
                    "effective_batch_size": getattr(model, "last_effective_batch_size", None),
                    "device": str(getattr(model, "device", "")),
                    "n_components": int(getattr(model, "n_components", 0) or 0),
                    "lagtime": int(getattr(model, "lagtime", 0) or 0),
                }
            )

        # Restore global RNG state to preserve exploration randomness
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        self.is_trained = True
        print("[EnsembleDeepTICA] Training complete.")

        self.last_fit_outcomes = fit_outcomes
        try:
            # Print a compact summary of the outcomes of all fits.
            print("[EnsembleDeepTICA] Fit outcomes:")
            for rec in fit_outcomes:
                idx = int(rec["model_idx"]) + 1
                vloss = rec["val_loss"]
                vamp2 = rec["val_vamp2"]
                step = rec["best_step"]
                bs = rec["effective_batch_size"]
                vloss_s = "n/a" if vloss is None else f"{float(vloss):.4f}"
                vamp2_s = "n/a" if vamp2 is None else f"{float(vamp2):.4f}"
                step_s = "n/a" if step is None else str(int(step))
                bs_s = "n/a" if bs is None else str(int(bs))
                print(f"  - model {idx}: ValLoss={vloss_s}  VAMP-2={vamp2_s}  best_step={step_s}  batch={bs_s}")
        except Exception as exc_summary:
            print(f"[EnsembleDeepTICA] Warning: failed to summarize outcomes: {exc_summary}")
        
        # Choose best model for state assignment.
        # Prefer highest validation VAMP-2 when available; otherwise prefer lowest validation loss.
        self.best_model_idx = 0
        try:
            cand_v = [
                (int(rec.get("model_idx", i)), rec.get("val_vamp2"))
                for i, rec in enumerate(fit_outcomes)
                if rec.get("val_vamp2") is not None
            ]
            if cand_v:
                self.best_model_idx = int(max(cand_v, key=lambda t: float(t[1]))[0])
            else:
                cand_l = [
                    (int(rec.get("model_idx", i)), rec.get("val_loss"))
                    for i, rec in enumerate(fit_outcomes)
                    if rec.get("val_loss") is not None
                ]
                if cand_l:
                    self.best_model_idx = int(min(cand_l, key=lambda t: float(t[1]))[0])
        except Exception:
            self.best_model_idx = 0
        print(f"[EnsembleDeepTICA] Selected best_model_idx={self.best_model_idx} for state assignment.")
        
        # Use the best model's soft outputs for state assignment (softmax -> argmax)
        print(f"[EnsembleDeepTICA] Assigning states via best model softmax (model {self.best_model_idx})...")
        best_model = self.models[self.best_model_idx].model
        if best_model is None:
            raise RuntimeError("Best DeepTICA model not initialized")
        with torch.no_grad():
            device = _model_device(best_model)
            x_t = torch.as_tensor(features, dtype=torch.float32, device=device)
            probs = _state_probs(best_model, x_t).cpu().numpy()
            state_labels = np.argmax(probs, axis=1)

        # Align ensemble state dimension with projector output
        prob_dim = probs.shape[1] if probs.ndim == 2 else self.n_states
        if prob_dim != self.n_states:
            print(f"[EnsembleDeepTICA] Adjusting n_states from {self.n_states} to {prob_dim} to match projector output")
            self.n_states = prob_dim
        
        # Track simple state statistics
        self.state_populations = np.bincount(state_labels, minlength=self.n_states).astype(np.float32)
        
        # Compute average energy per state
        if energies is not None:
            self.state_energies = np.zeros(self.n_states, dtype=np.float32)
            for s in range(self.n_states):
                mask = (state_labels == s)
                if mask.sum() > 0:
                    self.state_energies[s] = np.mean(energies[mask])
        else:
            self.state_energies = np.zeros(self.n_states, dtype=np.float32)
        
        # Optional: store mean softmax per state as a lightweight centroid
        state_means = []
        for s in range(self.n_states):
            mask = (state_labels == s)
            if mask.sum() > 0:
                vec = probs[mask].mean(axis=0)
                # Ensure centroid shape matches n_states
                if vec.shape[0] != self.n_states:
                    if vec.shape[0] < self.n_states:
                        vec = np.pad(vec, (0, self.n_states - vec.shape[0]))
                    else:
                        vec = vec[:self.n_states]
                state_means.append(vec)
            else:
                state_means.append(np.zeros(self.n_states, dtype=np.float32))
        self.state_centers = np.stack(state_means, axis=0)
        
        print(f"[EnsembleDeepTICA] State assignment complete. Population range: [{self.state_populations.min():.0f}, {self.state_populations.max():.0f}]")
        
    def assign_states(self, features: np.ndarray) -> np.ndarray:
        """
        Assign each frame to a discrete state based on projection.
        
        Args:
            features: [N, feature_dim] - input features
            
        Returns:
            state_labels: [N] - state assignment for each frame (0 to n_states-1)
        """
        if not self.is_trained or self.state_centers is None:
            raise RuntimeError("Ensemble not trained or states not discretized yet")
        
        # Project features with the best model softmax and take argmax
        best_model = self.models[self.best_model_idx].model
        if best_model is None:
            raise RuntimeError("Best DeepTICA model not initialized")
        with torch.no_grad():
            device = _model_device(best_model)
            x_t = torch.as_tensor(features, dtype=torch.float32, device=device)
            probs = _state_probs(best_model, x_t).cpu().numpy()
            state_labels = np.argmax(probs, axis=1)

        # Map raw labels to canonical labels if a permutation map exists
        perm = getattr(self, "state_inverse_permutation_map", None)
        if perm is not None and len(perm) == self.n_states:
            state_labels = perm[state_labels]
        
        return state_labels
    
    def get_state_values(self) -> np.ndarray:
        """
        Get value for each state (used for policy).
        Currently uses negative energy as value.
        
        Returns:
            state_values: [n_states] - value for each state
        """
        if self.state_energies is None:
            return np.zeros(self.n_states, dtype=np.float32)
        return -self.state_energies  # Lower energy = higher value
    
    def transform(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project features through all models and return mean + std.
        
        Returns:
            mean_projection: [N, n_components] - mean across ensemble
            std_projection: [N, n_components] - std across ensemble (uncertainty)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained yet")
            
        projections = []
        for model in self.models:
            proj = model.transform(features)
            projections.append(proj)

        projections = np.stack(projections, axis=0)  # [n_models, N, n_components]
        mean_proj = projections.mean(axis=0)
        std_proj = projections.std(axis=0)
        
        return mean_proj, std_proj
        
    def compute_loo_vamp_scores(self, features: np.ndarray, times: np.ndarray,
                                 trajectory_lengths: List[int],
                                 return_ensemble: bool = False) -> np.ndarray:
        """
        Compute leave-one-out VAMP-2 contribution for each trajectory, averaged across the ensemble.

        Returns the marginal contribution: delta_i = VAMP(all) - VAMP(without i)
        Higher values indicate trajectory i adds more unique slow dynamics information.

        Args:
            features: [N, feature_dim] - input features
            times: [N] - rescaled time for each frame
            trajectory_lengths: List of trajectory lengths
            return_ensemble: If True, return [n_models, n_trajectories] array; else return mean

        Returns:
            loo_contributions: [n_trajectories] or [n_models, n_trajectories] - marginal VAMP contribution
        """
        if not trajectory_lengths:
            return np.zeros((self.n_models, 0) if return_ensemble else 0)

        n_traj = len(trajectory_lengths)
        ensemble_loo_contributions = np.zeros((self.n_models, n_traj), dtype=np.float32)

        for m_idx, model in enumerate(self.models):
            projections = model.transform(features)
            lagtime = model.lagtime

            traj_start = 0
            traj_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
            for traj_len in trajectory_lengths:
                traj_end = traj_start + traj_len
                traj_proj = projections[traj_start:traj_end]
                if traj_len > lagtime:
                    x = traj_proj[:-lagtime]
                    y = traj_proj[lagtime:]
                    traj_pairs.append((x, y))
                else:
                    traj_pairs.append(None)
                traj_start = traj_end

            dim = projections.shape[1]
            stats = SufficientStatistics(dim)
            for pair in traj_pairs:
                if pair is not None:
                    stats.add_trajectory(pair[0], pair[1])

            # Compute baseline VAMP score with all trajectories
            baseline_vamp = stats.compute_vamp2_score()

            # Compute contribution as: delta_i = baseline - VAMP(without i)
            for i, pair in enumerate(traj_pairs):
                if pair is None:
                    ensemble_loo_contributions[m_idx, i] = 0.0
                    continue
                stats.remove_trajectory(pair[0], pair[1])
                vamp_without_i = stats.compute_vamp2_score()
                ensemble_loo_contributions[m_idx, i] = baseline_vamp - vamp_without_i
                stats.add_trajectory(pair[0], pair[1])

        if return_ensemble:
            return ensemble_loo_contributions
        return ensemble_loo_contributions.mean(axis=0)
            
    def save(self, path: str):
        """Save all models in ensemble and metadata."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights directly via torch.save
        for i, model in enumerate(self.models):
            model_path = save_dir / f"model_{i}.pth"
            model_obj = getattr(model, "model", None)
            if model_obj is not None:
                torch.save(model_obj.state_dict(), model_path)
            else:
                torch.save(None, model_path)

        # Save metadata (state centers, populations, permutation maps, etc.)
        metadata = {
            'state_centers': self.state_centers,
            'state_populations': self.state_populations,
            'state_energies': self.state_energies,
            'n_states': self.n_states,
            'best_model_idx': self.best_model_idx,
            'state_permutation_map': getattr(self, 'state_permutation_map', None),
            'state_inverse_permutation_map': getattr(self, 'state_inverse_permutation_map', None),
        }
        torch.save(metadata, str(save_dir / "ensemble_metadata.pt"))

    def load(self, path: str):
        """Load all models in ensemble and metadata."""
        load_dir = Path(path)

        for i, model in enumerate(self.models):
            model_path = load_dir / f"model_{i}.pth"
            if model_path.exists():
                state = torch.load(str(model_path), map_location='cpu')
                model_obj = getattr(model, "model", None)
                if model_obj is not None and state is not None:
                    model_obj.load_state_dict(state)

        # Load metadata
        meta_path = load_dir / "ensemble_metadata.pt"
        if meta_path.exists():
            metadata = torch.load(str(meta_path))
            self.state_centers = metadata.get('state_centers')
            self.state_populations = metadata.get('state_populations')
            self.state_energies = metadata.get('state_energies')
            self.n_states = metadata.get('n_states', self.n_states)
            self.best_model_idx = metadata.get('best_model_idx', 0)

            # Restore permutation maps if available
            perm_map = metadata.get('state_permutation_map')
            if perm_map is not None:
                self.state_permutation_map = perm_map
            inv_perm_map = metadata.get('state_inverse_permutation_map')
            if inv_perm_map is not None:
                self.state_inverse_permutation_map = inv_perm_map

            print(f"[EnsembleDeepTICA] Loaded metadata with {self.n_states} states (best model {self.best_model_idx}).")

        self.is_trained = True


class RobustRewardCalculator:
    """
    Reward calculator using LOO VAMP contribution with UCB exploration bonus.

    Reward_i = μ_i + λ * σ_i + β * projection_uncertainty_i

    Where:
    - μ_i: Mean LOO VAMP contribution across ensemble (delta_i = VAMP(all) - VAMP(without i))
    - σ_i: Std of LOO VAMP contribution across ensemble (epistemic uncertainty)
    - projection_uncertainty_i: Ensemble disagreement in projection space (optional)
    - λ: Exploration coefficient for LOO uncertainty (default 1.0)
    - β: Coefficient for projection disagreement bonus (default 0.0)
    """

    def __init__(self, ensemble: EnsembleDeepTICA,
                 exploration_coef: float = 1.0,
                 projection_uncertainty_coef: float = 0.0):
        self.ensemble = ensemble
        self.exploration_coef = exploration_coef
        self.projection_uncertainty_coef = projection_uncertainty_coef

    def compute_rewards(self, features: np.ndarray, times: np.ndarray,
                       trajectory_lengths: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust rewards using UCB on LOO VAMP contributions.

        Returns:
            frame_rewards: [sum(trajectory_lengths)] - per-frame rewards
            traj_scores:   [n_trajectories] - per-trajectory UCB scores (after normalization)
        """
        if not trajectory_lengths:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Compute LOO VAMP contributions per trajectory, per ensemble member
        # Shape: [n_models, n_trajectories]
        loo_ensemble = self.ensemble.compute_loo_vamp_scores(
            features, times, trajectory_lengths, return_ensemble=True
        )

        # Identify and downweight ill-conditioned trajectories: if any
        # model reports an abnormally large or non-finite contribution
        # for a trajectory, set its LOO contribution to zero so it does
        # not dominate the UCB scores.
        abs_vals = np.abs(loo_ensemble)
        finite_mask = np.isfinite(abs_vals)
        if not finite_mask.all():
            abs_vals[~finite_mask] = 0.0
        global_median = float(np.median(abs_vals)) if abs_vals.size > 0 else 0.0
        if global_median > 0.0:
            outlier_factor = 10.0
            thresh = outlier_factor * global_median
            bad_traj = (~finite_mask.all(axis=0)) | (abs_vals.max(axis=0) > thresh)
            if bad_traj.any():
                loo_ensemble[:, bad_traj] = 0.0

        # UCB formula: μ + λ * σ (both computed on the same quantity)
        loo_mean = loo_ensemble.mean(axis=0)  # [n_trajectories]
        loo_std = loo_ensemble.std(axis=0)    # [n_trajectories]
        ucb_scores = loo_mean + self.exploration_coef * loo_std

        # Broadcast trajectory UCB scores to frames
        frame_rewards = np.zeros(features.shape[0], dtype=np.float32)
        traj_start = 0
        for i, traj_len in enumerate(trajectory_lengths):
            traj_end = traj_start + traj_len
            frame_rewards[traj_start:traj_end] = ucb_scores[i]
            traj_start = traj_end

        # Optional: Add projection disagreement bonus (different from LOO uncertainty)
        if self.projection_uncertainty_coef > 0:
            _, std_proj = self.ensemble.transform(features)
            proj_uncertainty = np.linalg.norm(std_proj, axis=1)

            # Normalize to similar scale as UCB scores
            if proj_uncertainty.std() > 1e-9:
                proj_uncertainty = (proj_uncertainty - proj_uncertainty.mean()) / proj_uncertainty.std()

            bonus = self.projection_uncertainty_coef * proj_uncertainty
            frame_rewards += bonus

            # Aggregate projection bonus per trajectory and fold into traj_scores
            traj_start = 0
            traj_bonus = np.zeros_like(ucb_scores)
            for i, traj_len in enumerate(trajectory_lengths):
                traj_end = traj_start + traj_len
                if traj_len > 0:
                    traj_bonus[i] = bonus[traj_start:traj_end].mean()
                traj_start = traj_end
            ucb_scores = ucb_scores + traj_bonus

        return frame_rewards, ucb_scores


def _align_states_by_overlap(
    old_labels: np.ndarray,
    new_labels: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Compute a permutation that best aligns new states to previous ones
    based on a confusion matrix of frame-wise assignments.

    Only aligns states that are actually populated in the data.
    Unpopulated states use identity mapping.

    Args:
        old_labels: [N] - state labels under the previous projector
        new_labels: [N] - raw state labels under the refreshed projector
        n_states:   int - number of discrete states (assumed consistent)

    Returns:
        permutation: [n_states] - mapping old_label -> new_label
    """
    old_labels = np.asarray(old_labels, dtype=int)
    new_labels = np.asarray(new_labels, dtype=int)
    if old_labels.shape[0] != new_labels.shape[0]:
        raise ValueError(
            f"old_labels and new_labels must have same length, "
            f"got {old_labels.shape[0]} vs {new_labels.shape[0]}"
        )

    # Identify which states are actually populated in old and new
    old_populated = np.unique(old_labels)
    new_populated = np.unique(new_labels)
    old_populated = old_populated[(old_populated >= 0) & (old_populated < n_states)]
    new_populated = new_populated[(new_populated >= 0) & (new_populated < n_states)]

    if len(old_populated) == 0 or len(new_populated) == 0:
        # No valid states to align
        return np.arange(n_states, dtype=int)

    # Build confusion matrix ONLY for populated states
    # overlap[i, j] = count of frames where old=old_populated[i] and new=new_populated[j]
    overlap = np.zeros((len(old_populated), len(new_populated)), dtype=np.int64)
    old_to_idx = {state: idx for idx, state in enumerate(old_populated)}
    new_to_idx = {state: idx for idx, state in enumerate(new_populated)}

    for o, n in zip(old_labels, new_labels):
        if o in old_to_idx and n in new_to_idx:
            overlap[old_to_idx[o], new_to_idx[n]] += 1

    if overlap.sum() == 0:
        # No usable overlap information; fall back to identity mapping.
        return np.arange(n_states, dtype=int)

    # Hungarian solver minimizes cost, so use negative overlap as cost
    # to maximize agreement between old and new labels.
    cost_matrix = -overlap.astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build permutation: start with identity for all states
    permutation = np.arange(n_states, dtype=int)

    # Only update permutation for populated states that were matched
    for old_idx, new_idx in zip(row_ind, col_ind):
        old_state = old_populated[old_idx]
        new_state = new_populated[new_idx]
        permutation[old_state] = new_state

    # Ensure permutation is a true bijection (no collisions). After the
    # Hungarian assignment, some old states may remain unmapped and
    # some new states may be unused. We fill the remaining old states
    # so that every new state appears exactly once in the image.
    used_new = set(permutation[old_populated])
    all_new = set(range(n_states))
    unused_new = [s for s in sorted(all_new) if s not in used_new]

    for old_state in range(n_states):
        if old_state in old_populated:
            continue
        if old_state not in used_new:
            permutation[old_state] = old_state
            used_new.add(old_state)
        elif unused_new:
            new_state = unused_new.pop(0)
            permutation[old_state] = new_state
            used_new.add(new_state)

    print(f"[State Alignment] Matched {len(row_ind)} populated states: "
          f"{len(old_populated)} old → {len(new_populated)} new")

    return permutation


def train(args: argparse.Namespace):
    """
    Main training loop for GESPO with LOO VAMP + Ensemble Uncertainty rewards.
    """
    config = TrainingConfig.from_args(args)
    pdb_b = str(getattr(config, "pdb_b", "") or "").strip()
    if bool(getattr(config, "path_sampling", False)) and not pdb_b:
        raise ValueError("--path-sampling requires --pdb-b.")
    if pdb_b:
        from ..utils.endpoints import validate_endpoint_pdbs

        validate_endpoint_pdbs(str(getattr(config, "pdb_a", config.pdb) or config.pdb), pdb_b)

    # Back-compat: this loop historically used `tica_dim`/`tica_lagtime`.
    # Normalize both legacy and current field names so downstream components
    # (including older checkpoints/configs) behave consistently.
    if not hasattr(config, "geom2vec_dim"):
        try:
            setattr(config, "geom2vec_dim", int(getattr(config, "tica_dim", 2) or 2))
        except Exception:
            setattr(config, "geom2vec_dim", 2)
    if not hasattr(config, "geom2vec_lagtime"):
        try:
            setattr(config, "geom2vec_lagtime", int(getattr(config, "tica_lagtime", 20) or 20))
        except Exception:
            setattr(config, "geom2vec_lagtime", 20)
    if not hasattr(config, "tica_dim"):
        try:
            setattr(config, "tica_dim", int(getattr(config, "geom2vec_dim", 2) or 2))
        except Exception:
            setattr(config, "tica_dim", 2)
    if not hasattr(config, "tica_lagtime"):
        try:
            setattr(config, "tica_lagtime", int(getattr(config, "geom2vec_lagtime", 20) or 20))
        except Exception:
            setattr(config, "tica_lagtime", 20)
    proj_mode = str(getattr(config, "projection", "geom2vec_vamp")).lower()
    if proj_mode != "geom2vec_vamp":
        raise ValueError(
            "This discrete-start loop requires a discrete-state projector. "
            f"Got --projection={proj_mode!r}; supported: geom2vec_vamp."
        )
    if not str(getattr(config, "geom2vec_checkpoint", "") or "").strip():
        raise ValueError("--geom2vec-ckpt is required when --projection=geom2vec_vamp.")
    # Force categorical discrete-state start policy
    config.discrete_state_starts = True
    config.policy_type = "mlp"
    config.use_transformer_policy = False
    
    if not hasattr(config, 'max_pca_train') or config.max_pca_train < 100000:
        config.max_pca_train = 100000
    if not hasattr(config, 'history_reward_weight'):
        config.history_reward_weight = 5.0
    
    torch_device = "cuda" if torch.cuda.is_available() and config.platform != "CPU" else "cpu"
    gen_device = resolve_device(config.gen_device)
        
    print(f"GESPO Training with LOO VAMP + Ensemble Uncertainty")
    print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")
    print(f"Ensemble Models: 3, Exploration Coef: 1.0")
    
    # Initialize Environments
    print("Initializing environments...")
    n_env_gpus = 0
    openmm_device_ids: Optional[List[int]] = None
    geom2vec_reserved_device: Optional[int] = None
    geom2vec_reserved_devices: set[int] = set()
    try:
        geom2vec_dev = str(getattr(config, "geom2vec_device", "") or "").strip()
        if geom2vec_dev.startswith("cuda:"):
            geom2vec_reserved_device = int(geom2vec_dev.split(":", 1)[1])
    except Exception:
        geom2vec_reserved_device = None
    try:
        if bool(getattr(config, "enable_multi_gpu", False)) and torch.cuda.is_available() and config.platform != "CPU":
            n_env_gpus = int(torch.cuda.device_count())
    except Exception:
        n_env_gpus = 0
    if n_env_gpus > 1:
        openmm_device_ids = list(range(int(n_env_gpus)))
        reserve_geom2vec = bool(getattr(config, "geom2vec_reserve_gpus", False))
        dp_enabled = bool(getattr(config, "geom2vec_data_parallel", False))
        dp_devices_s = str(getattr(config, "geom2vec_data_parallel_devices", "") or "").strip()
        if dp_enabled:
            dp_ids: List[int] = []
            for part in dp_devices_s.split(","):
                p = part.strip()
                if not p:
                    continue
                try:
                    di = int(p)
                except Exception:
                    continue
                if 0 <= di < int(n_env_gpus) and di not in dp_ids:
                    dp_ids.append(di)
            if dp_ids:
                geom2vec_reserved_devices = set(int(x) for x in dp_ids)
            else:
                geom2vec_reserved_devices = set(int(i) for i in range(int(n_env_gpus)))
                print(
                    "[Multi-GPU] Warning: --geom2vec-data-parallel enabled with no explicit device list; "
                    "geom2vec will use all visible GPUs and share with OpenMM walkers (default), which may contend.",
                    flush=True,
                )
        elif geom2vec_reserved_device is not None and 0 <= int(geom2vec_reserved_device) < int(n_env_gpus):
            geom2vec_reserved_devices = {int(geom2vec_reserved_device)}

        if geom2vec_reserved_devices and int(n_env_gpus) > 1:
            if reserve_geom2vec:
                openmm_device_ids = [i for i in openmm_device_ids if int(i) not in geom2vec_reserved_devices]
                if openmm_device_ids:
                    print(
                        f"[Multi-GPU] Reserving CUDA devices {sorted(geom2vec_reserved_devices)} for geom2vec; "
                        f"assigning {config.grp_size} walkers across OpenMM devices {openmm_device_ids} (round-robin).",
                        flush=True,
                    )
                else:
                    openmm_device_ids = list(range(int(n_env_gpus)))
                    print(
                        f"[Multi-GPU] Warning: geom2vec reserved devices {sorted(geom2vec_reserved_devices)} leave no GPUs for OpenMM; "
                        "sharing devices.",
                        flush=True,
                    )
            else:
                print(
                    f"[Multi-GPU] geom2vec will use CUDA devices {sorted(geom2vec_reserved_devices)} and share with OpenMM walkers "
                    "(default). Set --geom2vec-reserve-gpus to dedicate GPUs.",
                    flush=True,
                )
        if openmm_device_ids:
            print(f"[Multi-GPU] OpenMM device pool: {openmm_device_ids}")
    envs = []
    for env_idx in range(int(config.grp_size)):
        env_args = args
        if n_env_gpus > 1 and openmm_device_ids:
            import copy

            env_args = copy.copy(args)
            env_args.device = int(openmm_device_ids[int(env_idx % len(openmm_device_ids))])
        envs.append(PeptideSampler(env_args, generator_device=gen_device))
    env_ref = envs[0]
    
    # TODO: Load secondary reference PDBs (from loop_refactored.py lines 56-125)
    # TODO: Build CV/pair seeds for anchoring (from loop_refactored.py lines 91-201)
    
    # Initialize Policy
    print("Initializing policy...")
    agent, agent_old, policy_opts, policy_params = initialize_policy(config, envs, torch_device)
    if not (isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy)):
        raise RuntimeError("Discrete state starts require StateLogitPolicy for both current and old policies")
    optimizer = GRPOOptimizer(config, agent, agent_old, policy_opts)
    
    # Checkpointing & History
    ckpt_manager = CheckpointManager(config)
    all_epoch_data = []
    all_trajectory_lengths = []
    all_epoch_traj_dcd_paths: List[List[Optional[str]]] = []
    # Optional: for epochs/trajectories backed by a DCD on disk, store the
    # original frame indices used for each imported frame (e.g., after applying
    # history_dcd_stride or RMSD filtering). For trajectories produced by the
    # current run, this is typically None (implies 0..L-1).
    all_epoch_traj_dcd_frame_indices: List[List[Optional[np.ndarray]]] = []
    # Per-epoch flag: True = include in replay pool for discrete-state starts
    # but EXCLUDE from VAMP training / LOO-VAMP reward scoring.
    # Use case: pretrain trajectories with different time spacing.
    all_epoch_vamp_exclude: List[bool] = []
    history_warm_started = False
    ep_start = 0
    prev_state_centers: Optional[np.ndarray] = None

    # Global (history-aware) LOO-VAMP sufficient statistics so trajectory rewards reflect
    # the marginal improvement to VAMP-2 over the accumulated dataset.
    loo_vamp_stats_by_model: Optional[List[SufficientStatistics]] = None
    loo_vamp_stats_keys: Set[str] = set()
    loo_vamp_stats_signature: Optional[Tuple[int, int, int]] = None  # (n_models, n_states, lag_embed)

    def _invalidate_loo_vamp_stats(tag: str) -> None:
        nonlocal loo_vamp_stats_by_model, loo_vamp_stats_keys, loo_vamp_stats_signature
        loo_vamp_stats_by_model = None
        loo_vamp_stats_keys = set()
        loo_vamp_stats_signature = None
        print(f"[LOO VAMP] Invalidated global stats ({tag}).")

    def _ensure_loo_vamp_history_stats(
        *,
        upto_epoch_exclusive: int,
        n_models: int,
        n_states: int,
        lag_embed: int,
        perm_maps: List[np.ndarray],
        tag: str,
    ) -> None:
        """
        Ensure `loo_vamp_stats_by_model` contains sufficient statistics for all cached
        trajectories in `all_epoch_data[:upto_epoch_exclusive]` under the *current* VAMP lobe.
        """
        nonlocal loo_vamp_stats_by_model, loo_vamp_stats_keys, loo_vamp_stats_signature

        if geom2vec_cache is None or geom2vec_projector is None or not _geom2vec_is_trained():
            return

        sig = (int(n_models), int(n_states), int(lag_embed))
        if loo_vamp_stats_by_model is None or loo_vamp_stats_signature != sig:
            loo_vamp_stats_by_model = [SufficientStatistics(int(n_states)) for _ in range(int(n_models))]
            loo_vamp_stats_keys = set()
            loo_vamp_stats_signature = sig

        upto = max(0, min(int(upto_epoch_exclusive), len(all_epoch_data)))
        if upto == 0:
            return

        ensemble = getattr(geom2vec_projector, "trained_ensemble", None)
        use_member_idx = ensemble is not None and len(ensemble) > 0

        added = 0
        skipped_short = 0
        skipped_missing = 0

        for epoch_idx in range(upto):
            # Skip epochs quarantined from VAMP scoring (e.g. pretrain trajs
            # with incompatible time spacing).
            if epoch_idx < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[epoch_idx]:
                continue

            epoch_data = all_epoch_data[epoch_idx]
            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            if not traj_lens:
                continue

            cursor = 0
            for traj_idx, L in enumerate(traj_lens):
                L_i = int(L)
                if L_i <= 0:
                    continue
                dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                key = _geom2vec_cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)
                cursor_next = cursor + L_i

                if key in loo_vamp_stats_keys:
                    cursor = cursor_next
                    continue

                cached_key = None
                if _geom2vec_cache_has_key(key):
                    cached_key = key
                else:
                    coords_nm = None
                    if not dcd_path:
                        try:
                            coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor:cursor_next]]
                            coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                        except Exception:
                            coords_nm = None
                    cached_key = _geom2vec_ensure_cached(
                        key=key,
                        dcd_path=dcd_path,
                        coords_nm=coords_nm,
                        overwrite=False,
                        meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                    )

                cursor = cursor_next

                if cached_key is None:
                    skipped_missing += 1
                    continue
                if cached_key in loo_vamp_stats_keys:
                    continue

                try:
                    rec = geom2vec_cache.get(cached_key)
                    if int(rec.frames) <= int(lag_embed):
                        loo_vamp_stats_keys.add(str(cached_key))
                        skipped_short += 1
                        continue
                except Exception:
                    pass

                try:
                    graph, ca = geom2vec_cache.load(cached_key)
                except Exception:
                    skipped_missing += 1
                    continue

                for m in range(int(n_models)):
                    try:
                        probs_raw = geom2vec_projector.state_probs_from_embeddings(
                            graph,
                            ca,
                            member_idx=m if use_member_idx else None,
                        )
                        probs = probs_raw[:, perm_maps[m]]
                        if probs.shape[1] != n_states:
                            if probs.shape[1] > n_states:
                                probs = probs[:, :n_states]
                            else:
                                probs = np.pad(probs, ((0, 0), (0, n_states - probs.shape[1])), mode="constant")
                        T = int(probs.shape[0])
                        if T > lag_embed:
                            loo_vamp_stats_by_model[m].add_trajectory(probs[:-lag_embed], probs[lag_embed:])
                    except Exception:
                        continue

                loo_vamp_stats_keys.add(str(cached_key))
                added += 1

        if added:
            print(
                f"[LOO VAMP] Seeded global stats ({tag}): added_trajs={added}, "
                f"skipped_short={skipped_short}, skipped_missing={skipped_missing}"
            )

    # Geom2Vec VAMP backbone state (only used when proj_mode == "geom2vec_vamp")
    geom2vec_cache: Optional[EmbeddingCache] = None
    # Softmax VAMP lobe: used for state labels and LOO-VAMP scoring.
    geom2vec_projector: Optional[Geom2VecVAMPProjector] = None
    # Linear VAMP lobe: optional diagnostics (kept for consistency with generative loop).
    geom2vec_projector_linear: Optional[Geom2VecVAMPProjector] = None
    geom2vec_state_permutation_map = np.arange(
        int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2), dtype=int
    )
    geom2vec_state_inverse_permutation_map = np.arange(
        int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2), dtype=int
    )
    geom2vec_lagtime_embed = 1
    geom2vec_embed_stride = int(getattr(config, "geom2vec_embed_stride", 2) or 2)
    geom2vec_embed_batch_size = int(getattr(config, "geom2vec_embed_batch_size", 32) or 32)
    geom2vec_reduction = str(getattr(config, "geom2vec_reduction", "sum") or "sum")
    geom2vec_center_ca = bool(getattr(config, "geom2vec_center_ca", True))
    geom2vec_hidden_channels_opt: Optional[int] = None
    geom2vec_md_top_protein = None
    geom2vec_topology_cache: Optional[Geom2VecTopologyCache] = None
    geom2vec_cache_root_path: Optional[Path] = None
    geom2vec_device_str = ""

    def _geom2vec_is_trained() -> bool:
        return geom2vec_projector is not None and getattr(geom2vec_projector, "trained", None) is not None

    def _geom2vec_import_module():
        try:
            import geom2vec  # type: ignore

            if hasattr(geom2vec, "create_model_from_checkpoint"):
                return geom2vec
        except ImportError:
            pass

        import importlib
        import sys

        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "apps" / "geom2vec" / "src"
            if (candidate / "geom2vec").exists():
                sys.path.insert(0, str(candidate))
                # If a namespace package shadowed the real geom2vec, purge it before retrying.
                for name in list(sys.modules.keys()):
                    if name == "geom2vec" or name.startswith("geom2vec."):
                        sys.modules.pop(name, None)
                importlib.invalidate_caches()
                import geom2vec  # type: ignore

                if hasattr(geom2vec, "create_model_from_checkpoint"):
                    return geom2vec
                raise ImportError(
                    "Imported `geom2vec` but it does not expose `create_model_from_checkpoint` "
                    "(possible namespace-package shadowing)."
                )

        raise ImportError(
            "Could not import `geom2vec`. Install it (e.g. `pip install -e apps/geom2vec`) "
            "or ensure `apps/geom2vec/src` is on PYTHONPATH."
        )

    def _geom2vec_cache_key_for_history_traj(
        epoch_idx: int,
        traj_idx: int,
        dcd_path: Optional[str],
    ) -> str:
        if dcd_path:
            p = Path(str(dcd_path))
            if geom2vec_cache_root_path is not None:
                try:
                    rel = p.relative_to(geom2vec_cache_root_path)
                    return rel.with_suffix("").as_posix()
                except Exception:
                    pass
            # Fallback: place under an "external" bucket to avoid absolute-path keys,
            # but keep the key unique even if basenames collide.
            try:
                import hashlib

                abs_path = str(p.resolve())
                payload = abs_path.encode("utf-8")
                try:
                    h = hashlib.md5(payload, usedforsecurity=False)
                except TypeError:
                    h = hashlib.md5(payload)
                digest = h.hexdigest()[:10]
                return (Path("external") / f"{digest}_{p.stem}").as_posix()
            except Exception:
                return (Path("external") / p.name).with_suffix("").as_posix()
        return (Path("history_bootstrap") / f"epoch_{epoch_idx:04d}" / f"traj_{traj_idx:03d}").as_posix()

    def _geom2vec_cache_has_key(key: str) -> bool:
        if geom2vec_cache is None:
            return False
        try:
            rec = geom2vec_cache.get(key)
        except KeyError:
            return False
        try:
            return Path(rec.embedding_path).exists() and Path(rec.ca_path).exists()
        except Exception:
            return False

    def _geom2vec_ensure_cached(
        *,
        key: str,
        dcd_path: Optional[str],
        coords_nm: Optional[np.ndarray],
        overwrite: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if geom2vec_cache is None or geom2vec_projector is None or geom2vec_topology_cache is None:
            return None

        effective_stride = int(geom2vec_embed_stride)

        expected_raw_frames: Optional[int] = None
        try:
            if meta is not None and "raw_frames" in meta and meta["raw_frames"] is not None:
                expected_raw_frames = int(meta["raw_frames"])
        except Exception:
            expected_raw_frames = None
        expected_embed_frames: Optional[int] = None
        if expected_raw_frames is not None and effective_stride > 0:
            expected_embed_frames = int((expected_raw_frames + effective_stride - 1) // effective_stride)

        def _stale_cache_reason(rec: Any) -> Optional[str]:
            try:
                rec_frames = int(getattr(rec, "frames", 0) or 0)
                if expected_embed_frames is not None and rec_frames != int(expected_embed_frames):
                    return f"cached_frames={rec_frames} expected_frames={int(expected_embed_frames)}"
            except Exception:
                pass

            if dcd_path:
                try:
                    dcd_p = Path(str(dcd_path))
                    if dcd_p.exists():
                        st = dcd_p.stat()
                        dcd_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                        dcd_size = int(st.st_size)

                        meta_dcd_mtime_ns = None
                        meta_dcd_size = None
                        try:
                            meta_dcd_mtime_ns = getattr(rec, "meta", {}).get("source_dcd_mtime_ns", None)
                            meta_dcd_size = getattr(rec, "meta", {}).get("source_dcd_size", None)
                        except Exception:
                            meta_dcd_mtime_ns = None
                            meta_dcd_size = None

                        if meta_dcd_mtime_ns is not None and int(meta_dcd_mtime_ns) != dcd_mtime_ns:
                            return "source_dcd_mtime_ns changed"
                        if meta_dcd_size is not None and int(meta_dcd_size) != dcd_size:
                            return "source_dcd_size changed"

                        try:
                            emb_mtime_ns = int(Path(str(getattr(rec, "embedding_path"))).stat().st_mtime_ns)
                            if dcd_mtime_ns > emb_mtime_ns:
                                return "source_dcd newer than cached embeddings"
                        except Exception:
                            pass
                except Exception:
                    pass
            return None

        if not overwrite and _geom2vec_cache_has_key(key):
            try:
                rec_existing = geom2vec_cache.get(key)
                reason = _stale_cache_reason(rec_existing)
                if reason is None:
                    return key
                print(f"[Geom2Vec Cache] Stale cache entry for key={key} ({reason}); recomputing.", flush=True)
                overwrite = True
            except Exception:
                return key

        if coords_nm is None:
            if not dcd_path:
                return None
            if geom2vec_md_top_protein is None:
                return None
            try:
                traj = md.load(str(dcd_path), top=geom2vec_md_top_protein, stride=int(effective_stride))
                coords_nm = np.asarray(traj.xyz, dtype=np.float32)
            except Exception:
                # In explicit-solvent full-coordinate replay mode, DCDs may contain the
                # full system. Fall back to loading with the full topology and slicing.
                try:
                    md_top_full = env_ref.md_topology
                    protein_indices = md_top_full.select("protein")
                    if protein_indices is None or len(protein_indices) == 0:
                        from ..utils.selection import infer_protein_atom_indices

                        protein_indices = infer_protein_atom_indices(md_top_full, include_caps=True, allow_empty=True)
                    traj_full = md.load(str(dcd_path), top=md_top_full, stride=int(effective_stride))
                    traj = traj_full.atom_slice(protein_indices)
                    coords_nm = np.asarray(traj.xyz, dtype=np.float32)
                except Exception as exc2:
                    print(f"[Geom2Vec Cache] Failed to load DCD {dcd_path}: {exc2}")
                    return None
            # We already applied mdtraj stride; do not stride again in inference.
            infer_stride = 1
        else:
            infer_stride = int(effective_stride)

        meta_extra: Dict[str, Any] = {"stride": int(effective_stride), "source_dcd": str(dcd_path) if dcd_path else ""}
        if dcd_path:
            try:
                st = Path(str(dcd_path)).stat()
                meta_extra["source_dcd_mtime_ns"] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                meta_extra["source_dcd_size"] = int(st.st_size)
            except Exception:
                pass

        try:
            rec = infer_and_cache(
                geom2vec_cache,
                key=key,
                model=geom2vec_projector.rep_model,
                topology=geom2vec_topology_cache,
                coords=coords_nm,
                coords_in_nm=True,
                stride=infer_stride,
                batch_size=int(geom2vec_embed_batch_size),
                reduction=str(geom2vec_reduction),
                hidden_channels=geom2vec_hidden_channels_opt,
                device=torch.device(str(geom2vec_device_str or torch_device)),
                center_ca=bool(geom2vec_center_ca),
                overwrite=overwrite,
                meta={
                    **(meta or {}),
                    **meta_extra,
                },
            )
            return rec.key
        except Exception as exc:
            print(f"[Geom2Vec Cache] Failed to compute/store embeddings for key={key}: {exc}")
            return None

    if proj_mode == "geom2vec_vamp":
        geom2vec_device_str = str(getattr(config, "geom2vec_device", "") or "").strip() or str(torch_device)
        geom2vec_hidden_channels = int(getattr(config, "geom2vec_hidden_channels", 0) or 0)
        geom2vec_hidden_channels_opt = int(geom2vec_hidden_channels) if geom2vec_hidden_channels > 0 else None

        cache_root = str(getattr(config, "geom2vec_cache_root", "") or "").strip() or str(
            getattr(config, "snapshot_dir", "") or "."
        )
        manifest_name = str(getattr(config, "geom2vec_manifest_name", "geom2vec_manifest.jsonl") or "geom2vec_manifest.jsonl")
        geom2vec_cache = EmbeddingCache(cache_root, manifest_name=manifest_name)
        geom2vec_cache_root_path = Path(cache_root).resolve()

        geom2vec = _geom2vec_import_module()
        rep_ckpt = str(getattr(config, "geom2vec_checkpoint", "") or "").strip()
        rep_model = geom2vec.create_model_from_checkpoint(rep_ckpt, device=geom2vec_device_str)

        # Build a protein-only topology cache aligned with the rollout DCDs.
        try:
            md_top_full = env_ref.md_topology
            protein_indices = md_top_full.select("protein")
            if protein_indices is None or len(protein_indices) == 0:
                from ..utils.selection import infer_protein_atom_indices

                protein_indices = infer_protein_atom_indices(md_top_full, include_caps=True, allow_empty=True)
            geom2vec_md_top_protein = md_top_full.subset(protein_indices)
        except Exception:
            geom2vec_md_top_protein = None

        if geom2vec_md_top_protein is None:
            raise RuntimeError("[Geom2Vec] Failed to build protein-only mdtraj topology for embedding inference.")

        geom2vec_topology_cache = Geom2VecTopologyCache.from_mdtraj(
            geom2vec_md_top_protein,
            exclude_hydrogens=bool(getattr(config, "geom2vec_exclude_hydrogens", True)),
            meta={"selection": "protein"},
        )

        geom2vec_projector = Geom2VecVAMPProjector(
            rep_model=rep_model,
            topology=geom2vec_topology_cache,
            cache=geom2vec_cache,
            device=torch.device(geom2vec_device_str),
            hidden_channels=geom2vec_hidden_channels_opt,
        )
        geom2vec_projector_linear = Geom2VecVAMPProjector(
            rep_model=rep_model,
            topology=geom2vec_topology_cache,
            cache=geom2vec_cache,
            device=torch.device(geom2vec_device_str),
            hidden_channels=geom2vec_hidden_channels_opt,
        )

        # Effective VAMP lag in embedding frames.
        # Convert lagtime from saved-frame units -> embedded-frame units.
        lag_raw = int(getattr(config, "geom2vec_lagtime", getattr(config, "tica_lagtime", 1)) or 1)
        embed_stride_i = max(1, int(geom2vec_embed_stride))
        geom2vec_lagtime_embed = max(1, int(round(float(max(1, int(lag_raw))) / float(embed_stride_i))))

    def _refresh_deeptica(reason: str):
        """Train ensemble DeepTICA on accumulated history and save."""
        nonlocal prev_state_centers
        if not all_epoch_data:
            print(f"[DeepTICA Refresh] Skipped ({reason}): no history frames available.")
            return

        # Gather historical data (skip VAMP-excluded epochs, e.g. pretrain trajs)
        hist_cv = [item[0] for ei, epoch_data in enumerate(all_epoch_data) for item in epoch_data
                    if not (ei < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[ei])]
        hist_coords = [item[1] for ei, epoch_data in enumerate(all_epoch_data) for item in epoch_data
                       if not (ei < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[ei])]
        hist_energy = [item[2] if len(item) > 2 else 0.0 for ei, epoch_data in enumerate(all_epoch_data) for item in epoch_data
                       if not (ei < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[ei])]
        hist_bias = [item[3] if len(item) > 3 else 0.0 for ei, epoch_data in enumerate(all_epoch_data) for item in epoch_data
                     if not (ei < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[ei])]

        flat_traj_lengths: List[int] = []
        for ei, epoch_traj_lens in enumerate(all_trajectory_lengths):
            if ei < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[ei]:
                continue
            flat_traj_lengths.extend(epoch_traj_lens)

        n_frames = len(hist_cv)
        lagtime = getattr(config, "geom2vec_lagtime", getattr(config, "tica_lagtime", 10))
        print(
            f"[DeepTICA Refresh] {reason}: frames={n_frames}, trajs={len(flat_traj_lengths)}, lagtime={lagtime}"
        )

        if n_frames == 0:
            print("[DeepTICA Refresh] No data available.")
            return

        # Subsample if too large
        max_train = config.max_pca_train if config.max_pca_train > 0 else 0
        if max_train and n_frames > max_train:
            start_idx = n_frames - max_train
            # Trim the flat trajectory lengths to match tail selection
            trimmed_lengths: List[int] = []
            drop = start_idx
            for L in flat_traj_lengths:
                if drop >= L:
                    drop -= L
                    continue
                trimmed_lengths.append(L - drop)
                drop = 0
            flat_traj_lengths = trimmed_lengths

            hist_cv = hist_cv[start_idx:]
            hist_coords = hist_coords[start_idx:]
            hist_energy = hist_energy[start_idx:]
            hist_bias = hist_bias[start_idx:]

        # Build features
        features = []
        for cv_val, coord_val in zip(hist_cv, hist_coords):
            feat = feature_builder.build_projection_features(
                cv_vec=np.asarray(cv_val, dtype=np.float32),
                coords_nm=np.asarray(coord_val, dtype=np.float32),
            )
            feat = np.asarray(feat, dtype=np.float32)
            if feat.ndim == 2 and feat.shape[0] == 1:
                feat = feat.ravel()
            features.append(feat)

        feats_arr = np.asarray(features, dtype=np.float32)
        if feats_arr.ndim > 2:
            feats_arr = feats_arr.reshape(feats_arr.shape[0], -1)

        # If an ensemble is already trained, capture previous labels on the
        # current history frames so we can build a confusion matrix after
        # the refresh. This provides a kinetically meaningful alignment
        # across projector updates.
        old_labels: Optional[np.ndarray] = None
        if ensemble_deeptica.is_trained and prev_state_centers is not None:
            try:
                old_labels = ensemble_deeptica.assign_states(feats_arr)
            except Exception as exc_old:
                print(
                    "[DeepTICA Refresh] Warning: failed to compute old state labels "
                    f"for overlap alignment: {exc_old}"
                )
                old_labels = None

        # Energies (kept for state statistics)
        energy_arr = np.asarray(hist_energy, dtype=np.float32)

        # Optionally override effective batch size for DeepTICA training
        # from the current config on every refresh, so changes to
        # --tica-batch-size are respected even when resuming from an
        # older checkpoint that stored a different value.
        try:
            bs_override = int(getattr(config, "tica_batch_size", 0) or 0)
        except Exception:
            bs_override = 0
        if bs_override > 0:
            for m in ensemble_deeptica.models:
                try:
                    m.batch_size = bs_override
                except Exception:
                    pass

        # Train ensemble (including state discretization)
        ensemble_deeptica.fit(
            feats_arr,
            times=None,  # use frame indices to ensure monotonic times for lag pairs
            energies=energy_arr,
            trajectory_lengths=flat_traj_lengths if flat_traj_lengths else None,
        )

        # Align new states to previous epoch using confusion-matrix matching
        # to keep policy/state semantics consistent across refreshes.
        if prev_state_centers is not None and ensemble_deeptica.state_centers is not None:
            permutation: Optional[np.ndarray] = None

            # Use confusion-matrix alignment when we have valid old labels
            if old_labels is not None and ensemble_deeptica.is_trained:
                try:
                    # Compute raw new labels from the refreshed best model
                    # WITHOUT applying any existing permutation map.
                    best_model = ensemble_deeptica.models[ensemble_deeptica.best_model_idx].model
                    if best_model is not None:
                        with torch.no_grad():
                            device = _model_device(best_model)
                            x_t = torch.as_tensor(feats_arr, dtype=torch.float32, device=device)
                            probs_new = _state_probs(best_model, x_t).cpu().numpy()
                        new_labels = np.argmax(probs_new, axis=1)
                        if len(new_labels) == len(old_labels):
                            permutation = _align_states_by_overlap(
                                old_labels=old_labels,
                                new_labels=new_labels,
                                n_states=ensemble_deeptica.n_states,
                            )
                            print(f"[DeepTICA Refresh] Applied confusion-matrix alignment: {permutation}")
                        else:
                            print(f"[DeepTICA Refresh] Warning: label length mismatch "
                                  f"({len(new_labels)} vs {len(old_labels)}), using identity permutation")
                    else:
                        print("[DeepTICA Refresh] Warning: best model is None, using identity permutation")
                except Exception as exc_overlap:
                    print(f"[DeepTICA Refresh] Warning: confusion-matrix alignment failed: {exc_overlap}")
                    print("[DeepTICA Refresh] Using identity permutation (no relabeling)")
            else:
                if old_labels is None:
                    print("[DeepTICA Refresh] First refresh, using identity permutation")
                else:
                    print("[DeepTICA Refresh] Ensemble not trained, using identity permutation")

            # Apply permutation (or identity if permutation is None)
            if permutation is not None:
                # Reorder all state metadata to match previous labels
                ensemble_deeptica.state_centers = ensemble_deeptica.state_centers[permutation]
                if ensemble_deeptica.state_populations is not None:
                    ensemble_deeptica.state_populations = ensemble_deeptica.state_populations[permutation]
                if ensemble_deeptica.state_energies is not None:
                    ensemble_deeptica.state_energies = ensemble_deeptica.state_energies[permutation]

                # Store permutation maps for downstream assignment
                ensemble_deeptica.state_permutation_map = permutation
                inv_perm = np.zeros_like(permutation)
                inv_perm[permutation] = np.arange(len(permutation))
                ensemble_deeptica.state_inverse_permutation_map = inv_perm
            else:
                # No permutation applied, keep identity
                identity = np.arange(ensemble_deeptica.n_states, dtype=int)
                ensemble_deeptica.state_permutation_map = identity
                ensemble_deeptica.state_inverse_permutation_map = identity

        # Cache centers for next refresh alignment (used for tracking, not for matching)
        if ensemble_deeptica.state_centers is not None:
            prev_state_centers = ensemble_deeptica.state_centers.copy()

        # Attach best DeepTICA model (removed)
        try:
            # Seed global DeepTICA state populations for logging/rarity
            if reward_calculator is not None:
                try:
                    reward_calculator.seed_deeptica_state_counts_from_history(
                        all_epoch_data,
                        all_trajectory_lengths,
                    )
                except Exception as exc_seed:
                    print(f"[DeepTICA Refresh] Warning: failed to seed DeepTICA state counts: {exc_seed}")
        except Exception as exc_attach:
            print(f"[DeepTICA Refresh] Warning: failed to attach ensemble model: {exc_attach}")

        # Save ensemble
        if config.snapshot_dir:
            ensemble_path = Path(config.snapshot_dir) / "ensemble_deeptica"
            ensemble_deeptica.save(str(ensemble_path))
            print(f"[DeepTICA Refresh] Saved ensemble to {ensemble_path}")
            # Also save a compact summary of the per-member fit outcomes (if available).
            try:
                outcomes = getattr(ensemble_deeptica, "last_fit_outcomes", None)
                if outcomes:
                    (ensemble_path / "fit_outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
                    print(f"[DeepTICA Refresh] Saved fit outcomes to {ensemble_path / 'fit_outcomes.json'}")
            except Exception as exc_save_outcomes:
                print(f"[DeepTICA Refresh] Warning: failed to save fit outcomes: {exc_save_outcomes}")

        # Optionally reset the discrete start policy after each DeepTICA
        # refresh so that the categorical start distribution does not
        # accumulate learning on drifting state labels across projector
        # updates. This is primarily intended for StateLogitPolicy runs
        # with discrete_state_starts enabled.
        if getattr(config, "reset_policy_on_projector_refresh", False):
            try:
                if isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy):
                    with torch.no_grad():
                        if getattr(agent, "logits", None) is not None:
                            agent.logits.zero_()
                        if getattr(agent_old, "logits", None) is not None:
                            agent_old.logits.zero_()
                    # Clear optimizer state so it can adapt cleanly to the
                    # reset parameters on subsequent GRPO updates.
                    for opt in policy_opts:
                        opt.state.clear()
                    print("[DeepTICA Refresh] Reset StateLogitPolicy logits to uniform after refresh.")
                else:
                    print("[DeepTICA Refresh] Policy reset flag is set, but current agent is not StateLogitPolicy; skipping reset.")
            except Exception as exc_reset:
                print(f"[DeepTICA Refresh] Warning: failed to reset policy after DeepTICA refresh: {exc_reset}")

    def _refresh_geom2vec_vamp(reason: str):
        """Train geom2vec VAMP lobe on cached embeddings and keep stable state indexing."""
        nonlocal geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, geom2vec_lagtime_embed

        if geom2vec_cache is None or geom2vec_projector is None:
            print(f"[Geom2Vec Refresh] Skipped ({reason}): projector/cache not initialized.")
            return
        if not all_epoch_data:
            print(f"[Geom2Vec Refresh] Skipped ({reason}): no history frames available.")
            return

        lag_raw = int(getattr(config, "geom2vec_lagtime", getattr(config, "tica_lagtime", 1)) or 1)
        embed_stride_i = max(1, int(geom2vec_embed_stride))
        geom2vec_lagtime_embed = max(1, int(round(float(max(1, int(lag_raw))) / float(embed_stride_i))))
        lag_embed = int(geom2vec_lagtime_embed)

        def _natural_sort_key(text: str) -> Tuple[Tuple[int, object], ...]:
            import re

            parts = re.split(r"(\d+)", text)
            key: List[Tuple[int, object]] = []
            for part in parts:
                if not part:
                    continue
                if part.isdigit():
                    key.append((1, int(part)))
                else:
                    key.append((0, part.lower()))
            return tuple(key)

        # Ensure embeddings exist for all historical trajectories, then select a training subset.
        # NOTE: we still cache embeddings for VAMP-excluded epochs (they are used
        # for state-labelling in the replay pool), but we skip them for VAMP training.
        all_keys: List[str] = []
        for epoch_idx, epoch_data in enumerate(all_epoch_data):
            # Skip epochs quarantined from VAMP (e.g. pretrain trajs with
            # incompatible time spacing).  We still cache their embeddings
            # below so the discrete-state start picker can label them.
            vamp_excluded = epoch_idx < len(all_epoch_vamp_exclude) and all_epoch_vamp_exclude[epoch_idx]

            traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
            traj_paths = all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
            if not traj_lens:
                continue

            cursor = 0
            for traj_idx, L in enumerate(traj_lens):
                L_i = int(L)
                if L_i <= 0:
                    continue
                dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                key = _geom2vec_cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)

                if _geom2vec_cache_has_key(key):
                    if not vamp_excluded:
                        all_keys.append(str(key))
                    cursor += L_i
                    continue

                coords_nm = None
                try:
                    coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor : cursor + L_i]]
                    coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                except Exception:
                    coords_nm = None

                cached_key = _geom2vec_ensure_cached(
                    key=key,
                    dcd_path=dcd_path,
                    coords_nm=coords_nm,
                    overwrite=False,
                    meta={
                        "epoch_idx": int(epoch_idx),
                        "traj_idx": int(traj_idx),
                        "raw_frames": int(L_i),
                    },
                )
                if cached_key is not None and not vamp_excluded:
                    all_keys.append(str(cached_key))

                cursor += L_i

        if not all_keys:
            print(f"[Geom2Vec Refresh] Skipped ({reason}): no trajectories could be cached.")
            return

        # De-duplicate and sort keys by path so "recent" corresponds to later epochs.
        keys_sorted = sorted(set(all_keys), key=_natural_sort_key)

        # Apply a frame-cap (similar intent as max_pca_train) in embedding frames.
        max_train_raw = int(getattr(config, "max_pca_train", 0) or 0)
        max_train_embed = int(math.ceil(max_train_raw / max(1, int(geom2vec_embed_stride)))) if max_train_raw > 0 else 0

        train_keys: List[str] = []
        total_embed_frames = 0
        for key in reversed(keys_sorted):
            try:
                rec = geom2vec_cache.get(key)
            except KeyError:
                continue
            if int(rec.frames) <= lag_embed:
                continue
            if max_train_embed > 0 and total_embed_frames + int(rec.frames) > max_train_embed and train_keys:
                break
            train_keys.append(key)
            total_embed_frames += int(rec.frames)
        train_keys = list(reversed(train_keys))

        if not train_keys:
            print(
                f"[Geom2Vec Refresh] Skipped ({reason}): no cached trajectories with frames > lag ({lag_embed})."
            )
            return

        # Capture old canonical labels for overlap alignment before retraining.
        old_labels_by_key: Optional[Dict[str, np.ndarray]] = None
        if _geom2vec_is_trained():
            old_labels_by_key = {}
            for key in train_keys:
                try:
                    probs_raw = geom2vec_projector.state_probs_from_cache_key(key)
                    probs_canon = probs_raw[:, geom2vec_state_permutation_map]
                    old_labels_by_key[key] = np.argmax(probs_canon, axis=1).astype(np.int64, copy=False)
                except Exception:
                    continue

        n_states = int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2)
        # Pair subsampling stride is intentionally fixed to 1.
        # We already control temporal resolution via record_stride (MD) and geom2vec_embed_stride (embedding).
        pair_stride = 1
        train_fraction = float(getattr(config, "geom2vec_vamp_train_fraction", 0.8) or 0.8)
        n_epochs = int(getattr(config, "geom2vec_vamp_epochs", 10) or 10)
        batch_size = int(getattr(config, "geom2vec_vamp_batch_size", 256) or 256)
        learning_rate = float(getattr(config, "geom2vec_vamp_learning_rate", 5e-4) or 5e-4)
        weight_decay = float(getattr(config, "geom2vec_vamp_weight_decay", 0.0) or 0.0)
        optimizer_name = str(getattr(config, "geom2vec_vamp_optimizer", "Adam") or "Adam")

        print(
            f"[Geom2Vec Refresh] {reason}: trajs={len(train_keys)}, "
            f"embed_frames~{total_embed_frames}, lag_embed={lag_embed}, pair_stride={pair_stride}"
        )

        try:
            seed_base = int(getattr(config, "geom2vec_vamp_seed_base", 42) or 42)
            epsilon = float(getattr(config, "geom2vec_vamp_epsilon", 1e-6) or 1e-6)
            inverse_mode = str(getattr(config, "geom2vec_vamp_inverse_mode", "regularize") or "regularize")
            vamp_merger = bool(getattr(config, "geom2vec_vamp_merger", False))
            vamp_merger_window = int(getattr(config, "geom2vec_vamp_merger_window", 6) or 6)
            vamp_double_lobe = bool(getattr(config, "geom2vec_vamp_double_lobe", False))
            lobe_invariant = bool(getattr(config, "geom2vec_lobe_invariant", False))

            lobe_kwargs: Dict[str, Any] = {}
            try:
                hidden_channels = int(getattr(config, "geom2vec_hidden_channels", 0) or 0)
                if hidden_channels > 0:
                    lobe_kwargs["hidden_channels"] = int(hidden_channels)
            except Exception:
                pass
            # Train a softmax lobe for LOO-VAMP scoring + stable state labels.
            geom2vec_projector.fit_from_cache(
                keys=train_keys,
                n_states=n_states,
                lag_time=lag_embed,
                seed=int(seed_base),
                pair_stride=pair_stride,
                train_fraction=train_fraction,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer=optimizer_name,
                epsilon=epsilon,
                inverse_mode=inverse_mode,
                softmax_output=True,
                vamp_merger=vamp_merger,
                vamp_merger_window=vamp_merger_window,
                double_lobe=vamp_double_lobe,
                lobe_invariant=lobe_invariant,
                lobe_kwargs=lobe_kwargs,
            )
            # Optionally train a linear lobe (kept for consistency with the generative loop).
            if geom2vec_projector_linear is not None:
                try:
                    geom2vec_projector_linear.fit_from_cache(
                        keys=train_keys,
                        n_states=n_states,
                        lag_time=lag_embed,
                        seed=int(seed_base) + 1000,
                        pair_stride=pair_stride,
                        train_fraction=train_fraction,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        optimizer=optimizer_name,
                        epsilon=epsilon,
                        inverse_mode=inverse_mode,
                        softmax_output=False,
                        vamp_merger=vamp_merger,
                        vamp_merger_window=vamp_merger_window,
                        double_lobe=vamp_double_lobe,
                        lobe_invariant=lobe_invariant,
                        lobe_kwargs=lobe_kwargs,
                    )
                except Exception as exc_lin:
                    print(f"[Geom2Vec Refresh] Warning: failed to train linear VAMP lobe (keeping previous): {exc_lin}")
        except Exception as exc:
            print(f"[Geom2Vec Refresh] CRITICAL ERROR: Failed to train geom2vec softmax VAMP lobe: {exc}")
            try:
                outcomes = getattr(geom2vec_projector, "last_fit_outcomes", None)
                if outcomes:
                    print("[Geom2Vec Refresh] Fit outcomes:")
                    for rec in outcomes:
                        if rec.get("ok"):
                            continue
                        midx = rec.get("model_idx", "?")
                        err = rec.get("error", rec)
                        print(f"  - model {midx}: {err}")
            except Exception:
                pass
            import traceback

            traceback.print_exc()
            return

        try:
            outcomes_sm = getattr(geom2vec_projector, "last_fit_outcomes", None)
            if outcomes_sm:
                rec = outcomes_sm[0]
                sm_flag = rec.get("softmax_output")
                print(
                    f"[Geom2Vec Refresh] Softmax fit (softmax_output={sm_flag}): "
                    f"val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
                )
            if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "last_fit_outcomes", None):
                rec = geom2vec_projector_linear.last_fit_outcomes[0]
                lin_flag = rec.get("softmax_output")
                print(
                    f"[Geom2Vec Refresh] Linear fit (softmax_output={lin_flag}): "
                    f"val={rec.get('val_score')} train={rec.get('train_score')} seed={rec.get('seed')}"
                )
        except Exception:
            pass

        # Align new raw labels back to the previous canonical indexing.
        if old_labels_by_key is not None:
            old_all: List[np.ndarray] = []
            new_all: List[np.ndarray] = []
            for key in train_keys:
                old_lbl = old_labels_by_key.get(key)
                if old_lbl is None:
                    continue
                try:
                    probs_new_raw = geom2vec_projector.state_probs_from_cache_key(key)
                except Exception:
                    continue
                new_lbl = np.argmax(probs_new_raw, axis=1).astype(np.int64, copy=False)
                if new_lbl.shape[0] != old_lbl.shape[0]:
                    continue
                old_all.append(old_lbl)
                new_all.append(new_lbl)

            if old_all and new_all:
                old_flat = np.concatenate(old_all, axis=0)
                new_flat = np.concatenate(new_all, axis=0)
                try:
                    perm = _align_states_by_overlap(old_labels=old_flat, new_labels=new_flat, n_states=n_states)
                    geom2vec_state_permutation_map = perm
                    inv_perm = np.zeros_like(perm)
                    inv_perm[perm] = np.arange(len(perm))
                    geom2vec_state_inverse_permutation_map = inv_perm
                    print(f"[Geom2Vec Refresh] Applied confusion-matrix alignment: {perm}")
                except Exception as exc_align:
                    print(f"[Geom2Vec Refresh] Warning: overlap alignment failed: {exc_align}; using identity mapping.")
                    identity = np.arange(n_states, dtype=int)
                    geom2vec_state_permutation_map = identity
                    geom2vec_state_inverse_permutation_map = identity
        else:
            identity = np.arange(n_states, dtype=int)
            geom2vec_state_permutation_map = identity
            geom2vec_state_inverse_permutation_map = identity

        # Persist lobe weights + metadata for debugging / resume.
        if config.snapshot_dir:
            out_dir = Path(config.snapshot_dir) / "geom2vec_vamp"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                lobe_softmax = geom2vec_projector._require_lobe()
                # Keep the legacy filename pointing at the softmax lobe.
                torch.save(lobe_softmax.state_dict(), out_dir / "lobe_state_dict.pt")
                torch.save(lobe_softmax.state_dict(), out_dir / "lobe_state_dict_softmax.pt")
                if geom2vec_projector_linear is not None and getattr(geom2vec_projector_linear, "trained", None) is not None:
                    lobe_lin = geom2vec_projector_linear._require_lobe()
                    torch.save(lobe_lin.state_dict(), out_dir / "lobe_state_dict_linear.pt")
                torch.save(
                    {
                        "train_keys": train_keys,
                        "lag_embed": int(lag_embed),
                        "embed_stride": int(geom2vec_embed_stride),
                        "state_permutation_map": np.asarray(geom2vec_state_permutation_map, dtype=np.int64),
                        "state_inverse_permutation_map": np.asarray(geom2vec_state_inverse_permutation_map, dtype=np.int64),
                        "softmax_fit_outcomes": getattr(geom2vec_projector, "last_fit_outcomes", None),
                        "linear_fit_outcomes": (
                            getattr(geom2vec_projector_linear, "last_fit_outcomes", None) if geom2vec_projector_linear is not None else None
                        ),
                    },
                    out_dir / "lobe_meta.pt",
                )
                print(f"[Geom2Vec Refresh] Saved lobe to {out_dir}")
            except Exception as exc_save:
                print(f"[Geom2Vec Refresh] Warning: failed to save lobe weights/meta: {exc_save}")

        # The VAMP lobe weights (and potentially the state permutation) have changed;
        # invalidate any cached global LOO statistics so rewards stay consistent.
        _invalidate_loo_vamp_stats(tag=f"geom2vec refresh: {reason}")

        # Optional: reset the start policy after each projector refresh to avoid
        # accumulating logits on drifting labels.
        if getattr(config, "reset_policy_on_projector_refresh", False):
            try:
                if isinstance(agent, StateLogitPolicy) and isinstance(agent_old, StateLogitPolicy):
                    with torch.no_grad():
                        if getattr(agent, "logits", None) is not None:
                            agent.logits.zero_()
                        if getattr(agent_old, "logits", None) is not None:
                            agent_old.logits.zero_()
                    for opt in policy_opts:
                        opt.state.clear()
                    print("[Geom2Vec Refresh] Reset StateLogitPolicy logits to uniform after refresh.")
            except Exception as exc_reset:
                print(f"[Geom2Vec Refresh] Warning: failed to reset policy after refresh: {exc_reset}")

    def _run_anchoring_round():
        """Seed history with short PDB-grounded trajectories before training."""
        if not getattr(config, "anchoring_round", False):
            return

        want_path_sampling = bool(getattr(config, "path_sampling", False)) and bool(str(getattr(config, "pdb_b", "") or "").strip())

        def _list_traj_files(d: Optional[Path]) -> List[Path]:
            if d is None:
                return []
            try:
                files = list(d.rglob("*.dcd")) + list(d.rglob("*.xtc"))
                # Deterministic ordering (epoch index is in filename for our writers).
                try:
                    import re

                    def _natural_key(p: Path):
                        parts = re.split(r"(\d+)", p.as_posix())
                        out = []
                        for part in parts:
                            if not part:
                                continue
                            if part.isdigit():
                                out.append((1, int(part)))
                            else:
                                out.append((0, part.lower()))
                        return tuple(out)

                    return sorted(set(files), key=_natural_key)
                except Exception:
                    return sorted(set(files), key=lambda p: p.as_posix())
            except Exception:
                return []

        def _max_epoch_index(files: List[Path]) -> int:
            try:
                import re

                best = 0
                pat = re.compile(r"epoch_(\\d+)_walker_\\d+\\.(?:dcd|xtc)$")
                for f in files:
                    m = pat.search(f.name)
                    if not m:
                        continue
                    try:
                        best = max(best, int(m.group(1)))
                    except Exception:
                        continue
                return int(best)
            except Exception:
                return 0

        anchor_dir = None
        anchor_dir_a = None
        anchor_dir_b = None
        if config.snapshot_dir:
            if want_path_sampling:
                anchor_dir_a = Path(config.snapshot_dir) / "anchoring_A"
                anchor_dir_b = Path(config.snapshot_dir) / "anchoring_B"
                anchor_dir_a.mkdir(parents=True, exist_ok=True)
                anchor_dir_b.mkdir(parents=True, exist_ok=True)
            else:
                anchor_dir = Path(config.snapshot_dir) / "anchoring"
                anchor_dir.mkdir(parents=True, exist_ok=True)

            # Prefer an explicit source directory for reusing existing anchoring
            # trajectories if provided (can be comma/colon-separated list).
            default_source_dir = ""
            if want_path_sampling and anchor_dir_a is not None and anchor_dir_b is not None:
                default_source_dir = f"{anchor_dir_a},{anchor_dir_b}"
            elif anchor_dir is not None:
                default_source_dir = str(anchor_dir)
            source_dir = getattr(config, "anchoring_source_dir", "") or default_source_dir

            # Import existing anchoring trajectories (if present), but do not
            # auto-skip unless the output directories already contain enough
            # trajectories to satisfy the requested count. This makes anchoring
            # resumable/top-up friendly.
            imported = 0
            imported_lens: List[int] = []
            if source_dir:
                try:
                    imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
                        all_epoch_data,
                        env_ref,
                        base_dir=source_dir,
                    )
                except Exception:
                    imported = 0
                    imported_lens = []
                if imported > 0:
                    all_trajectory_lengths.append(imported_lens)
                    all_epoch_vamp_exclude.append(False)
                    traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
                    if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                        all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
                    else:
                        all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
                    traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
                    if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                        all_epoch_traj_dcd_frame_indices.append(
                            [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                        )
                    else:
                        all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))
                else:
                    # Fallback: legacy NPZ history (single dir only).
                    if ("," not in str(source_dir)) and (":" not in str(source_dir)):
                        restored = 0
                        restored_lens: List[List[int]] = []
                        try:
                            restored, restored_lens = ckpt_manager.rebuild_history_from_npz(
                                all_epoch_data, base_dir=source_dir
                            )
                        except Exception:
                            restored = 0
                            restored_lens = []
                        if restored > 0:
                            all_trajectory_lengths.extend(restored_lens)
                            for lens in restored_lens:
                                all_epoch_traj_dcd_paths.append([None] * len(lens))
                                all_epoch_traj_dcd_frame_indices.append([None] * len(lens))
                            print(
                                f"[Anchoring] Detected existing anchoring data in {source_dir}; "
                                f"skipped rerun ({restored} NPZ epochs loaded)."
                            )
                            return

        print("[Anchoring] Running initial anchoring trajectories...", flush=True)

        desired_traj_count = int(getattr(config, "anchoring_num_trajs", 0) or len(envs))
        if desired_traj_count <= 0:
            desired_traj_count = len(envs)

        epoch_data: List[Any] = []
        traj_lengths: List[int] = []
        traj_dcd_paths: List[Optional[str]] = []

        pdb_b = str(getattr(config, "pdb_b", "") or "").strip()
        envs_b: Optional[List[PeptideSampler]] = None
        if want_path_sampling and pdb_b:
            b_args = copy.copy(args)
            b_args.pdb = pdb_b
            if anchor_dir_b is not None:
                b_args.snapshot_dir = str(anchor_dir_b)
            envs_b = [PeptideSampler(b_args, generator_device=gen_device)]

        # Split requested trajectories across endpoints (50/50 in path-sampling mode).
        n_total = int(desired_traj_count)
        n_b = (n_total // 2) if want_path_sampling else 0
        n_a = n_total - n_b

        # If the anchoring output dirs already have enough trajectories, skip
        # rerunning (but we may still have imported history above).
        existing_a = len(_list_traj_files(anchor_dir_a if want_path_sampling else anchor_dir))
        existing_b = len(_list_traj_files(anchor_dir_b)) if want_path_sampling else 0
        if want_path_sampling:
            if existing_a >= n_a and existing_b >= n_b:
                print(
                    f"[Anchoring] Found existing trajectories (A={existing_a}/{n_a}, B={existing_b}/{n_b}); skipping rerun.",
                    flush=True,
                )
                return
        else:
            if existing_a >= n_a:
                print(
                    f"[Anchoring] Found existing trajectories ({existing_a}/{n_a}) in {anchor_dir}; skipping rerun.",
                    flush=True,
                )
                return

        def _run_endpoint(label: str, env_pool: List[PeptideSampler], out_dir: Optional[Path], n_trajs: int):
            nonlocal epoch_data, traj_lengths, traj_dcd_paths
            if n_trajs <= 0:
                return
            existing_files = _list_traj_files(out_dir)
            existing_n = int(len(existing_files))
            if existing_n >= n_trajs:
                print(
                    f"[Anchoring {label}] Found existing trajectories ({existing_n}/{n_trajs}) in {out_dir}; skipping.",
                    flush=True,
                )
                return

            epoch_start = _max_epoch_index(existing_files) + 1
            n_to_run = int(n_trajs) - existing_n
            print(
                f"[Anchoring {label}] Resuming: found {existing_n}/{n_trajs} in {out_dir}; running {n_to_run} more (epoch_start={epoch_start}).",
                flush=True,
            )

            for run_idx in range(n_to_run):
                epoch_index = int(epoch_start + run_idx)
                seq_idx = int(epoch_index - 1)
                env = env_pool[seq_idx % len(env_pool)]

                # Check if we have multiple conformers in the PDB
                has_conformers = getattr(env, "has_multiple_conformers", False)
                num_conformers = len(getattr(env, "pdb_conformers", []))

                # Anchoring: condition the generator on the reference PDB torsions so
                # the first trajectories start from a physically meaningful prior.
                # If the generator reset fails, fall back to replaying the PDB coords.
                # IMPORTANT: If multiple conformers are available, use them directly instead of the generator.
                cv_targets: List[float]
                conformer_idx: Optional[int] = None

                if has_conformers and num_conformers > 0:
                    # Use conformers from the multi-conformer PDB instead of the generator
                    conformer_idx = seq_idx % num_conformers
                    print(
                        f"[Anchoring {label}] Using conformer {conformer_idx}/{num_conformers-1} for trajectory {existing_n + run_idx + 1}/{n_trajs}.",
                        flush=True,
                    )
                    # Get torsions from the specific conformer
                    try:
                        conformer_pos_nm = env.get_conformer_positions(conformer_idx)
                        if conformer_pos_nm is not None:
                            cv_targets = [float(_dihedral(conformer_pos_nm, atoms)) for _, atoms in env.cv_definitions]
                        else:
                            pdb_pos_nm = np.array(
                                env.clean_modeller.getPositions().value_in_unit(unit.nanometer),
                                dtype=np.float32,
                            )
                            cv_targets = [float(_dihedral(pdb_pos_nm, atoms)) for _, atoms in env.cv_definitions]
                            conformer_idx = None
                    except Exception:
                        n_cvs = int(getattr(env, "n_cvs", 0) or 0)
                        cv_targets = [0.0] * n_cvs
                        conformer_idx = None
                else:
                    try:
                        pdb_pos_nm = np.array(
                            env.clean_modeller.getPositions().value_in_unit(unit.nanometer),
                            dtype=np.float32,
                        )
                        cv_targets = [float(_dihedral(pdb_pos_nm, atoms)) for _, atoms in env.cv_definitions]
                    except Exception:
                        n_cvs = int(getattr(env, "n_cvs", 0) or 0)
                        cv_targets = [0.0] * n_cvs

                # Reset behavior:
                # - If multiple conformers: always use PDB replay with specific conformer (no generator)
                # - For endpoint B anchoring: always use PDB replay (stay in the endpoint basin).
                # - Otherwise: mix generator and PDB launches.
                if has_conformers and num_conformers > 0:
                    env.reset(cv_targets, force_pdb_coords=True, conformer_index=conformer_idx)
                else:
                    pdb_only_launch = (label == "B") or bool(getattr(config, "anchoring_pdb_only", False)) or ((seq_idx % 4) == 0)
                    if pdb_only_launch:
                        env.reset(cv_targets, force_pdb_coords=True)
                    else:
                        reset_cvs = env.reset(cv_targets, force_pdb_coords=False)
                        if getattr(env, "reset_failed", False) or reset_cvs is None:
                            print("[Anchoring] Generator reset failed; falling back to PDB replay.", flush=True)
                            env.reset(cv_targets, force_pdb_coords=True)

                h_scales = [0.0] * config.num_mixtures
                sigmas = [[max(config.sigma_min, 1e-3)] * env.n_cvs for _ in range(config.num_mixtures)]

                original_steps = env.n_md_steps
                try:
                    anchor_steps = int(getattr(config, "anchoring_steps", 0) or env.n_md_steps)
                    env.n_md_steps = anchor_steps
                    traj_data, is_stable, dcd_path = env.step(
                        h_scales,
                        sigmas,
                        record_frequency=config.record_stride,
                        output_dir=out_dir,
                        epoch_index=epoch_index,
                        walker_idx=seq_idx % max(1, len(envs)),
                        equilibration_steps=config.equilibration_steps,
                        write_full_dcd=getattr(config, "anchoring_write_full_dcd", None),
                    )
                finally:
                    env.n_md_steps = original_steps

                if not (traj_data and is_stable):
                    print(
                        f"[Anchoring {label}] Trajectory {existing_n + run_idx + 1}/{n_trajs} unstable or empty; skipping.",
                        flush=True,
                    )
                    continue

                epoch_data.extend(traj_data)
                traj_lengths.append(len(traj_data))
                traj_dcd_paths.append(str(dcd_path) if dcd_path else None)
                print(f"[Anchoring {label}] Completed trajectory {existing_n + run_idx + 1}/{n_trajs}.", flush=True)

        out_dir_a = anchor_dir_a if want_path_sampling else anchor_dir
        _run_endpoint("A", envs, out_dir_a, n_a)
        if want_path_sampling and envs_b is not None:
            _run_endpoint("B", envs_b, anchor_dir_b, n_b)

        if epoch_data:
            all_epoch_data.append(epoch_data)
            all_trajectory_lengths.append(traj_lengths)
            all_epoch_traj_dcd_paths.append(traj_dcd_paths)
            all_epoch_traj_dcd_frame_indices.append([None] * len(traj_dcd_paths))
            all_epoch_vamp_exclude.append(False)

        print("[Anchoring] Done.")
    
    # Load Checkpoint
    latest_ckpt, ckpt_data = ckpt_manager.load_latest()
    pca_state = None
    
    if config.resume and ckpt_data:
        ep_start, _, _ = ckpt_manager.restore_states(
            agent, agent_old, policy_opts, ckpt_data, torch_device
        )
        imported_frames, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.snapshot_dir or "."
        )
        if imported_frames > 0:
            print(f"[Resume] Rebuilt history from {imported_frames} frames (DCD).")
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(False)
            traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
            if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
            else:
                all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
            traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
            if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                all_epoch_traj_dcd_frame_indices.append(
                    [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                )
            else:
                all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))
        else:
            restored, restored_lens = ckpt_manager.rebuild_history_from_npz(
                all_epoch_data, base_dir=config.snapshot_dir or "."
            )
            if restored > 0:
                print(f"[Resume] Rebuilt history from {restored} NPZ epochs.")
                all_trajectory_lengths.extend(restored_lens)
                for lens in restored_lens:
                    all_epoch_traj_dcd_paths.append([None] * len(lens))
                    all_epoch_traj_dcd_frame_indices.append([None] * len(lens))
                    all_epoch_vamp_exclude.append(False)

    # Bootstrap history from external DCDs if provided (e.g., previous runs).
    if getattr(config, "history_dcd_dir", ""):
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.history_dcd_dir
        )
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(False)  # normal history: include in VAMP
            traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
            if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
            else:
                all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
            traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
            if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                all_epoch_traj_dcd_frame_indices.append(
                    [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                )
            else:
                all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))

    # Bootstrap replay-only history from external DCDs (e.g. pretrain trajs with
    # different time spacing).  These frames are available for discrete-state
    # start retrieval but are EXCLUDED from VAMP training & LOO-VAMP rewards.
    if getattr(config, "replay_only_dcd_dir", ""):
        print("[Replay-Only History] Loading pretrain trajs for replay pool (VAMP-excluded)...")
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.replay_only_dcd_dir
        )
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
            all_epoch_vamp_exclude.append(True)  # quarantine: exclude from VAMP
            traj_paths = getattr(ckpt_manager, "last_bootstrap_dcd_paths", None)
            if isinstance(traj_paths, list) and len(traj_paths) == len(imported_lens):
                all_epoch_traj_dcd_paths.append([str(p) for p in traj_paths])
            else:
                all_epoch_traj_dcd_paths.append([None] * len(imported_lens))
            traj_frame_idx = getattr(ckpt_manager, "last_bootstrap_frame_indices", None)
            if isinstance(traj_frame_idx, list) and len(traj_frame_idx) == len(imported_lens):
                all_epoch_traj_dcd_frame_indices.append(
                    [np.asarray(x, dtype=np.int64) if x is not None else None for x in traj_frame_idx]
                )
            else:
                all_epoch_traj_dcd_frame_indices.append([None] * len(imported_lens))
            print(f"[Replay-Only History] Imported {imported} frames from replay-only pool.")
    
    # Discrete state dimension (number of VAMP states)
    n_states = int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2)
    config.n_discrete_states = n_states
    n_discrete_states = n_states
    print("Initializing Geom2Vec VAMP backbone...")

    # Initialize PCA scorer (removed)
    
    # Feature Builder
    feature_builder = FeatureBuilder(config, generator=envs[0].generator, env=envs[0])
    
    # Core reward calculator (trajectory + launch rewards)
    # Vendi-based novelty estimators are disabled in the streamlined loop.
    vendi_launch_pca = None
    vendi_hill_pca = None
    vendi_launch_angle = None
    vendi_hill_angle = None
    # Rewards are computed directly in this loop; Sampler keeps the
    # reward_calculator argument only for API compatibility.
    reward_calculator = None

    sampler = Sampler(config, envs, reward_calculator, torch_device)
        
    # Stats Log
    stats_path = Path(config.stats_log) if config.stats_log else None
    if stats_path and not stats_path.exists():
        header = (
            "epoch,inner,valid_count,pg_h,pg_s,ent_coef,ent_val,loss,"
            "adv_h_mean,adv_h_std,adv_s_mean,adv_s_std,"
            "dlogp_h_mean,dlogp_h_std,ratio_h_mean,ratio_h_std,"
            "dlogp_s_mean,dlogp_s_std,ratio_s_mean,ratio_s_std,"
            "kl_h,kl_s,grad_norm,pre_clip_norm,avg_launch,avg_traj,avg_novelty,avg_pca,avg_loo_vamp\n"
        )
        stats_path.write_text(header)
    
    # Optional anchoring to seed history
    _run_anchoring_round()

    # Warm-start projector immediately after anchoring so VAMP rewards are ready for epoch 1
    if all_epoch_data and not _geom2vec_is_trained():
        _refresh_geom2vec_vamp(reason="post-anchoring warm-start")

    # Plotting for "this round": exclude any bootstrapped history/anchoring epochs.
    round_epoch_data_start = len(all_epoch_data)
    
    print(f"Starting training from epoch {ep_start+1}...")
    
    # Main Training Loop
    start_state_actions: Optional[torch.Tensor] = None

    for ep in range(ep_start, config.episodes):
        print(f"\n{'='*60}")
        print(f"Starting Epoch {ep+1}/{config.episodes}")
        print(f"{'='*60}")
        epoch_start_time = time.time()
        
        # A. Refresh projector (every N epochs)
        refresh_interval = int(getattr(config, "pca_refresh_interval", 1) or 1)
        model_ready = _geom2vec_is_trained()

        if refresh_interval <= 0:
            should_update = not model_ready
        else:
            should_update = ((ep + 1) % refresh_interval == 0) or (not model_ready)

        if should_update and all_epoch_data:
            print(f"[Epoch {ep+1}] Refreshing Geom2Vec VAMP lobe...")
            _refresh_geom2vec_vamp(reason=f"epoch {ep+1}")
            model_ready = _geom2vec_is_trained()
            # TODO: Warm start policy after refresh (from loop_refactored.py lines 724-971)

        # Ensure history-aware LOO stats match the current VAMP lobe before sampling.
        if model_ready:
            try:
                n_states_hist = int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2)
                lag_hist = int(geom2vec_lagtime_embed)
                _ensure_loo_vamp_history_stats(
                    upto_epoch_exclusive=len(all_epoch_data),
                    n_models=1,
                    n_states=n_states_hist,
                    lag_embed=lag_hist,
                    perm_maps=[np.asarray(geom2vec_state_permutation_map, dtype=int)],
                    tag=f"epoch {ep+1} pre-sample",
                )
            except Exception as exc_loo_seed:
                print(f"[LOO VAMP] Warning: failed to seed global stats ({exc_loo_seed})")
        
        # B. Sample Discrete State Starts from Policy
        print(f"[Epoch {ep+1}] Phase B: Sampling discrete state starts...")

        # Sample discrete states from categorical policy
        # If projector is not trained yet, fall back to random exploration
        start_state_actions = None
        if not model_ready or not all_epoch_data:
            print("[Discrete Starts] Projector not ready, using random exploration")
            # Random state indices (uniform over available history)
            if all_epoch_data:
                n_available = len([item for epoch_data in all_epoch_data for item in epoch_data])
                state_indices = torch.randint(0, n_available, (config.grp_size,), device=torch_device)
            else:
                # No history yet, use None (will trigger generator-based reset)
                state_indices = None
            rand_mask = torch.ones(config.grp_size, dtype=torch.bool, device=torch_device)
            start_state_actions = torch.zeros(config.grp_size, dtype=torch.long, device=torch_device)
            for env in envs:
                try:
                    env._launch_state_id = None
                except Exception:
                    pass
        else:
            # Flatten history frames (in the same order used by global-frame indexing).
            hist_cv: List[Any] = []
            hist_coords: List[Any] = []
            hist_energy: List[float] = []
            for epoch_data in all_epoch_data:
                for entry in epoch_data:
                    hist_cv.append(entry[0])
                    hist_coords.append(entry[1])
                    hist_energy.append(entry[2] if len(entry) > 2 else 0.0)

            boundary_scores = None

            # Geom2Vec: label a subset of frames (every `embed_stride`) by running the VAMP lobe
            # on cached embeddings, but keep indices in the *raw history* frame space.
            n_hist = len(hist_cv)
            n_states = int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2)
            state_labels = np.full((n_hist,), -1, dtype=np.int64)

            use_boundary = bool(getattr(config, "use_boundary_replay", False))
            boundary_scores = np.zeros((n_hist,), dtype=np.float32) if use_boundary else None

            if geom2vec_cache is None or geom2vec_projector is None:
                raise RuntimeError("[Geom2Vec] Projector not initialized.")

            global_offset = 0
            for epoch_idx, epoch_data in enumerate(all_epoch_data):
                traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
                traj_paths = (
                    all_epoch_traj_dcd_paths[epoch_idx] if epoch_idx < len(all_epoch_traj_dcd_paths) else []
                )
                if not traj_lens:
                    global_offset += len(epoch_data)
                    continue

                cursor = 0
                for traj_idx, L in enumerate(traj_lens):
                    L_i = int(L)
                    if L_i <= 0:
                        continue
                    dcd_path = traj_paths[traj_idx] if traj_idx < len(traj_paths) else None
                    key = _geom2vec_cache_key_for_history_traj(epoch_idx, traj_idx, dcd_path)

                    coords_nm = None
                    if not dcd_path:
                        try:
                            coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in epoch_data[cursor : cursor + L_i]]
                            coords_nm = np.asarray(coords_list, dtype=np.float32) if coords_list else None
                        except Exception:
                            coords_nm = None

                    cached_key = _geom2vec_ensure_cached(
                        key=key,
                        dcd_path=dcd_path,
                        coords_nm=coords_nm,
                        overwrite=False,
                        meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
                    )
                    if cached_key is None:
                        cursor += L_i
                        continue

                    try:
                        rec = geom2vec_cache.get(cached_key)
                        stride_eff = int(rec.meta.get("stride", geom2vec_embed_stride))
                    except Exception:
                        stride_eff = int(geom2vec_embed_stride)

                    try:
                        probs_raw = geom2vec_projector.state_probs_from_cache_key(cached_key)
                    except Exception:
                        cursor += L_i
                        continue

                    # Canonicalize state indexing so the start policy remains consistent across refreshes.
                    probs = probs_raw[:, geom2vec_state_permutation_map]
                    labels = np.argmax(probs, axis=1).astype(np.int64, copy=False)

                    ent = None
                    if boundary_scores is not None:
                        p = np.clip(probs, 1e-12, 1.0)
                        ent = (-(p * np.log(p)).sum(axis=1)).astype(np.float32, copy=False)

                    for e_i, lbl in enumerate(labels):
                        raw_local = int(e_i) * stride_eff
                        if raw_local >= L_i:
                            break
                        g_idx = global_offset + cursor + raw_local
                        if 0 <= g_idx < n_hist:
                            state_labels[g_idx] = int(lbl)
                            if boundary_scores is not None and ent is not None:
                                boundary_scores[g_idx] = float(ent[e_i])

                    cursor += L_i

                global_offset += len(epoch_data)

            labeled = int(np.count_nonzero(state_labels >= 0))
            pops = np.bincount(state_labels[state_labels >= 0], minlength=n_states).astype(np.float64) if labeled else None
            if pops is not None:
                print(
                    f"[Geom2Vec Starts] Labeled {labeled}/{n_hist} history frames (stride={geom2vec_embed_stride}); "
                    f"pop_min={pops.min():.0f}, pop_max={pops.max():.0f}"
                )

            # Policy over states: categorical start head.
            # We sample from agent_old (the PPO behavior policy) but log both the
            # behavior and current policies so learning is visible even when
            # sync_interval > 1.
            state_policy_feats = torch.zeros(config.grp_size, 1, device=torch_device)
            with torch.no_grad():
                state_dist_old = agent_old(state_policy_feats)
                sampled_states = state_dist_old.sample()
                state_dist_new = agent(state_policy_feats)

            def _mean_probs_str(dist: torch.distributions.Categorical) -> Optional[str]:
                try:
                    probs = getattr(dist, "probs", None)
                    if probs is None and getattr(dist, "logits", None) is not None:
                        probs = torch.softmax(dist.logits, dim=-1)
                    if probs is None:
                        return None
                    probs_np = probs.detach().cpu().numpy()
                    if probs_np.ndim != 2:
                        return None
                    mean_probs = probs_np.mean(axis=0)
                    return ", ".join(f"{float(p):.3f}" for p in mean_probs)
                except Exception:
                    return None

            probs_old = _mean_probs_str(state_dist_old)
            probs_new = _mean_probs_str(state_dist_new)
            if probs_old is not None:
                print(f"[Discrete Starts] Start distribution (agent_old sampling): [{probs_old}]")
            if probs_new is not None:
                print(f"[Discrete Starts] Start distribution (agent current):      [{probs_new}]")
            start_state_actions = sampled_states.clone().to(torch_device)
            # Mark these launches as policy-driven (not random) so PPO updates include them
            rand_mask = torch.zeros(config.grp_size, dtype=torch.bool, device=torch_device)

            # Map states to frame indices, preferring boundary frames (high entropy)
            state_indices = torch.zeros(config.grp_size, dtype=torch.long, device=torch_device)
            for w_idx in range(config.grp_size):
                state_id = int(sampled_states[w_idx].item())
                # Get all frames in this state
                state_mask = (state_labels == state_id)
                state_frame_ids = np.where(state_mask)[0]

                if len(state_frame_ids) > 0:
                    # Prefer boundary frames (high entropy) within this state
                    if boundary_scores is not None:
                        # Get boundary scores for frames in this state
                        state_boundary_scores = boundary_scores[state_frame_ids]
                        # Sample from top 50% boundary scores (or top 20 frames, whichever is smaller)
                        n_top = max(1, min(len(state_frame_ids) // 2, 20))
                        top_indices = np.argsort(state_boundary_scores)[-n_top:]
                        chosen_frame = np.random.choice(state_frame_ids[top_indices])
                    else:
                        # Fallback: no boundary scores, randomly sample from state
                        chosen_frame = np.random.choice(state_frame_ids)

                    state_indices[w_idx] = chosen_frame
                else:
                    # Fallback to random frame
                    state_indices[w_idx] = np.random.randint(0, len(hist_cv))

                # Record the chosen discrete state on the env for launch logging/reward
                try:
                    envs[w_idx]._launch_state_id = state_id
                except Exception:
                    pass
            
            print(f"[Discrete Starts] Sampled from {int(n_discrete_states)} states via categorical policy")
        
        # Extract replay coordinates from selected states
        replay_coords = None
        starts = None  # Will be derived from replay coords
        
        if state_indices is not None:
            # Reuse flattened history (aligned with state_indices) and keep anchor preference already applied
            hist_cv_flat = hist_cv if 'hist_cv' in locals() else [item[0] for epoch_data in all_epoch_data for item in epoch_data]
            hist_coords_flat = hist_coords if 'hist_coords' in locals() else [item[1] for epoch_data in all_epoch_data for item in epoch_data]

            md_top_protein = None
            try:
                md_top_full = env_ref.md_topology
                protein_indices = md_top_full.select("protein")
                if protein_indices is None or len(protein_indices) == 0:
                    from ..utils.selection import infer_protein_atom_indices

                    protein_indices = infer_protein_atom_indices(md_top_full, include_caps=True, allow_empty=True)
                md_top_protein = md_top_full.subset(protein_indices)
            except Exception:
                md_top_protein = None

            use_full_replay_coords = bool(getattr(config, "explicit_solvent_replay_full_coords", False)) and bool(
                getattr(env_ref, "explicit_solvent", False)
            )

            def _locate_history_frame(global_frame: int) -> Optional[Tuple[int, int, int]]:
                if global_frame < 0:
                    return None
                offset = 0
                for epoch_idx, epoch_data in enumerate(all_epoch_data):
                    n_frames_epoch = len(epoch_data)
                    if global_frame < offset + n_frames_epoch:
                        within = global_frame - offset
                        traj_lens = all_trajectory_lengths[epoch_idx] if epoch_idx < len(all_trajectory_lengths) else []
                        if not traj_lens:
                            return None
                        cum = 0
                        for traj_idx, L in enumerate(traj_lens):
                            if within < cum + L:
                                return epoch_idx, traj_idx, within - cum
                            cum += L
                        return None
                    offset += n_frames_epoch
                return None

            def _coords_for_global_frame(global_frame: int) -> Optional[Any]:
                loc = _locate_history_frame(global_frame)
                if loc is None:
                    return None
                epoch_idx, traj_idx, local_frame = loc
                if md_top_protein is None:
                    return None
                if epoch_idx >= len(all_epoch_traj_dcd_paths):
                    return None
                traj_paths = all_epoch_traj_dcd_paths[epoch_idx]
                if traj_idx >= len(traj_paths):
                    return None
                dcd_path = traj_paths[traj_idx]
                if not dcd_path:
                    return None
                # Map local frame index (within stored history) back to the original
                # index in the source DCD (after stride/filtering).
                src_frame_idx = int(local_frame)
                try:
                    if epoch_idx < len(all_epoch_traj_dcd_frame_indices):
                        seqs = all_epoch_traj_dcd_frame_indices[epoch_idx]
                        if traj_idx < len(seqs) and seqs[traj_idx] is not None:
                            seq = np.asarray(seqs[traj_idx], dtype=np.int64).reshape(-1)  # type: ignore[arg-type]
                            if 0 <= int(local_frame) < int(seq.shape[0]):
                                src_frame_idx = int(seq[int(local_frame)])
                except Exception:
                    src_frame_idx = int(local_frame)
                if use_full_replay_coords:
                    try:
                        coords_nm = np.asarray(hist_coords_flat[global_frame], dtype=np.float32)
                    except Exception:
                        coords_nm = None
                    return {
                        "protein_coords_nm": coords_nm,
                        "dcd_path": str(dcd_path),
                        "dcd_frame_idx": int(src_frame_idx),
                    }
                try:
                    tr = md.load_frame(str(dcd_path), index=int(src_frame_idx), top=md_top_protein)
                    if tr.n_frames == 0:
                        return None
                    return np.asarray(tr.xyz[0], dtype=np.float32)
                except Exception:
                    return None

            replay_coords = []
            start_cvs: List[Optional[np.ndarray]] = []
            for w_idx in range(config.grp_size):
                frame_idx = int(state_indices[w_idx].item())
                if 0 <= frame_idx < len(hist_cv_flat):
                    start_cvs.append(np.asarray(hist_cv_flat[frame_idx], dtype=np.float32))
                else:
                    start_cvs.append(None)

                coords = None
                if 0 <= frame_idx < len(hist_coords_flat):
                    coords = _coords_for_global_frame(frame_idx)
                    if coords is None:
                        coords = hist_coords_flat[frame_idx]
                replay_coords.append(coords)

            # Convert CVs to tensor (preserve walker order)
            n_cvs = int(envs[0].n_cvs)
            starts_arr = np.zeros((config.grp_size, n_cvs), dtype=np.float32)
            for w_idx in range(config.grp_size):
                cv = start_cvs[w_idx]
                if cv is None or cv.shape[0] != n_cvs:
                    starts_arr[w_idx] = (np.random.randn(n_cvs).astype(np.float32) * np.pi)
                else:
                    starts_arr[w_idx] = cv
            starts = torch.tensor(starts_arr, device=torch_device, dtype=torch.float32)
        
        # Hill bias disabled; provide zero scales/sigmas to satisfy sampler
        a_h_scales = torch.zeros(config.grp_size, config.num_mixtures, device=torch_device)
        a_sigmas = torch.zeros(config.grp_size, config.num_mixtures, envs[0].n_cvs, device=torch_device)
        
        # C. Sampling
        epoch_dir = None
        if config.save_epoch_coords and config.snapshot_dir:
            epoch_dir = Path(config.snapshot_dir) / f"epoch_{ep+1:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
        
        current_epoch_data, reward_tensors, metrics, active_mask_list = sampler.sample_epoch(
            ep,
            starts,
            None,  # pairwise_starts
            rand_mask,
            a_h_scales,
            a_sigmas,
            epoch_dir,
            replay_coords=replay_coords,
        )

        # Track launch reward baseline even before VAMP augmentation
        launch_rewards = reward_tensors.get('launch', torch.zeros(config.grp_size, device=torch_device))
        if launch_rewards.numel() == 0:
            launch_rewards = torch.zeros(config.grp_size, device=torch_device)
        metrics['avg_launch'] = float(launch_rewards.detach().mean().item())
        
        all_epoch_data.append(current_epoch_data)
        all_epoch_vamp_exclude.append(False)

        if 'trajectory_lengths' in metrics:
            all_trajectory_lengths.append(metrics['trajectory_lengths'])
        else:
            all_trajectory_lengths.append([])

        traj_lens = all_trajectory_lengths[-1]
        traj_dcd_paths = metrics.get('trajectory_dcd_paths', None)
        if traj_dcd_paths is None or len(traj_dcd_paths) != len(traj_lens):
            traj_dcd_paths = [None] * len(traj_lens)
        all_epoch_traj_dcd_paths.append(traj_dcd_paths)
        traj_dcd_frame_idx = [None] * len(traj_dcd_paths)
        all_epoch_traj_dcd_frame_indices.append(traj_dcd_frame_idx)

        # D. Compute LOO VAMP + Uncertainty Rewards (if ensemble is trained)
        loo_vamp_rewards = np.zeros(config.grp_size, dtype=np.float32)
        loo_vamp_contributions = None  # Store raw contributions for statistics
        vamp_baseline = None
        if len(current_epoch_data) > 0 and 'trajectory_lengths' in metrics:
            if proj_mode == "geom2vec_vamp":
                # Geom2Vec VAMP: compute LOO contributions using the lobe probabilities.
                if _geom2vec_is_trained() and geom2vec_cache is not None and geom2vec_projector is not None:
                    try:
                        lag_embed = int(geom2vec_lagtime_embed)
                        n_states = int(getattr(config, "geom2vec_dim", getattr(config, "tica_dim", 2)) or 2)

                        # Segment current_epoch_data into per-walker coordinate arrays (protein coords in nm).
                        traj_lengths_raw = [int(x) for x in metrics.get('trajectory_lengths', [])]
                        cursor = 0
                        cached_keys_epoch: List[Optional[str]] = []
                        traj_pairs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []

                        for walker_idx in range(config.grp_size):
                            L_i = traj_lengths_raw[walker_idx] if walker_idx < len(traj_lengths_raw) else 0
                            dcd_path = traj_dcd_paths[walker_idx] if walker_idx < len(traj_dcd_paths) else None

                            coords_nm = None
                            if L_i > 0:
                                seg = current_epoch_data[cursor : cursor + L_i]
                                try:
                                    coords_list = [np.asarray(fr[1], dtype=np.float32) for fr in seg if fr and fr[1] is not None]
                                    if len(coords_list) == L_i:
                                        coords_nm = np.asarray(coords_list, dtype=np.float32)
                                except Exception:
                                    coords_nm = None
                            cursor += max(0, L_i)

                            key = _geom2vec_cache_key_for_history_traj(len(all_epoch_data) - 1, walker_idx, dcd_path)
                            cached_key = _geom2vec_ensure_cached(
                                key=key,
                                dcd_path=dcd_path,
                                coords_nm=coords_nm,
                                overwrite=False,
                                meta={"epoch_idx": int(len(all_epoch_data) - 1), "traj_idx": int(walker_idx), "raw_frames": int(L_i)},
                            )
                            cached_keys_epoch.append(cached_key)

                        # Single-model LOO VAMP (no ensemble/UCB).
                        lam = 0.0
                        ensemble = None
                        n_models = 1
                        perm_maps = [np.asarray(geom2vec_state_permutation_map, dtype=int)]

                        # Seed global (history) sufficient statistics, excluding the just-appended epoch.
                        _ensure_loo_vamp_history_stats(
                            upto_epoch_exclusive=max(0, len(all_epoch_data) - 1),
                            n_models=n_models,
                            n_states=n_states,
                            lag_embed=lag_embed,
                            perm_maps=perm_maps,
                            tag=f"epoch {ep+1} pre-reward",
                        )
                        if loo_vamp_stats_by_model is None or loo_vamp_stats_signature != (n_models, n_states, lag_embed):
                            raise RuntimeError("Global LOO VAMP stats not initialized.")

                        per_model_rewards = np.zeros((n_models, config.grp_size), dtype=np.float32)
                        per_model_contribs = np.zeros((n_models, config.grp_size), dtype=np.float32)
                        vamp_baselines: List[float] = []
                        pair_counts = np.zeros(config.grp_size, dtype=np.int64)

                        # Build per-trajectory (current epoch) prob pairs for all ensemble members, reusing
                        # a single embedding load per trajectory key.
                        use_member_idx = ensemble is not None and len(ensemble) > 0
                        traj_pairs_by_model: List[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = [
                            [None for _ in range(len(cached_keys_epoch))] for _ in range(n_models)
                        ]

                        for t_i, cached_key in enumerate(cached_keys_epoch):
                            if cached_key is None:
                                continue
                            try:
                                graph, ca = geom2vec_cache.load(cached_key)
                            except Exception:
                                continue
                            for m in range(n_models):
                                try:
                                    probs_raw = geom2vec_projector.state_probs_from_embeddings(
                                        graph,
                                        ca,
                                        member_idx=m if use_member_idx else None,
                                    )
                                    probs = probs_raw[:, perm_maps[m]]
                                    if probs.shape[1] != n_states:
                                        if probs.shape[1] > n_states:
                                            probs = probs[:, :n_states]
                                        else:
                                            probs = np.pad(
                                                probs, ((0, 0), (0, n_states - probs.shape[1])), mode="constant"
                                            )
                                    T = int(probs.shape[0])
                                    if T > lag_embed:
                                        traj_pairs_by_model[m][t_i] = (probs[:-lag_embed], probs[lag_embed:])
                                except Exception:
                                    traj_pairs_by_model[m][t_i] = None

                        # Add current epoch trajectories to global stats (once per key, across all models).
                        for t_i, cached_key in enumerate(cached_keys_epoch):
                            if cached_key is None:
                                continue
                            if cached_key in loo_vamp_stats_keys:
                                continue
                            for m in range(n_models):
                                pair = traj_pairs_by_model[m][t_i]
                                if pair is not None:
                                    loo_vamp_stats_by_model[m].add_trajectory(pair[0], pair[1])
                            loo_vamp_stats_keys.add(str(cached_key))

                        for m in range(n_models):
                            stats = loo_vamp_stats_by_model[m]
                            vamp_baseline = float(stats.compute_vamp2_score())
                            vamp_baselines.append(vamp_baseline)

                            for i, pair in enumerate(traj_pairs_by_model[m]):
                                if pair is None:
                                    continue
                                stats.remove_trajectory(pair[0], pair[1])
                                vamp_wo = float(stats.compute_vamp2_score())
                                contrib = float(vamp_baseline - vamp_wo)
                                per_model_contribs[m, i] = contrib
                                stats.add_trajectory(pair[0], pair[1])
                                if m == 0:
                                    pair_counts[i] = max(1, int(pair[0].shape[0]))
                                # No pair normalization for reward computation.
                                per_model_rewards[m, i] = contrib

                        loo_mean = per_model_rewards.mean(axis=0)
                        loo_std = per_model_rewards.std(axis=0)
                        loo_vamp_rewards = (loo_mean + lam * loo_std).astype(np.float32, copy=False)
                        loo_vamp_contributions = per_model_contribs.mean(axis=0).astype(np.float32, copy=False)

                        print(f"\n[LOO VAMP Statistics - Epoch {ep+1}]")
                        if n_models > 1:
                            print(f"  Geom2Vec VAMP ensemble: n_models={n_models}, lambda={lam:.3f}")
                            print(
                                "  Baseline VAMP-2 per model (history+epoch trajectories): "
                                + ", ".join(f"{b:.4f}" for b in vamp_baselines)
                            )
                        else:
                            print(f"  Baseline VAMP-2 (history+epoch trajectories): {float(vamp_baselines[0]):.4f}")

                        # Per-trajectory breakdown (raw Δ, pair-normalized Δ, and final UCB reward).
                        raw_mean = per_model_contribs.mean(axis=0)
                        raw_std = per_model_contribs.std(axis=0)
                        score_label = f"ucb(μ+λσ,λ={lam:.2f})" if lam != 0.0 else "score(μ)"
                        print(f"  Per-trajectory contributions (delta_i = S_all - S_minus_i):")
                        for i in range(config.grp_size):
                            if pair_counts[i] <= 0:
                                continue
                            print(
                                f"    Traj {i:2d}: "
                                f"raw={float(raw_mean[i]):+.4f}±{float(raw_std[i]):.4f}  "
                                f"pairs={int(pair_counts[i])}  "
                                f"{score_label}={float(loo_vamp_rewards[i]):+.4f}"
                            )
                        print(
                            f"  Raw Δ summary: mean={float(raw_mean.mean()):.4f}, std={float(raw_mean.std()):.4f}, "
                            f"min={float(raw_mean.min()):.4f}, max={float(raw_mean.max()):.4f}"
                        )
                        if lam != 0.0:
                            print(
                                f"  UCB rewards (μ+λσ over models): mean={loo_vamp_rewards.mean():.4f}, std={loo_vamp_rewards.std():.4f}"
                            )
                        else:
                            print(f"  Mean rewards (μ over models): mean={loo_vamp_rewards.mean():.4f}, std={loo_vamp_rewards.std():.4f}")
                    except Exception as exc:
                        print(f"[LOO VAMP] Warning: Failed to compute geom2vec LOO rewards: {exc}")
                        import traceback
                        traceback.print_exc()
                        loo_vamp_rewards = np.zeros(config.grp_size, dtype=np.float32)

        # E. Policy Update (GRPO)
        if len(current_epoch_data) > 0:
            hill_data = None
            
            # Prepare Start Data (categorical over discrete states)
            start_features = None
            raw_start_actions = None
            pair_actions_for_opt = None
            chan_mask = None

            # Ensure launch rewards live on the training device
            launch_rewards = reward_tensors.get('launch', torch.zeros(config.grp_size, device=torch_device))
            if launch_rewards.device != torch_device:
                launch_rewards = launch_rewards.to(torch_device)
                reward_tensors['launch'] = launch_rewards

            # Combine launch rewards with LOO VAMP rewards
            loo_weight = getattr(config, 'loo_vamp_reward_weight', 1.0)
            if loo_vamp_rewards is not None and np.any(np.abs(loo_vamp_rewards) > 1e-8):
                loo_vamp_tensor = torch.tensor(loo_vamp_rewards, device=torch_device, dtype=torch.float32)
                combined_rewards = launch_rewards + loo_weight * loo_vamp_tensor
                reward_tensors['launch'] = combined_rewards
                metrics['avg_hill'] = float(combined_rewards.mean().item())
                print(f"[Rewards] Combined launch + LOO VAMP (weight={loo_weight:.2f}): "
                      f"launch_mean={launch_rewards.mean():.4f}, loo_mean={loo_vamp_tensor.mean():.4f}, "
                      f"combined_mean={combined_rewards.mean():.4f}")
            else:
                # Keep logging consistent: `avg_hill` reflects the scalar reward
                # used for start-policy updates in this geom2vec_vamp-only loop.
                try:
                    metrics['avg_hill'] = float(launch_rewards.mean().item())
                except Exception:
                    pass

            # Optional off-action discount: downweight rewards for starts whose requested
            # torsion/CV targets cannot be realized by reset/steering (e.g., unphysical jumps).
            off_coef = float(getattr(config, "off_action_penalty_coef", 0.0) or 0.0)
            if off_coef > 0.0:
                reset_mae = metrics.get("reset_cv_mae", None)
                if reset_mae is not None:
                    try:
                        reset_mae_t = torch.as_tensor(reset_mae, device=torch_device, dtype=torch.float32)
                        reset_mae_t = torch.nan_to_num(reset_mae_t, nan=0.0, posinf=0.0, neginf=0.0)
                        reset_mae_t = torch.clamp(reset_mae_t, min=0.0)
                    except Exception:
                        reset_mae_t = torch.zeros(config.grp_size, device=torch_device, dtype=torch.float32)

                    active_mask = metrics.get(
                        "active_mask",
                        torch.ones(config.grp_size, dtype=torch.bool, device=torch_device),
                    )
                    if torch.is_tensor(active_mask) and active_mask.device != torch_device:
                        active_mask = active_mask.to(torch_device)
                    if torch.is_tensor(active_mask) and active_mask.dtype != torch.bool:
                        active_mask = active_mask.bool()

                    rewards_for_start = reward_tensors.get("launch", launch_rewards)
                    if rewards_for_start.device != torch_device:
                        rewards_for_start = rewards_for_start.to(torch_device)
                    discount = torch.exp((-off_coef) * reset_mae_t)
                    discount = torch.clamp(discount, min=0.0, max=1.0)
                    discount = torch.where(active_mask, discount, torch.ones_like(discount))
                    rewards_for_start = rewards_for_start * discount
                    reward_tensors["launch"] = rewards_for_start
                    try:
                        metrics["avg_hill"] = float(rewards_for_start.mean().item())
                    except Exception:
                        pass

                    try:
                        mae_active = reset_mae_t[active_mask]
                        disc_active = discount[active_mask]
                        metrics["avg_reset_cv_mae"] = float(mae_active.mean().item()) if mae_active.numel() else 0.0
                        metrics["avg_off_action_discount"] = float(disc_active.mean().item()) if disc_active.numel() else 1.0
                        print(
                            f"[Rewards] Off-action discount (coef={off_coef:.2e}): "
                            f"reset_mae_mean={metrics['avg_reset_cv_mae']:.4f} rad, "
                            f"discount_mean={metrics['avg_off_action_discount']:.4f}"
                        )
                    except Exception:
                        print(f"[Rewards] Off-action discount enabled (coef={off_coef:.2e}).")

            if getattr(config, 'discrete_state_starts', False):
                start_features = torch.zeros(config.grp_size, 1, device=torch_device)
                raw_start_actions = start_state_actions.to(torch_device) if start_state_actions is not None else torch.zeros(config.grp_size, dtype=torch.long, device=torch_device)
            elif config.use_transformer_policy:
                # Reconstruct residue features
                feats_list = []
                for w_idx, env in enumerate(envs):
                    feat = env.per_residue_torsion_features()
                    feats_list.append(feat)
                residue_feature_batch = torch.tensor(np.array(feats_list), device=torch_device, dtype=torch.float32)
                start_features = residue_feature_batch
                
                # Sample start actions from old policy
                with torch.no_grad():
                    dist, pair_dist = agent_old(residue_feature_batch)
                    raw_start_actions = dist.sample()
                    
                    if pair_dist is not None:
                        pair_actions_raw = pair_dist.sample()
                        # Reshape for optimizer: [B, N*K] -> [B, N, K]
                        N = envs[0].n_residues
                        K = envs[0].max_partners
                        pair_actions_for_opt = pair_actions_raw.view(config.grp_size, N, K)
                
                # Build channel mask
                phi_exists = torch.from_numpy(envs[0]._phi_exists).to(torch_device)
                psi_exists = torch.from_numpy(envs[0]._psi_exists).to(torch_device)
                chan_mask = torch.stack([phi_exists, phi_exists, psi_exists, psi_exists], dim=-1).unsqueeze(0)
                
            else:
                # MLP mode
                start_features = torch.stack([torch.tensor(env.state(), device=torch_device, dtype=torch.float32) for env in envs])
                
                with torch.no_grad():
                    dist_old_s = agent_old.start_distribution(config.grp_size)
                    raw_start_actions = dist_old_s.rsample()
            
            # Align mask device with rewards
            valid_mask = metrics.get('valid_mask', torch.ones(config.grp_size, dtype=torch.bool, device=torch_device))
            if valid_mask.device != launch_rewards.device:
                valid_mask = valid_mask.to(launch_rewards.device)

            # Exclude random starts from policy updates (they weren't from the policy)
            # rand_mask is True for random starts, so we need to mask them out
            if rand_mask is not None:
                if rand_mask.device != valid_mask.device:
                    rand_mask = rand_mask.to(valid_mask.device)
                # Only train on non-random starts
                valid_mask = valid_mask & (~rand_mask)
                n_random = rand_mask.sum().item()
                n_policy = (~rand_mask).sum().item()
                if n_random > 0:
                    print(f"[Policy Update] Masking {n_random} random starts, training on {n_policy} policy starts")

            start_data = {
                'features': start_features,
                'actions': raw_start_actions,
                'pair_actions': pair_actions_for_opt,
                'rewards': reward_tensors.get('launch', torch.zeros(config.grp_size, device=torch_device)),
                'valid_mask': valid_mask,
                'use_transformer': config.use_transformer_policy,
                'chan_mask': chan_mask
            }
            
            # Perform GRPO update
            update_stats = optimizer.update(start_data, hill_data)
            
            # Sync Old Policy (every sync_interval epochs)
            if config.sync_interval > 0 and (ep + 1) % config.sync_interval == 0:
                agent_old.load_state_dict(agent.state_dict())
                print(f"[Policy] Synced old policy at epoch {ep+1}")
        else:
            update_stats = {}
        
        # F. Logging & Checkpointing
        print(f"[Epoch {ep+1}] Phase E: Logging & checkpointing...")
        print(f"Epoch {ep+1} Complete. Avg Reward: {metrics.get('avg_hill', 0.0):.4f}")
        
        if stats_path:
            avg_loo_vamp = loo_vamp_rewards.mean() if loo_vamp_rewards is not None else 0.0
            pg_s_val = update_stats.get('pg_s', 0.0) if update_stats else 0.0
            ent_val = update_stats.get('entropy', 0.0) if update_stats else 0.0
            loss_val = update_stats.get('loss', 0.0) if update_stats else 0.0
            adv_s_mean = update_stats.get('adv_s_mean', 0.0) if update_stats else 0.0
            adv_s_std = update_stats.get('adv_s_std', 0.0) if update_stats else 0.0
            kl_s_val = update_stats.get('kl_s', 0.0) if update_stats else 0.0
            avg_launch = metrics.get('avg_launch', 0.0)

            # Keep header alignment but zero unused hill/diagnostic fields
            row = (
                f"{ep+1},0,{len(current_epoch_data)},0.0,{pg_s_val:.6f},"
                f"0.0,{ent_val:.6f},{loss_val:.6f},0.0,0.0,"
                f"{adv_s_mean:.6f},{adv_s_std:.6f},"
                f"0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
                f"0.0,{kl_s_val:.6f},0.0,0.0,"
                f"{avg_launch:.6f},{metrics.get('avg_hill', 0.0):.6f},"
                f"0.000000,0.000000,{avg_loo_vamp:.6f}\n"
            )
            with stats_path.open("a") as f:
                f.write(row)
        
        
        # H. Checkpointing
        if config.checkpoint_interval > 0 and (ep + 1) % config.checkpoint_interval == 0:
            ckpt_manager.save(
                ep, agent, agent_old, policy_opts,
                {},
                None,
                vars(args)
            )
            
        
        # I. Plotting
        if config.plot_freq > 0 and (ep + 1) % config.plot_freq == 0:
            all_cv_samples = [item[0] for epoch_data in all_epoch_data for item in epoch_data]
            all_cv_array = np.asarray(all_cv_samples, dtype=np.float32)
            
            plot_ramachandran_by_residue(
                all_cv_array,
                envs[0].cv_names,
                f"ramachandran_ep{ep+1:03d}.png"
            )
            
            plot_analysis_summary(
                all_epoch_data[round_epoch_data_start:],
                None,
                envs[0].cv_names,
                config.pdb,
                f"analysis_summary_ep{ep+1:03d}.png",
                feature_builder=feature_builder
            )
        
        print(f"[Epoch {ep+1}] Complete in {time.time() - epoch_start_time:.1f}s.")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Training Complete.")
