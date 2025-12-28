import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import TrainingConfig
from .rewards import RewardCalculator
from ..utils.constants import MOLAR_GAS_CONSTANT_R
from openmm.unit import kilojoule_per_mole, kelvin

class Sampler:
    """
    Handles the sampling loop: environment reset, stepping, and reward collection.
    """
    def __init__(self, 
                 config: TrainingConfig, 
                 envs: List, 
                 reward_calculator: RewardCalculator,
                 device: str):
        self.config = config
        self.envs = envs
        self.rewards = reward_calculator
        self.device = device

    @staticmethod
    def _to_kjmol(x) -> float:
        try:
            return float(x.value_in_unit(kilojoule_per_mole))
        except Exception:
            return float(x)

    def _save_weights(
        self,
        *,
        epoch_dir: Optional[Path],
        ep: int,
        walker_idx: int,
        trajectory_data: List[Any],
    ) -> None:
        if epoch_dir is None or not trajectory_data:
            return
        try:
            energies = np.asarray([self._to_kjmol(frame[2]) for frame in trajectory_data], dtype=np.float32)
            biases = np.asarray([self._to_kjmol(frame[3]) for frame in trajectory_data], dtype=np.float32)
            beta = 1.0 / (
                MOLAR_GAS_CONSTANT_R.value_in_unit(kilojoule_per_mole / kelvin) * float(self.config.temperature)
            )
            logw_bias = beta * biases
            weights_path = epoch_dir / f"epoch_{ep+1:04d}_walker_{walker_idx:02d}_weights.npz"
            np.savez_compressed(weights_path, energy=energies, bias=biases, logw_bias=logw_bias)
        except Exception:
            return

    def _run_gpu_walkers(self, gpu_id: int, walker_indices: List[int],
                         active_mask_list: List[bool], a_h_scales: torch.Tensor,
                         a_sigmas: torch.Tensor, epoch_dir: Optional[Path],
                         ep: int, starts: torch.Tensor) -> Dict[int, Tuple]:
        """
        Run all walkers assigned to a single GPU sequentially.
        
        Returns:
            Dict mapping walker_idx -> (hill_reward, trajectory_data, dcd_path, traj_length,
                                        r_vendi, r_pca, r_state)
        """
        results = {}

        for walker_idx in walker_indices:
            if not active_mask_list[walker_idx]:
                results[walker_idx] = (0.0, [], None, 0, 0.0, 0.0, 0.0)
                continue
            
            env = self.envs[walker_idx]
            h_scales = a_h_scales[walker_idx].tolist()
            sigmas = a_sigmas[walker_idx].tolist()
            
            trajectory_data, is_stable, dcd_path = env.step(
                h_scales, sigmas,
                record_frequency=self.config.record_stride,
                output_dir=epoch_dir,
                epoch_index=ep + 1,
                walker_idx=walker_idx,
                equilibration_steps=self.config.equilibration_steps,
            )

            self._save_weights(epoch_dir=epoch_dir, ep=ep, walker_idx=walker_idx, trajectory_data=trajectory_data)
            
            if not is_stable:
                results[walker_idx] = (0.0, trajectory_data, dcd_path, 0, 0.0, 0.0, 0.0)
                continue
            
            # Compute reward
            if len(trajectory_data) > 0:
                start_cv = starts[walker_idx]
                if torch.is_tensor(start_cv):
                    start_cv = start_cv.detach().cpu().numpy()
                else:
                    start_cv = np.asarray(start_cv)
                
                r_total, r_vendi, r_pca, r_state = self.rewards.compute_trajectory_reward(
                    trajectory_data, env, start_cv,
                    None,
                    None,
                    walker_idx,
                )
                
                results[walker_idx] = (r_total, trajectory_data, dcd_path, len(trajectory_data),
                                      r_vendi, r_pca, r_state)
            else:
                results[walker_idx] = (0.0, trajectory_data, dcd_path, 0, 0.0, 0.0, 0.0)
        
        return results
    
    def sample_epoch(self,
                     ep: int,
                     starts: torch.Tensor,
                     pairwise_starts: Optional[torch.Tensor],
                     rand_mask: torch.Tensor,
                     a_h_scales: torch.Tensor,
                     a_sigmas: torch.Tensor,
                     epoch_dir: Optional[Path] = None,
                     replay_coords: Optional[List[Any]] = None,
                     replay_global_targets: Optional[torch.Tensor] = None,
                     replay_global_token_names: Optional[List[str]] = None) -> Tuple[List[Any], Dict[str, torch.Tensor], Dict[str, Any], List[bool]]:
        """
        Run a full sampling epoch across all walkers.
        
        Returns:
            - all_epoch_data: List of trajectory data (cv, coords) for history.
            - reward_tensors: Dict containing 'launch', 'hill' (trajectory) reward tensors.
            - metrics: Dict of scalar metrics for logging.
            - active_mask_list: List of booleans indicating active walkers.
        """
        # 1. Reset & Launch Rewards
        active_mask_list = []
        svd_reset_flags = []
        r_launch_vals = []
        print(f"Epoch {ep+1}/{self.config.episodes} - Resetting walkers...")
        
        svd_logged = False
        svd_triggered = False
        for idx, (env, s) in enumerate(zip(self.envs, starts)):
            # Reset
            p_target = None
            if pairwise_starts is not None:
                p_target = pairwise_starts[idx].detach().cpu().numpy()

            explicit_coords = None
            if replay_coords is not None and idx < len(replay_coords):
                payload = replay_coords[idx]
                if isinstance(payload, dict):
                    protein_coords = payload.get("protein_coords_nm", None)
                    dcd_path = payload.get("dcd_path", None)
                    dcd_frame_idx = payload.get("dcd_frame_idx", None)

                    loaded_full = None
                    if (
                        bool(getattr(self.config, "explicit_solvent_replay_full_coords", False))
                        and bool(getattr(env, "explicit_solvent", False))
                        and dcd_path
                        and dcd_frame_idx is not None
                    ):
                        try:
                            import mdtraj as md

                            tr = md.load_frame(str(dcd_path), index=int(dcd_frame_idx), top=env.md_topology)
                            if tr.n_frames > 0:
                                loaded_full = np.asarray(tr.xyz[0], dtype=np.float32)
                        except Exception:
                            loaded_full = None

                    if loaded_full is not None:
                        explicit_coords = loaded_full
                    elif protein_coords is not None:
                        explicit_coords = np.asarray(protein_coords, dtype=np.float32)
                    else:
                        explicit_coords = None
                else:
                    explicit_coords = payload

            global_targets = None
            if replay_global_targets is not None:
                try:
                    g = replay_global_targets[idx]
                    if torch.is_tensor(g):
                        global_targets = g.detach().cpu().numpy()
                    else:
                        global_targets = np.asarray(g)
                except Exception:
                    global_targets = None

            # If this walker was explicitly marked for dropping this epoch
            # (e.g., requested discrete state has no replay frames), skip
            # the reset entirely and treat it as inactive.
            drop_epoch = bool(getattr(env, "_drop_this_epoch", False))
            if drop_epoch:
                svd_reset_flags.append(False)
                active_mask_list.append(False)
                r_launch_vals.append(0.0)
                print(f"  Walker {idx:02d} requested empty replay state; skipping reset and launch.")
                continue

            if explicit_coords is not None:
                env.reset(
                    s.tolist(),
                    pairwise_targets=p_target,
                    explicit_coords_nm=explicit_coords,
                    target_globals_nm=global_targets,
                    target_global_names=replay_global_token_names,
                )
            else:
                env.reset(
                    s.tolist(),
                    pairwise_targets=p_target,
                    target_globals_nm=global_targets,
                    target_global_names=replay_global_token_names,
                )

            # Track if this reset encountered an SVD (Kabsch) failure internally
            err = getattr(env, "last_reset_error", None)
            has_svd_issue = bool(err is not None and "SVD did not converge" in str(err))
            if has_svd_issue:
                env.reset_failed = True
                if not svd_logged:
                    print("[SVD Guard] Immediate trigger: detected 'SVD did not converge' during reset; policy refresh running now.")
                    svd_logged = True
                svd_triggered = True
            # Drop walkers whose reset failed (e.g., geometry/SVD issues).
            failed = bool(getattr(env, "reset_failed", False))
            svd_reset_flags.append(has_svd_issue)
            active_mask_list.append(not failed)
            
            if failed:
                print(f"  Walker {idx:02d} reset failed or dropped; skipping.")
                r_launch_vals.append(0.0)
                continue
            
            # Launch Reward
            r, _ = self.rewards.compute_launch_reward(env)
            r_launch_vals.append(r)

            if svd_triggered:
                break

        # If an SVD reset issue was hit, stop the epoch immediately and return
        # minimal bookkeeping so the training loop can refresh policy/optimizers.
        if svd_triggered:
            # Pad lists to full size to avoid shape mismatch errors with rand_mask
            remaining = len(self.envs) - len(active_mask_list)
            if remaining > 0:
                active_mask_list.extend([False] * remaining)
                r_launch_vals.extend([0.0] * remaining)

            r_launch = torch.tensor(r_launch_vals, device=self.device)
            active_mask = torch.tensor(active_mask_list, dtype=torch.bool, device=self.device)
            valid_mask = (~rand_mask) & active_mask
            reward_tensors = {
                'launch': r_launch,
                'hill': torch.zeros((len(self.envs),), device=self.device),
            }
            metrics = {
                'avg_launch': r_launch.mean().item() if r_launch.numel() else 0.0,
                'avg_hill': 0.0,
                'avg_vendi': 0.0,
                'avg_pca': 0.0,
                'avg_state': 0.0,
                'active_count': int(active_mask.sum().item()),
                'valid_count': int(valid_mask.sum().item()),
                'valid_mask': valid_mask,
                'active_mask': active_mask,
                'epoch_dcd_paths': [],
                'trajectory_walker_ids': [],
                'trajectory_dcd_paths': [],
                'generation_records': [],
                'trajectory_lengths': [],
                'had_svd_issue': True,
            }
            return [], reward_tensors, metrics, active_mask_list

        r_launch = torch.tensor(r_launch_vals, device=self.device)
        active_mask = torch.tensor(active_mask_list, dtype=torch.bool, device=self.device)
        
        # Valid mask for PPO (not random AND active)
        valid_mask = (~rand_mask) & active_mask
        num_active = int(active_mask.sum().item())
        
        print(f"Epoch {ep+1}/{self.config.episodes} - Launch rewards: {r_launch.tolist()} (active: {num_active}/{self.config.grp_size})")

        # 2. Sampling Loop
        n_walkers = len(self.envs)
        hill_rewards = [0.0] * n_walkers
        trajectory_lengths = [0] * n_walkers
        trajectory_dcd_paths: List[Optional[str]] = [None] * n_walkers
        # Store per-walker trajectories for deterministic concatenation and bookkeeping.
        per_walker_traj: List[List[Any]] = [[] for _ in range(n_walkers)]
        
        # Metrics containers
        vendi_rewards = []
        pca_rewards = []
        state_rewards = []
        
        # Collect generation records for saving
        epoch_generation_records = []
        for walker_idx, env in enumerate(self.envs):
            if env.last_generation_record:
                rec = env.last_generation_record
                # Deep copy to avoid reference issues
                epoch_generation_records.append({
                    'walker': walker_idx,
                    'phi_angles': rec['phi_angles'].copy(),
                    'psi_angles': rec['psi_angles'].copy(),
                    'generated_heavy_angstrom': rec['generated_heavy_angstrom'].copy(),
                    'pre_min_full_angstrom': rec['pre_min_full_angstrom'].copy(),
                    'relaxed_full_angstrom': rec['relaxed_full_angstrom'].copy(),
                    'relaxed_heavy_angstrom': rec['relaxed_heavy_angstrom'].copy(),
                    'timing': rec.get('timing', {}).copy(),
                })

        # Save generation records immediately if needed (can be moved out if preferred)
        if epoch_dir and epoch_generation_records:
            for rec in epoch_generation_records:
                out_path = epoch_dir / f"walker_{rec['walker']:02d}.npz"
                np.savez_compressed(out_path, **rec) # Simplified saving

        # Group walkers by GPU for parallel execution
        use_parallel = getattr(self.config, 'enable_multi_gpu', False)
        n_gpus = torch.cuda.device_count() if use_parallel else 1
        
        if use_parallel and n_gpus > 1:
            # Multi-GPU parallel execution
            gpu_to_walkers = {gpu_id: [] for gpu_id in range(n_gpus)}
            for idx in range(len(self.envs)):
                gpu_id = idx % n_gpus
                gpu_to_walkers[gpu_id].append(idx)
            
            print(f"[Multi-GPU] Sampling with {n_gpus} GPUs")
            
            # Thread-safe containers for metrics
            metrics_lock = threading.Lock()
            
            # One thread per GPU
            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = {
                    executor.submit(
                        self._run_gpu_walkers,
                        gpu_id, walker_list, active_mask_list,
                        a_h_scales, a_sigmas, epoch_dir, ep, starts
                    ): gpu_id
                    for gpu_id, walker_list in gpu_to_walkers.items()
                }
                
                # Collect results
                for future in tqdm(as_completed(futures), total=n_gpus, desc=f"Epoch {ep+1} Multi-GPU"):
                    gpu_results = future.result()
                    
                    for walker_idx, (r_total, traj_data, dcd_path, traj_len, r_vendi, r_pca, r_state) in gpu_results.items():
                        hill_rewards[walker_idx] = float(r_total)
                        trajectory_lengths[walker_idx] = int(traj_len)
                        per_walker_traj[walker_idx] = traj_data or []
                        
                        if r_vendi != 0.0:
                            with metrics_lock:
                                vendi_rewards.append(r_vendi)
                                pca_rewards.append(r_pca)
                                state_rewards.append(r_state)
                        
                        if dcd_path:
                            trajectory_dcd_paths[walker_idx] = str(dcd_path)
        else:
            # Sequential execution (single GPU or multi-GPU disabled)
            for walker_idx, env in tqdm(enumerate(self.envs), total=len(self.envs), desc=f"Epoch {ep+1} Sampling"):
                if not active_mask_list[walker_idx]:
                    hill_rewards[walker_idx] = 0.0
                    trajectory_lengths[walker_idx] = 0
                    per_walker_traj[walker_idx] = []
                    continue
                    
                h_scales = a_h_scales[walker_idx].tolist()
                sigmas = a_sigmas[walker_idx].tolist()
                
                trajectory_data, is_stable, dcd_path = env.step(
                    h_scales, sigmas,
                    record_frequency=self.config.record_stride,
                    output_dir=epoch_dir,
                    epoch_index=ep + 1,
                    walker_idx=walker_idx,
                    equilibration_steps=self.config.equilibration_steps,
                )
                
                self._save_weights(epoch_dir=epoch_dir, ep=ep, walker_idx=walker_idx, trajectory_data=trajectory_data)
                
                if dcd_path:
                    trajectory_dcd_paths[walker_idx] = str(dcd_path)
                    
                if not is_stable:
                    hill_rewards[walker_idx] = 0.0
                    trajectory_lengths[walker_idx] = 0
                    per_walker_traj[walker_idx] = []
                else:
                    if len(trajectory_data) > 0:
                        per_walker_traj[walker_idx] = trajectory_data
                        start_cv = starts[walker_idx]
                        if torch.is_tensor(start_cv):
                            start_cv = start_cv.detach().cpu().numpy()
                        else:
                            start_cv = np.asarray(start_cv)
                        
                        r_total, r_vendi, r_pca, r_state = self.rewards.compute_trajectory_reward(
                            trajectory_data, env, start_cv,
                            None,
                            None,
                            walker_idx,
                        )
                        
                        hill_rewards[walker_idx] = float(r_total)
                        vendi_rewards.append(r_vendi)
                        pca_rewards.append(r_pca)
                        state_rewards.append(r_state)
                        trajectory_lengths[walker_idx] = len(trajectory_data)
                    else:
                        hill_rewards[walker_idx] = 0.0
                        trajectory_lengths[walker_idx] = 0
                        per_walker_traj[walker_idx] = []

        # Deterministic flattening: concatenate trajectories in walker order
        current_epoch_data: List[Any] = []
        for walker_idx in range(n_walkers):
            current_epoch_data.extend(per_walker_traj[walker_idx])

        epoch_dcd_paths = [(walker_idx, path) for walker_idx, path in enumerate(trajectory_dcd_paths) if path]
        trajectory_walker_ids = list(range(n_walkers))

        # 3. Package Results
        reward_tensors = {
            'launch': r_launch,
            'hill': torch.tensor(hill_rewards, device=self.device, dtype=torch.float32)
        }
        
        metrics = {
            'avg_launch': r_launch.mean().item(),
            'avg_hill': reward_tensors['hill'].mean().item(),
            'avg_vendi': float(np.mean(vendi_rewards)) if vendi_rewards else 0.0,
            'avg_pca': float(np.mean(pca_rewards)) if pca_rewards else 0.0,
            'avg_state': float(np.mean(state_rewards)) if state_rewards else 0.0,
            'active_count': num_active,
            'valid_count': int(valid_mask.sum().item()),
            'valid_mask': valid_mask,
            'active_mask': active_mask,
            'epoch_dcd_paths': epoch_dcd_paths,
            'trajectory_walker_ids': trajectory_walker_ids,
            'trajectory_dcd_paths': trajectory_dcd_paths,
            'generation_records': epoch_generation_records,
            'trajectory_lengths': trajectory_lengths,  # Include trajectory lengths
            'had_svd_issue': any(svd_reset_flags),
        }
        
        return current_epoch_data, reward_tensors, metrics, active_mask_list
