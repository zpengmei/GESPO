import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import TrainingConfig
from ..utils.constants import MOLAR_GAS_CONSTANT_R
from openmm.unit import kilojoule_per_mole, kelvin

class Sampler:
    """
    Handles the sampling loop: environment reset, stepping, and trajectory collection.
    Rewards are computed externally in the training loop.
    """
    def __init__(self, 
                 config: TrainingConfig, 
                 envs: List, 
                 reward_calculator: Optional[Any] = None,  # kept for API compat, unused
                 device: str = "cpu"):
        self.config = config
        self.envs = envs
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

    def _run_gpu_resets(
        self,
        gpu_id: int,
        walker_indices: List[int],
        starts: torch.Tensor,
        pairwise_starts: Optional[torch.Tensor],
        replay_coords: Optional[List[Any]],
        replay_torsion_targets_residue: Optional[torch.Tensor],
        replay_global_targets: Optional[torch.Tensor],
        replay_global_token_names: Optional[List[str]],
    ) -> Dict[int, Tuple[bool, bool, float, float, float, float]]:
        """
        Reset all walkers assigned to a single GPU sequentially.

        Returns:
            Dict mapping walker_idx -> (active, reset_mae, reset_rmse, reset_maxabs)
        """
        results: Dict[int, Tuple[bool, float, float, float]] = {}

        for idx in walker_indices:
            env = self.envs[idx]
            s = starts[idx]

            p_target = None
            if pairwise_starts is not None:
                try:
                    p_target = pairwise_starts[idx].detach().cpu().numpy()
                except Exception:
                    p_target = None

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

            torsion_targets_res = None
            if replay_torsion_targets_residue is not None:
                try:
                    t = replay_torsion_targets_residue[idx]
                    if torch.is_tensor(t):
                        torsion_targets_res = t.detach().cpu().numpy()
                    else:
                        torsion_targets_res = np.asarray(t)
                except Exception:
                    torsion_targets_res = None

            drop_epoch = bool(getattr(env, "_drop_this_epoch", False))
            if drop_epoch:
                results[idx] = (False, float("nan"), float("nan"), float("nan"))
                continue

            if explicit_coords is not None:
                env.reset(
                    s.tolist(),
                    pairwise_targets=p_target,
                    explicit_coords_nm=explicit_coords,
                    target_globals_nm=global_targets,
                    target_global_names=replay_global_token_names,
                    target_torsions_residue_rad=torsion_targets_res,
                )
            else:
                env.reset(
                    s.tolist(),
                    pairwise_targets=p_target,
                    target_globals_nm=global_targets,
                    target_global_names=replay_global_token_names,
                    target_torsions_residue_rad=torsion_targets_res,
                )

            failed = bool(getattr(env, "reset_failed", False))
            active = not failed

            mae = float("nan")
            rmse = float("nan")
            maxabs = float("nan")
            if active:
                try:
                    target_cv = np.asarray(s.detach().cpu().numpy() if torch.is_tensor(s) else s, dtype=np.float32)
                    actual_cv = np.asarray(getattr(env, "cv_values", None), dtype=np.float32)
                    if actual_cv.ndim != 1 or target_cv.ndim != 1 or actual_cv.shape[0] != target_cv.shape[0]:
                        raise ValueError("shape mismatch")
                    diff = target_cv - actual_cv
                    diff = ((diff + np.pi) % (2.0 * np.pi)) - np.pi
                    abs_diff = np.abs(diff)
                    mae = float(abs_diff.mean())
                    rmse = float(np.sqrt(np.mean(diff * diff)))
                    maxabs = float(abs_diff.max() if abs_diff.size else 0.0)
                except Exception:
                    mae = float("nan")
                    rmse = float("nan")
                    maxabs = float("nan")

            results[idx] = (active, mae, rmse, maxabs)

        return results

    def _run_gpu_walkers(self, gpu_id: int, walker_indices: List[int],
                         active_mask_list: List[bool], a_h_scales: torch.Tensor,
                         a_sigmas: torch.Tensor, epoch_dir: Optional[Path],
                         ep: int, starts: torch.Tensor) -> Dict[int, Tuple]:
        """
        Run all walkers assigned to a single GPU sequentially.
        
        Returns:
            Dict mapping walker_idx -> (trajectory_data, dcd_path, traj_length)
        """
        results = {}

        for walker_idx in walker_indices:
            if not active_mask_list[walker_idx]:
                results[walker_idx] = ([], None, 0)
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
                results[walker_idx] = (trajectory_data, dcd_path, 0)
            elif len(trajectory_data) > 0:
                results[walker_idx] = (trajectory_data, dcd_path, len(trajectory_data))
            else:
                results[walker_idx] = (trajectory_data, dcd_path, 0)
        
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
                     replay_torsion_targets_residue: Optional[torch.Tensor] = None,
                     replay_global_targets: Optional[torch.Tensor] = None,
                     replay_global_token_names: Optional[List[str]] = None) -> Tuple[List[Any], Dict[str, torch.Tensor], Dict[str, Any], List[bool]]:
        """
        Run a full sampling epoch across all walkers.
        
        Returns:
            - all_epoch_data: List of trajectory data (cv, coords) for history.
            - empty dict (kept for API compat).
            - metrics: Dict of scalar metrics for logging.
            - active_mask_list: List of booleans indicating active walkers.
        """
        # 1. Reset walkers
        active_mask_list = []
        # Track how closely the post-reset CVs match the requested targets.
        reset_cv_mae = []
        reset_cv_rmse = []
        reset_cv_maxabs = []

        print(f"Epoch {ep+1}/{self.config.episodes} - Resetting walkers...")

        n_walkers = len(self.envs)
        use_parallel_resets = bool(getattr(self.config, "enable_multi_gpu", False))
        # Guardrail: explicit-solvent replay full-coordinate resets can load DCD frames via mdtraj
        # (dcdplugin). That loader is not reliably thread-safe, and parallel resets can trigger
        # rare native crashes (e.g., "free(): invalid pointer"). Prefer sequential resets here.
        if bool(getattr(self.config, "explicit_solvent_replay_full_coords", False)) and use_parallel_resets:
            print("[Multi-GPU] Note: disabling parallel resets for explicit-solvent full-coordinate replay (mdtraj DCD loads are thread-unsafe).", flush=True)
            use_parallel_resets = False
        n_gpus_for_reset = torch.cuda.device_count() if use_parallel_resets else 1

        if use_parallel_resets and n_gpus_for_reset > 1:
            print(f"[Multi-GPU] Resetting {n_walkers} walkers across {n_gpus_for_reset} GPUs", flush=True)
            # Preallocate in walker order.
            active_mask_list = [False] * n_walkers
            reset_cv_mae = [float("nan")] * n_walkers
            reset_cv_rmse = [float("nan")] * n_walkers
            reset_cv_maxabs = [float("nan")] * n_walkers

            gpu_to_walkers = {gpu_id: [] for gpu_id in range(int(n_gpus_for_reset))}
            for idx in range(n_walkers):
                gpu_to_walkers[int(idx % n_gpus_for_reset)].append(int(idx))

            with ThreadPoolExecutor(max_workers=int(n_gpus_for_reset)) as executor:
                futures = {
                    executor.submit(
                        self._run_gpu_resets,
                        gpu_id,
                        walker_list,
                        starts,
                        pairwise_starts,
                        replay_coords,
                        replay_torsion_targets_residue,
                        replay_global_targets,
                        replay_global_token_names,
                    ): gpu_id
                    for gpu_id, walker_list in gpu_to_walkers.items()
                }
                for future in tqdm(as_completed(futures), total=int(n_gpus_for_reset), desc=f"Epoch {ep+1} Reset Multi-GPU"):
                    gpu_results = future.result()
                    for idx, (active, mae, rmse, maxabs) in gpu_results.items():
                        active_mask_list[int(idx)] = bool(active)
                        reset_cv_mae[int(idx)] = float(mae)
                        reset_cv_rmse[int(idx)] = float(rmse)
                        reset_cv_maxabs[int(idx)] = float(maxabs)
        else:
            for idx, (env, s) in enumerate(zip(self.envs, starts)):
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
                        explicit_solvent_replay = bool(getattr(self.config, "explicit_solvent_replay_full_coords", False))
                        env_explicit_solvent = bool(getattr(env, "explicit_solvent", False))

                        if (
                            explicit_solvent_replay
                            and env_explicit_solvent
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

                torsion_targets_res = None
                if replay_torsion_targets_residue is not None:
                    try:
                        t = replay_torsion_targets_residue[idx]
                        if torch.is_tensor(t):
                            torsion_targets_res = t.detach().cpu().numpy()
                        else:
                            torsion_targets_res = np.asarray(t)
                    except Exception:
                        torsion_targets_res = None

                # If this walker was explicitly marked for dropping this epoch
                # (e.g., requested discrete state has no replay frames), skip
                # the reset entirely and treat it as inactive.
                drop_epoch = bool(getattr(env, "_drop_this_epoch", False))
                if drop_epoch:
                    active_mask_list.append(False)
                    reset_cv_mae.append(float("nan"))
                    reset_cv_rmse.append(float("nan"))
                    reset_cv_maxabs.append(float("nan"))
                    print(f"  Walker {idx:02d} requested empty replay state; skipping reset.")
                    continue

                env.reset(
                    s.tolist(),
                    pairwise_targets=p_target,
                    explicit_coords_nm=explicit_coords,
                    target_globals_nm=global_targets,
                    target_global_names=replay_global_token_names,
                    target_torsions_residue_rad=torsion_targets_res,
                )

                # Drop walkers whose reset failed.
                failed = bool(getattr(env, "reset_failed", False))
                active_mask_list.append(not failed)

                if failed:
                    print(f"  Walker {idx:02d} reset failed; skipping.")
                    reset_cv_mae.append(float("nan"))
                    reset_cv_rmse.append(float("nan"))
                    reset_cv_maxabs.append(float("nan"))
                    continue

                # Deviation between requested CV targets and realized CVs after reset.
                try:
                    target_cv = np.asarray(s.detach().cpu().numpy() if torch.is_tensor(s) else s, dtype=np.float32)
                    actual_cv = np.asarray(getattr(env, "cv_values", None), dtype=np.float32)
                    if actual_cv.ndim != 1 or target_cv.ndim != 1 or actual_cv.shape[0] != target_cv.shape[0]:
                        raise ValueError("shape mismatch")
                    diff = target_cv - actual_cv
                    diff = ((diff + np.pi) % (2.0 * np.pi)) - np.pi
                    abs_diff = np.abs(diff)
                    reset_cv_mae.append(float(abs_diff.mean()))
                    reset_cv_rmse.append(float(np.sqrt(np.mean(diff * diff))))
                    reset_cv_maxabs.append(float(abs_diff.max() if abs_diff.size else 0.0))
                except Exception:
                    reset_cv_mae.append(float("nan"))
                    reset_cv_rmse.append(float("nan"))
                    reset_cv_maxabs.append(float("nan"))

        active_mask = torch.tensor(active_mask_list, dtype=torch.bool, device=self.device)
        
        # Valid mask for PPO (not random AND active)
        valid_mask = (~rand_mask) & active_mask
        num_active = int(active_mask.sum().item())
        
        print(f"Epoch {ep+1}/{self.config.episodes} - Reset complete (active: {num_active}/{self.config.grp_size})")

        # 2. Sampling Loop
        n_walkers = len(self.envs)
        trajectory_lengths = [0] * n_walkers
        trajectory_dcd_paths: List[Optional[str]] = [None] * n_walkers
        # Store per-walker trajectories for deterministic concatenation and bookkeeping.
        per_walker_traj: List[List[Any]] = [[] for _ in range(n_walkers)]
        
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
                    
                    for walker_idx, (traj_data, dcd_path, traj_len) in gpu_results.items():
                        trajectory_lengths[walker_idx] = int(traj_len)
                        per_walker_traj[walker_idx] = traj_data or []
                        
                        if dcd_path:
                            trajectory_dcd_paths[walker_idx] = str(dcd_path)
        else:
            # Sequential execution (single GPU or multi-GPU disabled)
            for walker_idx, env in tqdm(enumerate(self.envs), total=len(self.envs), desc=f"Epoch {ep+1} Sampling"):
                if not active_mask_list[walker_idx]:
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
                    
                if not is_stable or len(trajectory_data) == 0:
                    trajectory_lengths[walker_idx] = 0
                    per_walker_traj[walker_idx] = []
                else:
                    per_walker_traj[walker_idx] = trajectory_data
                    trajectory_lengths[walker_idx] = len(trajectory_data)

        # Deterministic flattening: concatenate trajectories in walker order
        current_epoch_data: List[Any] = []
        for walker_idx in range(n_walkers):
            current_epoch_data.extend(per_walker_traj[walker_idx])

        epoch_dcd_paths = [(walker_idx, path) for walker_idx, path in enumerate(trajectory_dcd_paths) if path]
        trajectory_walker_ids = list(range(n_walkers))

        # 3. Package Results
        metrics = {
            'active_count': num_active,
            'valid_count': int(valid_mask.sum().item()),
            'valid_mask': valid_mask,
            'active_mask': active_mask,
            'epoch_dcd_paths': epoch_dcd_paths,
            'trajectory_walker_ids': trajectory_walker_ids,
            'trajectory_dcd_paths': trajectory_dcd_paths,
            'generation_records': epoch_generation_records,
            'trajectory_lengths': trajectory_lengths,
            'reset_cv_mae': reset_cv_mae,
            'reset_cv_rmse': reset_cv_rmse,
            'reset_cv_maxabs': reset_cv_maxabs,
        }
        
        return current_epoch_data, {}, metrics, active_mask_list
