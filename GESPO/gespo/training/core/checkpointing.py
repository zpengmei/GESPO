import torch
import numpy as np
import glob
import os
import random
import mdtraj as md
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from ...config import TrainingConfig
from ...utils.geometry import calculate_rmsd
from simtk.unit import nanometer, kelvin
from gespo.utils.logging import log_print

class CheckpointManager:
    """Handles checkpoint saving, loading, and history bootstrapping."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None

    def save(self, 
             ep_done: int, 
             agent: torch.nn.Module, 
             agent_old: torch.nn.Module, 
             optimizers: list, 
             vendi_states: dict, 
             pca_scorer, 
             args_dict: dict):
        """Save training state to checkpoint."""
        if not self.ckpt_dir:
            return
            
        # Get Scorer State and strip large arrays
        scorer_state = pca_scorer.get_state() if hasattr(pca_scorer, 'get_state') else None
        if scorer_state is not None:
            scorer_state = scorer_state.copy()
            for k in ['base_features', 'feature_history', 'last_batch_coords']:
                if k in scorer_state:
                    scorer_state.pop(k)

        ckpt = {
            'ep_done': ep_done,
            'agent': agent.state_dict(),
            'agent_old': agent_old.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'vendi_launch_pca': vendi_states.get('launch_pca'),
            'vendi_hill_pca': vendi_states.get('hill_pca'),
            'vendi_launch_angle': vendi_states.get('launch_angle'),
            'vendi_hill_angle': vendi_states.get('hill_angle'),
            'pca_scorer': scorer_state,
            'np_rng': np.random.get_state(),
            'py_rng': random.getstate(),
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'args': args_dict,
        }
        
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"ckpt_ep_{ep_done+1:04d}.pt"
        tmp = path.with_suffix('.tmp')
        
        with tmp.open('wb') as f:
            torch.save(ckpt, f, pickle_protocol=4)
        tmp.replace(path)
        log_print(f"[Checkpoint] Saved to {path}")

        # Also save a standalone policy payload for convenience (easy loading
        # without parsing the full checkpoint dict).
        try:
            policy_payload: Dict[str, Any] = {
                "epoch": int(ep_done) + 1,
                "agent_state_dict": agent.state_dict(),
                "agent_old_state_dict": agent_old.state_dict(),
                "agent_class": agent.__class__.__name__,
                "agent_module": agent.__class__.__module__,
                "agent_old_class": agent_old.__class__.__name__,
                "agent_old_module": agent_old.__class__.__module__,
            }

            # Best-effort: include enough info to reconstruct common policy types.
            try:
                policy_cfg: Dict[str, Any] = {}
                for attr in (
                    "num_states",
                    "feature_dim",
                    "hidden_dim",
                    "num_residues",
                    "residue_vocab_size",
                    "d_model",
                    "nhead",
                    "num_layers",
                    "kmix",
                ):
                    if hasattr(agent, attr):
                        val = getattr(agent, attr)
                        try:
                            policy_cfg[attr] = int(val)
                        except Exception:
                            policy_cfg[attr] = val
                if policy_cfg:
                    policy_payload["policy_config"] = policy_cfg
            except Exception:
                pass

            policy_path = self.ckpt_dir / f"policy_ep_{ep_done+1:04d}.pt"
            policy_tmp = policy_path.with_suffix(".tmp")
            with policy_tmp.open("wb") as f:
                torch.save(policy_payload, f, pickle_protocol=4)
            policy_tmp.replace(policy_path)
            log_print(f"[Checkpoint] Saved policy to {policy_path}")
        except Exception as exc_policy:
            log_print(f"[Checkpoint] Warning: failed to save standalone policy: {exc_policy}")

    def load_latest(self):
        """Load the latest checkpoint if it exists."""
        if not self.ckpt_dir:
            return None, None
        
        ckpts = sorted(self.ckpt_dir.glob('ckpt_ep_*.pt'))
        if not ckpts:
            return None, None
            
        latest = ckpts[-1]
        log_print(f"[Checkpoint] Loading {latest}...")
        data = torch.load(latest, map_location='cpu')
        return latest, data

    def restore_states(self, 
                       agent: torch.nn.Module, 
                       agent_old: torch.nn.Module, 
                       policy_opts: List, 
                       data: Dict, 
                       torch_device: str) -> Tuple[int, Dict, Any]:
        """Restore model states from checkpoint data."""
        agent.load_state_dict(data['agent'])
        agent.to(torch_device)
        
        agent_old.load_state_dict(data['agent_old'])
        agent_old.to(torch_device).eval()
        
        for opt, state in zip(policy_opts, data.get('optimizers', [])):
            opt.load_state_dict(state)
        
        vendi_states = {}
        for key in ('vendi_launch_pca', 'vendi_hill_pca', 'vendi_launch_angle', 
                    'vendi_hill_angle'):
            if data.get(key):
                vendi_states[key] = data[key]
        
        pca_state = data.get('pca_scorer')
        
        # Restore RNG
        if data.get('np_rng') is not None: np.random.set_state(data['np_rng'])
        if data.get('py_rng') is not None: random.setstate(data['py_rng'])
        if data.get('torch_rng') is not None: torch.set_rng_state(data['torch_rng'])
        if torch.cuda.is_available() and data.get('torch_cuda_rng_all') is not None:
            torch.cuda.set_rng_state_all(data['torch_cuda_rng_all'])
        
        ep_start = int(data.get('ep_done', -1)) + 1
        return ep_start, vendi_states, pca_state

    def rebuild_history_from_npz(self, all_epoch_data: list, base_dir: Optional[str] = None):
        """Rebuild training history from existing NPZ epoch files.

        Returns:
            restored (int): number of NPZ epochs loaded
            traj_lengths (List[List[int]]): per-epoch trajectory-length lists
        """
        patterns = ['chignolin_epoch_*.npz', 'epoch_*.npz', 'history_epoch_*.npz']
        search_dir = Path(base_dir) if base_dir else Path(".")
        files = []
        for pat in patterns:
            files.extend(sorted(str(p) for p in search_dir.glob(pat)))
        files = sorted(set(files))
        if not files:
            return 0, []
        
        restored = 0
        traj_lengths = []
        for npz in files:
            try:
                d = np.load(npz, allow_pickle=True)
                if 'cvs' in d and 'coords' in d:
                    cvs = d['cvs']
                    coords = d['coords']
                    
                    # Validate non-empty arrays
                    if cvs.size == 0 or coords.size == 0:
                        log_print(f"[History] Skipping empty NPZ: {npz}")
                        continue
                    
                    # Validate shapes match
                    if len(cvs) != len(coords):
                        log_print(f"[History] Skipping NPZ with mismatched shapes: {npz}")
                        continue
                    
                    epoch_data = list(zip(list(cvs), list(coords)))
                    all_epoch_data.append(epoch_data)
                    restored += 1
                    # Treat the NPZ batch as a single trajectory for boundary tracking
                    traj_lengths.append([len(epoch_data)])
            except Exception as e:
                log_print(f"[History] Failed to load {npz}: {e}")
        return restored, traj_lengths

    def bootstrap_history_from_dcd(self, all_epoch_data: list, env, base_dir: str):
        """Bootstrap history from existing DCD trajectories.

        base_dir may be a single path or a comma/colon-separated list of paths.

        Returns:
            imported (int): total frames imported
            traj_lengths (List[int]): list of trajectory lengths aligned with the
                                      appended epoch in all_epoch_data (single
                                      entry representing the imported history).
        """
        if not base_dir:
            return 0, []
        
        dirs = []
        for token in str(base_dir).split(","):
            dirs.extend(token.split(":"))
        dirs = [Path(d) for d in dirs if d]
        dirs = [d for d in dirs if d.is_dir()]
        if not dirs:
            log_print(f"[History DCD] No valid directories found in '{base_dir}'.")
            return 0, []

        # Track DCD path ordering for downstream mapping/debugging.
        try:
            self.last_bootstrap_dcd_paths = []  # type: ignore[attr-defined]
        except Exception:
            pass
        # Track original frame indices (0-based) in each source DCD for the
        # imported frames (after applying stride and RMSD filtering). This is
        # required when replay resets need to restore full-system coordinates
        # from a particular frame in the source DCD.
        try:
            self.last_bootstrap_frame_indices = []  # type: ignore[attr-defined]
        except Exception:
            pass

        log_print("[History DCD] Bootstrapping from:")
        for d in dirs:
            log_print(f"  - {d}")
        
        # Trajectories written by env.step() are protein-only by default
        # (md_topology.subset("protein")), but in explicit-solvent full-coordinate
        # replay mode they may include the full solvated system. We load with a
        # protein-only topology when possible, and fall back to loading the full
        # system and slicing to protein when atom counts mismatch.
        md_top_full = env.md_topology
        protein_indices = md_top_full.select("protein")
        if protein_indices is None or len(protein_indices) == 0:
            protein_indices = np.arange(md_top_full.n_atoms, dtype=int)
        md_top = md_top_full.subset(protein_indices)

        state0 = env.sim.context.getState(getPositions=True)
        ref_full_nm = state0.getPositions(asNumpy=True).value_in_unit(nanometer)
        ref_positions_nm = np.asarray(ref_full_nm, dtype=np.float32)[np.asarray(protein_indices, dtype=int)]
        ref_traj = md.Trajectory(ref_positions_nm[None, :, :], md_top)
        ca_indices = [a.index for a in md_top.atoms if a.name == "CA"]

        dcd_files = []
        for d in dirs:
            dcd_files.extend(d.glob("**/*.dcd"))
        if not dcd_files:
            log_print("[History DCD] No .dcd files found.")
            return 0, []

        # Deterministic ordering so trajectory indices remain stable across runs.
        try:
            import re

            def _natural_sort_key(path: Path):
                text = path.as_posix()
                parts = re.split(r"(\d+)", text)
                key = []
                for part in parts:
                    if not part:
                        continue
                    if part.isdigit():
                        key.append((1, int(part)))
                    else:
                        key.append((0, part.lower()))
                return tuple(key)

            dcd_files = sorted(set(dcd_files), key=_natural_sort_key)
        except Exception:
            dcd_files = sorted(set(dcd_files), key=lambda p: p.as_posix())

        log_print(f"[History DCD] Found {len(dcd_files)} .dcd files.")

        # Optional progress bar if tqdm is available
        try:
            from tqdm.auto import tqdm  # type: ignore
        except Exception:
            tqdm = None

        imported = 0
        history_epoch = []
        traj_lengths = []
        traj_paths = []
        traj_frame_indices: List[List[int]] = []
        
        stride = max(1, self.config.history_dcd_stride)
        max_frames = self.config.history_dcd_max_frames
        rmsd_th = self.config.rmsd_filter
        # Batch frames to avoid long single-frame loops on large DCDs
        batch_size = max(1, int(getattr(self.config, "history_dcd_batch_size", 500)))

        # Precompute dihedral atom indices for vectorized torsion extraction.
        # env.cv_definitions are defined on the full simulation topology, so we map
        # them into the protein-only topology indices used for DCD loading.
        dihedral_indices = None
        try:
            dihedral_full = np.array([atoms for _, atoms in env.cv_definitions], dtype=int)
            # Map full atom indices -> protein topology indices
            full_to_protein = np.full((md_top_full.n_atoms,), -1, dtype=int)
            full_to_protein[np.asarray(protein_indices, dtype=int)] = np.arange(len(protein_indices), dtype=int)
            dihedral_indices = full_to_protein[dihedral_full]
            # If any dihedral cannot be mapped cleanly, fall back to env.get_cvs()
            # (slower, but keeps CV dimensionality consistent).
            if np.any(dihedral_indices < 0):
                log_print(
                    "[History DCD] Warning: failed to map dihedral indices into protein topology; "
                    "falling back to per-frame CV extraction."
                )
                dihedral_indices = None
        except Exception:
            dihedral_indices = None

        for fpath in dcd_files:
            try:
                file_count = 0  # total frames accepted from this file
                file_frame_idx: List[int] = []
                pbar = tqdm(desc=f"[History DCD] {fpath.name}", unit="frame") if tqdm else None
                # Stream in batches to avoid loading very large trajectories at once
                strided_cursor = 0  # number of frames yielded by iterload so far (post-stride)

                def _iter_protein_chunks():
                    # First try: protein-only DCD (common default).
                    for tr0 in md.iterload(str(fpath), top=md_top, stride=stride, chunk=batch_size):
                        yield tr0

                def _iter_full_then_slice_chunks():
                    # Fallback: full-system DCD (explicit-solvent replay mode).
                    for tr_full in md.iterload(str(fpath), top=md_top_full, stride=stride, chunk=batch_size):
                        yield tr_full.atom_slice(protein_indices)

                used_full_fallback = False
                try:
                    chunk_iter = _iter_protein_chunks()
                    for tr in chunk_iter:
                        if tr.n_frames == 0:
                            continue

                        tr = tr.superpose(ref_traj)

                        # RMSD Filter
                        keep_mask = None
                        if rmsd_th > 0 and ca_indices:
                            try:
                                r_vals = calculate_rmsd(tr.xyz, ref_positions_nm, atom_indices=ca_indices)
                                keep_mask = np.asarray(r_vals) <= rmsd_th
                                n_before = tr.n_frames
                                if keep_mask.sum() == 0:
                                    strided_cursor += tr.n_frames
                                    continue
                                if keep_mask.sum() < n_before:
                                    log_print(
                                        f"[History DCD] RMSD filter: kept {int(keep_mask.sum())}/{n_before} frames from {fpath.name}"
                                    )
                            except Exception as e:
                                log_print(f"[History DCD] RMSD filtering failed for {fpath.name}: {e}")

                        # Vectorized torsion computation when possible
                        torsions_chunk = None
                        if dihedral_indices is not None:
                            try:
                                torsions_chunk = md.compute_dihedrals(tr, dihedral_indices)
                            except Exception as e:
                                log_print(f"[History DCD] Dihedral computation failed for {fpath.name}: {e}")
                                torsions_chunk = None

                        for frame_idx in range(tr.n_frames):
                            if keep_mask is not None and not keep_mask[frame_idx]:
                                continue

                            coords_nm = tr.xyz[frame_idx]
                            try:
                                if torsions_chunk is not None:
                                    cv_vals = torsions_chunk[frame_idx]
                                else:
                                    # Fallback path: reconstruct full-system coordinates by
                                    # splicing protein coords into the current full state.
                                    coords_full = np.asarray(ref_full_nm, dtype=np.float32).copy()
                                    coords_full[np.asarray(protein_indices, dtype=int)] = coords_nm
                                    env.sim.context.setPositions(coords_full * nanometer)
                                    cv_vals = env.get_cvs()
                            except Exception:
                                continue

                            if np.any(np.isnan(cv_vals)):
                                continue

                            # Map this chunk-relative index back to the original DCD frame index.
                            src_frame = (int(strided_cursor) + int(frame_idx)) * int(stride)
                            file_frame_idx.append(int(src_frame))

                            history_epoch.append((cv_vals.copy(), coords_nm.copy()))
                            imported += 1
                            file_count += 1

                            if max_frames and imported >= max_frames:
                                break

                        if pbar is not None:
                            pbar.update(tr.n_frames)
                        strided_cursor += tr.n_frames
                        if max_frames and imported >= max_frames:
                            break
                except Exception as exc_first:
                    # Reset counter when retrying with a different topology.
                    strided_cursor = 0
                    try:
                        log_print(
                            f"[History DCD] Protein-topology load failed for {fpath.name} ({exc_first}); "
                            "retrying as full-system DCD and slicing to protein."
                        )
                    except Exception:
                        pass
                    for tr in _iter_full_then_slice_chunks():
                        if tr.n_frames == 0:
                            continue

                        tr = tr.superpose(ref_traj)

                        keep_mask = None
                        if rmsd_th > 0 and ca_indices:
                            try:
                                r_vals = calculate_rmsd(tr.xyz, ref_positions_nm, atom_indices=ca_indices)
                                keep_mask = np.asarray(r_vals) <= rmsd_th
                                n_before = tr.n_frames
                                if keep_mask.sum() == 0:
                                    strided_cursor += tr.n_frames
                                    continue
                                if keep_mask.sum() < n_before:
                                    log_print(
                                        f"[History DCD] RMSD filter: kept {int(keep_mask.sum())}/{n_before} frames from {fpath.name}"
                                    )
                            except Exception as e:
                                log_print(f"[History DCD] RMSD filtering failed for {fpath.name}: {e}")

                        torsions_chunk = None
                        if dihedral_indices is not None:
                            try:
                                torsions_chunk = md.compute_dihedrals(tr, dihedral_indices)
                            except Exception as e:
                                log_print(f"[History DCD] Dihedral computation failed for {fpath.name}: {e}")
                                torsions_chunk = None

                        for frame_idx in range(tr.n_frames):
                            if keep_mask is not None and not keep_mask[frame_idx]:
                                continue

                            coords_nm = tr.xyz[frame_idx]
                            try:
                                if torsions_chunk is not None:
                                    cv_vals = torsions_chunk[frame_idx]
                                else:
                                    coords_full = np.asarray(ref_full_nm, dtype=np.float32).copy()
                                    coords_full[np.asarray(protein_indices, dtype=int)] = coords_nm
                                    env.sim.context.setPositions(coords_full * nanometer)
                                    cv_vals = env.get_cvs()
                            except Exception:
                                continue

                            if np.any(np.isnan(cv_vals)):
                                continue

                            src_frame = (int(strided_cursor) + int(frame_idx)) * int(stride)
                            file_frame_idx.append(int(src_frame))

                            history_epoch.append((cv_vals.copy(), coords_nm.copy()))
                            imported += 1
                            file_count += 1

                            if max_frames and imported >= max_frames:
                                break

                        if pbar is not None:
                            pbar.update(tr.n_frames)
                        strided_cursor += tr.n_frames
                        if max_frames and imported >= max_frames:
                            break
                if file_count > 0:
                    traj_lengths.append(file_count)
                    traj_paths.append(str(fpath))
                    traj_frame_indices.append(file_frame_idx)
                if pbar is not None:
                    pbar.close()
            except Exception as e:
                log_print(f"[History DCD] Error loading {fpath}: {e}")
            
            if max_frames and imported >= max_frames:
                break
        if history_epoch:
            # Restore env state
            env.sim.context.setPositions(env._pdb_xyz)
            env.sim.context.setVelocitiesToTemperature(300 * kelvin)
            all_epoch_data.append(history_epoch)

        try:
            self.last_bootstrap_dcd_paths = traj_paths  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.last_bootstrap_frame_indices = traj_frame_indices  # type: ignore[attr-defined]
        except Exception:
            pass
        
        log_print(f"[History DCD] Imported {imported} frames.")
        return imported, traj_lengths
