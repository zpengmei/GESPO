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
from openmm import unit
from gespo.utils.logging import log_print

from ...config import TrainingConfig
from ...sampling import PeptideSampler
from ...utils.openmm_utils import resolve_device
from ...utils.plotting import plot_ramachandran_by_residue, plot_analysis_summary
from ...utils.geometry import _dihedral
from ..core.setup import initialize_policy, initialize_projection_scorer
from .anchoring import run_anchoring_round
from .geom2vec_refresh import PolicyResetHandles, refresh_geom2vec_vamp
from ..policies.rl import StateLogitPolicy
from ..core.features import FeatureBuilder
from ..core.sampler import Sampler
from ..core.checkpointing import CheckpointManager
from ..core.optimization import GRPOOptimizer
from ..core.rewards import RewardCalculator
from ..geom2vec_backbone import EmbeddingCache, Geom2VecTopologyCache, Geom2VecVAMPProjector, infer_and_cache
from ..geom2vec.utils import align_states_by_overlap
from ..core.stats import SufficientStatistics


def train(args: argparse.Namespace):
    """
    Main training loop for GESPO with LOO VAMP + Ensemble Uncertainty rewards.
    """
    config = TrainingConfig.from_args(args)
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
        
    log_print("GESPO Training with LOO VAMP (Geom2Vec, single-model)")
    log_print(f"Episodes: {config.episodes}, Walkers: {config.grp_size}")
    log_print(f"Platform: {config.platform}, Device: {torch_device}, Gen Device: {gen_device}")
    log_print("Geom2Vec VAMP lobe: single model (no UCB)")
    
    # Initialize Environments
    log_print("Initializing environments...")
    envs = [PeptideSampler(args, generator_device=gen_device) for _ in range(config.grp_size)]
    env_ref = envs[0]
    
    # Initialize Policy
    log_print("Initializing policy...")
    agent, agent_old, policy_opts, _ = initialize_policy(config, envs, torch_device)
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
    ep_start = 0

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
        log_print(f"[LOO VAMP] Invalidated global stats ({tag}).")

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
            log_print(
                f"[LOO VAMP] Seeded global stats ({tag}): added_trajs={added}, "
                f"skipped_short={skipped_short}, skipped_missing={skipped_missing}"
            )

    # Geom2Vec VAMP backbone state (only used when proj_mode == "geom2vec_vamp")
    geom2vec_cache: Optional[EmbeddingCache] = None
    # Softmax VAMP lobe: used for state labels and LOO-VAMP scoring.
    geom2vec_projector: Optional[Geom2VecVAMPProjector] = None
    # Linear VAMP lobe: optional diagnostics (kept for consistency with generative loop).
    geom2vec_projector_linear: Optional[Geom2VecVAMPProjector] = None
    geom2vec_state_permutation_map = np.arange(int(getattr(config, "geom2vec_dim", 2) or 2), dtype=int)
    geom2vec_state_inverse_permutation_map = np.arange(int(getattr(config, "geom2vec_dim", 2) or 2), dtype=int)
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

        if not overwrite and _geom2vec_cache_has_key(key):
            return key

        if coords_nm is None:
            if not dcd_path:
                return None
            if geom2vec_md_top_protein is None:
                return None
            try:
                traj = md.load(str(dcd_path), top=geom2vec_md_top_protein, stride=int(geom2vec_embed_stride))
                coords_nm = np.asarray(traj.xyz, dtype=np.float32)
            except Exception:
                # In explicit-solvent full-coordinate replay mode, DCDs may contain the
                # full system. Fall back to loading with the full topology and slicing.
                try:
                    md_top_full = env_ref.md_topology
                    protein_indices = md_top_full.select("protein")
                    if protein_indices is None or len(protein_indices) == 0:
                        protein_indices = np.arange(md_top_full.n_atoms, dtype=int)
                    traj_full = md.load(str(dcd_path), top=md_top_full, stride=int(geom2vec_embed_stride))
                    traj = traj_full.atom_slice(protein_indices)
                    coords_nm = np.asarray(traj.xyz, dtype=np.float32)
                except Exception as exc2:
                    log_print(f"[Geom2Vec Cache] Failed to load DCD {dcd_path}: {exc2}")
                    return None
            # We already applied mdtraj stride; do not stride again in inference.
            effective_stride = int(geom2vec_embed_stride)
            infer_stride = 1
        else:
            effective_stride = int(geom2vec_embed_stride)
            infer_stride = int(geom2vec_embed_stride)

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
                    "stride": int(effective_stride),
                    "source_dcd": str(dcd_path) if dcd_path else "",
                },
            )
            return rec.key
        except Exception as exc:
            log_print(f"[Geom2Vec Cache] Failed to compute/store embeddings for key={key}: {exc}")
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
                protein_indices = np.arange(md_top_full.n_atoms, dtype=int)
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

        # Effective VAMP lag in embedding frames (account for embedding stride).
        lag_raw = int(getattr(config, "geom2vec_lagtime", 1) or 1)
        geom2vec_lagtime_embed = max(1, int(round(lag_raw / max(1, int(geom2vec_embed_stride)))))


    # Load Checkpoint
    latest_ckpt, ckpt_data = ckpt_manager.load_latest()
    pca_state = None
    
    if config.resume and ckpt_data:
        ep_start, _, pca_state = ckpt_manager.restore_states(
            agent, agent_old, policy_opts, ckpt_data, torch_device
        )
        imported_frames, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.snapshot_dir or "."
        )
        if imported_frames > 0:
            log_print(f"[Resume] Rebuilt history from {imported_frames} frames (DCD).")
            all_trajectory_lengths.append(imported_lens)
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
                log_print(f"[Resume] Rebuilt history from {restored} NPZ epochs.")
                all_trajectory_lengths.extend(restored_lens)
                for lens in restored_lens:
                    all_epoch_traj_dcd_paths.append([None] * len(lens))
                    all_epoch_traj_dcd_frame_indices.append([None] * len(lens))

    # Bootstrap history from external DCDs if provided (e.g., previous runs).
    if getattr(config, "history_dcd_dir", ""):
        imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(
            all_epoch_data, env_ref, base_dir=config.history_dcd_dir
        )
        if imported > 0:
            all_trajectory_lengths.append(imported_lens)
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
    
    # Discrete state dimension (number of VAMP states)
    n_state_components = int(getattr(config, "geom2vec_dim", 2) or 2)
    config.n_discrete_states = n_state_components
    n_discrete_states = n_state_components
    log_print("Initializing Geom2Vec VAMP backbone...")

    # Initialize PCA scorer (for feature building)
    pca_scorer = initialize_projection_scorer(config, envs[0].n_cvs, args)
    if pca_state:
        pca_scorer.set_state(pca_state)
    
    # Feature Builder
    feature_builder = FeatureBuilder(config, generator=envs[0].generator, pca_scorer=pca_scorer, env=envs[0])
    
    # Core reward calculator (trajectory + launch rewards)
    # Vendi-based novelty estimators are disabled in the streamlined loop.
    vendi_launch_pca = None
    vendi_hill_pca = None
    vendi_launch_angle = None
    vendi_hill_angle = None
    reward_calculator = RewardCalculator(
        config,
        pca_scorer,
        feature_builder,
        vendi_launch_pca,
        vendi_hill_pca,
        vendi_launch_angle,
        vendi_hill_angle,
    )

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
    run_anchoring_round(
        config=config,
        ckpt_manager=ckpt_manager,
        all_epoch_data=all_epoch_data,
        all_trajectory_lengths=all_trajectory_lengths,
        all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
        all_epoch_traj_dcd_frame_indices=all_epoch_traj_dcd_frame_indices,
        env_ref=env_ref,
    )

    # Warm-start projector immediately after anchoring so VAMP rewards are ready for epoch 1
    if all_epoch_data and not _geom2vec_is_trained():
        geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, geom2vec_lagtime_embed = (
            refresh_geom2vec_vamp(
                reason="post-anchoring warm-start",
                config=config,
                all_epoch_data=all_epoch_data,
                all_trajectory_lengths=all_trajectory_lengths,
                all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
                geom2vec_cache=geom2vec_cache,
                geom2vec_projector=geom2vec_projector,
                geom2vec_projector_linear=geom2vec_projector_linear,
                geom2vec_embed_stride=geom2vec_embed_stride,
                geom2vec_state_permutation_map=geom2vec_state_permutation_map,
                geom2vec_state_inverse_permutation_map=geom2vec_state_inverse_permutation_map,
                geom2vec_cache_key_for_history_traj=_geom2vec_cache_key_for_history_traj,
                geom2vec_cache_has_key=_geom2vec_cache_has_key,
                geom2vec_ensure_cached=_geom2vec_ensure_cached,
                invalidate_callbacks=[_invalidate_loo_vamp_stats],
                policy_reset=PolicyResetHandles(agent=agent, agent_old=agent_old, policy_opts=policy_opts),
            )
        )

    # Plotting for "this round": exclude any bootstrapped history/anchoring epochs.
    round_epoch_data_start = len(all_epoch_data)
    
    log_print(f"Starting training from epoch {ep_start+1}...")
    
    # Main Training Loop
    start_state_actions: Optional[torch.Tensor] = None

    for ep in range(ep_start, config.episodes):
        log_print(f"\n{'='*60}")
        log_print(f"Starting Epoch {ep+1}/{config.episodes}")
        log_print(f"{'='*60}")
        epoch_start_time = time.time()
        
        # A. Refresh projector (every N epochs)
        refresh_interval = int(getattr(config, "pca_refresh_interval", 1) or 1)
        model_ready = _geom2vec_is_trained()

        if refresh_interval <= 0:
            should_update = not model_ready
        else:
            should_update = ((ep + 1) % refresh_interval == 0) or (not model_ready)

        if should_update and all_epoch_data:
            log_print(f"[Epoch {ep+1}] Refreshing Geom2Vec VAMP lobe...")
            geom2vec_state_permutation_map, geom2vec_state_inverse_permutation_map, geom2vec_lagtime_embed = (
                refresh_geom2vec_vamp(
                    reason=f"epoch {ep+1}",
                    config=config,
                    all_epoch_data=all_epoch_data,
                    all_trajectory_lengths=all_trajectory_lengths,
                    all_epoch_traj_dcd_paths=all_epoch_traj_dcd_paths,
                    geom2vec_cache=geom2vec_cache,
                    geom2vec_projector=geom2vec_projector,
                    geom2vec_projector_linear=geom2vec_projector_linear,
                    geom2vec_embed_stride=geom2vec_embed_stride,
                    geom2vec_state_permutation_map=geom2vec_state_permutation_map,
                    geom2vec_state_inverse_permutation_map=geom2vec_state_inverse_permutation_map,
                    geom2vec_cache_key_for_history_traj=_geom2vec_cache_key_for_history_traj,
                    geom2vec_cache_has_key=_geom2vec_cache_has_key,
                    geom2vec_ensure_cached=_geom2vec_ensure_cached,
                    invalidate_callbacks=[_invalidate_loo_vamp_stats],
                    policy_reset=PolicyResetHandles(agent=agent, agent_old=agent_old, policy_opts=policy_opts),
                )
            )
            model_ready = _geom2vec_is_trained()
        # Ensure history-aware LOO stats match the current VAMP lobe before sampling.
        if model_ready:
            try:
                n_states_hist = int(getattr(config, "geom2vec_dim", 2) or 2)
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
                log_print(f"[LOO VAMP] Warning: failed to seed global stats ({exc_loo_seed})")
        
        # B. Sample Discrete State Starts from Policy
        log_print(f"[Epoch {ep+1}] Phase B: Sampling discrete state starts...")

        # Sample discrete states from categorical policy
        # If projector is not trained yet, fall back to random exploration
        start_state_actions = None
        if not model_ready or not all_epoch_data:
            log_print("[Discrete Starts] Projector not ready, using random exploration")
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
            n_states = int(getattr(config, "geom2vec_dim", 2) or 2)
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
                log_print(
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
                log_print(f"[Discrete Starts] Start distribution (agent_old sampling): [{probs_old}]")
            if probs_new is not None:
                log_print(f"[Discrete Starts] Start distribution (agent current):      [{probs_new}]")
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
            
            log_print(f"[Discrete Starts] Sampled from {int(n_discrete_states)} states via categorical policy")
        
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
                    protein_indices = np.arange(md_top_full.n_atoms, dtype=int)
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
                        n_states = int(getattr(config, "geom2vec_dim", 2) or 2)

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

                        log_print(f"\n[LOO VAMP Statistics - Epoch {ep+1}]")
                        if n_models > 1:
                            log_print(f"  Geom2Vec VAMP ensemble: n_models={n_models}, lambda={lam:.3f}")
                            log_print(
                                "  Baseline VAMP-2 per model (history+epoch trajectories): "
                                + ", ".join(f"{b:.4f}" for b in vamp_baselines)
                            )
                        else:
                            log_print(f"  Baseline VAMP-2 (history+epoch trajectories): {float(vamp_baselines[0]):.4f}")

                        # Per-trajectory breakdown (raw Δ, pair-normalized Δ, and final UCB reward).
                        raw_mean = per_model_contribs.mean(axis=0)
                        raw_std = per_model_contribs.std(axis=0)
                        score_label = f"ucb(μ+λσ,λ={lam:.2f})" if lam != 0.0 else "score(μ)"
                        log_print(f"  Per-trajectory contributions (delta_i = S_all - S_minus_i):")
                        for i in range(config.grp_size):
                            if pair_counts[i] <= 0:
                                continue
                            log_print(
                                f"    Traj {i:2d}: "
                                f"raw={float(raw_mean[i]):+.4f}±{float(raw_std[i]):.4f}  "
                                f"pairs={int(pair_counts[i])}  "
                                f"{score_label}={float(loo_vamp_rewards[i]):+.4f}"
                            )
                        log_print(
                            f"  Raw Δ summary: mean={float(raw_mean.mean()):.4f}, std={float(raw_mean.std()):.4f}, "
                            f"min={float(raw_mean.min()):.4f}, max={float(raw_mean.max()):.4f}"
                        )
                        if lam != 0.0:
                            log_print(
                                f"  UCB rewards (μ+λσ over models): mean={loo_vamp_rewards.mean():.4f}, std={loo_vamp_rewards.std():.4f}"
                            )
                        else:
                            log_print(f"  Mean rewards (μ over models): mean={loo_vamp_rewards.mean():.4f}, std={loo_vamp_rewards.std():.4f}")
                    except Exception as exc:
                        log_print(f"[LOO VAMP] Warning: Failed to compute geom2vec LOO rewards: {exc}")
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
                log_print(f"[Rewards] Combined launch + LOO VAMP (weight={loo_weight:.2f}): "
                      f"launch_mean={launch_rewards.mean():.4f}, loo_mean={loo_vamp_tensor.mean():.4f}, "
                      f"combined_mean={combined_rewards.mean():.4f}")
            else:
                # Keep logging consistent: `avg_hill` reflects the scalar reward
                # used for start-policy updates in this geom2vec_vamp-only loop.
                try:
                    metrics['avg_hill'] = float(launch_rewards.mean().item())
                except Exception:
                    pass

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
                    log_print(f"[Policy Update] Masking {n_random} random starts, training on {n_policy} policy starts")

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
                log_print(f"[Policy] Synced old policy at epoch {ep+1}")
        else:
            update_stats = {}
        
        # F. Logging & Checkpointing
        log_print(f"[Epoch {ep+1}] Phase E: Logging & checkpointing...")
        log_print(f"Epoch {ep+1} Complete. Avg Reward: {metrics.get('avg_hill', 0.0):.4f}")
        
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
                pca_scorer,
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
                pca_scorer,
                envs[0].cv_names,
                config.pdb,
                f"analysis_summary_ep{ep+1:03d}.png",
                feature_builder=feature_builder
            )
        
        log_print(f"[Epoch {ep+1}] Complete in {time.time() - epoch_start_time:.1f}s.")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    log_print("Training Complete.")
