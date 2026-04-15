"""Anchoring round: seed history with short PDB-grounded trajectories.

Extracted from both loop_generative.py and loop_generative_path.py.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mdtraj as md
import numpy as np
import torch
from openmm import unit


def _run_anchoring_round(
    *,
    config,
    envs,
    sampler,
    agent,
    agent_old,
    env_ref,
    gen_device,
    torch_device,
    all_epoch_data,
    all_trajectory_lengths,
    all_epoch_traj_dcd_paths,
    all_epoch_traj_dcd_frame_indices,
    all_epoch_vamp_exclude,
    append_retrieval_cache_fn,
    rebalance_fn,
    refresh_geom2vec_fn,
    feature_builder,
    grpo_opt,
    geom2vec_mgr,
) -> None:
    """Seed history with short PDB-grounded trajectories before training."""
    if not getattr(config, "anchoring_round", False):
        return

    want_path_sampling = bool(getattr(config, "path_sampling", False)) and bool(str(getattr(config, "pdb_b", "") or "").strip())

    def _list_traj_files(d: Optional[Path]) -> List[Path]:
        if d is None:
            return []
        try:
            files = list(d.rglob("*.dcd")) + list(d.rglob("*.xtc"))
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

        default_source_dir = ""
        if want_path_sampling and anchor_dir_a is not None and anchor_dir_b is not None:
            default_source_dir = f"{anchor_dir_a},{anchor_dir_b}"
        elif anchor_dir is not None:
            default_source_dir = str(anchor_dir)
        source_dir = getattr(config, "anchoring_source_dir", "") or default_source_dir

        imported = 0
        imported_lens: List[int] = []
        if source_dir:
            try:
                imported, imported_lens = ckpt_manager.bootstrap_history_from_dcd(all_epoch_data, env_ref, base_dir=source_dir)
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
                            all_epoch_vamp_exclude.append(False)
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

    n_total = int(desired_traj_count)
    n_b = (n_total // 2) if want_path_sampling else 0
    n_a = n_total - n_b

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

    def _run_endpoint(label: str, env_pool: List[PeptideSampler], out_dir: Optional[Path], n_trajs: int) -> None:
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

            has_conformers = getattr(env, "has_multiple_conformers", False)
            num_conformers = len(getattr(env, "pdb_conformers", []))

            cv_targets: List[float]
            conformer_idx: Optional[int] = None

            if has_conformers and num_conformers > 0:
                conformer_idx = seq_idx % num_conformers
                print(
                    f"[Anchoring {label}] Using conformer {conformer_idx}/{num_conformers-1} for trajectory {existing_n + run_idx + 1}/{n_trajs}.",
                    flush=True,
                )
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

        try:
            cv_arr = np.asarray([x[0] for x in epoch_data], dtype=np.float32)
            feat_arr = np.asarray(
                [
                    np.asarray(
                        feature_builder.build_projection_features(
                            np.asarray(x[0], dtype=np.float32),
                            np.asarray(x[1], dtype=np.float32),
                        ),
                        dtype=np.float32,
                    )
                    for x in epoch_data
                ],
                dtype=np.float32,
            )
            if feat_arr.ndim > 2:
                feat_arr = feat_arr.reshape(feat_arr.shape[0], -1)
            # PCA scorer removed - feature builder is standalone now
            print("[Anchoring] Feature building complete.")
        except Exception as exc_feat:
            print(f"[Anchoring] Warning: feature building failed: {exc_feat}")

    print("[Anchoring] Done.")

# Optional anchoring to seed history
_run_anchoring_round()

# Build/refresh the in-memory retrieval cache from *all* loaded history sources
# (resume + external history DCDs + anchoring). This keeps replay starts stable
# and ensures older history frames are not silently dropped by load order.
if all_epoch_data:
    try:
        _rebuild_retrieval_cache_from_history(tag="pre-train")
    except Exception as exc_rebuild:
        print(f"[Retrieval Cache] Warning: failed to rebuild from history ({exc_rebuild})")

# Warm-start projector immediately after anchoring so LOO rewards are ready for epoch 1
if all_epoch_data and not geom2vec_mgr.is_trained():
    _refresh_geom2vec_vamp(reason="post-anchoring warm-start")

# Plotting for "this round": exclude any bootstrapped history/anchoring epochs.
round_epoch_data_start = len(all_epoch_data)

print(f"Starting training from epoch {ep_start+1}...")

