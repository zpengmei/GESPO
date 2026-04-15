"""Committor tube-cell history count accumulation (path-sampling mode)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from . import committor_utils as _cu
from . import graph_algorithms as _graph


def ensure_committor_tube_history_counts(
    *,
    upto_epoch_exclusive: int,
    n_bins: int,
    tag: str,
    config: Any,
    committor_state: Any,
    geom2vec_cache: Any,
    geom2vec_projector: Any,
    geom2vec_projector_linear: Any,
    geom2vec_mgr: Any,
    geom2vec_state_permutation_map: np.ndarray,
    geom2vec_embed_stride: int,
    n_states: int,
    iter_history_trajs: Callable[..., Iterator],
    rmsd_to_endpoints_fn: Callable,
) -> None:
    """Accumulate tube-cell visit weights from history.

    Tube-cell = (committor_bin, state), with per-frame weights
    w_hat(x) = 4 q(x) (1-q(x)) so basins contribute ~0.
    """
    # Aliases for committor utility functions
    _committor_from_rmsd = _cu.committor_from_rmsd
    _rmsd_state_labels_2d = _cu.rmsd_state_labels_2d
    _lift_center_for_delay = _cu.lift_center_for_delay
    _conn_shortest_path_len = _graph.shortest_path_len
    _rmsd_to_endpoints = rmsd_to_endpoints_fn

    """Accumulate tube-cell visit weights from history (excluding current epoch).

    Tube-cell = (committor_bin, state), with per-frame weights
    w_hat(x) = 4 q(x) (1-q(x)) so basins contribute ~0.
    """

    committor_state_mode = str(getattr(config, "committor_state_mode", "dga") or "dga").strip().lower()
    if committor_state_mode not in {"dga", "vamp"}:
        committor_state_mode = "dga"
    tube_state_mode = str(getattr(config, "committor_tube_state_mode", "model") or "model").strip().lower()
    if tube_state_mode in {"dga", "vamp"}:
        tube_state_mode = "model"
    if tube_state_mode not in {"model", "q_only", "rmsd_2d"}:
        tube_state_mode = "model"
    rmsd_bins = int(getattr(config, "committor_rmsd_state_bins", 10) or 10)
    rmsd_bins = int(max(1, rmsd_bins))
    try:
        rmsd_max_nm = float(getattr(config, "committor_rmsd_state_max_nm", 2.0) or 2.0)
    except Exception:
        rmsd_max_nm = 2.0
    if not np.isfinite(rmsd_max_nm) or rmsd_max_nm <= 0.0:
        rmsd_max_nm = 2.0
    force_rmsd_proxy = bool(getattr(config, "committor_use_rmsd_proxy", False))
    need_model = bool((not force_rmsd_proxy) or (str(tube_state_mode) == "model"))

    if geom2vec_cache is None:
        return
    if need_model:
        if geom2vec_projector_linear is None or (
            getattr(geom2vec_projector_linear, "trained", None) is None
            and not bool(getattr(geom2vec_projector_linear, "trained_ensemble", None))
        ):
            return
        if str(committor_state_mode) == "vamp":
            if committor_state.vamp_model is None or geom2vec_projector is None or (
                getattr(geom2vec_projector, "trained", None) is None
                and not bool(getattr(geom2vec_projector, "trained_ensemble", None))
            ):
                return
        else:
            if committor_state.dga_model is None:
                return

    need_tube = float(getattr(config, "committor_tube_diversity_reward_weight", 0.0) or 0.0) > 0.0
    need_edge = float(getattr(config, "committor_tube_edge_novelty_reward_weight", 0.0) or 0.0) > 0.0
    need_conn = float(getattr(config, "committor_connectivity_reward_weight", 0.0) or 0.0) > 0.0
    if not (need_tube or need_edge or need_conn):
        return

    M = int(n_bins)
    if M <= 1:
        return
    if str(tube_state_mode) == "q_only":
        K = 1
    elif str(tube_state_mode) == "rmsd_2d":
        K = int(rmsd_bins) * int(rmsd_bins)
    else:
        if str(committor_state_mode) == "vamp":
            K = int(np.asarray(getattr(committor_state.vamp_model, "q_state", np.zeros((0,)))).reshape(-1).shape[0])  # type: ignore[arg-type]
        else:
            K = int(getattr(committor_state.dga_model, "coeffs", np.zeros((0,))).shape[0])
    n_cells = int(M) * int(K)

    if need_model and str(committor_state_mode) == "vamp":
        dim = 0
        lag = int(getattr(committor_state.vamp_model, "lag", 1) or 1)  # type: ignore[arg-type]
        delay_n = 0
        delay_delta = 1
    elif need_model:
        dim = int(getattr(committor_state.dga_model, "dim", 0) or 0)
        lag = int(getattr(committor_state.dga_model, "lag", 0) or 0)
        delay_n = int(getattr(committor_state.dga_model, "delay_n_embed", 0) or 0)
        delay_delta = int(getattr(committor_state.dga_model, "delay_delta", 1) or 1)
    else:
        dim = 0
        lag = 0
        delay_n = 0
        delay_delta = 1

    sig_tube = (
        str(tube_state_mode),
        str(committor_state_mode),
        int(M),
        int(K),
        int(dim),
        int(lag),
        int(delay_n),
        int(delay_delta),
        str(boundary_mode),
        int(rmsd_bins),
        float(rmsd_max_nm),
        bool(force_rmsd_proxy),
    )
    if need_tube and (committor_state.tube_counts is None or committor_state.tube_signature != sig_tube):
        committor_state.tube_counts = np.zeros((int(M) * int(K),), dtype=np.float64)
        committor_state.tube_total_weight = 0.0
        committor_state.tube_counted_keys = set()
        committor_state.tube_signature = sig_tube

    edge_delta = int(getattr(config, "committor_tube_edge_novelty_lag", 1) or 1)
    edge_delta = int(max(1, edge_delta))
    edge_undirected = bool(getattr(config, "committor_tube_edge_novelty_undirected", True))
    edge_bin_change_only = bool(getattr(config, "committor_tube_edge_novelty_bin_change_only", False))
    sig_edge = (
        str(tube_state_mode),
        str(committor_state_mode),
        int(M),
        int(K),
        int(edge_delta),
        bool(edge_undirected),
        bool(edge_bin_change_only),
        int(dim),
        int(lag),
        int(delay_n),
        int(delay_delta),
        str(boundary_mode),
        int(rmsd_bins),
        float(rmsd_max_nm),
        bool(force_rmsd_proxy),
    )
    if need_edge and (committor_state.edge_counts is None or committor_state.edge_signature != sig_edge):
        max_dense_edges = 4_000_000  # ~32MB float64 (plus overhead)
        n_edges = int(n_cells) * int(n_cells)
        if n_edges <= int(max_dense_edges):
            committor_state.edge_counts = np.zeros((int(n_edges),), dtype=np.float64)
            mode = "dense"
        else:
            committor_state.edge_counts = {}
            mode = "sparse"
        committor_state.edge_total_count = 0.0
        committor_state.edge_counted_keys = set()
        committor_state.edge_signature = sig_edge
        print(
            f"[Committor Edge] Initialized edge counts ({mode}): cells={int(n_cells)} delta={int(edge_delta)} "
            f"undirected={bool(edge_undirected)} bin_change_only={bool(edge_bin_change_only)}",
            flush=True,
        )

    sig_conn = (
        int(M),
        int(K),
        int(edge_delta),
        bool(edge_undirected),
        bool(edge_bin_change_only),
        str(tube_state_mode),
        str(boundary_mode),
        int(rmsd_bins),
        float(rmsd_max_nm),
        bool(force_rmsd_proxy),
    )
    if need_conn and (committor_state.conn_adj is None or committor_state.conn_signature != sig_conn):
        committor_state.conn_adj = [set() for _ in range(int(n_cells))]
        committor_state.conn_edges = set()
        committor_state.conn_a_cells = set()
        committor_state.conn_b_cells = set()
        committor_state.conn_counted_keys = set()
        committor_state.conn_signature = sig_conn
        print(
            f"[Committor Conn] Initialized tube graph: cells={int(n_cells)} delta={int(edge_delta)} "
            f"undirected={bool(edge_undirected)} bin_change_only={bool(edge_bin_change_only)}",
            flush=True,
        )

    try:
        cutoff_a_nm = float(getattr(config, "path_boundary_cutoff_a_nm", 0.20) or 0.20)
        cutoff_b_nm = float(getattr(config, "path_boundary_cutoff_b_nm", cutoff_a_nm) or cutoff_a_nm)
    except Exception:
        cutoff_a_nm = 0.20
        cutoff_b_nm = 0.20

    added_tube = 0
    added_edge = 0
    added_conn = 0
    added_conn_edges = 0
    skipped = 0
    for epoch_idx, traj_idx, L_i, dcd_path, coords_nm in _iter_history_trajs(int(upto_epoch_exclusive)):
        key = geom2vec_mgr.cache_key_for_history_traj(int(epoch_idx), int(traj_idx), dcd_path)
        key_s0 = str(key)
        tube_done = (not need_tube) or (key_s0 in committor_state.tube_counted_keys)
        edge_done = (not need_edge) or (key_s0 in committor_state.edge_counted_keys)
        conn_done = (not need_conn) or (key_s0 in committor_state.conn_counted_keys)
        if tube_done and edge_done and conn_done:
            continue
        cached_key = geom2vec_mgr.ensure_cached(
            key=key,
            dcd_path=dcd_path,
            coords_nm=coords_nm,
            overwrite=False,
            meta={"epoch_idx": int(epoch_idx), "traj_idx": int(traj_idx), "raw_frames": int(L_i)},
        )
        if cached_key is None:
            skipped += 1
            continue
        key_s = str(cached_key)
        tube_done = (not need_tube) or (key_s in committor_state.tube_counted_keys)
        edge_done = (not need_edge) or (key_s in committor_state.edge_counted_keys)
        conn_done = (not need_conn) or (key_s in committor_state.conn_counted_keys)
        if tube_done and edge_done and conn_done:
            continue

        try:
            graph, ca = geom2vec_cache.load(cached_key)
            z_lin_arr = None
            if geom2vec_projector_linear is not None and (
                getattr(geom2vec_projector_linear, "trained", None) is not None
                or bool(getattr(geom2vec_projector_linear, "trained_ensemble", None))
            ):
                z_lin_raw = geom2vec_projector_linear.state_output_from_embeddings(graph, ca)
                z_lin_arr = np.asarray(z_lin_raw, dtype=np.float64)

            # DGA can optionally be fit on non-linear inputs (e.g., VAMP probabilities).
            # Keep history seeding consistent with the fitted model's input_mode.
            z_arr = z_lin_arr
            if need_model:
                try:
                    if (
                        str(committor_state_mode) != "vamp"
                        and committor_state.dga_model is not None
                        and str(getattr(committor_state.dga_model, "input_mode", "linear") or "linear") == "vamp_probs"
                    ):
                        probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)  # type: ignore[union-attr]
                        probs = np.asarray(probs_raw, dtype=np.float64)[:, geom2vec_state_permutation_map]
                        z_arr = probs
                except Exception:
                    z_arr = z_lin_arr
            try:
                if (
                    need_model
                    and str(committor_state_mode) != "vamp"
                    and committor_state.dga_model is not None
                    and bool(getattr(committor_state.dga_model, "include_global_tokens", False))
                    and z_arr is not None
                ):
                    z_arr = _augment_committor_dga_features(np.asarray(z_arr, dtype=np.float32), ca, use_globals=True)
            except Exception:
                pass
        except Exception:
            skipped += 1
            continue

        if need_model and (z_arr is None or z_arr.ndim != 2 or int(z_arr.shape[0]) <= 0):
            skipped += 1
            continue

        try:
            in_a, in_b, rmsd_a, rmsd_b = _boundary_masks_from_ca(
                ca,
                cutoff_a_nm=float(cutoff_a_nm),
                cutoff_b_nm=float(cutoff_b_nm),
                graph_features=graph,
                z_lin=z_lin_arr,
                cache_key=str(cached_key),
            )
        except Exception:
            T = 0
            try:
                T = int(np.asarray(ca).shape[0])
            except Exception:
                T = int(z_arr.shape[0]) if z_arr is not None and getattr(z_arr, "ndim", 0) == 2 else 0
            in_a = np.zeros((int(T),), dtype=bool)
            in_b = np.zeros((int(T),), dtype=bool)
            rmsd_a = np.full((int(T),), np.nan, dtype=np.float64)
            rmsd_b = np.full((int(T),), np.nan, dtype=np.float64)

        q_m = None
        labels_m = None
        if need_model:
            if str(committor_state_mode) == "vamp":
                try:
                    probs_raw = geom2vec_projector.state_probs_from_embeddings(graph, ca)  # type: ignore[union-attr]
                    probs = np.asarray(probs_raw, dtype=np.float32)[:, geom2vec_state_permutation_map]
                    K_model = int(np.asarray(getattr(committor_state.vamp_model, "q_state", np.zeros((0,)))).reshape(-1).shape[0])  # type: ignore[arg-type]
                    if probs.shape[1] != int(K_model):
                        if probs.shape[1] > int(K_model):
                            probs = probs[:, : int(K_model)]
                        else:
                            probs = np.pad(probs, ((0, 0), (0, int(K_model) - int(probs.shape[1]))), mode="constant")
                    labels_src = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                    q_m, labels_m = committor_state.vamp_model.predict_q_and_labels(labels_src, in_a=in_a, in_b=in_b)  # type: ignore[union-attr]
                except Exception:
                    skipped += 1
                    continue
            else:
                try:
                    q_m, labels_m = committor_state.dga_model.predict_q_and_labels(z_arr, in_a=in_a, in_b=in_b)
                except Exception:
                    skipped += 1
                    continue

        if force_rmsd_proxy:
            q_r = _committor_from_rmsd(rmsd_a, rmsd_b)
            q_r = np.asarray(q_r, dtype=np.float64).reshape(-1)
            ia_l = np.asarray(in_a, dtype=bool).reshape(-1)
            ib_l = np.asarray(in_b, dtype=bool).reshape(-1)
            if need_model and str(committor_state_mode) != "vamp" and int(delay_n) > 0 and q_m is not None:
                q_r = _lift_center_for_delay(
                    q_r,
                    n_embed=int(delay_n),
                    lag=int(delay_delta),
                    target_len=int(np.asarray(q_m).reshape(-1).shape[0]),
                )
                ia_l = _lift_center_for_delay(
                    ia_l, n_embed=int(delay_n), lag=int(delay_delta), target_len=int(q_r.shape[0])
                ).astype(bool, copy=False)
                ib_l = _lift_center_for_delay(
                    ib_l, n_embed=int(delay_n), lag=int(delay_delta), target_len=int(q_r.shape[0])
                ).astype(bool, copy=False)
            q_r = np.clip(q_r, 0.0, 1.0)
            if int(ia_l.shape[0]) == int(q_r.shape[0]):
                q_r[ia_l] = 0.0
            if int(ib_l.shape[0]) == int(q_r.shape[0]):
                q_r[ib_l] = 1.0
            q_f = np.asarray(q_r, dtype=np.float64).reshape(-1)
            ia_conn = np.asarray(ia_l, dtype=bool).reshape(-1)
            ib_conn = np.asarray(ib_l, dtype=bool).reshape(-1)
        else:
            q_f = np.asarray(q_m, dtype=np.float64).reshape(-1) if q_m is not None else np.zeros((0,), dtype=np.float64)
            ia_conn = np.asarray(in_a, dtype=bool).reshape(-1)
            ib_conn = np.asarray(in_b, dtype=bool).reshape(-1)

        if str(tube_state_mode) == "q_only":
            labels_f = np.zeros((int(q_f.shape[0]),), dtype=np.int64)
        elif str(tube_state_mode) == "rmsd_2d":
            ra_use = np.asarray(rmsd_a, dtype=np.float64).reshape(-1)
            rb_use = np.asarray(rmsd_b, dtype=np.float64).reshape(-1)
            if need_model and str(committor_state_mode) != "vamp" and int(delay_n) > 0 and q_f.size:
                ra_use = _lift_center_for_delay(
                    ra_use, n_embed=int(delay_n), lag=int(delay_delta), target_len=int(q_f.shape[0])
                )
                rb_use = _lift_center_for_delay(
                    rb_use, n_embed=int(delay_n), lag=int(delay_delta), target_len=int(q_f.shape[0])
                )
            labels_f = _rmsd_state_labels_2d(ra_use, rb_use, bins=int(rmsd_bins), max_nm=float(rmsd_max_nm))
        else:
            labels_f = np.asarray(labels_m, dtype=np.int64).reshape(-1) if labels_m is not None else np.zeros((int(q_f.shape[0]),), dtype=np.int64)

        n = int(min(int(q_f.shape[0]), int(labels_f.shape[0])))
        if n <= 0:
            if need_tube:
                committor_state.tube_counted_keys.add(key_s)
            if need_edge:
                committor_state.edge_counted_keys.add(key_s)
            if need_conn:
                committor_state.conn_counted_keys.add(key_s)
            continue
        q_f = q_f[:n]
        labels_f = labels_f[:n]
        try:
            if int(delay_n) > 0 and int(ia_conn.shape[0]) != int(n):
                ia_conn = _lift_center_for_delay(
                    ia_conn,
                    n_embed=int(delay_n),
                    lag=int(delay_delta),
                    target_len=int(n),
                ).astype(bool, copy=False)
                ib_conn = _lift_center_for_delay(
                    ib_conn,
                    n_embed=int(delay_n),
                    lag=int(delay_delta),
                    target_len=int(n),
                ).astype(bool, copy=False)
        except Exception:
            pass
        ia_conn = np.asarray(ia_conn, dtype=bool).reshape(-1)[:n]
        ib_conn = np.asarray(ib_conn, dtype=bool).reshape(-1)[:n]

        w_hat = 4.0 * q_f * (1.0 - q_f)
        w_hat = np.where(np.isfinite(w_hat), w_hat, 0.0)
        w_hat = np.clip(w_hat, 0.0, 1.0)
        if not np.any(w_hat > 0.0):
            if need_tube:
                committor_state.tube_counted_keys.add(key_s)
            if need_edge:
                committor_state.edge_counted_keys.add(key_s)
            if need_conn:
                committor_state.conn_counted_keys.add(key_s)
            continue

        m_bins = np.floor(float(M) * q_f).astype(np.int64, copy=False)
        m_bins = np.clip(m_bins, 0, int(M) - 1)
        labels_f = np.clip(labels_f, 0, int(K) - 1)
        cell = m_bins * int(K) + labels_f

        if need_tube and (not tube_done) and committor_state.tube_counts is not None:
            np.add.at(committor_state.tube_counts, cell, w_hat)
            committor_state.tube_total_weight += float(np.sum(w_hat))
            committor_state.tube_counted_keys.add(key_s)
            added_tube += 1

        if need_edge and (not edge_done) and committor_state.edge_counts is not None:
            if n > int(edge_delta):
                src = np.asarray(cell[: n - int(edge_delta)], dtype=np.int64)
                dst = np.asarray(cell[int(edge_delta) : n], dtype=np.int64)
                if edge_bin_change_only:
                    mb0 = np.asarray(m_bins[: n - int(edge_delta)], dtype=np.int64)
                    mb1 = np.asarray(m_bins[int(edge_delta) : n], dtype=np.int64)
                    mask = mb0 != mb1
                    src = src[mask]
                    dst = dst[mask]
                if src.size:
                    if edge_undirected:
                        a = np.minimum(src, dst)
                        b = np.maximum(src, dst)
                        src = a
                        dst = b
                    edge_idx = src.astype(np.int64) * int(n_cells) + dst.astype(np.int64)
                    if isinstance(committor_state.edge_counts, np.ndarray):
                        np.add.at(committor_state.edge_counts, edge_idx, 1.0)
                    else:
                        uniq, cnts = np.unique(edge_idx, return_counts=True)
                        d = committor_state.edge_counts
                        for u, c in zip(uniq.tolist(), cnts.tolist()):
                            d[int(u)] = float(d.get(int(u), 0.0)) + float(c)
                    committor_state.edge_total_count += float(edge_idx.size)
            committor_state.edge_counted_keys.add(key_s)
            added_edge += 1

        if need_conn and (not conn_done) and committor_state.conn_adj is not None:
            # Record which boundary-side tube cells are present (based on in_a/in_b masks).
            try:
                if int(M) > 1:
                    a_mask = ia_conn
                    b_mask = ib_conn
                    if bool(np.any(a_mask)):
                        committor_state.conn_a_cells.update(np.asarray(cell, dtype=np.int64)[a_mask].tolist())
                    if bool(np.any(b_mask)):
                        committor_state.conn_b_cells.update(np.asarray(cell, dtype=np.int64)[b_mask].tolist())
            except Exception:
                pass

            if n > int(edge_delta):
                src = np.asarray(cell[: n - int(edge_delta)], dtype=np.int64)
                dst = np.asarray(cell[int(edge_delta) : n], dtype=np.int64)
                if edge_bin_change_only:
                    mb0 = np.asarray(m_bins[: n - int(edge_delta)], dtype=np.int64)
                    mb1 = np.asarray(m_bins[int(edge_delta) : n], dtype=np.int64)
                    mask = mb0 != mb1
                    src = src[mask]
                    dst = dst[mask]
                if src.size:
                    if edge_undirected:
                        a = np.minimum(src, dst)
                        b = np.maximum(src, dst)
                        src = a
                        dst = b
                    edge_idx = src.astype(np.int64) * int(n_cells) + dst.astype(np.int64)
                    uniq = np.unique(edge_idx)
                    new = 0
                    for e in uniq.tolist():
                        ei = int(e)
                        if ei in committor_state.conn_edges:
                            continue
                        committor_state.conn_edges.add(ei)
                        u = int(ei // int(n_cells))
                        v = int(ei - u * int(n_cells))
                        if 0 <= u < int(n_cells) and 0 <= v < int(n_cells):
                            committor_state.conn_adj[u].add(v)
                            committor_state.conn_adj[v].add(u)
                        new += 1
                    if new:
                        added_conn_edges += int(new)
            committor_state.conn_counted_keys.add(key_s)
            added_conn += 1

    if added_tube:
        print(
            f"[Committor Tube] Seeded history counts ({tag}): added_trajs={added_tube} skipped={skipped} "
            f"M={int(M)} K={int(K)} total_w={float(committor_state.tube_total_weight):.1f}",
            flush=True,
        )
    if added_edge:
        mode = "dense" if isinstance(committor_state.edge_counts, np.ndarray) else "sparse"
        print(
            f"[Committor Edge] Seeded history edges ({tag}): added_trajs={added_edge} skipped={skipped} "
            f"delta={int(edge_delta)} mode={mode} total_edges={float(committor_state.edge_total_count):.1f}",
            flush=True,
        )
    if added_conn:
        try:
            n_a = int(len(committor_state.conn_a_cells))
            n_b = int(len(committor_state.conn_b_cells))
        except Exception:
            n_a = 0
            n_b = 0
        print(
            f"[Committor Conn] Seeded history graph ({tag}): added_trajs={added_conn} skipped={skipped} "
            f"added_edges={int(added_conn_edges)} A_cells={int(n_a)} B_cells={int(n_b)}",
            flush=True,
        )
        try:
            if committor_state.conn_adj is not None and committor_state.conn_a_cells and committor_state.conn_b_cells:
                dist_ab = _conn_shortest_path_len(
                    adj=committor_state.conn_adj,
                    sources=list(committor_state.conn_a_cells),
                    targets=set(committor_state.conn_b_cells),
                    overlay=None,
                )
                dist_s = "inf" if not np.isfinite(dist_ab) else str(int(dist_ab))
                print(f"[Committor Conn] A→B shortest path (history graph): {dist_s}", flush=True)
        except Exception:
            pass
