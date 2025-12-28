"""Plotting utilities: Ramachandran and exploration summary panels."""
from .._imports import *
from .geometry import calculate_rmsd
from ..training.scorer import featurize_for_pca
import os

def plot_ramachandran_by_residue(all_cvs, cv_names, fname, residue_ids=None):
    """
    Generate per-residue Ramachandran plots for any protein.

    - Infers residue indices from `cv_names` (expects names like 'phi_#'/'psi_#').
    - Dynamically sizes the subplot grid based on number of residues.
    - Optionally restrict to a provided list of `residue_ids` (1-based).
    """
    print(f"Generating per-residue Ramachandran plot: {fname}")

    # Infer residue IDs from cv_names if not provided. Prefer residues with both φ and ψ.
    if residue_ids is None:
        res_from_phi = sorted({int(n.split('_')[1]) for n in cv_names if n.startswith('phi_') and n.split('_')[1].isdigit()})
        res_from_psi = sorted({int(n.split('_')[1]) for n in cv_names if n.startswith('psi_') and n.split('_')[1].isdigit()})
        both = [r for r in res_from_phi if r in res_from_psi]
        # Uniformly choose up to 10 residues across the sequence
        if len(both) == 0:
            all_ids = []
        elif len(both) <= 10:
            all_ids = both
        else:
            targets = np.linspace(0, len(both) - 1, num=10)
            selected_idxs = []
            used = set()
            for t in targets:
                base = int(round(t))
                left = base
                right = base
                while True:
                    added = False
                    if 0 <= left and left not in used:
                        used.add(left); selected_idxs.append(left); added = True
                    elif right < len(both) and right not in used:
                        used.add(right); selected_idxs.append(right); added = True
                    if added:
                        break
                    left -= 1; right += 1
            selected_idxs = sorted(selected_idxs[:10])
            all_ids = [both[i] for i in selected_idxs]
    else:
        all_ids = sorted(list(set(int(r) for r in residue_ids)))

    if not all_ids:
        print("[WARN] No residue IDs inferred from cv_names; skipping plot.")
        return

    n_res = len(all_ids)
    # Default layout: up to 2 rows x 5 cols to show at most 10 residues
    n_cols = 5 if n_res > 5 else n_res
    n_rows = int(np.ceil(n_res / max(1, n_cols)))

    # Scale figure size with grid
    fig_w = 4.0 * n_cols
    fig_h = 4.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True,
                             subplot_kw={'aspect': 'equal'})
    axes = np.atleast_1d(axes).ravel()

    for idx, rid in enumerate(all_ids):
        ax = axes[idx]
        phi_name = f'phi_{rid}'
        psi_name = f'psi_{rid}'

        try:
            phi_idx = cv_names.index(phi_name)
            psi_idx = cv_names.index(psi_name)

            phi_data = all_cvs[:, phi_idx]
            psi_data = all_cvs[:, psi_idx]

            ax.hexbin(
                phi_data,
                psi_data,
                gridsize=80,
                cmap="viridis",
                norm=LogNorm(),
                extent=[-math.pi, math.pi, -math.pi, math.pi],
            )
            ax.set_title(f"Residue {rid}")
        except (ValueError, IndexError):
            ax.text(0.5, 0.5, "Data not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Residue {rid}")

        ax.set_xlim([-math.pi, math.pi])
        ax.set_ylim([-math.pi, math.pi])
        ax.axhline(0, color='grey', lw=0.5, linestyle='--')
        ax.axvline(0, color='grey', lw=0.5, linestyle='--')

    # Hide any unused axes
    for j in range(n_res, len(axes)):
        axes[j].axis('off')

    fig.supxlabel("φ (rad)", fontsize=12)
    fig.supylabel("ψ (rad)", fontsize=12)
    fig.suptitle("Per-Residue Ramachandran (10 uniform residues)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved per-residue Ramachandran plot to {fname}")

def plot_analysis_summary(
    all_epoch_data,
    pca_scorer,
    cv_names,
    ref_pdb_path,
    fname,
    bins=60,
    rama_residues=None,
    feature_builder=None,
    ep=None,
    vamp_bin_counts=None,
    vamp_bin_clip=None,
    vamp_bin_total=None,
    **_,
):
    """
    Generate a 2x2 analysis plot of trajectory data, generalized to any protein.
    """
    if not all_epoch_data: return

    print(f"Generating summary plot: {fname}")

    # --- Aggregate and (optionally) downsample to control memory ---
    flat = [item for epoch_data in all_epoch_data for item in epoch_data]
    if not flat: return

    max_plot_frames = int(os.environ.get("GESPO_MAX_PLOT_FRAMES", "20000"))
    total_frames = len(flat)
    if total_frames > max_plot_frames:
        sel_idx = np.linspace(0, total_frames - 1, num=max_plot_frames, dtype=int)
        print(f"Subsampling frames for plotting: {max_plot_frames}/{total_frames}")
    else:
        sel_idx = np.arange(total_frames, dtype=int)
        
    all_cvs = np.vstack([flat[i][0] for i in sel_idx])
    all_coords = np.array([flat[i][1] for i in sel_idx])

    # Extract per-frame energies in kJ/mol when available.
    # Frames where the energy is missing or non-finite are marked as NaN and
    # later excluded from the Energy/RMSD panel.
    energies = []
    for i in sel_idx:
        e_val = np.nan
        try:
            frame = flat[i]
            if len(frame) >= 3:
                e = frame[2]
                try:
                    # Convert OpenMM Quantity to kJ/mol if applicable
                    if hasattr(e, "value_in_unit"):
                        from openmm.unit import kilojoule_per_mole as _kj
                        e_val = float(e.value_in_unit(_kj))
                    else:
                        e_val = float(e)
                except Exception:
                    e_val = np.nan
        except Exception:
            e_val = np.nan
        energies.append(e_val)
    energies = np.asarray(energies, dtype=np.float32)
    # Debug: report how many frames have valid energies (finite values).
    valid_energy_mask = np.isfinite(energies)
    n_valid_energy = int(np.count_nonzero(valid_energy_mask))
    print(f"[Plot] Found {n_valid_energy}/{energies.shape[0]} frames with valid energies for Energy/RMSD diagnostics.")

    ref_pdb = PDBFile(ref_pdb_path)
    ref_positions = ref_pdb.positions.value_in_unit(nanometer)
    
    ca_indices = [a.index for a in ref_pdb.topology.atoms() if a.name == 'CA']
    rmsd_values = calculate_rmsd(all_coords, ref_positions, atom_indices=ca_indices)
    e2e_dist = None
    try:
        if len(ca_indices) >= 2:
            e2e_dist = np.linalg.norm(
                all_coords[:, ca_indices[0], :] - all_coords[:, ca_indices[-1], :],
                axis=1,
            )
    except Exception:
        e2e_dist = None
    
    # Optional RMSD filter
    try:
        rf = float(os.environ.get("GESPO_RMSD_FILTER", "0.0"))
    except: rf = 0.0
    if rf > 0:
        mask = rmsd_values <= rf
        kept = int(np.count_nonzero(mask))
        if kept >= 10:
            all_cvs = all_cvs[mask]
            all_coords = all_coords[mask]
            rmsd_values = rmsd_values[mask]
            energies = energies[mask]
            if e2e_dist is not None:
                e2e_dist = e2e_dist[mask]
            print(f"[RMSD filter] Plotting kept {kept} frames")

    # Detect TICA mode
    proj_name = getattr(pca_scorer, 'projection_name', 'PCA')
    use_tica_msm = bool(getattr(pca_scorer, 'use_msm_reweight', False) and str(proj_name).upper() == 'TICA')

    # --- Helper: Feature Builder (Must match Training Logic) ---
    def _build_features_for_pca(cvs, coords):
        mode = str(getattr(pca_scorer, 'feature_mode', 'torsion')).lower()
        
        # [FIX] Match the Sparse Pairwise logic from training
        if mode == 'pairwise':
            ca_idx = getattr(pca_scorer, 'ca_indices', None)
            if ca_idx is None:
                # Try to infer from ref_pdb if scorer doesn't have it
                ca_idx = [a.index for a in ref_pdb.topology.atoms() if a.name == 'CA']
            
            if ca_idx is None or len(ca_idx) == 0:
                return featurize_for_pca(cvs) # Fallback

            n_res = len(ca_idx)
            out_feats = []
            
            for t in range(coords.shape[0]):
                # coords is (N, 3)
                ca_pos = coords[t][ca_idx] 
                dists = []
                for i in range(n_res):
                    for j in range(i + 3, n_res): # Sparse Lag > 2
                        d = float(np.linalg.norm(ca_pos[i] - ca_pos[j]))
                        if d > 100.0: d = 100.0
                        dists.append(d)
                out_feats.append(np.array(dists, dtype=np.float32))
            
            return np.array(out_feats, dtype=np.float32)

        # [Existing Logic]
        elif mode == 'tica_ca':
            ca_idx = getattr(pca_scorer, 'ca_indices', None)
            ref_ca_nm = getattr(pca_scorer, 'ref_ca_nm', None)
            if ca_idx is None or ref_ca_nm is None:
                return featurize_for_pca(cvs)
            feats = []
            for t in range(coords.shape[0]):
                pos_nm = coords[t]
                cur_ca = pos_nm[ca_idx]
                try:
                    from .geometry import kabsch_align as _kab
                    aligned = _kab(cur_ca, ref_ca_nm)
                except: aligned = cur_ca
                feats.append(aligned.reshape(-1).astype(np.float32))
            return np.asarray(feats, dtype=np.float32)
        
        else:
            # Default Torsion
            return featurize_for_pca(cvs)

    # --- Optional: filter extreme projection outliers for plotting ---
    # A small number of frames with very large IC values can squeeze the
    # color scale/axes along the edges of the summary panels. For plotting
    # only (not for training), we allow an optional radial cutoff in the
    # primary projection space.
    try:
        q = float(os.environ.get("GESPO_PLOT_IC_RADIUS_Q", "0.0"))
    except Exception:
        q = 0.0

    # 0 < q <= 1: keep frames within the q-quantile radius; q==0 disables.
    plot_radius_quantile = q if 0.0 < q <= 1.0 else 0.0

    # --- Optional: VAMP-bin PMF panel (linear VAMP coords, z-scored) ---
    vamp_pmf = None
    vamp_pmf_edges = None
    vamp_pmf_vmax = None
    vamp_pmf_visited = None
    vamp_pmf_total = None
    try:
        if vamp_bin_counts is not None:
            counts = np.asarray(vamp_bin_counts, dtype=np.float64)
            if counts.ndim == 2 and counts.shape[0] == counts.shape[1] and counts.shape[0] > 1:
                bins_v = int(counts.shape[0])
                clip_v = float(vamp_bin_clip) if vamp_bin_clip is not None else 3.0
                if clip_v > 0:
                    total_v = float(vamp_bin_total) if vamp_bin_total is not None else float(counts.sum())
                    if not np.isfinite(total_v) or total_v <= 0:
                        total_v = float(counts.sum())
                    nbins = int(bins_v) * int(bins_v)
                    p = (counts + 1.0) / (total_v + float(nbins))
                    pmf_v = -np.log(p + 1e-12)
                    pmf_v -= float(np.nanmin(pmf_v))
                    try:
                        vmax = float(os.environ.get("GESPO_VAMP_PMF_VMAX", "10.0"))
                    except Exception:
                        vmax = 10.0
                    pmf_v = np.minimum(pmf_v, vmax)
                    edges = np.linspace(-clip_v, clip_v, num=bins_v + 1, dtype=np.float64)
                    vamp_pmf = pmf_v.astype(np.float32, copy=False)
                    vamp_pmf_edges = edges.astype(np.float32, copy=False)
                    vamp_pmf_vmax = float(vmax)
                    vamp_pmf_visited = int(np.count_nonzero(counts > 0))
                    vamp_pmf_total = float(total_v)
    except Exception as exc_vamp_pmf:
        print(f"[Plot] Warning: failed to prepare VAMP-bin PMF panel: {exc_vamp_pmf}")
        vamp_pmf = None

    # --- Create Figure ---
    # 3x2 layout (optional +1 row for VAMP PMF):
    #   (A) Primary projection (TICA/PCA)
    #   (B) MSM-reweighted PMF (if available) or RMSD/E2E
    #   (C) Plain PCA projection
    #   (D) PCA density (counts)
    #   (E,F) Two Ramachandran panels
    #   (G) VAMP-bin PMF (if provided)
    if vamp_pmf is not None and vamp_pmf_edges is not None:
        fig = plt.figure(figsize=(18, 28))
        gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 22))
        gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # --- Build features and projections once ---
    cv_pca = None
    cv_pca_plain = None
    macro_labels = None
    macro_centers = None
    frame_weights = None
    vamp_labels = None
    feats = None
    if pca_scorer.is_ready:
        try:
            if feature_builder is not None:
                feats = feature_builder.build_projection_features(all_cvs, all_coords)
                feats = np.asarray(feats, dtype=np.float32)
                if feats.ndim == 3 and feats.shape[0] == all_cvs.shape[0]:
                    feats = feats.reshape(all_cvs.shape[0], -1)
            else:
                feats = _build_features_for_pca(all_cvs, all_coords)
            feats = np.asarray(feats, dtype=np.float32)
            cv_pca = pca_scorer.pca.transform(feats)
            # Plain PCA on the same features for comparison
            try:
                cv_pca_plain = PCA(n_components=2).fit_transform(feats)
            except Exception:
                cv_pca_plain = None
        except Exception:
            cv_pca = None
            cv_pca_plain = None

        # Apply outlier filter in IC space for plotting only (if enabled).
        if cv_pca is not None and cv_pca.shape[0] == all_cvs.shape[0] and plot_radius_quantile > 0.0:
            try:
                radii = np.linalg.norm(cv_pca[:, :2], axis=1)
                r_cut = np.quantile(radii[np.isfinite(radii)], plot_radius_quantile)
                mask = radii <= r_cut
                if mask.any() and mask.sum() >= 10:
                    cv_pca = cv_pca[mask]
                    if cv_pca_plain is not None and cv_pca_plain.shape[0] == mask.shape[0]:
                        # If cv_pca_plain was built from the same feats, keep it in sync
                        cv_pca_plain = cv_pca_plain[mask]
                    if macro_labels is not None and macro_labels.shape[0] == mask.shape[0]:
                        macro_labels = macro_labels[mask]
                    if frame_weights is not None and frame_weights.shape[0] == mask.shape[0]:
                        frame_weights = frame_weights[mask]
                    if rmsd_values.shape[0] == mask.shape[0]:
                        rmsd_values = rmsd_values[mask]
                    if energies.shape[0] == mask.shape[0]:
                        energies = energies[mask]
                    if e2e_dist is not None and e2e_dist.shape[0] == mask.shape[0]:
                        e2e_dist = e2e_dist[mask]
                    all_cvs = all_cvs[mask]
                    all_coords = all_coords[mask]
                    if feats is not None and feats.shape[0] == mask.shape[0]:
                        feats = feats[mask]
                    print(f"[Plot] Applied IC radius quantile filter q={plot_radius_quantile:.3f}; kept {int(mask.sum())}/{len(mask)} frames for plotting.")
            except Exception as exc_filter:
                print(f"[Plot] Failed to apply IC outlier filter: {exc_filter}")

        # If TICA+MSM/PCCA metadata is available, build per-frame macrostate labels
        # and MSM-inspired frame weights for plotting.
        if use_tica_msm and cv_pca is not None:
            try:
                pcca_centers = getattr(pca_scorer, "pcca_centers", None)
                micro_to_macro = getattr(pca_scorer, "pcca_micro_to_macro", None)
                pi_full = getattr(pca_scorer, "msm_pi_full", None)
                if (
                    pcca_centers is not None
                    and micro_to_macro is not None
                    and pi_full is not None
                    and len(pcca_centers) > 0
                ):
                    centers_arr = np.asarray(pcca_centers, dtype=np.float32)
                    m2M = np.asarray(micro_to_macro, dtype=int)
                    pi_arr = np.asarray(pi_full, dtype=float)

                    # Map each frame to its nearest microstate center
                    d2 = ((cv_pca[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
                    micro_labels = d2.argmin(axis=1).astype(int)
                    micro_labels_clipped = np.clip(micro_labels, 0, len(m2M) - 1)
                    macro_labels = m2M[micro_labels_clipped]

                    # Frame weights: distribute pi_full over frames in each microstate
                    counts = np.bincount(micro_labels, minlength=len(pi_arr))
                    w_micro = np.zeros_like(pi_arr, dtype=float)
                    visited = counts > 0
                    w_micro[visited] = pi_arr[visited] / counts[visited]
                    frame_weights = w_micro[micro_labels]
                    total_w = frame_weights.sum()
                    if total_w > 0 and np.isfinite(total_w):
                        frame_weights = frame_weights / total_w
                    else:
                        frame_weights = np.ones_like(frame_weights, dtype=float) / len(frame_weights)

                    # Macro centers for annotation
                    n_sets = int(max(0, m2M.max() + 1)) if np.any(m2M >= 0) else 0
                    if n_sets > 0:
                        macro_centers = np.full((n_sets, 2), np.nan, dtype=float)
                        for m in range(n_sets):
                            mask = macro_labels == m
                            if np.any(mask):
                                macro_centers[m] = cv_pca[mask].mean(axis=0)
                    else:
                        macro_centers = None
            except Exception as exc_pcca_plot:
                print(f"[Plot] Failed to construct PCCA macrostate labels for plotting: {exc_pcca_plot}")
                macro_labels = None
                macro_centers = None
                frame_weights = None

        # If we are in DeepTICA/VAMP mode and the projector exposes a model
        # with soft VAMP state outputs, build per-frame hard state labels for
        # coloring diagnostics (e.g., Energy vs RMSD).
        try:
            proj_name_upper = str(proj_name).upper()
            projector = getattr(pca_scorer, "pca", None)
            model = getattr(projector, "model", None) if projector is not None else None
            if (
                proj_name_upper == "DEEPTICA"
                and model is not None
                and feats is not None
                and feats.shape[0] == all_cvs.shape[0]
                and hasattr(model, "get_hidden")
            ):
                import torch as _torch
                x_t = _torch.tensor(feats, dtype=_torch.float32, device=model.average.device)
                with _torch.no_grad():
                    z = model.get_hidden(x_t)
                    probs = z
                probs_np = probs.detach().cpu().numpy().astype(np.float32)
                vamp_labels = probs_np.argmax(axis=1).astype(int)
        except Exception as exc_vlab:
            print(f"[Plot] Failed to construct VAMP cluster labels for Energy/RMSD panel: {exc_vlab}")
            vamp_labels = None

    # --- Panel A: Primary Projection (TICA/PCA) ---
    ax0 = plt.subplot(gs[0, 0])
    if pca_scorer.is_ready and cv_pca is not None:
        try:
            if use_tica_msm:
                # If PCCA metadata is available, color by macrostate cluster.
                if macro_labels is not None and macro_centers is not None:
                    scatter = ax0.scatter(
                        cv_pca[:, 0],
                        cv_pca[:, 1],
                        c=macro_labels,
                        s=1,
                        alpha=0.5,
                        cmap="tab10",
                    )
                    ax0.scatter(
                        macro_centers[:, 0],
                        macro_centers[:, 1],
                        c="red",
                        s=200,
                        marker="X",
                        edgecolors="white",
                        linewidths=2,
                        zorder=5,
                        label="PCCA Centers",
                    )
                    for m, (x_c, y_c) in enumerate(macro_centers):
                        if np.isfinite(x_c) and np.isfinite(y_c):
                            ax0.text(
                                x_c,
                                y_c,
                                str(m),
                                fontsize=10,
                                ha="center",
                                va="center",
                                color="white",
                                weight="bold",
                            )
                    fig.colorbar(scatter, ax=ax0, label="Macrostate ID")
                    ax0.set_title(f"(A) PCCA Macrostates in {proj_name}", pad=15)
                else:
                    # Fallback: high-resolution unweighted PMF
                    H, x_e, y_e = np.histogram2d(cv_pca[:, 0], cv_pca[:, 1], bins=100)
                    H = H.astype(float)
                    pmf = -np.log(H / H.sum() + 1e-12)
                    pmf -= np.nanmin(pmf)
                    pmf[H == 0] = np.nan
                    X, Y = np.meshgrid(x_e[:-1], y_e[:-1])
                    im0 = ax0.pcolormesh(X.T, Y.T, pmf, cmap="viridis", shading="auto", vmax=10.0)
                    fig.colorbar(im0, ax=ax0, label="PMF (kT)")
                    ax0.set_title(f"(A) Unweighted PMF in {proj_name}", pad=15)
            else:
                # Non-TICA or MSM disabled: show high-resolution counts.
                H, x_e, y_e = np.histogram2d(cv_pca[:, 0], cv_pca[:, 1], bins=100)
                H = H.astype(float)
                im0 = ax0.pcolormesh(
                    x_e[:-1],
                    y_e[:-1],
                    H.T,
                    cmap="magma",
                    shading="auto",
                    norm=LogNorm(),
                )
                fig.colorbar(im0, ax=ax0, label="Counts")
                ax0.set_title(f"(A) {proj_name} Projection", pad=15)
            ax0.set_xlabel("IC 1"); ax0.set_ylabel("IC 2")
        except Exception as e:
            ax0.text(0.5, 0.5, f"Projection Error:\n{str(e)}", ha="center")
    else:
        ax0.text(0.5, 0.5, "Model not ready", ha="center")

    # --- Panel B: MSM-Reweighted PMF or RMSD/E2E ---
    ax1 = plt.subplot(gs[0, 1])
    if use_tica_msm and pca_scorer.is_ready and cv_pca is not None:
        try:
            if frame_weights is not None:
                # High-resolution MSM-reweighted PMF as in the notebook snippet
                hist_rw, xedges_rw, yedges_rw = np.histogram2d(
                    cv_pca[:, 0],
                    cv_pca[:, 1],
                    bins=100,
                    weights=frame_weights,
                    density=True,
                )
                pmf_rw = -np.log(hist_rw + 1e-10)
                pmf_rw -= np.nanmin(pmf_rw)
                VMAX = 5.0
                pmf_rw[pmf_rw > VMAX] = np.inf
                X_rw, Y_rw = np.meshgrid(xedges_rw[:-1], yedges_rw[:-1])
                contourf = ax1.contourf(
                    X_rw,
                    Y_rw,
                    pmf_rw.T,
                    levels=20,
                    cmap="viridis",
                    vmin=0,
                    vmax=VMAX,
                )
                ax1.contour(
                    X_rw,
                    Y_rw,
                    pmf_rw.T,
                    levels=5,
                    colors="white",
                    linewidths=0.5,
                    alpha=0.6,
                )
                if macro_centers is not None:
                    ax1.scatter(
                        macro_centers[:, 0],
                        macro_centers[:, 1],
                        c="red",
                        s=200,
                        marker="X",
                        edgecolors="white",
                        linewidths=2,
                        zorder=5,
                    )
                fig.colorbar(contourf, ax=ax1, label="PMF (kT)")
                ax1.set_title("(B) MSM-Reweighted PMF", pad=15)
                ax1.set_xlabel("IC 1"); ax1.set_ylabel("IC 2")
            else:
                ax1.text(0.5, 0.5, "MSM/PCCA metadata unavailable", ha="center")
        except Exception as e:
            ax1.text(0.5, 0.5, f"Reweighted PMF error:\n{str(e)}", ha="center")
    else:
        # Fallback: RMSD vs End-to-End
        if e2e_dist is None:
            try:
                e2e_dist = np.zeros_like(rmsd_values)
            except Exception:
                e2e_dist = np.zeros(len(all_cvs))
            print("[Plot] e2e_dist unavailable; using zeros as fallback for panel B.")
        ax1.hexbin(e2e_dist, rmsd_values, gridsize=100, cmap="cividis", norm=LogNorm())
        ax1.set_xlabel("End-to-End (nm)"); ax1.set_ylabel("RMSD (nm)")
        ax1.set_title("(B) RMSD vs E2E", pad=15)

    # --- Panel C: Plain PCA Projection (unweighted) ---
    ax2 = plt.subplot(gs[1, 0])
    if cv_pca_plain is not None:
        try:
            hb2 = ax2.hexbin(cv_pca_plain[:, 0], cv_pca_plain[:, 1], gridsize=100, cmap="magma", norm=LogNorm())
            fig.colorbar(hb2, ax=ax2, label="Counts")
            ax2.set_title("(C) Plain PCA Projection", pad=15)
            ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2")
        except Exception as e:
            ax2.text(0.5, 0.5, f"PCA projection error:\n{str(e)}", ha="center")
    else:
        ax2.text(0.5, 0.5, "Plain PCA not available", ha="center")

    # --- Panel D: PCA PMF (MSM-reweighted when available) ---
    ax3 = plt.subplot(gs[1, 1])
    if cv_pca_plain is not None:
        try:
            if use_tica_msm and pca_scorer.is_ready and cv_pca is not None:
                centers = getattr(pca_scorer, "pcca_centers", None)
                pi_full = getattr(pca_scorer, "msm_pi_full", None)
                if centers is not None and pi_full is not None:
                    # Map frames to MSM microstates in the primary (TICA/PCA) space
                    d2 = ((cv_pca[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                    labels = d2.argmin(axis=1)
                    weights = pi_full[labels]
                    # Histogram in plain PCA space, weighted by MSM stationary probs
                    H_p, x_p, y_p = np.histogram2d(
                        cv_pca_plain[:, 0], cv_pca_plain[:, 1],
                        bins=bins,
                        weights=weights,
                    )
                    H_p = H_p.astype(float)
                    pmf_p = -np.log(H_p / H_p.sum() + 1e-12)
                    pmf_p -= np.nanmin(pmf_p)
                    pmf_p[H_p == 0] = np.nan
                    Xp, Yp = np.meshgrid(x_p[:-1], y_p[:-1])
                    im3 = ax3.pcolormesh(
                        Xp.T, Yp.T, pmf_p,
                        cmap="viridis",
                        shading="auto",
                        vmax=10.0,
                    )
                    fig.colorbar(im3, ax=ax3, label="Reweighted PMF (kT)")
                    ax3.set_title("(D) PCA PMF (MSM-reweighted)", pad=15)
                    ax3.set_xlabel("PC 1"); ax3.set_ylabel("PC 2")
                else:
                    raise RuntimeError("MSM/PCCA metadata unavailable for PCA reweighting")
            else:
                # Fallback: unweighted PCA PMF
                H_p, x_p, y_p = np.histogram2d(
                    cv_pca_plain[:, 0], cv_pca_plain[:, 1], bins=bins
                )
                H_p = H_p.astype(float)
                pmf_p = -np.log(H_p / H_p.sum() + 1e-12)
                pmf_p -= np.nanmin(pmf_p)
                pmf_p[H_p == 0] = np.nan
                Xp, Yp = np.meshgrid(x_p[:-1], y_p[:-1])
                im3 = ax3.pcolormesh(
                    Xp.T, Yp.T, pmf_p,
                    cmap="viridis",
                    shading="auto",
                    vmax=10.0,
                )
                fig.colorbar(im3, ax=ax3, label="PMF (kT)")
                ax3.set_title("(D) Plain PCA PMF", pad=15)
                ax3.set_xlabel("PC 1"); ax3.set_ylabel("PC 2")
        except Exception as e:
            ax3.text(0.5, 0.5, f"PCA PMF error:\n{str(e)}", ha="center")
    else:
        ax3.text(0.5, 0.5, "Plain PCA not available", ha="center")

    # --- Panel E: Energy vs RMSD colored by VAMP cluster (DeepTICA) ---
    ax4 = plt.subplot(gs[2, 0])
    try:
        if energies.size and rmsd_values.size and energies.shape[0] == rmsd_values.shape[0]:
            # Restrict to frames with valid (finite) energies only.
            mask_e = np.isfinite(energies)
            if not np.any(mask_e):
                ax4.text(0.5, 0.5, "No valid energies for Energy/RMSD panel", ha="center")
            else:
                e_plot = energies[mask_e]
                r_plot = rmsd_values[mask_e]
                v_plot = None
                if vamp_labels is not None and vamp_labels.shape[0] == energies.shape[0]:
                    v_plot = vamp_labels[mask_e]

                if v_plot is not None:
                    sc = ax4.scatter(
                        r_plot,
                        e_plot,
                        c=v_plot,
                        s=5,
                        cmap="tab10",
                        alpha=0.7,
                        edgecolors="none",
                    )
                    cbar = fig.colorbar(sc, ax=ax4)
                    cbar.set_label("VAMP State ID")
                    ax4.set_title("(E) Energy vs RMSD (VAMP states)", pad=15)
                else:
                    hb_e = ax4.hexbin(
                        r_plot,
                        e_plot,
                        gridsize=100,
                        cmap="viridis",
                        norm=LogNorm(),
                    )
                    fig.colorbar(hb_e, ax=ax4, label="Counts")
                    ax4.set_title("(E) Energy vs RMSD", pad=15)
                ax4.set_xlabel("RMSD (nm)")
                ax4.set_ylabel("Energy (kJ/mol)")
        else:
            ax4.text(0.5, 0.5, "Energy/RMSD unavailable", ha="center")
    except Exception as exc_e:
        ax4.text(0.5, 0.5, f"Energy/RMSD panel error:\n{str(exc_e)}", ha="center")

    # --- Panel F: Single-residue Ramachandran (for quick angle view) ---
    res_phi = sorted({int(n.split('_')[1]) for n in cv_names if 'phi' in n})
    r2 = res_phi[-2] if len(res_phi) > 2 else (res_phi[0] if res_phi else 0)
    ax5 = plt.subplot(gs[2, 1])
    try:
        if res_phi:
            idx_p = cv_names.index(f'phi_{r2}')
            idx_s = cv_names.index(f'psi_{r2}')
            ax5.hexbin(
                all_cvs[:, idx_p],
                all_cvs[:, idx_s],
                gridsize=100,
                cmap="viridis",
                norm=LogNorm(),
                extent=[-np.pi, np.pi, -np.pi, np.pi],
            )
            ax5.set_title(f"(F) Res {r2}")
        else:
            ax5.text(0.5, 0.5, "No Ramachandran data", ha="center")
    except Exception as exc_f:
        ax5.text(0.5, 0.5, f"Ramachandran panel error:\n{str(exc_f)}", ha="center")

    # --- Panel G: Linear VAMP PMF used for bin counting (if available) ---
    if vamp_pmf is not None and vamp_pmf_edges is not None:
        ax6 = plt.subplot(gs[3, :])
        try:
            pmf_plot = np.asarray(vamp_pmf, dtype=np.float32).T
            edges = np.asarray(vamp_pmf_edges, dtype=np.float32)
            vmax = float(vamp_pmf_vmax) if vamp_pmf_vmax is not None else 10.0
            im6 = ax6.pcolormesh(
                edges,
                edges,
                pmf_plot,
                cmap="viridis",
                shading="auto",
                vmin=0.0,
                vmax=vmax,
            )
            fig.colorbar(im6, ax=ax6, label="PMF (kT)")
            nbins = int(vamp_pmf.shape[0]) * int(vamp_pmf.shape[1])
            visited = int(vamp_pmf_visited) if vamp_pmf_visited is not None else None
            total = float(vamp_pmf_total) if vamp_pmf_total is not None else None
            extra = []
            if visited is not None:
                extra.append(f"visited={visited}/{nbins}")
            if total is not None:
                extra.append(f"frames~{int(total)}")
            extra_s = (" (" + ", ".join(extra) + ")") if extra else ""
            ax6.set_title(f"(G) Linear VAMP PMF (bin-count space){extra_s}", pad=15)
            ax6.set_xlabel("Linear VAMP 1 (z-score)")
            ax6.set_ylabel("Linear VAMP 2 (z-score)")
        except Exception as exc_g:
            ax6.text(0.5, 0.5, f"VAMP PMF error:\n{str(exc_g)}", ha="center")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved plot to {fname}")


def plot_transition_diagnostics(
    sym_matrix: np.ndarray,
    sym_counts: np.ndarray,
    raw_matrix: np.ndarray,
    raw_counts: np.ndarray,
    pca_coords: np.ndarray,
    cluster_labels: np.ndarray,
    point_scores: np.ndarray,
    fname: Union[str, Path],
    title: str = "",
    proj_name: str = "PCA",
):
    """
    Save a figure with heatmaps (sym/raw) and PCA scatters colored by cluster and transition score.
    """
    sym_matrix = np.asarray(sym_matrix, dtype=float)
    raw_matrix = np.asarray(raw_matrix, dtype=float)
    sym_counts = np.asarray(sym_counts, dtype=int)
    raw_counts = np.asarray(raw_counts, dtype=int)
    pca_coords = np.asarray(pca_coords, dtype=float)
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    point_scores = np.asarray(point_scores, dtype=float)
    num_states = sym_matrix.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    vmin, vmax = 0.0, max(sym_matrix.max() if sym_matrix.size else 1.0, raw_matrix.max() if raw_matrix.size else 1.0)
    panels = [
        (axes[0, 0], sym_matrix, sym_counts, "Symmetrized"),
        (axes[0, 1], raw_matrix, raw_counts, "Raw"),
    ]

    for ax, matrix, counts, panel_title in panels:
        im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Destination Cluster")
        ax.set_ylabel("Source Cluster")
        ax.set_title(panel_title)
        for i in range(num_states):
            for j in range(num_states):
                prob = matrix[i, j]
                count = counts[i, j] if counts.size else 0
                text_color = "white" if prob > 0.5 else "black"
                ax.text(j, i, f"{prob:.2f}\n({count})", ha="center", va="center", color=text_color, fontsize=8)
        fig.colorbar(im, ax=ax, label="Transition Probability", fraction=0.046, pad=0.04)

    scatter_ax = axes[1, 0]
    unique_clusters = np.unique(cluster_labels)
    cmap = plt.get_cmap("tab20", max(len(unique_clusters), 1))
    norm = plt.Normalize(vmin=unique_clusters.min() if unique_clusters.size else 0, vmax=unique_clusters.max() if unique_clusters.size else 1)
    scatter = scatter_ax.scatter(
        pca_coords[:, 0],
        pca_coords[:, 1],
        c=cluster_labels if unique_clusters.size else np.zeros_like(cluster_labels),
        cmap=cmap,
        norm=norm,
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    # Dynamic axis/labels depending on projection
    label1 = "IC 1" if str(proj_name).upper() == "TICA" else ("PC 1" if str(proj_name).upper() == "PCA" else "Comp 1")
    label2 = "IC 2" if str(proj_name).upper() == "TICA" else ("PC 2" if str(proj_name).upper() == "PCA" else "Comp 2")
    scatter_ax.set_title(f"{proj_name} colored by cluster ID")
    scatter_ax.set_xlabel(label1)
    scatter_ax.set_ylabel(label2)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(norm(lbl)),
            label=f"C{int(lbl):02d}",
            markersize=6,
        )
        for lbl in unique_clusters
    ]
    if handles:
        scatter_ax.legend(
            handles=handles,
            title="Cluster",
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
        )

    score_ax = axes[1, 1]
    score_scatter = score_ax.scatter(
        pca_coords[:, 0],
        pca_coords[:, 1],
        c=point_scores,
        cmap="plasma",
        s=20,
        alpha=0.7,
        edgecolors="none",
    )
    score_ax.set_title(f"{proj_name} colored by transition score")
    score_ax.set_xlabel(label1)
    score_ax.set_ylabel(label2)
    fig.colorbar(score_scatter, ax=score_ax, label="Transition score", fraction=0.046, pad=0.04)

    fig.suptitle(title or "Transition Diagnostics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = Path(fname)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved transition diagnostics figure to {fname}")
