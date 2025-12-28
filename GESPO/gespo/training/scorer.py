"""Novelty scoring (Vendi) and PCA-based exploration utilities."""
from typing import Tuple
from .._imports import *

def featurize_for_pca(cv_data: np.ndarray) -> np.ndarray:
    """Converts an array of dihedral angles to its sine/cosine representation."""
    if cv_data.ndim == 1:
        cv_data = cv_data.reshape(1, -1)
    sines = np.sin(cv_data)
    cosines = np.cos(cv_data)
    return np.concatenate([sines, cosines], axis=1)

def weight_K(K, p=None): return K / K.shape[0] if p is None else K * torch.outer(torch.sqrt(p), torch.sqrt(p))

def normalize_K(K): d = torch.sqrt(torch.diagonal(K)); return K / torch.outer(d, d)

def entropy_q(w): w = torch.clamp(w[w > 0], min=1e-12); return -(w * torch.log(w)).sum()

def vendi_score(K, p=None, normalize=False, jitter=1e-6, max_tries=7):
    if normalize:
        K = normalize_K(K)
    K = weight_K(K, p)
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    for _ in range(max_tries):
        try:
            w, _ = torch.linalg.eigh(K)
            break
        except RuntimeError:
            K = K + jitter * eye
            jitter *= 10
    else:
        warnings.warn("vendi_score: eigen decomposition failed, using uniform fallback.")
        w = torch.full((K.shape[-1],), 1.0 / K.shape[-1], device=K.device, dtype=K.dtype)
        return torch.exp(entropy_q(w))
    w = torch.clamp(w, min=1e-12)
    return torch.exp(entropy_q(w))

class PCAScorer:
    """
    Coverage/novelty scoring in a 2D projected space.

    - By default fits a 2D PCA on provided features.
    - If a fixed projector (e.g., TICA) is provided, uses its .transform() without refitting.
    """
    def __init__(self,
                 n_cvs: int,
                 n_components: int = 2,
                 bins: int = 50,
                 completion_threshold: float = 0.9,
                 projector=None,
                 projection_name: Optional[str] = None,
                 freeze_projector: bool = False):
        self.n_cvs = n_cvs
        self.n_components = n_components
        self.bins = bins
        self.completion_threshold = completion_threshold
        # Projector can be a fitted PCA/TICA-like object exposing .transform(...)
        self.pca = projector
        self.projection_name = projection_name or ("PCA" if projector is None else "Projection")
        self.freeze_projector = bool(freeze_projector and projector is not None)
        self.edges = None
        self.counts = None
        # Smoothed density for coverage rewards (post-convolution)
        self.smoothed_counts = None
        # Optional secondary PCA (plain PCA on features) so we can
        # simultaneously score coverage in the primary projection
        # (e.g., TICA/VAMPNet) and in a standard PCA space.
        self.plain_pca = None
        self.plain_edges = None
        self.plain_counts = None
        self.is_ready = False
        self.exploration_is_complete = False
        self.last_update_ep = 0

    def load_tica_model(self, path: str) -> None:
        """
        Load a pre-trained TICA model from disk and register it as the projector.

        The loaded object is expected to expose a scikit-learn–style
        .transform(...) method (and optionally .fit(...) if online refits are
        desired).
        """
        if not path:
            return
        try:
            import os as _os
            import pickle as _pickle
            if not _os.path.isfile(path):
                print(f"[TICA] Requested TICA model '{path}' not found; skipping load.")
                return
            
            # Check if it's a DeepTICA checkpoint (pt) or pickle
            if path.endswith(".pt") or path.endswith(".pth"):
                # Assume DeepTICA
                # We need to instantiate a DeepTICAProjector first
                # Note: We don't know the input dim or architecture from just the file easily
                # unless we save metadata. For now, we assume the user provides a pickle wrapper
                # OR we just support loading pickle for now as per original code.
                # If the user wants to load a DeepTICA model, they should probably wrap it.
                # BUT, let's try to load it if it's a DeepTICAProjector pickle.
                pass

            with open(path, "rb") as f:
                projector = _pickle.load(f)
            
            self.pca = projector
            self.projection_name = "TICA"
            # Allow refits unless the caller explicitly freezes the projector
            self.freeze_projector = False
            print(f"[TICA] Loaded existing TICA model from {path}.")
        except Exception as exc:
            print(f"[TICA] Failed to load TICA model from {path}: {exc}")

    def update_model(self, cv_data: np.ndarray, current_ep: int, features: Optional[np.ndarray] = None, trajectory_lengths: Optional[List[int]] = None, times: Optional[np.ndarray] = None, is_anchoring: bool = False):
        """Update PCA/TICA projection model (now split into smaller helpers)."""
        if self.exploration_is_complete or len(cv_data) < self.n_components + 1:
            return

        featurized_data, traj_lengths = self._prepare_features(cv_data, features, trajectory_lengths)
        cv_pca = self._fit_projector_and_transform(featurized_data, traj_lengths, times=times, is_anchoring=is_anchoring)
        if cv_pca is None:
            return
        self.last_update_ep = current_ep

        # MSM reweighting and MSM/PCCA-based extras have been retired from the
        # active training loop. Keep a simple unweighted histogram update for
        # coverage and exploration scoring.
        weights = None

        self._update_histogram(cv_pca, weights)
        fraction_visited = np.sum(self.counts > 0) / (self.bins * self.bins)

        if fraction_visited > self.completion_threshold:
            self.exploration_is_complete = True
            print(f"--- PCA Exploration Complete ({fraction_visited:.1%}). Deactivating PCAScorer. ---")
        else:
            print(f"PCA Scorer updated ({fraction_visited:.1%} bins visited). Last updated Ep. {current_ep}.")

        # Also fit a plain PCA on the same features for secondary
        # coverage scoring when the primary projector is not already
        # PCA (e.g., TICA or VAMPNet modes).
        try:
            if str(self.projection_name).upper() != "PCA":
                self.plain_pca = PCA(n_components=self.n_components)
                cv_plain = self.plain_pca.fit_transform(featurized_data)
                # Build a separate histogram in this PCA space.
                try:
                    import os as _os
                    qlow = float(_os.environ.get("GESPO_PCA_QLOW", "1.0"))
                    qhigh = float(_os.environ.get("GESPO_PCA_QHIGH", "99.0"))
                except Exception:
                    qlow, qhigh = 1.0, 99.0
                qlow = max(0.0, min(qlow, 50.0))
                qhigh = max(50.0, min(qhigh, 100.0))
                try:
                    x_lo, x_hi = np.percentile(cv_plain[:, 0], [qlow, qhigh])
                    y_lo, y_hi = np.percentile(cv_plain[:, 1], [qlow, qhigh])
                    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo >= x_hi:
                        raise ValueError
                    if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_lo >= y_hi:
                        raise ValueError
                    x_edges = np.linspace(float(x_lo), float(x_hi), num=self.bins + 1)
                    y_edges = np.linspace(float(y_lo), float(y_hi), num=self.bins + 1)
                    plain_counts, x_edges, y_edges = np.histogram2d(
                        cv_plain[:, 0], cv_plain[:, 1], bins=[x_edges, y_edges]
                    )
                    self.plain_counts = plain_counts
                    self.plain_edges = [x_edges, y_edges]
                except Exception:
                    plain_counts, x_edges, y_edges = np.histogram2d(
                        cv_plain[:, 0], cv_plain[:, 1], bins=self.bins
                    )
                    self.plain_counts = plain_counts
                    self.plain_edges = [x_edges, y_edges]
            else:
                # In pure PCA mode, secondary PCA is redundant.
                self.plain_pca = None
                self.plain_edges = None
                self.plain_counts = None
        except Exception as exc_plain:
            print(f"[PCA-secondary] Failed to fit plain PCA for secondary coverage ({exc_plain}); disabling secondary PCA rewards.")
            self.plain_pca = None
            self.plain_edges = None
            self.plain_counts = None

    # --- Internal helpers -------------------------------------------------
    def _prepare_features(
        self,
        cv_data: np.ndarray,
        features: Optional[np.ndarray],
        trajectory_lengths: Optional[List[int]],
    ) -> Tuple[np.ndarray, Optional[List[int]]]:
        """Build input features, apply optional subsampling, and prepend base data."""
        if features is None:
            featurized_data = featurize_for_pca(cv_data)
        else:
            featurized_data = np.asarray(features, dtype=np.float32)

        try:
            import os as _os
            max_train = int(_os.environ.get("GESPO_MAX_PCA_TRAIN", "0"))
        except Exception:
            max_train = 0
        if max_train and featurized_data.shape[0] > max_train:
            if trajectory_lengths is not None and len(trajectory_lengths) > 0:
                # Use provided trajectory lengths; randomly drop whole trajectories to stay under cap
                total_frames = sum(trajectory_lengths)
                n_traj = len(trajectory_lengths)
                
                if total_frames > max_train:
                    # Compute how many trajectories to keep
                    cumsum = np.cumsum([0] + trajectory_lengths)
                    n_keep = 1
                    for i in range(1, n_traj + 1):
                        if cumsum[i] <= max_train:
                            n_keep = i
                        else:
                            break
                    
                    # Randomly select n_keep trajectories
                    keep_traj_idx = np.random.choice(n_traj, size=n_keep, replace=False)
                    keep_traj_idx.sort()
                    
                    # Build mask to keep selected trajectories
                    keep_mask = np.zeros(featurized_data.shape[0], dtype=bool)
                    offset = 0
                    new_traj_lengths = []
                    for t_idx in range(n_traj):
                        traj_len = trajectory_lengths[t_idx]
                        start = offset
                        end = offset + traj_len
                        if t_idx in keep_traj_idx:
                            keep_mask[start:end] = True
                            new_traj_lengths.append(traj_len)
                        offset = end
                    
                    kept_frames = int(np.count_nonzero(keep_mask))
                    featurized_data = featurized_data[keep_mask]
                    trajectory_lengths = new_traj_lengths
                    print(f"[TICA Subsample] Kept {n_keep}/{n_traj} trajectories ({kept_frames} frames) to respect cap {max_train}.")
            else:
                # No trajectory_lengths provided; try to infer from environment variables
                try:
                    n_md_steps = int(_os.environ.get("GESPO_N_MD_STEPS", "0"))
                    record_stride = int(_os.environ.get("GESPO_RECORD_STRIDE", "1"))
                    if n_md_steps > 0 and record_stride > 0:
                        traj_len = (n_md_steps // record_stride) + 1
                        n_traj = featurized_data.shape[0] // traj_len
                        if n_traj > 0:
                            # Randomly drop whole trajectories
                            n_keep = max(1, max_train // traj_len)
                            n_keep = min(n_traj, n_keep)
                            keep_traj_idx = np.random.choice(n_traj, size=n_keep, replace=False)
                            keep_traj_idx.sort()
                            
                            keep_mask = np.zeros(featurized_data.shape[0], dtype=bool)
                            for t_idx in keep_traj_idx:
                                start = t_idx * traj_len
                                end = min(start + traj_len, featurized_data.shape[0])
                                keep_mask[start:end] = True
                            
                            featurized_data = featurized_data[keep_mask]
                            trajectory_lengths = [traj_len] * n_keep
                            print(f"[TICA Subsample] Inferred traj_len={traj_len}, kept {n_keep}/{n_traj} trajectories to respect cap {max_train}.")
                        else:
                            print(f"[TICA Subsample] Warning: Could not infer valid trajectories; keeping all {featurized_data.shape[0]} frames.")
                    else:
                        print(f"[TICA Subsample] Warning: Could not infer trajectory structure; keeping all {featurized_data.shape[0]} frames.")
                except Exception as exc:
                    print(f"[TICA Subsample] Warning: Failed to infer trajectory structure ({exc}); keeping all frames.")

        base_feats = getattr(self, "base_features", None)
        base_traj_lens = getattr(self, "base_traj_lengths", None)
        if base_feats is not None:
            try:
                base_arr = np.asarray(base_feats, dtype=np.float32)
                if base_arr.size and base_arr.shape[1] == featurized_data.shape[1]:
                    featurized_data = np.concatenate([base_arr, featurized_data], axis=0)
                    if base_traj_lens is not None and trajectory_lengths is not None:
                        trajectory_lengths = list(base_traj_lens) + list(trajectory_lengths)
                    elif base_traj_lens is not None:
                        trajectory_lengths = list(base_traj_lens)
            except Exception:
                pass
        return featurized_data, trajectory_lengths

    def _fit_projector_and_transform(
        self, featurized_data: np.ndarray, trajectory_lengths: Optional[List[int]], times: Optional[np.ndarray] = None, is_anchoring: bool = False
    ) -> Optional[np.ndarray]:
        """Fit (or refit) the PCA/TICA projector and return projected data."""
        print(f"[Debug Scorer] _fit_projector_and_transform called. pca={type(self.pca)}, projection_name={self.projection_name}, freeze={self.freeze_projector}")
        if self.pca is not None and not self.freeze_projector:
            try:
                if hasattr(self.pca, "fit"):
                    print(f"[Debug Scorer] Attempting to fit projector. Has {len(trajectory_lengths) if trajectory_lengths else 0} trajectories.")
                    if trajectory_lengths is not None and len(trajectory_lengths) > 0:
                        traj_list = []
                        time_list = []
                        start_idx = 0
                        for traj_len in trajectory_lengths:
                            if traj_len > 0:
                                traj_list.append(featurized_data[start_idx:start_idx + traj_len])
                                if times is not None:
                                    time_list.append(times[start_idx:start_idx + traj_len])
                                start_idx += traj_len
                        
                        print(f"[Debug Scorer] Built {len(traj_list)} trajectory lists. Data shape: {featurized_data.shape}")
                        if times is not None and len(time_list) == len(traj_list):
                             # Pass times to fit if supported (DeepTICAProjector supports it)
                             # We check signature or just try passing it if it's DeepTICA
                             if str(self.projection_name).lower() == "deeptica":
                                 print(f"[Debug Scorer] Calling DeepTICA fit with times and is_anchoring={is_anchoring}")
                                 self.pca.fit(traj_list, times=time_list, is_anchoring=is_anchoring)
                             else:
                                 self.pca.fit(traj_list)
                        else:
                             print(f"[Debug Scorer] Calling fit without times, is_anchoring={is_anchoring}")
                             if str(self.projection_name).lower() == "deeptica":
                                 self.pca.fit(traj_list, is_anchoring=is_anchoring)
                             else:
                                 self.pca.fit(traj_list)
                        print(f"[Projector] Fitted with {len(traj_list)} trajectories (respecting boundaries).")
                    else:
                        if str(self.projection_name).lower() == "deeptica":
                            self.pca.fit(featurized_data, is_anchoring=is_anchoring)
                        else:
                            self.pca.fit(featurized_data)
                        print("[Projector] Warning: No trajectory_lengths; fitted as single trajectory (suboptimal for TICA).")
                print(f"[Debug Scorer] Fit successful, calling transform")
                return self.pca.transform(featurized_data)
            except Exception as exc:
                print(f"[Projector] fit failed ({exc}), falling back to PCA")
                import traceback
                traceback.print_exc()
                self.pca = PCA(n_components=self.n_components)
                self.projection_name = "PCA"
                return self.pca.fit_transform(featurized_data)

        if self.freeze_projector and self.pca is not None:
            try:
                return self.pca.transform(featurized_data)
            except Exception:
                return None

        # Instantiate DeepTICA if requested
        if str(self.projection_name).lower() == "deeptica" and self.pca is None:
             # We need to know lagtime etc. These should be passed or configured.
             # For now, we use defaults or try to read from config if possible.
             # Ideally PCAScorer should accept these in __init__.
             # We'll assume they are passed via kwargs or similar, but PCAScorer signature is fixed.
             # We will use hardcoded defaults or env vars for now as a quick fix, 
             # or better: initialize_projection_scorer should have set self.pca to DeepTICAProjector.
             pass

        if self.pca is None:
             self.pca = PCA(n_components=self.n_components)
             self.projection_name = "PCA"
        
        return self.pca.fit_transform(featurized_data)

    def _apply_msm_pcca(self, cv_pca: np.ndarray, trajectory_lengths: Optional[List[int]]) -> Optional[np.ndarray]:
        """Legacy MSM/PCCA helper (no-op).

        MSM/PCCA-based reweighting has been retired from the active
        training loop. This stub is kept only for backward compatibility
        with older code paths and always returns ``None``.
        """
        return None
        try:
            from deeptime.clustering import KMeans
            from deeptime.markov.msm import MaximumLikelihoodMSM
        except Exception as exc:
            print(f"[TICA-MSM] Failed to import MSM tooling ({exc}); using unweighted coverage.")
            return None

        weights = None
        try:
            n_clusters = int(getattr(self, 'msm_n_clusters', 200) or 200)
            lag = int(getattr(self, 'msm_lagtime', 50) or 50)
            kmeans = KMeans(n_clusters=n_clusters, max_iter=500, fixed_seed=True)
            kmeans_model = kmeans.fit_fetch(cv_pca)
            dtrajs_pooled = np.asarray(kmeans_model.transform(cv_pca), dtype=int)

            if trajectory_lengths is not None and len(trajectory_lengths) > 0:
                total_expected = sum(trajectory_lengths)
                if total_expected != len(dtrajs_pooled):
                    print(f"[TICA-MSM] Warning: trajectory_lengths sum ({total_expected}) != data length ({len(dtrajs_pooled)}); using single trajectory.")
                    dtrajs_list = [dtrajs_pooled]
                else:
                    dtrajs_list = []
                    start_idx = 0
                    for traj_len in trajectory_lengths:
                        if traj_len > 0:
                            dtrajs_list.append(dtrajs_pooled[start_idx:start_idx + traj_len])
                            start_idx += traj_len
                    print(f"[TICA-MSM] Split into {len(dtrajs_list)} trajectories for MSM fitting.")
            else:
                dtrajs_list = [dtrajs_pooled]
                print("[TICA-MSM] Warning: No trajectory_lengths provided; treating pooled data as single trajectory (suboptimal).")

            msm_est = MaximumLikelihoodMSM(lagtime=lag, reversible=True)
            msm = msm_est.fit_fetch(dtrajs_list)
            pi = np.asarray(msm.stationary_distribution, dtype=float)

            active_labels = getattr(msm, 'state_symbols', None)
            if active_labels is not None and callable(active_labels):
                try:
                    active_labels = active_labels()
                except Exception:
                    active_labels = None
            try:
                if active_labels is not None:
                    active_labels = np.asarray(active_labels, dtype=int).reshape(-1)
            except Exception:
                active_labels = None
            if active_labels is None:
                try:
                    active_labels = np.asarray(sorted(set(int(x) for x in dtrajs_pooled.tolist())), dtype=int)
                except Exception:
                    active_labels = np.unique(dtrajs_pooled).astype(int)

            centers = np.asarray(getattr(kmeans_model, "cluster_centers", None))
            n_centers = int(centers.shape[0]) if centers is not None else int(active_labels.max() + 1)
            label_to_active = -np.ones((n_centers,), dtype=int)
            for ai, lab in enumerate(active_labels.tolist()):
                if 0 <= lab < n_centers:
                    label_to_active[lab] = int(ai)

            pi_full = np.zeros((n_centers,), dtype=float)
            for ai, lab in enumerate(active_labels.tolist()):
                if 0 <= lab < n_centers:
                    pi_full[lab] = float(pi[ai])
            self.msm_pi_full = pi_full.astype(np.float32)

            mapped = label_to_active[np.clip(dtrajs_pooled, 0, n_centers - 1)]
            valid_mask = mapped >= 0
            weights = np.zeros(dtrajs_pooled.shape[0], dtype=float)
            if np.any(valid_mask):
                weights[valid_mask] = pi[mapped[valid_mask]]
            print(f"[TICA-MSM] Reweighting coverage histogram with MSM (clusters={n_clusters}, lag={lag}).")

            requested_sets = int(getattr(self, 'pcca_n_states', 10) or 10)
            n_sets = requested_sets
            pcca = None
            while n_sets >= 2 and pcca is None:
                try:
                    pcca = msm.pcca(n_sets)
                except Exception as exc_pcca:
                    print(f"[TICA-PCCA] Failed to compute PCCA metastable sets (n_sets={n_sets}, {exc_pcca}); halving and retrying.")
                    n_sets = n_sets // 2
                    if n_sets < 2:
                        break
            if pcca is None:
                print("[TICA-PCCA] Giving up on PCCA; state/committor rewards disabled for this update.")
                self._reset_pcca_fields()
                return weights

            if n_sets != requested_sets:
                print(f"[TICA-PCCA] Using reduced number of metastable sets (requested={requested_sets}, used={n_sets}).")
            memberships = np.asarray(pcca.memberships)
            mm_active = None
            micro_to_macro_full = -np.ones((n_centers,), dtype=np.int32)
            if memberships.ndim == 2 and memberships.shape[0] == active_labels.shape[0]:
                mm_active = memberships.argmax(axis=1).astype(np.int32)
                for ai, lab in enumerate(active_labels.tolist()):
                    if 0 <= lab < n_centers:
                        micro_to_macro_full[lab] = int(mm_active[ai])
            self.pcca_micro_to_macro = micro_to_macro_full
            self.pcca_active_macro = mm_active.astype(np.int32) if mm_active is not None else None
            self.msm_active_labels = active_labels.astype(np.int32)
            self.pcca_centers = np.asarray(centers, dtype=np.float32) if centers is not None else None
            
            # CRITICAL: Always reset visit counts when PCCA states are redefined.
            # The physical meaning of "macrostate i" has changed after refitting,
            # so carrying over old counts would penalize/reward the wrong states.
            self.pcca_visit_counts = np.zeros(n_sets, dtype=np.int64)
            print(f"[TICA-PCCA] Reset visit counts (macrostates redefined after MSM refit).")

            try:
                sets = [np.asarray(s, dtype=int) for s in pcca.sets]
                if len(sets) == n_sets:
                    self.pcca_sets = sets
                else:
                    self.pcca_sets = None
            except Exception:
                self.pcca_sets = None

            try:
                P = getattr(msm, "transition_matrix", None)
                if P is None:
                    P = getattr(msm, "transition_matrix_", None)
                P = np.asarray(P, dtype=float) if P is not None else None
                sets = self.pcca_sets
                if P is not None and sets is not None and len(sets) == n_sets:
                    T_macro = np.zeros((n_sets, n_sets), dtype=float)
                    for i_set, S_i in enumerate(sets):
                        S_i = np.asarray(S_i, dtype=int)
                        if S_i.size == 0:
                            continue
                        pi_i_total = float(pi[S_i].sum())
                        if pi_i_total <= 0.0:
                            continue
                        flow_i = (pi[S_i][:, None] * P[S_i, :]).sum(axis=0)
                        for j_set, S_j in enumerate(sets):
                            S_j = np.asarray(S_j, dtype=int)
                            if S_j.size == 0:
                                continue
                            T_macro[i_set, j_set] = float(flow_i[S_j].sum() / max(pi_i_total, 1e-12))
                    row_sums = T_macro.sum(axis=1, keepdims=True)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        T_macro = np.divide(
                            T_macro,
                            row_sums,
                            out=np.zeros_like(T_macro),
                            where=row_sums != 0,
                        )

                    scores = []
                    for idx in range(n_sets):
                        row = T_macro[idx]
                        stay_prob = row[idx] if np.isfinite(row[idx]) else 0.0
                        stay_prob = max(0.0, min(1.0, stay_prob))
                        off_diag_prob = max(0.0, 1.0 - stay_prob)
                        row_entropy = 0.0
                        for prob in row:
                            if prob > 0.0:
                                row_entropy -= float(prob * math.log(prob + 1e-12))
                        scores.append(off_diag_prob * row_entropy)
                    scores = np.asarray(scores, dtype=np.float32)
                    if scores.size > 0:
                        s_min = float(np.min(scores))
                        scores = scores - s_min
                        s_max = float(np.max(scores))
                        if s_max > 0.0:
                            scores = scores / s_max
                    self.pcca_state_scores = scores
                else:
                    self.pcca_state_scores = None
            except Exception as exc_meta:
                print(f"[TICA-PCCA] Failed to build metastable transition/committor data ({exc_meta}); disabling committor/transition rewards.")
                self.pcca_state_scores = None
            print(f"[TICA-PCCA] Initialized {n_sets} metastable sets.")
        except Exception as exc:
            print(f"[TICA-MSM] Failed to apply MSM reweighting ({exc}); using unweighted coverage.")
            weights = None
            self.msm_pi_full = None
        return weights

    def _reset_pcca_fields(self) -> None:
        """Safely clear PCCA-derived fields after a failure."""
        self.pcca_micro_to_macro = None
        self.pcca_centers = None
        self.pcca_visit_counts = None
        self.pcca_sets = None
        self.pcca_state_scores = None
        self.pcca_bridge_scores = None
        self.pcca_geom_bridge_scores = None
        self.pcca_components = None
        self.pcca_main_component = None
        self.pcca_island_components = None
        self.pcca_island_macrostates = None

    def _update_histogram(self, cv_pca: np.ndarray, weights: Optional[np.ndarray]) -> None:
        """Compute/refresh 2D histogram edges and counts in projection space."""
        # Use IQR-based outlier filtering for robust grid boundaries
        try:
            # Compute quartiles for both dimensions
            q1_x, q3_x = np.percentile(cv_pca[:, 0], [25, 75])
            q1_y, q3_y = np.percentile(cv_pca[:, 1], [25, 75])
            
            # Compute IQR
            iqr_x = q3_x - q1_x
            iqr_y = q3_y - q1_y
            
            # Define outlier fences (1.5 * IQR rule)
            x_lo = q1_x - 1.5 * iqr_x
            x_hi = q3_x + 1.5 * iqr_x
            y_lo = q1_y - 1.5 * iqr_y
            y_hi = q3_y + 1.5 * iqr_y
            
            # Sanity check: ensure valid range
            if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo >= x_hi:
                raise ValueError("Invalid X range")
            if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_lo >= y_hi:
                raise ValueError("Invalid Y range")
            
            x_edges = np.linspace(float(x_lo), float(x_hi), num=self.bins + 1)
            y_edges = np.linspace(float(y_lo), float(y_hi), num=self.bins + 1)
            self.counts, x_edges, y_edges = np.histogram2d(
                cv_pca[:, 0], cv_pca[:, 1], bins=[x_edges, y_edges], weights=weights
            )
            self.edges = [x_edges, y_edges]
        except Exception:
            # Fallback to automatic binning if IQR fails
            self.counts, x_edges, y_edges = np.histogram2d(
                cv_pca[:, 0], cv_pca[:, 1], bins=self.bins, weights=weights
            )
            self.edges = [x_edges, y_edges]

        # Smooth counts with a simple 3x3 mean kernel to reduce the impact
        # of isolated singletons dominating novelty.
        try:
            counts = np.asarray(self.counts, dtype=np.float64)
            if counts.ndim == 2 and counts.size > 0:
                padded = np.pad(counts, 1, mode="edge")
                smoothed = np.zeros_like(counts, dtype=np.float64)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        smoothed += padded[
                            1 + dx : 1 + dx + counts.shape[0],
                            1 + dy : 1 + dy + counts.shape[1],
                        ]
                smoothed /= 9.0
                self.smoothed_counts = smoothed.astype(np.float32)
            else:
                self.smoothed_counts = None
        except Exception:
            self.smoothed_counts = None

        self.is_ready = True

    def __call__(self, cv_point: Optional[np.ndarray], feature_point: Optional[np.ndarray] = None) -> float:
        if not self.is_ready or self.exploration_is_complete:
            return 0.0
        if feature_point is not None:
            featurized_point = np.asarray(feature_point, dtype=np.float32).reshape(1, -1)
        else:
            if cv_point is None:
                return 0.0
            featurized_point = featurize_for_pca(cv_point.reshape(1, -1))
        
        point_pca = self.pca.transform(featurized_point)
        # Ensure point_pca is 2D for consistent indexing
        if point_pca.ndim == 1:
            point_pca = point_pca.reshape(1, -1)
        
        x_idx = np.digitize(point_pca[0, 0], self.edges[0]) - 1
        y_idx = np.digitize(point_pca[0, 1], self.edges[1]) - 1

        # Out-of-bounds points get exploration bonus (they're in uncharted territory)
        if not (0 <= x_idx < self.bins and 0 <= y_idx < self.bins):
            return 5.0  # Exploration bonus for outliers

        # Use smoothed counts when available to avoid singletons dominating.
        if self.smoothed_counts is not None:
            c = float(self.smoothed_counts[x_idx, y_idx])
        else:
            c = float(self.counts[x_idx, y_idx])
        # Floor effective count at 3 to bound novelty from ultra-sparse bins.
        c_eff = max(c, 3.0)
        reward = -np.log(c_eff + 1e-9)
        return float(reward)

    def plain_coverage_score(
        self,
        cv_point: Optional[np.ndarray],
        feature_point: Optional[np.ndarray] = None,
    ) -> float:
        """Coverage-style score in the secondary plain-PCA space (if available).

        This mirrors __call__ but uses the plain PCA histogram. Returns 0.0
        if no secondary PCA has been fitted (e.g., projection=='pca').
        """
        if self.plain_pca is None or self.plain_counts is None or self.plain_edges is None:
            return 0.0
        if feature_point is not None:
            featurized_point = np.asarray(feature_point, dtype=np.float32).reshape(1, -1)
        else:
            if cv_point is None:
                return 0.0
            featurized_point = featurize_for_pca(cv_point.reshape(1, -1))

        point_pca = self.plain_pca.transform(featurized_point)
        if point_pca.ndim == 1:
            point_pca = point_pca.reshape(1, -1)

        x_idx = np.digitize(point_pca[0, 0], self.plain_edges[0]) - 1
        y_idx = np.digitize(point_pca[0, 1], self.plain_edges[1]) - 1

        if not (0 <= x_idx < self.bins and 0 <= y_idx < self.bins):
            return 5.0

        reward = -np.log(self.plain_counts[x_idx, y_idx] + 1e-9)
        return float(reward)

    def state_reward_from_pca_point(self, point_pca: np.ndarray) -> float:
        """Legacy PCCA-based coverage reward (no-op).

        PCCA/MSM-based state rewards are no longer used in the
        active training code and this method always returns 0.0.
        """
        return 0.0
    
    def metastable_transition_score_from_pca_point(self, point_pca: np.ndarray) -> float:
        """Legacy PCCA-based transition score (no-op).

        Metastable-set transition rewards have been removed from the
        active training loop. This method is kept for backward
        compatibility and always returns 0.0.
        """
        return 0.0

    def sample_unexplored_points(self, n_samples: int) -> np.ndarray:
        if not self.is_ready:
            return (np.random.rand(n_samples, self.n_cvs) * 2 * np.pi) - np.pi

        unexplored_indices = np.argwhere(self.counts == 0)
        if len(unexplored_indices) == 0:
            return (np.random.rand(n_samples, self.n_cvs) * 2 * np.pi) - np.pi

        chosen_bin_indices = unexplored_indices[np.random.choice(len(unexplored_indices), size=n_samples, replace=True)]
        x_centers = (self.edges[0][chosen_bin_indices[:, 0]] + self.edges[0][chosen_bin_indices[:, 0] + 1]) / 2.0
        y_centers = (self.edges[1][chosen_bin_indices[:, 1]] + self.edges[1][chosen_bin_indices[:, 1] + 1]) / 2.0
        
        x_width = self.edges[0][1] - self.edges[0][0]
        y_width = self.edges[1][1] - self.edges[1][0]
        sampled_pca_points = np.vstack([x_centers, y_centers]).T
        sampled_pca_points += (np.random.rand(n_samples, 2) - 0.5) * np.array([x_width, y_width])
        
        # If inverse_transform is unavailable (e.g., TICA projector), fallback to random torsions
        if not hasattr(self.pca, 'inverse_transform'):
            return (np.random.rand(n_samples, self.n_cvs) * 2 * np.pi) - np.pi
        sampled_featurized_points = self.pca.inverse_transform(sampled_pca_points)
        sines, cosines = sampled_featurized_points[:, :self.n_cvs], sampled_featurized_points[:, self.n_cvs:]
        return np.arctan2(sines, cosines)

    def get_state(self) -> dict:
        # Avoid serializing non-picklable deep projectors (e.g., VAMPNet).
        # For those, we only persist coverage metadata; the projector itself
        # will be reconstructed on resume by initialize_projection_scorer.
        pca_obj = self.pca
        try:
            if isinstance(pca_obj, VAMPNetProjector):
                pca_obj = None
        except NameError:
            # VAMPNetProjector not defined yet; fall back to raw object.
            pass
        return {
            'n_cvs': self.n_cvs,
            'n_components': self.n_components,
            'bins': self.bins,
            'completion_threshold': self.completion_threshold,
            'pca': pca_obj,
            'projection_name': self.projection_name,
            'freeze_projector': self.freeze_projector,
            'plain_pca': self.plain_pca,
            'plain_edges': self.plain_edges,
            'plain_counts': self.plain_counts,
            'edges': self.edges,
            'counts': self.counts,
            'is_ready': self.is_ready,
            'exploration_is_complete': self.exploration_is_complete,
            'last_update_ep': self.last_update_ep,
        }

    def load_state(self, state: dict):
        if not state:
            return
        self.n_cvs = state.get('n_cvs', self.n_cvs)
        self.n_components = state.get('n_components', self.n_components)
        self.bins = state.get('bins', self.bins)
        self.completion_threshold = state.get('completion_threshold', self.completion_threshold)
        # Only overwrite self.pca if a serialized projector is present.
        # For VAMPNet, we intentionally do not serialize the deep model, so
        # self.pca (a freshly constructed VAMPNetProjector) should be kept.
        if 'pca' in state and state['pca'] is not None:
            self.pca = state['pca']
        self.projection_name = state.get('projection_name', self.projection_name if hasattr(self, 'projection_name') else 'PCA')
        self.freeze_projector = bool(state.get('freeze_projector', getattr(self, 'freeze_projector', False)))
        self.plain_pca = state.get('plain_pca', None)
        self.plain_edges = state.get('plain_edges', None)
        self.plain_counts = state.get('plain_counts', None)
        self.edges = state.get('edges', None)
        self.counts = state.get('counts', None)
        self.is_ready = state.get('is_ready', False)
        self.exploration_is_complete = state.get('exploration_is_complete', False)
        self.last_update_ep = state.get('last_update_ep', 0)

    def set_state(self, state: dict):
        """Alias for load_state to match older checkpoint interfaces."""
        self.load_state(state)

class VAMPNetProjector:
    """
    Wrapper around deeptime's VAMPNet deep estimator exposing a simple
    scikit-learn style interface:

    - ``fit(trajectories)`` where ``trajectories`` is a list of arrays
      with shape [T_i, n_features]. Trajectory boundaries are respected.
    - ``transform(X)`` which returns 2D coordinates for each frame.

    Internally, the lobe outputs fuzzy state probabilities which are
    then reduced to 2D via an internal PCA on the probability vectors.
    """

    def __init__(
        self,
        lagtime: int = 1,
        n_states: int = 2,
        hidden_dim: int = 20,
        n_layers: int = 4,
        learning_rate: float = 5e-3,
        n_epochs: int = 30,
        batch_size: int = 10000,
        device: str = "cpu",
    ):
        self.lagtime = int(max(1, lagtime))
        self.n_states = int(max(2, n_states))
        self.hidden_dim = int(max(1, hidden_dim))
        self.n_layers = int(max(1, n_layers))
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(max(1, n_epochs))
        self.batch_size = int(max(1, batch_size))
        self.device = device

        self._feature_dim: Optional[int] = None
        self._initialized: bool = False
        self._vampnet = None
        self._model = None
        self._pca_reducer = None
        self._is_trained: bool = False

    def _init_model(self, feature_dim: int):
        """Lazy initialization of the VAMPNet estimator and lobe."""
        self._feature_dim = int(feature_dim)
        try:
            from deeptime.decomposition.deep import VAMPNet as _VAMPNet
            from deeptime.util.data import TrajectoriesDataset as _TrajectoriesDataset
        except Exception as exc:
            raise RuntimeError(
                "deeptime with deep-learning extras is required for 'vampnet' "
                "projection but could not be imported. Install `deeptime[deep]` "
                "or set projection to 'pca' or 'tica'."
            ) from exc

        try:
            import torch as _torch
            import torch.nn as _nn
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required for 'vampnet' projection but could not be imported."
            ) from exc

        dev = _torch.device(self.device) if isinstance(self.device, str) else self.device

        layers = []
        layers.append(_nn.BatchNorm1d(self._feature_dim))
        in_dim = self._feature_dim
        for _ in range(self.n_layers):
            layers.append(_nn.Linear(in_dim, self.hidden_dim))
            layers.append(_nn.ELU())
            in_dim = self.hidden_dim
        # Final linear layer outputs a 2D embedding (n_states)
        # without Softmax so that the projector behaves like
        # a 2D PCA/TICA map.
        layers.append(_nn.Linear(in_dim, self.n_states))
        lobe = _nn.Sequential(*layers).to(device=dev)

        vampnet = _VAMPNet(lobe=lobe, learning_rate=self.learning_rate, device=dev)

        self._torch = _torch
        self._TrajectoriesDataset = _TrajectoriesDataset
        self._vampnet = vampnet
        self._device = dev
        self._initialized = True

    def fit(self, trajectories: List[np.ndarray]):
        """
        Fit the VAMPNet model on a list of trajectories.

        Each element of ``trajectories`` must be an array of shape
        [T_i, n_features]. Trajectories shorter than lagtime+1 are ignored.
        """
        if trajectories is None:
            return self

        traj_clean = []
        feature_dim = None
        for arr in trajectories:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] <= self.lagtime:
                continue
            if feature_dim is None:
                feature_dim = arr.shape[1]
            if arr.shape[1] != feature_dim:
                continue
            traj_clean.append(arr)

        if not traj_clean or feature_dim is None:
            print("[VAMPNet] No valid trajectories for training; skipping update.")
            return self

        if not self._initialized or self._feature_dim != feature_dim:
            self._init_model(feature_dim)

        from torch.utils.data import DataLoader

        DatasetCls = self._TrajectoriesDataset

        # --- Trajectory-wise train/validation split ------------------------
        # We split on the trajectory index (not individual windows) to avoid
        # leakage of temporally correlated frames between train and validation.
        n_traj = len(traj_clean)
        if n_traj == 1:
            # With a single trajectory, fall back to using all data for both
            # train and validation to keep behaviour simple and robust.
            train_trajs = traj_clean
            val_trajs = traj_clean
        else:
            # Default: use 10% of trajectories for validation, but ensure at
            # least one trajectory in each split when possible.
            import numpy as _np

            indices = _np.random.permutation(n_traj)
            n_val_traj = max(1, int(0.1 * n_traj))
            if n_val_traj >= n_traj:
                n_val_traj = max(1, n_traj // 5)
            val_idx = indices[:n_val_traj]
            train_idx = indices[n_val_traj:]
            if train_idx.size == 0:
                # Degenerate case: fall back to using all trajectories for both
                train_trajs = traj_clean
                val_trajs = traj_clean
            else:
                train_trajs = [traj_clean[i] for i in train_idx]
                val_trajs = [traj_clean[i] for i in val_idx]

        train_data = DatasetCls.from_numpy(int(self.lagtime), train_trajs)
        val_data = DatasetCls.from_numpy(int(self.lagtime), val_trajs)

        if len(train_data) == 0 or len(val_data) == 0:
            # If windowing removed everything from one split, use all data
            # for both train and validation as a safe fallback.
            dataset_all = DatasetCls.from_numpy(int(self.lagtime), traj_clean)
            if len(dataset_all) == 0:
                print("[VAMPNet] Empty dataset after windowing; skipping update.")
                return self
            train_data = dataset_all
            val_data = dataset_all

        train_bs = min(self.batch_size, len(train_data))
        loader_train = DataLoader(train_data, batch_size=train_bs, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

        self._vampnet.fit(
            loader_train,
            n_epochs=self.n_epochs,
            validation_loader=loader_val,
            progress=None,
        )
        self._model = self._vampnet.fetch_model()

        self._is_trained = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features into 2D VAMPNet coordinates."""
        if self._model is None or not self._is_trained:
            raise RuntimeError("VAMPNetProjector.transform() called before fit().")

        X = np.asarray(X, dtype=np.float32)
        try:
            emb = self._model.transform(X)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim != 2:
                emb = emb.reshape(emb.shape[0], -1)
        except Exception as exc:
            raise RuntimeError(f"[VAMPNet] transform failed: {exc}") from exc

        # Ensure a 2D output, matching PCA/TICA usage.
        if emb.shape[1] >= 2:
            return emb[:, :2]

        out = np.zeros((emb.shape[0], 2), dtype=np.float32)
        out[:, :emb.shape[1]] = emb
        return out

class VendiNovelty:
    def __init__(self, cv_dim: int, kernel_sigma=0.4, window=500, device="cpu"):
        self.s2 = kernel_sigma**2
        self.buf = deque(maxlen=window)
        self.device = device
        self.cv_dim = cv_dim

    def _d2(self, A, B):
        delta = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))
        delta = torch.minimum(delta, 2*math.pi - delta)
        return (delta**2).sum(-1)

    def _kernel_matrix(self, X):
        d2 = self._d2(X, X)
        return torch.exp(-0.5 * d2 / self.s2)

    def __call__(self, cv_np):
        x = torch.as_tensor(cv_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not self.buf:
            self.buf.append(x);
            return 5.0

        S = torch.cat(list(self.buf), dim=0)
        K_S = self._kernel_matrix(S)
        vs_old = torch.log(vendi_score(K_S))

        Sx = torch.cat([S, x], dim=0)
        K_Sx = self._kernel_matrix(Sx)
        vs_new = torch.log(vendi_score(K_Sx))

        self.buf.append(x)
        return (vs_new - vs_old).item()

    def get_state(self) -> dict:
        return {
            's2': self.s2,
            'window': self.buf.maxlen,
            'cv_dim': self.cv_dim,
            'buf': [t.detach().cpu().numpy() for t in self.buf],
        }

    def load_state(self, state: dict):
        if not state:
            return
        window = state.get('window', self.buf.maxlen)
        self.buf = deque(maxlen=window)
        self.cv_dim = state.get('cv_dim', self.cv_dim)
        self.s2 = state.get('s2', self.s2)
        for arr in state.get('buf', []):
            t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
            self.buf.append(t)

class VendiNoveltyEuclid:
    """Vendi-style novelty on Euclidean space (e.g., 2D PCA coordinates).

    Maintains a sliding window buffer of recent points x in R^d and
    returns the change in Vendi score when a new point is added.
    """
    def __init__(self, dim: int, kernel_sigma=1.0, window=500, device="cpu"):
        self.s2 = float(kernel_sigma) ** 2
        self.buf = deque(maxlen=int(window))
        self.device = device
        self.dim = int(dim)

    def _d2(self, A, B):
        delta = A.unsqueeze(1) - B.unsqueeze(0)
        return (delta**2).sum(-1)

    def _kernel_matrix(self, X):
        d2 = self._d2(X, X)
        return torch.exp(-0.5 * d2 / self.s2)

    def __call__(self, x_np: Optional[np.ndarray]):
        if x_np is None:
            return 0.0
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device).reshape(1, -1)
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected point dim {self.dim}, got {x.shape[-1]}")
        if not self.buf:
            self.buf.append(x)
            return 5.0
        S = torch.cat(list(self.buf), dim=0)
        K_S = self._kernel_matrix(S)
        vs_old = torch.log(vendi_score(K_S))
        Sx = torch.cat([S, x], dim=0)
        K_Sx = self._kernel_matrix(Sx)
        vs_new = torch.log(vendi_score(K_Sx))
        self.buf.append(x)
        return (vs_new - vs_old).item()

    def get_state(self):
        """Return state dict for checkpointing."""
        return {
            's2': self.s2,
            'dim': self.dim,
            'device': self.device,
            'window': self.buf.maxlen,
            'buffer': [x.cpu().numpy() for x in self.buf]
        }

    def load_state(self, state):
        """Restore from checkpoint."""
        self.s2 = state['s2']
        self.dim = state['dim']
        self.device = state.get('device', 'cpu')
        self.buf = deque(maxlen=state['window'])
        for x_np in state.get('buffer', []):
            x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
            self.buf.append(x)
