"""Novelty scoring (Vendi) and PCA-based exploration utilities."""
from typing import Tuple
from .._imports import *
from gespo.utils.logging import log_print

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
    - If a fixed projector is provided, uses its .transform() without refitting.
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
        # Projector can be a fitted PCA-like object exposing .transform(...)
        self.pca = projector
        self.projection_name = projection_name or ("PCA" if projector is None else "Projection")
        self.freeze_projector = bool(freeze_projector and projector is not None)
        self.edges = None
        self.counts = None
        # Smoothed density for coverage rewards (post-convolution)
        self.smoothed_counts = None
        # Optional secondary PCA (plain PCA on features) so we can
        # simultaneously score coverage in the primary projection
        # and in a standard PCA space.
        self.plain_pca = None
        self.plain_edges = None
        self.plain_counts = None
        self.is_ready = False
        self.exploration_is_complete = False
        self.last_update_ep = 0
    def update_model(self, cv_data: np.ndarray, current_ep: int, features: Optional[np.ndarray] = None, trajectory_lengths: Optional[List[int]] = None, times: Optional[np.ndarray] = None, is_anchoring: bool = False):
        """Update PCA projection model (now split into smaller helpers)."""
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
            log_print(f"--- PCA Exploration Complete ({fraction_visited:.1%}). Deactivating PCAScorer. ---")
        else:
            log_print(f"PCA Scorer updated ({fraction_visited:.1%} bins visited). Last updated Ep. {current_ep}.")

        # Also fit a plain PCA on the same features for secondary
        # coverage scoring when the primary projector is not already
        # PCA.
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
            log_print(f"[PCA-secondary] Failed to fit plain PCA for secondary coverage ({exc_plain}); disabling secondary PCA rewards.")
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
                    log_print(f"[Projection Subsample] Kept {n_keep}/{n_traj} trajectories ({kept_frames} frames) to respect cap {max_train}.")
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
                            log_print(f"[Projection Subsample] Inferred traj_len={traj_len}, kept {n_keep}/{n_traj} trajectories to respect cap {max_train}.")
                        else:
                            log_print(f"[Projection Subsample] Warning: Could not infer valid trajectories; keeping all {featurized_data.shape[0]} frames.")
                    else:
                        log_print(f"[Projection Subsample] Warning: Could not infer trajectory structure; keeping all {featurized_data.shape[0]} frames.")
                except Exception as exc:
                    log_print(f"[Projection Subsample] Warning: Failed to infer trajectory structure ({exc}); keeping all frames.")

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
        """Fit (or refit) the PCA projector and return projected data."""
        log_print(f"[Debug Scorer] _fit_projector_and_transform called. pca={type(self.pca)}, projection_name={self.projection_name}, freeze={self.freeze_projector}")
        if self.pca is not None and not self.freeze_projector:
            try:
                if hasattr(self.pca, "fit"):
                    log_print(f"[Debug Scorer] Attempting to fit projector. Has {len(trajectory_lengths) if trajectory_lengths else 0} trajectories.")
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
                        
                        log_print(f"[Debug Scorer] Built {len(traj_list)} trajectory lists. Data shape: {featurized_data.shape}")
                        if times is not None and len(time_list) == len(traj_list):
                             log_print(f"[Debug Scorer] Calling fit with time indices.")
                             self.pca.fit(traj_list)
                        else:
                             log_print(f"[Debug Scorer] Calling fit without time indices.")
                             self.pca.fit(traj_list)
                        log_print(f"[Projector] Fitted with {len(traj_list)} trajectories (respecting boundaries).")
                    else:
                        self.pca.fit(featurized_data)
                        log_print("[Projector] Warning: No trajectory_lengths; fitted as single trajectory (suboptimal for projection).")
                log_print(f"[Debug Scorer] Fit successful, calling transform")
                return self.pca.transform(featurized_data)
            except Exception as exc:
                log_print(f"[Projector] fit failed ({exc}), falling back to PCA")
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

        if self.pca is None:
             self.pca = PCA(n_components=self.n_components)
             self.projection_name = "PCA"
        
        return self.pca.fit_transform(featurized_data)
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
        
        # If inverse_transform is unavailable, fallback to random torsions.
        if not hasattr(self.pca, 'inverse_transform'):
            return (np.random.rand(n_samples, self.n_cvs) * 2 * np.pi) - np.pi
        sampled_featurized_points = self.pca.inverse_transform(sampled_pca_points)
        sines, cosines = sampled_featurized_points[:, :self.n_cvs], sampled_featurized_points[:, self.n_cvs:]
        return np.arctan2(sines, cosines)

    def get_state(self) -> dict:
        return {
            'n_cvs': self.n_cvs,
            'n_components': self.n_components,
            'bins': self.bins,
            'completion_threshold': self.completion_threshold,
            'pca': self.pca,
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
