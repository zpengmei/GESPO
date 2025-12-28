import numpy as np
from typing import Optional
from ...config import TrainingConfig

class FeatureBuilder:
    """
    Builds feature vectors for:
    1. Policy/Generator inputs (based on policy_conditioning_mode).
    2. PCA projection features (for coverage/diagnostics).
    """
    def __init__(self, config: TrainingConfig, generator=None, pca_scorer=None, env=None):
        self.config = config
        self.env = env
        
        # Metadata from Generator (for Policy/PCA matching)
        self.conditioning_mode = "torsion"
        self.partner_residues = None
        self.pair_mask = None
        self.hybrid_pair_dim = 0
        self.pair_feature_dim = 0
        self.cond_mean = None
        self.cond_std = None
        
        if generator:
            # Use config mode if specified, otherwise fallback to generator's mode or "hybrid"
            # Actually, config should be the source of truth now.
            self.conditioning_mode = config.policy_conditioning_mode.lower()
            self.partner_residues = (
                np.asarray(getattr(generator, "partner_residues", None))
                if getattr(generator, "partner_residues", None) is not None
                else None
            )
            self.pair_mask = (
                np.asarray(getattr(generator, "pair_mask", None))
                if getattr(generator, "pair_mask", None) is not None
                else None
            )
            # Some checkpoints store hybrid_pair_dim / pair_feature_dim as None;
            # treat those as 0 to avoid int(None) crashes when pairwise
            # conditioning is not actually used.
            if hasattr(generator, "hybrid_pair_dim"):
                _h = getattr(generator, "hybrid_pair_dim", 0)
                self.hybrid_pair_dim = int(_h) if isinstance(_h, (int, float, np.integer, np.floating)) else 0
            else:
                self.hybrid_pair_dim = 0
            if hasattr(generator, "pair_feature_dim"):
                _p = getattr(generator, "pair_feature_dim", 0)
                self.pair_feature_dim = int(_p) if isinstance(_p, (int, float, np.integer, np.floating)) else 0
            else:
                self.pair_feature_dim = 0
            
            # Pre-fetch mean/std as numpy
            cm = getattr(generator, "cond_mean", None)
            cs = getattr(generator, "cond_std", None)
            self.cond_mean = cm.detach().cpu().numpy() if cm is not None else None
            self.cond_std = cs.detach().cpu().numpy() if cs is not None else None

        # Metadata for projection (CA indices)
        self.ca_indices = None
        if pca_scorer and hasattr(pca_scorer, "ca_indices") and pca_scorer.ca_indices is not None:
            self.ca_indices = pca_scorer.ca_indices
        elif env and hasattr(env, "ca_atom_indices"):
            self.ca_indices = np.asarray(env.ca_atom_indices, dtype=np.int64)

    def _pairwise_from_coords(self, coords_nm: np.ndarray) -> Optional[np.ndarray]:
        """Compute pairwise features matching the policy's conditioning."""
        if self.partner_residues is None or self.pair_mask is None or self.ca_indices is None:
            return None
        
        try:
            pos_angstrom = coords_nm * 10.0
            n_res, max_partners = self.partner_residues.shape
            feats = np.zeros((n_res, max_partners), dtype=np.float32)
            
            for i in range(n_res):
                i_ca = self.ca_indices[i]
                if i_ca < 0: continue
                for j in range(max_partners):
                    if not self.pair_mask[i, j]: continue
                    k = int(self.partner_residues[i, j])
                    if k <= i or k >= n_res: continue
                    k_ca = self.ca_indices[k]
                    if k_ca < 0 or k_ca >= pos_angstrom.shape[0]: continue
                    
                    d = float(np.linalg.norm(pos_angstrom[i_ca] - pos_angstrom[k_ca]))
                    if not np.isfinite(d) or d > 100.0: continue
                    feats[i, j] = np.float32(d)
            
            flat = feats.reshape(-1)
            
            # Normalize
            if self.cond_mean is not None and self.cond_std is not None:
                mean = self.cond_mean.astype(np.float32)
                std = self.cond_std.astype(np.float32)
                
                if self.conditioning_mode == "hybrid" and self.hybrid_pair_dim > 0:
                    m = mean[-self.hybrid_pair_dim:]
                    s = std[-self.hybrid_pair_dim:]
                else:
                    m = mean[:flat.size] if mean.size >= flat.size else None
                    s = std[:flat.size] if std.size >= flat.size else None
                
                if m is not None and s is not None and m.size == flat.size and s.size == flat.size:
                    s = np.where(s < 1e-6, 1.0, s)
                    flat = ((flat - m) / s).astype(np.float32)
            
            return flat
        except Exception:
            return None

    def build_policy_features(self, cv_vec: np.ndarray, coords_nm: Optional[np.ndarray]) -> np.ndarray:
        """Build features for the Policy Network."""
        # Always compute torsions
        sines = np.sin(cv_vec).astype(np.float32)
        cosines = np.cos(cv_vec).astype(np.float32)
        torsion_feats = np.concatenate([sines, cosines], axis=0).astype(np.float32)
        
        mode = self.conditioning_mode
        
        if mode == "torsion":
            return torsion_feats
            
        pw = self._pairwise_from_coords(coords_nm) if coords_nm is not None else None
        
        if mode == "pairwise":
            return (pw if pw is not None else np.zeros((self.pair_feature_dim,), dtype=np.float32))
            
        if mode == "hybrid":
            if pw is None:
                return torsion_feats # Fallback? Or zero pad?
            return np.concatenate([torsion_feats, pw], axis=0).astype(np.float32)
            
        # CA modes (not fully implemented in _pairwise_from_coords but placeholder)
        if mode == "ca":
             # Flattened CA coords
             if coords_nm is not None and self.ca_indices is not None:
                 return coords_nm[self.ca_indices].flatten().astype(np.float32)
             return np.zeros((1,), dtype=np.float32)

        return torsion_feats

    def build_projection_features(self, cv_vec: np.ndarray, coords_nm: Optional[np.ndarray]) -> np.ndarray:
        """Build features for PCA/projection scorer.
        
        Returns:
            Features as 2D array [n_samples, n_features] for sklearn compatibility.
            If input is 1D (single sample), output is [1, n_features].
        """
        cv_batch = np.asarray(cv_vec, dtype=np.float32)
        coords_batch = None if coords_nm is None else np.asarray(coords_nm, dtype=np.float32)

        if cv_batch.ndim == 1:
            cv_batch = cv_batch.reshape(1, -1)
            if coords_batch is not None and coords_batch.ndim == 2:
                coords_batch = coords_batch.reshape(1, -1, 3)

        # Use policy features for projection/PCA so diagnostics are aligned with
        # the conditioning used by the generator/policy.
        policy_feats = [
            self.build_policy_features(cv_batch[i], coords_batch[i] if coords_batch is not None else None)
            for i in range(cv_batch.shape[0])
        ]
        return np.asarray(policy_feats, dtype=np.float32).reshape(cv_batch.shape[0], -1)
    
    def build(self, cv_vec: np.ndarray, coords_nm: Optional[np.ndarray]) -> np.ndarray:
        """Deprecated alias for build_projection_features (for backward compat if needed)."""
        return self.build_projection_features(cv_vec, coords_nm)
