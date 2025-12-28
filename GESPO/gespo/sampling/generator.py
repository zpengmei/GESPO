"""Wrapper around the trained diffusion model to generate heavy-atom coordinates given conditioning features."""
from .._imports import *
from typing import Dict, Tuple, Optional
from ..utils.geometry import build_residue_maps, kabsch_align
from ..models.diffusion import ConditionalHierarchicalCartesianDiffusion
from ..diffusion import create_diffusion
from gespo.utils.logging import log_print

class ConditionalGenerator:
    def __init__(self, checkpoint_path: str, data_path: str, pdb_path: str, device: Union[torch.device, str] = 'cpu'):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        data = np.load(data_path)
        self.conditioning_mode = str(config.get('conditioning_mode', data.get('conditioning_mode', 'torsion')))
        dataset_mode = str(data.get('conditioning_mode', 'torsion'))
        if self.conditioning_mode != dataset_mode:
            raise ValueError(f"Checkpoint expects conditioning mode '{self.conditioning_mode}' but dataset provides '{dataset_mode}'")
        self.cond_injection_mode = str(config.get('cond_injection_mode', data.get('cond_injection_mode', 'global')))
        self.per_res_torsion = bool(config.get('per_res_torsion', data.get('per_res_torsion', False)))
        atomic_numbers = data['atomic_numbers']
        mask = atomic_numbers != 1
        self.atomic_numbers = torch.from_numpy(atomic_numbers[mask]).long()

        traj = md.load(pdb_path)
        self.res_idx_pad, self.res_mask, self.heavy_atom_indices = build_residue_maps(traj)
        self.reference_coords_angstrom = traj.atom_slice(self.heavy_atom_indices).xyz[0] * 10.0
        if self.reference_coords_angstrom.shape[0] != len(self.atomic_numbers):
            raise ValueError('Mismatch between NPZ heavy atoms and PDB heavy atoms.')

        conditioning = data['conditioning'] if 'conditioning' in data.files else None
        cond_dim = conditioning.shape[1] if conditioning is not None else int(config.get('cond_dim', 0))
        self.pair_feature_dim = None
        # Preserve dataset pair selection metadata (for consistent online features)
        self.pair_mask = data.get('pair_mask') if 'pair_mask' in data.files else None
        self.partner_residues = data.get('partner_residues') if 'partner_residues' in data.files else None
        self.pair_step = int(data.get('pair_step', np.array(0))) if 'pair_step' in data.files else None
        self.pair_min_separation = int(data.get('pair_min_separation', np.array(0))) if 'pair_min_separation' in data.files else None

        self.n_residues = int(self.res_idx_pad.shape[0])
        if self.conditioning_mode in ('torsion', 'hybrid'):
            subset = traj.atom_slice(self.heavy_atom_indices)
            self.phi_idx, _ = md.compute_phi(subset)
            self.psi_idx, _ = md.compute_psi(subset)
            self.n_phi = self.phi_idx.shape[0]
            self.n_psi = self.psi_idx.shape[0]
            self.phi_res_idx = self._torsion_residue_indices(subset, self.phi_idx)
            self.psi_res_idx = self._torsion_residue_indices(subset, self.psi_idx)

        if self.conditioning_mode == 'torsion':
            if self.per_res_torsion:
                expected_dim = 4 * self.n_residues
            else:
                expected_dim = 2 * (self.n_phi + self.n_psi)
            cond_dim = cond_dim or expected_dim
            if cond_dim != expected_dim:
                log_print(f"[generator] Warning: conditioning dim {cond_dim} != expected torsion dim {expected_dim}")
            self.conditioning_dim = cond_dim
        elif self.conditioning_mode == 'pairwise':
            if conditioning is None:
                raise ValueError("Dataset is missing 'conditioning' required for pairwise mode.")
            self.pair_feature_dim = cond_dim
            self.conditioning_dim = cond_dim
        elif self.conditioning_mode == 'hybrid':
            if conditioning is None:
                raise ValueError("Dataset is missing 'conditioning' required for hybrid mode.")
            self.hybrid_torsion_dim = int(config.get('hybrid_torsion_dim', data.get('hybrid_torsion_dim', 0)))
            if self.hybrid_torsion_dim <= 0:
                self.hybrid_torsion_dim = 2 * (self.n_phi + self.n_psi)
            cond_dim = cond_dim or conditioning.shape[1]
            self.hybrid_pair_dim = cond_dim - self.hybrid_torsion_dim
            if self.hybrid_pair_dim <= 0:
                raise ValueError("Hybrid conditioning requires positive pairwise feature dimension.")
            self.pair_feature_dim = self.hybrid_pair_dim
            self.conditioning_dim = cond_dim
        else:
            raise NotImplementedError(f"Unsupported conditioning mode '{self.conditioning_mode}'.")
        atom_mixer = config.get('atom_mixer', 'transformer')
        num_residue_sinks = int(config.get('num_residue_sinks', 0))  # backward-compatible default
        self.model = ConditionalHierarchicalCartesianDiffusion(
            n_atoms=len(self.atomic_numbers),
            n_atom_types=int(self.atomic_numbers.max().item()),
            R=self.res_idx_pad.shape[0],
            max_atoms_res=int(self.res_idx_pad.shape[1]),
            res_idx_pad=self.res_idx_pad,
            res_mask=self.res_mask,
            atom_types=self.atomic_numbers,
            d_model=config.get('d_model', 128),
            latent_dim=config.get('latent_dim', 16),
            nhead=config.get('nhead', 8),
            num_layers_intra=config.get('num_layers_intra', 4),
            num_layers_inter=config.get('num_layers_inter', 4),
            dropout=config.get('dropout', 0.0),
            latent_level=config.get('latent_level', 'atom'),
            use_atom_id=True,
            use_intra_pos_enc=True,
            use_dec_intra_pos_enc=True,
            ungroup_mode=config.get('ungroup_mode', 'split'),
            learn_sigma=True,
            cond_dim=cond_dim,
            cond_injection_mode=self.cond_injection_mode,
            atom_mixer=atom_mixer,
            num_residue_sinks=num_residue_sinks,
        ).to(self.device)
        # Allow backward compatibility (e.g., sinks added after checkpoint)
        load_res = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        missing, unexpected = load_res.missing_keys, load_res.unexpected_keys
        if missing or unexpected:
            log_print(f"[generator] Loaded diffusion weights with non-strict match. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        self.model.eval()

        self.cond_mean = checkpoint['cond_mean'].to(self.device)
        self.cond_std = checkpoint['cond_std'].to(self.device)
        self.cond_std = torch.where(self.cond_std < 1e-6, torch.ones_like(self.cond_std), self.cond_std)

        diffusion_steps = config.get('diffusion_steps', 1000)
        self.diffusion = create_diffusion(timestep_respacing='', learn_sigma=self.model.learn_sigma, diffusion_steps=diffusion_steps)

    def _torsion_residue_indices(self, traj: md.Trajectory, torsion_idx: np.ndarray) -> np.ndarray:
        if torsion_idx.size == 0:
            return np.zeros((0,), dtype=np.int64)
        res_idx = []
        for torsion in torsion_idx:
            atom = traj.top.atom(int(torsion[1]))
            res_idx.append(atom.residue.index)
        return np.asarray(res_idx, dtype=np.int64)

    def build_condition_torsion(self, phi_angles: np.ndarray, psi_angles: np.ndarray) -> np.ndarray:
        phi = np.atleast_2d(phi_angles)
        psi = np.atleast_2d(psi_angles)
        if phi.shape[1] != self.n_phi or psi.shape[1] != self.n_psi:
            raise ValueError(f'Expected phi/psi lengths {(self.n_phi, self.n_psi)}, got {(phi.shape[1], psi.shape[1])}')
        phi_sin = np.sin(phi)
        phi_cos = np.cos(phi)
        psi_sin = np.sin(psi)
        psi_cos = np.cos(psi)
        if self.per_res_torsion:
            B = phi.shape[0]
            feats = np.zeros((B, self.n_residues, 4), dtype=np.float32)
            feats[:, :, 1] = 1.0
            feats[:, :, 3] = 1.0
            for col, res_idx in enumerate(self.phi_res_idx):
                feats[:, res_idx, 0] = phi_sin[:, col]
                feats[:, res_idx, 1] = phi_cos[:, col]
            for col, res_idx in enumerate(self.psi_res_idx):
                feats[:, res_idx, 2] = psi_sin[:, col]
                feats[:, res_idx, 3] = psi_cos[:, col]
            feats = feats.reshape(B, -1)
        else:
            feats = np.concatenate([phi_sin, phi_cos, psi_sin, psi_cos], axis=1)
        feats = feats.astype(np.float32)
        if phi_angles.ndim == 1:
            return feats[0]
        return feats

    def build_condition_pairwise(self, pair_features: np.ndarray) -> np.ndarray:
        feats = np.atleast_2d(pair_features)
        if feats.shape[1] != self.pair_feature_dim:
            raise ValueError(f"Expected pairwise feature dimension {self.pair_feature_dim}, got {feats.shape[1]}")
        feats = feats.astype(np.float32)
        if pair_features.ndim == 1:
            return feats[0]
        return feats

    def _prepare_condition_batch(
        self,
        phi_angles_batch: Optional[np.ndarray],
        psi_angles_batch: Optional[np.ndarray],
        condition_batch: Optional[np.ndarray],
    ) -> torch.Tensor:
        if self.conditioning_mode == 'torsion':
            if phi_angles_batch is None or psi_angles_batch is None:
                raise ValueError("phi_angles_batch and psi_angles_batch are required for torsion conditioning.")
            if phi_angles_batch.ndim != 2 or psi_angles_batch.ndim != 2:
                raise ValueError('phi/psi batches must be 2D arrays.')
            if phi_angles_batch.shape != psi_angles_batch.shape:
                raise ValueError('phi/psi batches must have matching shapes.')
            cond_np = self.build_condition_torsion(phi_angles_batch, psi_angles_batch)
        elif self.conditioning_mode == 'pairwise':
            if condition_batch is None:
                raise ValueError("condition_batch is required for pairwise conditioning.")
            cond_np = self.build_condition_pairwise(condition_batch)
        else:
            # hybrid mode: allow omitting condition_batch during MD sampling and default
            # the pairwise block to the dataset mean (which normalizes to zeros).
            if phi_angles_batch is None or psi_angles_batch is None:
                raise ValueError("phi/psi batches are required for hybrid conditioning.")
            if phi_angles_batch.shape != psi_angles_batch.shape:
                raise ValueError('phi/psi batches must have matching shapes.')
            torsion_feats = self.build_condition_torsion(phi_angles_batch, psi_angles_batch)
            if condition_batch is None:
                # Use dataset mean for the pairwise block so that after normalization the
                # pairwise inputs are zero, matching a neutral prior.
                if not hasattr(self, 'hybrid_pair_dim') or self.hybrid_pair_dim <= 0:
                    raise ValueError("Hybrid pairwise dim is not set; cannot default pairwise features.")
                pair_mean = self.cond_mean[-self.hybrid_pair_dim:].detach().cpu().numpy().reshape(1, -1)
                pair_feats = np.repeat(pair_mean, torsion_feats.shape[0], axis=0)
            else:
                pair_feats = self.build_condition_pairwise(condition_batch)
            cond_np = np.concatenate([torsion_feats, pair_feats], axis=1)
        cond_tensor = torch.from_numpy(cond_np).to(self.device)
        cond_tensor = (cond_tensor - self.cond_mean) / self.cond_std
        return cond_tensor

    @torch.no_grad()
    def sample(
        self,
        phi_angles: Optional[np.ndarray] = None,
        psi_angles: Optional[np.ndarray] = None,
        condition_vector: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        samples = self.sample_batch(
            phi_angles_batch=None if phi_angles is None else phi_angles[None, :],
            psi_angles_batch=None if psi_angles is None else psi_angles[None, :],
            condition_batch=None if condition_vector is None else condition_vector[None, :],
            seed=seed,
        )
        return samples[0]

    @torch.no_grad()
    def sample_batch(
        self,
        phi_angles_batch: Optional[np.ndarray] = None,
        psi_angles_batch: Optional[np.ndarray] = None,
        condition_batch: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        cond_tensor = self._prepare_condition_batch(phi_angles_batch, psi_angles_batch, condition_batch)
        B = cond_tensor.shape[0]
        shape = (B, 3, self.model.n_atoms, 1)
        samples = self.diffusion.p_sample_loop(
            self.model,
            shape,
            clip_denoised=False,
            progress=False,
            device=self.device,
            model_kwargs={'cond': cond_tensor},
        )
        coords = samples.squeeze(-1).permute(0, 2, 1).cpu().numpy()
        return coords

generator_cache: Dict[Tuple[str, str, str, str], "ConditionalGenerator"] = {}
def get_conditional_generator(checkpoint_path: str, data_path: str, pdb_path: str, device: torch.device) -> ConditionalGenerator:
    key = (checkpoint_path, data_path, pdb_path, str(device))
    if key not in generator_cache:
        generator_cache[key] = ConditionalGenerator(checkpoint_path, data_path, pdb_path, device=device)
    return generator_cache[key]
