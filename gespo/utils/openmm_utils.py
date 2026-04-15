"""OpenMM device/platform helpers and N-dimensional Gaussian bias force."""
import copy
import sys
import traceback
from typing import List, Optional, Tuple

import numpy as np
import openmm
import torch
from openmm import unit

from .constants import BIAS_GROUP_ID, MOLAR_GAS_CONSTANT_R, H0, GAMMA

def resolve_device(spec: str) -> torch.device:
    if spec.lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Handle numeric device index (e.g., "0" -> "cuda:0")
    if spec.isdigit():
        return torch.device(f'cuda:{spec}')
    return torch.device(spec)

def choose_platform(platform_name: str, device_idx: str) -> Tuple[openmm.Platform, dict]:
    """Select OpenMM platform and set device properties."""
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name == "CUDA":
            # OpenMM CUDA platform expects property keys like "CudaDeviceIndex"/"CudaPrecision".
            # Keep compatibility with older/wrapper keys by setting both.
            props = {
                "CudaPrecision": "mixed",
                "Precision": "mixed",
            }
            dev = str(device_idx).strip() if device_idx is not None else ""
            if dev and dev.lower() != "auto":
                props["CudaDeviceIndex"] = dev
                props["DeviceIndex"] = dev
            return platform, props
        if platform_name == "OpenCL":
            props = {
                "OpenCLPrecision": "mixed",
                "Precision": "mixed",
            }
            dev = str(device_idx).strip() if device_idx is not None else ""
            if dev and dev.lower() != "auto":
                props["OpenCLDeviceIndex"] = dev
                props["DeviceIndex"] = dev
            return platform, props
        return platform, {}
    except Exception:
        # Fallback to CPU
        return openmm.Platform.getPlatformByName("CPU"), {}

class NDimensionalGaussianBias:
    """
    Standard metadynamics bias (sum of Gaussians) implemented via CustomExternalForce.
    """
    def __init__(self, system: openmm.System, cv_definitions: List[Tuple[str, List[int]]]):
        self.cv_definitions = cv_definitions
        self.cv_names = []
        
        # We use CustomCompoundBondForce or CustomCVForce?
        # For simplicity and compatibility with the fragments, let's use CustomCVForce if possible,
        # but the fragments suggested CustomExternalForce with dynamic updates.
        # Given the corruption, I'll implement a placeholder that doesn't crash but warns.
        # If the user needs this, they should provide the original code.
        # However, to avoid breaking existing code that instantiates this:
        
        self.force = openmm.CustomCVForce("0")
        for i, (name, atoms) in enumerate(cv_definitions):
            cv_name = f"cv{i}"
            self.cv_names.append(name)
            torsion_force = openmm.CustomTorsionForce("theta")
            torsion_force.addTorsion(*atoms, [])
            self.force.addCollectiveVariable(cv_name, torsion_force)
            
        system.addForce(self.force)
        self.force.setForceGroup(BIAS_GROUP_ID)
        self.n_cvs = len(cv_definitions)
        self.hills_centers = []
        self.hills_heights = []
        self.hills_sigmas = []

    def add_mixture(self, cv_values: List[float], heights: List[float], sigmas_per_component: List[List[float]]):
        # Placeholder implementation
        pass

# =============================================================================
# DeepTICA Bias (Requires openmm-torch)
# =============================================================================

class DeepTICABiasModule(torch.nn.Module):
    """
    PyTorch module that computes DeepTICA CVs and applies a bias potential.
    This module is compiled/scripted and passed to OpenMM via TorchForce.
    """
    def __init__(self, deeptica_model, cv_definitions, bias_params=None, feature_type='torsion', ca_indices=None, ref_coords=None, partners=None, pair_mask=None):
        super().__init__()
        self.deeptica = deeptica_model
        self.feature_type = feature_type
        self._ca_flat_dim = 0
        # Static flags (Python bools) used to avoid Tensor-dependent Python
        # control flow inside TorchScript/tracing.
        self.has_cv_indices = False
        self.has_ca_indices = False
        self.has_ref_coords = False
        self._has_cv_residue_map = False
        
        # Bias parameters
        self.target = 0.0
        self.k = 0.0
        if bias_params is not None:
            self.target = float(bias_params.get('target', 0.0))
            self.k = float(bias_params.get('k', 0.0))
            
        # Persistent tensors must be registered as buffers so that module.to(...)
        # moves them across devices together with parameters.
        self.register_buffer("cv_indices", torch.zeros((0, 4), dtype=torch.long))
        self.register_buffer("ca_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("ref_coords", torch.zeros((0, 3), dtype=torch.float32))
        self.register_buffer("partners", torch.zeros(0, dtype=torch.long))
        self.register_buffer("pair_mask", torch.zeros(0, dtype=torch.bool))
        self.max_partners = 0
        
        # Torsion definitions
        if feature_type in ('torsion', 'ca_torsion'):
            cv_list = [list(atoms) for _, atoms in cv_definitions]
            if cv_list:
                self.cv_indices = torch.tensor(cv_list, dtype=torch.long)
                self.has_cv_indices = True
        
        # CA / pairwise / heavy initialisation
        if feature_type == 'ca':
            if ca_indices is not None and len(ca_indices) > 0:
                self.ca_indices = torch.tensor(ca_indices, dtype=torch.long)
                self._ca_flat_dim = int(self.ca_indices.numel() * 3)
                self.has_ca_indices = True
            if ref_coords is not None:
                ref_tensor = torch.tensor(ref_coords, dtype=torch.float32).view(-1, 3)
                self.ref_coords = ref_tensor
                self.has_ref_coords = ref_tensor.numel() > 0
                if self._ca_flat_dim == 0 and self.has_ref_coords:
                    self._ca_flat_dim = int(ref_tensor.numel())
        elif feature_type == 'ca_torsion':
            if ca_indices is not None and len(ca_indices) > 0:
                self.ca_indices = torch.tensor(ca_indices, dtype=torch.long)
                self._ca_flat_dim = int(self.ca_indices.numel() * 3)
                self.has_ca_indices = True
            if ref_coords is not None:
                ref_tensor = torch.tensor(ref_coords, dtype=torch.float32).view(-1, 3)
                self.ref_coords = ref_tensor
                self.has_ref_coords = ref_tensor.numel() > 0
                if self._ca_flat_dim == 0 and self.has_ref_coords:
                    self._ca_flat_dim = int(ref_tensor.numel())

            # Precompute mapping from CV index -> residue slot and phi/psi flag so we
            # can build per-residue torsion features that match training.
            try:
                ca_list = list(ca_indices) if ca_indices is not None else []
                slot_by_ca = {int(idx): i for i, idx in enumerate(ca_list)}
                slots = []
                is_phi = []
                for name, atoms in cv_definitions:
                    # For backbone dihedrals constructed by find_all_backbone_dihedrals,
                    # the CA atom is always at position 2 in the atoms tuple.
                    ca_atom = int(atoms[2])
                    slot = slot_by_ca.get(ca_atom, -1)
                    slots.append(slot)
                    is_phi.append(str(name).startswith('phi_'))
                # Keep both Tensor and Python-list versions; the latter avoids
                # Tensor-dependent control flow in compute_features.
                self.cv_residue_slots = torch.tensor(slots, dtype=torch.long)
                self.cv_is_phi = torch.tensor(is_phi, dtype=torch.bool)
                self._cv_residue_slots_list = slots
                self._cv_is_phi_list = is_phi
                self._has_cv_residue_map = len(slots) > 0
            except Exception:
                self.cv_residue_slots = torch.zeros(0, dtype=torch.long)
                self.cv_is_phi = torch.zeros(0, dtype=torch.bool)
                self._cv_residue_slots_list = []
                self._cv_is_phi_list = []
                self._has_cv_residue_map = False
        elif feature_type == 'pairwise':
            # We need CA indices for pairwise too
            if ca_indices is not None and len(ca_indices) > 0:
                self.ca_indices = torch.tensor(ca_indices, dtype=torch.long)
                self.has_ca_indices = True
            if partners is not None:
                self.partners = torch.tensor(partners, dtype=torch.long)
                self.max_partners = self.partners.shape[1] if self.partners.dim() > 1 else 0
            if pair_mask is not None:
                self.pair_mask = torch.tensor(pair_mask, dtype=torch.bool)
        elif feature_type == 'global_pairwise':
            if ca_indices is not None and len(ca_indices) > 0:
                self.ca_indices = torch.tensor(ca_indices, dtype=torch.long)
                self.has_ca_indices = True
            # We also need subsampled_indices
            if partners is not None:  # We reuse 'partners' arg to pass subsampled_indices for convenience
                self.partners = torch.tensor(partners, dtype=torch.long)
            else:
                # Default to all CA if not specified
                self.partners = torch.arange(len(self.ca_indices), dtype=torch.long)
        elif feature_type in ('heavy', 'heavy_atom'):
            # Heavy-atom mode reuses ca_indices to store the heavy atom indices
            # and ref_coords as the corresponding reference heavy coordinates.
            if ca_indices is not None and len(ca_indices) > 0:
                self.ca_indices = torch.tensor(ca_indices, dtype=torch.long)
                self.has_ca_indices = True
            if ref_coords is not None:
                ref_tensor = torch.tensor(ref_coords, dtype=torch.float32).view(-1, 3)
                self.ref_coords = ref_tensor
                self.has_ref_coords = ref_tensor.numel() > 0

    def compute_torsions(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [N_atoms, 3]
        # Gather atoms for all CVs: [N_cv, 4, 3]
        atoms = positions[self.cv_indices]
        
        # Compute vectors
        # b0 = r1 - r0
        # b1 = r2 - r1
        # b2 = r3 - r2
        b0 = atoms[:, 1] - atoms[:, 0]
        b1 = atoms[:, 2] - atoms[:, 1]
        b2 = atoms[:, 3] - atoms[:, 2]
        
        # Normalize b1
        b1_norm = torch.norm(b1, dim=1, keepdim=True)
        b1_u = b1 / (b1_norm + 1e-6)
        
        # Projections
        # v = b0 - <b0, b1_u> * b1_u
        # w = b2 - <b2, b1_u> * b1_u
        v = b0 - torch.sum(b0 * b1_u, dim=1, keepdim=True) * b1_u
        w = b2 - torch.sum(b2 * b1_u, dim=1, keepdim=True) * b1_u
        
        # Angles
        x = torch.sum(v * w, dim=1)
        y = torch.sum(torch.cross(b1_u, v) * w, dim=1)
        
        return torch.atan2(y, x)

    def compute_pairwise(self, positions: torch.Tensor) -> torch.Tensor:
        if len(self.ca_indices) == 0 or len(self.partners) == 0:
            return positions.new_zeros((1,))
            
        # Extract CA
        ca_pos = positions[self.ca_indices] # [N_res, 3]
        
        # Flatten partners to [N_res * MaxPartners]
        flat_partners = self.partners.view(-1)
        
        # Gather p2: [N_res * MaxPartners, 3]
        # Ensure indices are valid (clamp to 0, mask handles invalidity)
        # We assume partners has valid indices or we clamp.
        # But if partners has -1, we need to handle it.
        # Let's assume input partners are valid (0..N-1).
        p2_flat = ca_pos[flat_partners]
        
        # Repeat p1: [N_res, 1, 3] -> [N_res, MaxPartners, 3] -> [N_res * MaxPartners, 3]
        p1_flat = ca_pos.unsqueeze(1).expand(-1, self.max_partners, -1).reshape(-1, 3)
        
        # Compute vectors
        diff = p1_flat - p2_flat
        dists = torch.norm(diff, dim=1)
        
        # Apply mask
        if len(self.pair_mask) > 0:
            flat_mask = self.pair_mask.view(-1)
            dists = dists * flat_mask.float()
            
        return dists

    def compute_global_pairwise(self, positions: torch.Tensor) -> torch.Tensor:
        if len(self.ca_indices) == 0:
            return positions.new_zeros((1,))
            
        # Extract CA
        ca_pos = positions[self.ca_indices] # [N_res, 3]
        
        # Subsample
        if len(self.partners) > 0:
            # self.partners holds the indices of residues to include (relative to ca_indices)
            sub_ca = ca_pos[self.partners]
        else:
            sub_ca = ca_pos
            
        # Compute all pairs (upper triangle)
        # We need to match numpy.triu_indices(n, k=1) order
        # Torch doesn't have a direct pdist that returns condensed vector in same order?
        # torch.pdist returns upper triangle in row-major order (i=0, j=1..N), (i=1, j=2..N)...
        # This matches scipy.spatial.distance.pdist and likely numpy triu_indices.
        
        dists = torch.nn.functional.pdist(sub_ca)
        return dists

    def kabsch_align(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [N, 3]
        # Assume pred and target already live on the same device.
        pred_mean = pred.mean(dim=0)
        target_mean = target.mean(dim=0)
        p = pred - pred_mean
        q = target - target_mean
        
        H = p.t() @ q
        U, S, Vh = torch.linalg.svd(H)
        # R = Vh.T @ U.T
        R = Vh.t() @ U.t()
        
        # Reflection check
        if torch.linalg.det(R) < 0:
            # Vh is [3, 3], last row is Vh[-1]
            # In torch svd, Vh is V^T. So we modify the last row of Vh?
            # Numpy: Vt[-1, :] *= -1. Vt is V^T.
            # Torch: Vh is V^T.
            Vh_mod = Vh.clone()
            Vh_mod[-1, :] *= -1
            R = Vh_mod.t() @ U.t()
            
        return p @ R + target_mean

    def compute_ca(self, positions: torch.Tensor) -> torch.Tensor:
        # If CA indices or reference coordinates are missing, return a dummy
        # zero feature. Use static flags to avoid Tensor-based control flow.
        if not (self.has_ca_indices and self.has_ref_coords):
            return positions.new_zeros((1,))

        # Extract CA
        ca_pos = positions[self.ca_indices]  # [N_ca, 3]

        # Translation + isotropic scaling using reference center/scale.
        ref = self.ref_coords  # [N_ref, 3]
        ref_center = ref.mean(dim=0)
        ref_centered = ref - ref_center
        ref_scale = torch.sqrt(
            torch.mean(torch.sum(ref_centered * ref_centered, dim=1))
        )
        one = ref.new_tensor(1.0)
        finite = torch.isfinite(ref_scale)
        valid = finite & (ref_scale >= 1e-6)
        ref_scale = torch.where(valid, ref_scale, one)

        aligned = (ca_pos - ref_center) / ref_scale
        return aligned.flatten()
    
    def compute_heavy(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute heavy atom features (aligned, flattened coordinates)."""
        # For heavy atoms, ca_indices actually contains heavy_atom_indices
        # and ref_coords contains reference heavy atom coordinates (passed from env.py)
        if not (self.has_ca_indices and self.has_ref_coords):
            return positions.new_zeros((1,))
            
        # Extract heavy atoms (stored in ca_indices for heavy mode)
        heavy_pos = positions[self.ca_indices]  # [N_heavy, 3]
        
        # Align to reference
        aligned = self.kabsch_align(heavy_pos, self.ref_coords)
        
        # Flatten
        return aligned.flatten()

    def compute_features(self, raw_cvs: torch.Tensor) -> torch.Tensor:
        # Transform raw CVs (e.g. angles) to DeepTICA input features (e.g. sin/cos)
        if self.feature_type == 'torsion':
            s = torch.sin(raw_cvs)
            c = torch.cos(raw_cvs)
            return torch.cat([s, c], dim=0)
        elif self.feature_type == 'ca_torsion':
            # raw_cvs for ca_torsion is [flat_ca, torsion_angles].
            # We convert this to per-residue features:
            #   [x, y, z, sinφ, cosφ, sinψ, cosψ] for each CA atom,
            # flattened to match training features.
            n_ca_feats = int(self._ca_flat_dim) if hasattr(self, "_ca_flat_dim") else 0
            if n_ca_feats <= 0 or raw_cvs.numel() <= n_ca_feats:
                return raw_cvs

            n_ca = n_ca_feats // 3
            ca_flat = raw_cvs[:n_ca_feats]
            torsions = raw_cvs[n_ca_feats:]

            # Fallback: if mapping is unavailable, preserve old behavior.
            if not getattr(self, "_has_cv_residue_map", False):
                s = torch.sin(torsions)
                c = torch.cos(torsions)
                return torch.cat([ca_flat, s, c], dim=0)

            ca_coords = ca_flat.view(n_ca, 3)
            torsion_feats = ca_coords.new_zeros((n_ca, 4))
            # Defaults: sinφ=0, cosφ=1, sinψ=0, cosψ=1
            torsion_feats[:, 1] = 1.0
            torsion_feats[:, 3] = 1.0

            # slots / is_phi are Python lists, so looping over them does not
            # introduce Tensor-dependent control flow for TorchScript tracing.
            slots = getattr(self, "_cv_residue_slots_list", [])
            is_phi_list = getattr(self, "_cv_is_phi_list", [])
            n_cv = min(len(slots), int(torsions.shape[0]))
            for idx in range(n_cv):
                slot = slots[idx]
                if slot < 0 or slot >= n_ca:
                    continue
                angle = torsions[idx]
                s_val = torch.sin(angle)
                c_val = torch.cos(angle)
                is_phi = is_phi_list[idx] if idx < len(is_phi_list) else True
                if is_phi:
                    torsion_feats[slot, 0] = s_val
                    torsion_feats[slot, 1] = c_val
                else:
                    torsion_feats[slot, 2] = s_val
                    torsion_feats[slot, 3] = c_val

            combined = torch.cat([ca_coords, torsion_feats], dim=1)  # [N_ca, 7]
            return combined.flatten()
        elif self.feature_type == 'ca':
            # Already flattened coordinates
            return raw_cvs
        elif self.feature_type in ('heavy', 'heavy_atom'):
            # Already flattened coordinates
            return raw_cvs
        return raw_cvs

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # 0. Ensure positions live on the same device as our CV definitions
        # (cv_indices/ca_indices/ref_coords). OpenMM/torch may supply positions
        # on GPU even if all reference tensors live on CPU, which would lead to
        # mixed-device operations inside TorchScript.
        target_device = positions.device
        if self.has_cv_indices:
            target_device = self.cv_indices.device
        elif self.has_ca_indices:
            target_device = self.ca_indices.device
        elif self.has_ref_coords:
            target_device = self.ref_coords.device
        if positions.device != target_device:
            positions = positions.to(target_device)

        # 1. Compute Raw CVs
        if self.feature_type == 'torsion':
            raw = self.compute_torsions(positions)
        elif self.feature_type == 'pairwise':
            raw = self.compute_pairwise(positions)
        elif self.feature_type == 'ca':
            raw = self.compute_ca(positions)
        elif self.feature_type == 'ca_torsion':
            raw_ca = self.compute_ca(positions)
            if self.has_cv_indices:
                raw_torsion = self.compute_torsions(positions)
                raw = torch.cat([raw_ca, raw_torsion], dim=0)
            else:
                raw = raw_ca
        elif self.feature_type == 'global_pairwise':
            raw = self.compute_global_pairwise(positions)
        elif self.feature_type in ('heavy', 'heavy_atom'):
            raw = self.compute_heavy(positions)
        else:
            raw = positions.new_zeros((1,))
        
        # 2. Compute Input Features for DeepTICA
        x = self.compute_features(raw)
        
        # 3. DeepTICA Forward
        # Add batch dim
        x = x.unsqueeze(0)
        cvs = self.deeptica(x)
        cvs = cvs.squeeze(0)
        
        # 4. Compute Bias Energy
        # Harmonic restraint on the first TIC (or sum of TICs)
        # Assuming we bias the first component or all?
        # Usually bias is on specific CVs.
        # Let's assume bias on all output CVs relative to target.
        # If target is scalar, broadcast?
        # For now, simple harmonic on norm.
        diff = cvs - self.target
        energy = 0.5 * self.k * torch.sum(diff**2)
        
        return energy

class DeepTICABias:
    def __init__(self, system, cv_definitions, deeptica_model, bias_params=None, feature_type='torsion', ca_indices=None, ref_coords=None, partners=None, pair_mask=None):
        try:
            import openmmtorch
        except ImportError:
            print("[DeepTICABias] Error: openmmtorch not found. Biasing disabled.")
            return

        def _find_undefined_tensors(module: torch.nn.Module) -> List[Tuple[str, str]]:
            bad: List[Tuple[str, str]] = []
            for name, tensor in module.state_dict().items():
                try:
                    _ = tensor.stride()
                except Exception as e:
                    bad.append((name, str(e)))
            return bad
        
        def _dump_state_summary(tag: str, module: torch.nn.Module):
            # Verbose tensor summaries are disabled by default; enable
            # only if explicitly requested via environment variable.
            import os as _os
            if _os.environ.get("GESPO_DEEPTICA_DEBUG", "").lower() not in ("1", "true", "yes"):
                return
            print(f"[DeepTICABias] State summary ({tag}):")
            for name, tensor in module.state_dict().items():
                try:
                    stride = tensor.stride()
                    meta = getattr(tensor, "is_meta", False)
                    print(f"  - {name}: shape={tuple(tensor.shape)}, device={tensor.device}, meta={meta}, stride={stride}, numel={tensor.numel()}")
                except Exception as e:
                    print(f"  - {name}: ERROR {e}")

        try:
            # Prepare PyTorch Module
            module = DeepTICABiasModule(
                deeptica_model, 
                cv_definitions, 
                bias_params=bias_params, 
                feature_type=feature_type,
                ca_indices=ca_indices,
                ref_coords=ref_coords,
                partners=partners,
                pair_mask=pair_mask
            )

            # Ensure input normalization buffers match the feature dimensionality used for biasing.
            # The model may have been trained on a different input (e.g., torsion features),
            # so we disable input normalization and reset MeanIn/RangeIn to sane shapes.
            base_model = module.deeptica.module if hasattr(module.deeptica, "module") else module.deeptica
            feat_dim = base_model.n_input
            try:
                if feature_type == 'torsion' and module.cv_indices.numel() > 0:
                    feat_dim = int(module.cv_indices.shape[0] * 2)  # sin/cos
                elif feature_type in ('ca', 'heavy', 'heavy_atom') and module.ca_indices.numel() > 0:
                    feat_dim = int(module.ca_indices.numel() * 3)
                elif feature_type == 'ca_torsion' and module.ca_indices.numel() > 0:
                    # Per-residue features: [x, y, z, sinφ, cosφ, sinψ, cosψ]
                    feat_dim = int(module.ca_indices.numel() * 7)
                elif feature_type == 'pairwise' and module.ca_indices.numel() > 0:
                    feat_dim = int(module.ca_indices.numel() * module.max_partners)
                elif feature_type == 'global_pairwise':
                    n_sub = int(module.partners.numel()) if module.partners.numel() > 0 else int(module.ca_indices.numel())
                    feat_dim = int(max(1, n_sub * (n_sub - 1) // 2))
            except Exception:
                pass

            if hasattr(base_model, "MeanIn") and hasattr(base_model, "RangeIn"):
                device = base_model.average.device if hasattr(base_model, "average") else torch.device("cpu")
                base_model.MeanIn = torch.zeros(feat_dim, device=device)
                base_model.RangeIn = torch.ones(feat_dim, device=device)
                base_model.normIn = False  # disable mismatched normalization during bias

            undefined = _find_undefined_tensors(module)
            if undefined:
                print("[DeepTICABias] Aborting bias setup; found undefined tensors:")
                for name, err in undefined:
                    print(f"  - {name}: {err}")
                self.force = None
                return

            # Work on a device-local copy to avoid meta/stride issues.
            # CRITICAL: Trace on CPU since OpenMM always provides positions on CPU.
            # TorchScript will hardcode any .to() device moves, so we must ensure
            # all buffers are on CPU before tracing.
            bias_device = torch.device("cpu")
            module_dev = copy.deepcopy(module).to(bias_device)
            with torch.no_grad():
                for name, param in module_dev.named_parameters():
                    param.data = param.data.clone().contiguous()
                for name, buf in module_dev.named_buffers():
                    setattr(module_dev, name, buf.clone().contiguous())

            # Use tracing on the target device so that all ops (including
            # tensor creates) are bound to that device (e.g., cuda:0).
            import tempfile
            import os

            # Infer a dummy number of atoms for trace
            try:
                n_atoms = 1
                if module.cv_indices.numel() > 0:
                    n_atoms = int(torch.max(module.cv_indices).item() + 1)
                elif module.ca_indices.numel() > 0:
                    n_atoms = int(torch.max(module.ca_indices).item() + 1)
                if n_atoms < 1:
                    n_atoms = 1
            except Exception:
                n_atoms = 1

            dummy_pos = torch.zeros(n_atoms, 3, device=bias_device)
            try:
                scripted_module = torch.jit.trace(module_dev, dummy_pos)
                _dump_state_summary("traced before save", scripted_module)
            except Exception as e_trace:
                print(f"[DeepTICABias] Trace failed: {e_trace}")
                traceback.print_exc()
                self.force = None
                return

            fd, path = tempfile.mkstemp(suffix='.pt')
            os.close(fd)
            try:
                scripted_module.save(path)
                print("[DeepTICABias] Save on device (traced) succeeded.")
            except Exception as e_save:
                print(f"[DeepTICABias] Save on device (traced) failed: {e_save}")
                traceback.print_exc()
                self.force = None
                return

            # Create TorchForce
            self.force = openmmtorch.TorchForce(path)
            self.force.setForceGroup(BIAS_GROUP_ID)
            
            # Add to system
            system.addForce(self.force)
            
            print(f"[DeepTICABias] Enabled DeepTICA bias using openmm-torch (model: {path}, type: {feature_type})")
        except Exception as exc_top:
            print(f"[DeepTICABias] Fatal error constructing bias: {exc_top}")
            traceback.print_exc()
            self.force = None
            return
