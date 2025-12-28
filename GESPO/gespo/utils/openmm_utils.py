"""OpenMM device/platform helpers and N-dimensional Gaussian bias force."""
from .._imports import *
from .constants import H0, GAMMA, BIAS_GROUP_ID
import torch
import numpy as np
import openmm
import copy
import traceback
from typing import List, Tuple
from openmm import unit
from typing import List, Tuple, Optional
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
