"""Environment for MD rollout: OpenMM simulation + CV tracking."""
import math
import sys
import time
import traceback
import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import torch
from openmm import LangevinIntegrator, MonteCarloBarostat, Vec3
from openmm.app import ForceField, Modeller, NoCutoff, PME, PDBFile, Simulation
from openmm.app.forcefield import HBonds
from openmm.unit import (
    bar, femtoseconds, kelvin, kilojoule_per_mole, molar, nanometer, picosecond,
)

from ..utils.openmm_utils import choose_platform, NDimensionalGaussianBias
from ..utils.geometry import (
    find_all_backbone_dihedrals,
    find_all_sidechain_chi_dihedrals,
    _dihedral,
    cv_targets_to_phi_psi,
    round_and_wrap_angles,
    kabsch_align,
    calculate_rmsd,
)
from ..utils.constants import H0, GAMMA, BIAS_GROUP_ID
from ..utils.adk_metrics import ADK_GLOBAL_TOKEN_NAMES, build_adk_token_indices
from .targeted_md import add_replay_steering_forces, steer_replay_to_targets

_OPENMM_CONTEXT_BUILD_LOCK = threading.Lock()
_OPENMM_MINIMIZE_LOCK = threading.Lock()

def _is_fft_friendly(n: int) -> bool:
    n = int(n)
    if n <= 0:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def _next_fft_friendly(n: int, *, max_increase: int = 1024) -> int:
    n0 = int(max(1, n))
    for k in range(n0, n0 + int(max_increase) + 1):
        if _is_fft_friendly(k):
            return int(k)
    return int(n0)


def _pme_alpha_from_tol(cutoff_nm: float, tol: float) -> float:
    # Match OpenMM's typical alpha scaling:
    #   erfc(alpha * rc) ~ tol  => alpha ~ sqrt(-ln(2*tol)) / rc
    # (This is approximate but consistent with common practice and close to OpenMM defaults.)
    rc = float(cutoff_nm)
    t = float(tol)
    if rc <= 0 or not np.isfinite(rc):
        return 2.2
    if t <= 0 or not np.isfinite(t):
        t = 5e-4
    return float(np.sqrt(max(0.0, -np.log(max(1e-16, 2.0 * t)))) / rc)

def _build_simulation_or_fallback(topology, system, integrator, platform_name: str, platform_device):
    """
    Construct an OpenMM Simulation, retrying/falling back when CUDA/OpenCL
    device selection fails (common on misconfigured nodes or when device
    indices don't match the job's visible devices).
    """
    platform_name = str(platform_name or "CPU")
    try:
        platform, platform_props = choose_platform(platform_name, platform_device)
        with _OPENMM_CONTEXT_BUILD_LOCK:
            return Simulation(topology, system, integrator, platform, platform_props or None), platform_name
    except Exception as exc:
        msg = str(exc)
        is_device_load = "requested cuda device could not be loaded" in msg.lower()
        if platform_name in ("CUDA", "OpenCL") and is_device_load:
            print(f"[OpenMM] {platform_name} device init failed ({exc}); retrying without device pinning.", flush=True)
            try:
                platform, platform_props = choose_platform(platform_name, "")
                with _OPENMM_CONTEXT_BUILD_LOCK:
                    return Simulation(topology, system, integrator, platform, platform_props or None), platform_name
            except Exception as exc2:
                print(f"[OpenMM] {platform_name} retry failed ({exc2}); falling back to CPU.", flush=True)
        elif platform_name in ("CUDA", "OpenCL"):
            print(f"[OpenMM] {platform_name} init failed ({exc}); falling back to CPU.", flush=True)
        platform, platform_props = choose_platform("CPU", 0)
        with _OPENMM_CONTEXT_BUILD_LOCK:
            return Simulation(topology, system, integrator, platform, platform_props or None), "CPU"

class PeptideSampler:
    def __init__(self, args, generator_device: torch.device):
        self.n_md_steps = args.n_md_steps
        # Policy input mode controls what state() returns
        self.policy_mode = str(getattr(args, "policy_conditioning_mode", "torsion")).lower()
        self.use_bias = bool(getattr(args, "use_bias", False))

        # Store PDB path for later initialization (after multi-conformer detection)
        self._generator_args = {
            'pdb': args.pdb,
            'device': generator_device
        }
        # Optional endpoint-B PDB for path sampling (used for RMSD-to-A/B globals + steering).
        self._endpoint_b_pdb = str(getattr(args, "pdb_b", "") or "")
        self.generator = None  # Will be initialized later if needed
        self.generator_device = generator_device
        self.heavy_atom_indices = None  # Will be set from generator or PDB
        self.reference_heavy_angstrom = None  # Will be set from generator or PDB
        self.last_generation_record: Optional[dict] = None
        # Flag indicating whether the most recent conditional reset fell back
        # to the original PDB structure after exhausting resample attempts.
        self.reset_failed: bool = False
        self.energy_threshold_kj = float(args.energy_threshold)
        self.max_resample_attempts = max(1, int(args.max_resample_attempts))
        self.torsion_round_decimals = int(args.torsion_decimals)
        self.minimize_iterations = max(0, int(getattr(args, "minimize_iters", 200)))
        self.heat_num_stages = max(0, int(getattr(args, "heat_stages", 4)))
        self.heat_stage_steps = max(0, int(getattr(args, "heat_steps", 100)))
        self.friction = float(getattr(args, "friction", 1.0))
        self.timestep_fs = float(getattr(args, "timestep_fs", 2.0))
        self.temperature = float(getattr(args, "temperature", 300.0))
        self.minimize_tolerance_kj_mol_nm = getattr(args, "minimize_tolerance_kj_mol_nm", None)
        self.platform_name = args.platform
        self.platform_device = args.device
        # Targeted replay steering: match CA RMSD + E2E (+ optional Rg), then torsions.
        self.steer_replay_starts = bool(getattr(args, "steer_replay_starts", True))
        self.steer_replay_verbose = bool(getattr(args, "steer_replay_verbose", False))
        self.steer_global_steps = max(0, int(getattr(args, "steer_global_steps", 2000) or 0))
        self.steer_global_updates = max(1, int(getattr(args, "steer_global_updates", 25) or 1))
        self.steer_torsion_steps = max(0, int(getattr(args, "steer_torsion_steps", 1000) or 0))
        self.steer_torsion_updates = max(1, int(getattr(args, "steer_torsion_updates", 25) or 1))
        self.steer_release_steps = max(0, int(getattr(args, "steer_release_steps", 200) or 0))
        self.steer_refine_minimize_iters = max(0, int(getattr(args, "steer_refine_minimize_iters", 0) or 0))
        self.steer_k_rmsd = float(getattr(args, "steer_k_rmsd", 5000.0) or 0.0)
        self.steer_k_e2e = float(getattr(args, "steer_k_e2e", 5000.0) or 0.0)
        self.steer_max_energy_kj_mol = float(getattr(args, "steer_max_energy_kj_mol", 0.0) or 0.0)
        if not np.isfinite(self.steer_max_energy_kj_mol) or self.steer_max_energy_kj_mol < 0.0:
            self.steer_max_energy_kj_mol = 0.0
        self.steer_k_rg = float(getattr(args, "steer_k_rg", 0.0) or 0.0)
        self.steer_k_torsion = float(getattr(args, "steer_k_torsion", 200.0) or 0.0)
        # Sidechain χ torsion steering weight (multiplies each χ torsion term). Set to 0 to disable χ steering.
        self.steer_chi_weight = float(getattr(args, "steer_chi_weight", 0.1) or 0.0)
        if not np.isfinite(self.steer_chi_weight) or self.steer_chi_weight < 0.0:
            self.steer_chi_weight = 0.0
        # Ligand rotatable dihedral steering weight. Default 1.0 (steer ligand to AR-proposed torsion).
        self.steer_ligand_weight = float(getattr(args, "steer_ligand_weight", 1.0) or 0.0)
        if not np.isfinite(self.steer_ligand_weight) or self.steer_ligand_weight < 0.0:
            self.steer_ligand_weight = 0.0
        # Additional global metrics (e.g., ADK distances/hinge angles).
        self.steer_k_dist = float(getattr(args, "steer_k_dist", self.steer_k_e2e) or 0.0)
        self.steer_k_theta = float(getattr(args, "steer_k_theta", self.steer_k_torsion) or 0.0)
        self.steer_allow_large_explicit_extra_globals = bool(
            getattr(args, "steer_allow_large_explicit_extra_globals", False) or False
        )
        self.steer_global_lambda_during_torsion = float(getattr(args, "steer_global_lambda_during_torsion", 0.2) or 0.0)
        # Number of candidate conditional samples to consider at reset when PCCA guidance is available
        try:
            self.pcca_guided_start_samples = max(1, int(getattr(args, "pcca_guided_start_samples", 4)))
        except Exception:
            self.pcca_guided_start_samples = 4
        platform, platform_props = choose_platform(self.platform_name, self.platform_device)
        self.implicit_solvent = bool(getattr(args, "implicit_solvent", False))
        explicit_flag = bool(getattr(args, "explicit_solvent", False))
        # --implicit-solvent takes precedence over the explicit_solvent default.
        if self.implicit_solvent:
            self.explicit_solvent = False
        else:
            self.explicit_solvent = explicit_flag or True
        self.explicit_solvent_replay_full_coords = bool(getattr(args, "explicit_solvent_replay_full_coords", False))
        # Explicit-solvent: box/ions
        self.solvent_padding_nm = float(getattr(args, "solvent_padding_nm", 1.0))
        self.box_length_nm = getattr(args, "box_length_nm", None)
        self.ionic_strength_m = float(getattr(args, "ionic_strength_m", 0.1))
        self.solvent_positive_ion = str(getattr(args, "solvent_positive_ion", "Na+"))
        self.solvent_negative_ion = str(getattr(args, "solvent_negative_ion", "Cl-"))
        self.solvent_neutralize = bool(getattr(args, "solvent_neutralize", True))
        # Explicit-solvent: nonbonded settings
        self.nonbonded_cutoff_nm = getattr(args, "nonbonded_cutoff_nm", None)
        self.use_switching_function = bool(getattr(args, "use_switching_function", False))
        self.switch_distance_nm = getattr(args, "switch_distance_nm", None)
        self.ewald_error_tolerance = getattr(args, "ewald_error_tolerance", None)
        # Optional NPT
        self.use_barostat = bool(getattr(args, "use_barostat", False))
        self.pressure_bar = float(getattr(args, "pressure_bar", 1.0))
        self.barostat_frequency = int(getattr(args, "barostat_frequency", 25))

        # Snapshot folder + saved topology (used for explicit-solvent full-coordinate replay).
        self.snapshot_root: Optional[Path] = None
        snapshot_dir = getattr(args, "snapshot_dir", "")
        if snapshot_dir:
            snapshot_root = Path(snapshot_dir)
            if not snapshot_root.is_absolute():
                snapshot_root = Path.cwd() / snapshot_root
            snapshot_root.mkdir(parents=True, exist_ok=True)
            self.snapshot_root = snapshot_root
            self.topology_pdb_path = snapshot_root / "simulation_topology.pdb"
        else:
            self.topology_pdb_path = None

        # Simulation topology source:
        # - Default: use the provided protein PDB and solvate in OpenMM.
        # - Full-replay mode: if a previously saved solvated topology exists, reuse it
        #   so DCD frame atom ordering matches across restarts.
        pdb_path = args.pdb
        use_snapshot_topology = False
        snapshot_topology_was_stale = False
        if (
            self.explicit_solvent
            and self.explicit_solvent_replay_full_coords
            and self.topology_pdb_path is not None
            and self.topology_pdb_path.exists()
        ):
            # Sanity check: ensure the saved topology actually contains solvent.
            # If a prior run wrote a protein-only topology under this snapshot_dir,
            # reusing it would silently disable explicit solvation because we skip
            # Modeller.addSolvent() when use_snapshot_topology=True.
            has_solvent = False
            try:
                with self.topology_pdb_path.open("r", errors="ignore") as handle:
                    for line in handle:
                        if not (line.startswith("ATOM") or line.startswith("HETATM")):
                            continue
                        resname = line[17:20].strip().upper()
                        if resname in ("HOH", "WAT", "SOL", "TIP3", "TIP3P"):
                            has_solvent = True
                            break
            except Exception:
                has_solvent = False

            if has_solvent:
                pdb_path = str(self.topology_pdb_path)
                use_snapshot_topology = True
                print(
                    f"[INFO] Using existing simulation topology for explicit-solvent replay: {self.topology_pdb_path}",
                    flush=True,
                )
            else:
                snapshot_topology_was_stale = True
                print(
                    f"[WARN] Found existing topology PDB but it contains no solvent ({self.topology_pdb_path}). "
                    "Ignoring it and re-solvating, and will overwrite it with a new solvated topology.",
                    flush=True,
                )

        # Detect multiple conformers in the PDB file
        # IMPORTANT: Always check the ORIGINAL PDB path from args, not the topology path
        # The topology path might be a saved single-frame solvated structure
        self.pdb_conformers = []  # List of positions for each conformer
        self.has_multiple_conformers = False
        original_pdb_path = self._generator_args['pdb']  # Use original PDB from args
        try:
            import mdtraj as md
            pdb_traj = md.load(original_pdb_path)
            if pdb_traj.n_frames > 1:
                self.has_multiple_conformers = True
                print(f"[INFO] Detected {pdb_traj.n_frames} conformers in PDB file. Will use these for anchoring instead of generative model.")
                # Store all conformer positions (convert from nm to the units expected by OpenMM)
                for frame_idx in range(pdb_traj.n_frames):
                    # MDTraj uses nm, OpenMM PDBFile uses Angstroms internally but we'll match the topology
                    frame_positions = pdb_traj.xyz[frame_idx] * 10.0  # nm to Angstrom
                    self.pdb_conformers.append(frame_positions)
            else:
                print(f"[INFO] Single conformer detected in PDB file.")
        except Exception as e:
            print(f"[WARN] Could not check for multiple conformers with MDTraj: {e}")
            print(f"[WARN] Falling back to standard single-conformer loading.")

        self.pdb = PDBFile(pdb_path)
        
        # Configurable force fields
        ff_protein = str(getattr(args, "forcefield_protein", "amber14-all.xml"))
        ff_water = str(getattr(args, "forcefield_water", "amber14/tip3pfb.xml"))
        ff_implicit = str(getattr(args, "forcefield_implicit", "implicit/gbn2.xml"))
        
        if self.implicit_solvent:
            print(f"[INFO] Using implicit solvent ({ff_implicit}) at {self.temperature}K.")
            self.forcefield = ForceField(ff_protein, ff_implicit)
            self.ff_args = {'nonbondedMethod': NoCutoff, 'constraints': HBonds}
        else:
            print(f"[INFO] Using explicit solvent ({ff_protein} + {ff_water}) at {self.temperature}K.")
            self.forcefield = ForceField(ff_protein, ff_water)
            self.ff_args = {'nonbondedMethod': PME, 'constraints': HBonds, 'rigidWater': True}

        # Register GAFF template generator for ligand residues if needed.
        ligand_smiles = str(getattr(args, "ligand_smiles", "") or "").strip()
        if not ligand_smiles:
            # Auto-detect: check if topology has non-protein, non-solvent residues
            _non_std = {"HOH","WAT","TIP3","NA","CL","K","CA2","NA+","CL-"}
            _std_aa = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY",
                       "HIS","HID","HIE","HIP","ILE","LEU","LYS","MET","PHE",
                       "PRO","SER","THR","TRP","TYR","VAL","ACE","NME"}
            _has_unk = False
            for _r in self.pdb.topology.residues():
                if _r.name not in _std_aa and _r.name not in _non_std:
                    _has_unk = True
                    break
            if _has_unk:
                ligand_smiles = str(getattr(args, "ligand_smiles", "") or "").strip()
        if ligand_smiles:
            try:
                from openmmforcefields.generators import GAFFTemplateGenerator
                from openff.toolkit import Molecule
                lig_mol = Molecule.from_smiles(ligand_smiles, allow_undefined_stereo=True)
                gaff_gen = GAFFTemplateGenerator(molecules=[lig_mol], forcefield="gaff-2.11")
                self.forcefield.registerTemplateGenerator(gaff_gen.generator)
                print(f"[INFO] Registered GAFF-2.11 for ligand: {ligand_smiles}")
            except Exception as _gaff_err:
                print(f"[WARN] Failed to register GAFF for ligand '{ligand_smiles}': {_gaff_err}")

            cutoff_nm = self.nonbonded_cutoff_nm
            switch_nm = self.switch_distance_nm
            use_switch = bool(self.use_switching_function)
            if switch_nm is not None and (not use_switch):
                use_switch = True
            if cutoff_nm is not None:
                self.ff_args["nonbondedCutoff"] = float(cutoff_nm) * nanometer
            if use_switch:
                # OpenMM enables LJ switching via the `switchDistance` kwarg.
                # (There is no separate `useSwitchingFunction` kwarg in ForceField.createSystem.)
                if switch_nm is None:
                    if cutoff_nm is None:
                        raise ValueError(
                            "LJ switching requested but no cutoff is set. "
                            "Provide --nonbonded-cutoff-nm and --switch-distance-nm."
                        )
                    guessed = float(cutoff_nm) - 0.2
                    if guessed <= 0:
                        raise ValueError(
                            "LJ switching requested but cutoff is too small to infer a switch distance. "
                            "Provide --switch-distance-nm explicitly."
                        )
                    switch_nm = guessed
                self.ff_args["switchDistance"] = float(switch_nm) * nanometer
            if self.ewald_error_tolerance is not None:
                self.ff_args["ewaldErrorTolerance"] = float(self.ewald_error_tolerance)

            # Basic sanity checks when user explicitly sets values.
            if cutoff_nm is not None and switch_nm is not None and float(switch_nm) > float(cutoff_nm):
                raise ValueError(
                    f"switchDistance ({switch_nm} nm) must be <= nonbondedCutoff ({cutoff_nm} nm)."
                )

        self.solvent_padding = float(self.solvent_padding_nm) * nanometer
        self.ionic_strength = float(self.ionic_strength_m) * molar
        self._box_size = None
        if self.box_length_nm is not None:
            L = float(self.box_length_nm)
            if L <= 0:
                raise ValueError(f"--box-length-nm must be > 0; got {L}.")
            self._box_size = Vec3(L, L, L) * nanometer

        self._minimize_tolerance = None
        if self.minimize_tolerance_kj_mol_nm is not None:
            self._minimize_tolerance = float(self.minimize_tolerance_kj_mol_nm) * kilojoule_per_mole / nanometer

        if use_snapshot_topology and self.explicit_solvent:
            if self._box_size is not None or self.solvent_padding_nm != 1.0 or self.ionic_strength_m != 0.1:
                print(
                    "[WARN] Explicit-solvent replay is using an existing solvated topology; "
                    "box/ion settings (box-length/padding/ionic-strength/ion types) will be ignored. "
                    "Delete the saved topology PDB or pass --no-explicit-solvent-replay-full-coords to re-solvate.",
                    flush=True,
                )

        # Keep a clean protein-only topology to rebuild solvent around new conformations.
        self.clean_modeller = Modeller(self.pdb.topology, self.pdb.positions)
        self.clean_modeller.deleteWater()
        self.clean_topology = self.clean_modeller.topology

        # Store the clean topology for later use with different conformers
        self._clean_topology_for_conformers = self.clean_modeller.topology

        # Heavy atom indices are computed after system creation (below) to match
        # the actual simulation particle count, which may differ from the PDB
        # if ions were stripped/re-added or waters deleted.

        modeller = Modeller(self.pdb.topology, self.pdb.positions)
        if self.explicit_solvent and not use_snapshot_topology:
            solvent_kwargs = {
                "ionicStrength": self.ionic_strength,
                "positiveIon": self.solvent_positive_ion,
                "negativeIon": self.solvent_negative_ion,
                "neutralize": self.solvent_neutralize,
            }
            if self._box_size is not None:
                solvent_kwargs["boxSize"] = self._box_size
            else:
                solvent_kwargs["padding"] = self.solvent_padding
            modeller.addSolvent(self.forcefield, **solvent_kwargs)
        if self.explicit_solvent:
            try:
                water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
                n_waters = sum(1 for res in modeller.topology.residues() if str(res.name).upper() in water_names)
                n_atoms = int(modeller.topology.getNumAtoms())
                print(f"[INFO] Explicit-solvent topology: n_atoms={n_atoms}, n_waters={int(n_waters)}", flush=True)
            except Exception:
                pass
        # Strip unparameterized ions (e.g. CA2 for Ca²⁺) before createSystem,
        # then add them back with manual nonbonded parameters.
        _ion_residues = [r for r in modeller.topology.residues() if r.name == "CA2"]
        _ion_positions = {}
        if _ion_residues:
            from openmm.app import Element as _Elem
            from openmm import NonbondedForce as _NBF
            for _ir in _ion_residues:
                for _ia in _ir.atoms():
                    _ion_positions[_ir.name] = modeller.positions[_ia.index]
            modeller.delete(_ion_residues)
            print(f"[INFO] Stripped {len(_ion_residues)} ion residue(s) for parameterization")

        system = self.forcefield.createSystem(modeller.topology, **self.ff_args)

        # Re-add stripped ions with AMBER parameters
        if _ion_positions:
            from openmm.app import Element as _Elem
            from openmm import NonbondedForce as _NBF
            _ion_params = {
                "CA2": (2.0, 0.2412, 0.459789),  # charge(e), sigma(nm), epsilon(kJ/mol)
            }
            for _iname, _ipos in _ion_positions.items():
                _q, _sig, _eps = _ion_params.get(_iname, (0.0, 0.3, 0.0))
                system.addParticle(40.078 if _iname == "CA2" else 1.0)
                for _f in system.getForces():
                    if isinstance(_f, _NBF):
                        from openmm.unit import elementary_charge as _ec
                        _f.addParticle(
                            _q * _ec, _sig * nanometer, _eps * kilojoule_per_mole,
                        )
                        break
                _chain = modeller.topology.addChain()
                _res = modeller.topology.addResidue(_iname, _chain)
                modeller.topology.addAtom("CA", _Elem.getBySymbol("Ca"), _res)
                _new_pos = list(modeller.positions)
                _new_pos.append(_ipos)
                modeller.positions = _new_pos
            print(f"[INFO] Re-added {len(_ion_positions)} ion(s) with manual parameters")

        # Compute heavy atom indices from the actual simulation topology.
        try:
            _sim_md_top = md.Topology.from_openmm(modeller.topology)
            heavy_atoms = _sim_md_top.select("not element H")
            self.heavy_atom_indices = heavy_atoms
            _pos_nm = np.array(
                [[p.x, p.y, p.z] for p in modeller.positions.value_in_unit(nanometer)],
                dtype=np.float32,
            )
            self.reference_heavy_angstrom = _pos_nm[heavy_atoms] * 10.0
        except Exception as _ha_err:
            print(f"[WARN] Could not extract heavy atoms from simulation topology: {_ha_err}")
            self.heavy_atom_indices = np.array([], dtype=np.int64)
            self.reference_heavy_angstrom = np.array([]).reshape(0, 3).astype(np.float32)

        # Explicit solvent: enforce PME grid spacing and ensure long-range dispersion correction.
        if self.explicit_solvent:
            try:
                pme_spacing_a = getattr(args, "pme_grid_spacing_a", None)
            except Exception:
                pme_spacing_a = None
            try:
                pme_spacing_a = float(pme_spacing_a) if pme_spacing_a is not None else None
            except Exception:
                pme_spacing_a = None
            if pme_spacing_a is not None and np.isfinite(pme_spacing_a) and float(pme_spacing_a) > 0:
                try:
                    nb_force = None
                    for fi in range(int(system.getNumForces())):
                        f = system.getForce(int(fi))
                        if f.__class__.__name__ == "NonbondedForce":
                            nb_force = f
                            break
                    if nb_force is not None:
                        # Always enable dispersion correction (long-range vdW continuum correction).
                        try:
                            nb_force.setUseDispersionCorrection(True)
                        except Exception:
                            pass

                        # Determine cubic box edge length (nm).
                        a_nm = None
                        try:
                            vecs = modeller.topology.getPeriodicBoxVectors()
                            if vecs is not None:
                                a_nm = float(vecs[0][0].value_in_unit(nanometer))
                        except Exception:
                            a_nm = None

                        if a_nm is not None and a_nm > 0:
                            # Target dims from requested spacing (Å).
                            target = int(np.ceil((a_nm * 10.0) / float(pme_spacing_a)))
                            n_grid = _next_fft_friendly(target)

                            cutoff_nm = float(self.nonbonded_cutoff_nm) if self.nonbonded_cutoff_nm is not None else 1.2
                            tol = float(self.ewald_error_tolerance) if self.ewald_error_tolerance is not None else 5e-4
                            alpha_nm_inv = _pme_alpha_from_tol(cutoff_nm, tol)
                            nb_force.setPMEParameters(alpha_nm_inv / nanometer, int(n_grid), int(n_grid), int(n_grid))

                            try:
                                actual_a = (a_nm * 10.0) / float(n_grid)
                                print(
                                    f"[PME] Set PME grid to {n_grid}^3 for box ~{a_nm*10.0:.1f} Å "
                                    f"(spacing≈{actual_a:.3f} Å), alpha≈{alpha_nm_inv:.3f} 1/nm, "
                                    f"cutoff={cutoff_nm:.3f} nm, tol={tol:g}",
                                    flush=True,
                                )
                            except Exception:
                                pass
                except Exception as exc:
                    print(f"[PME] Warning: failed to set PME grid spacing ({exc})", flush=True)

        if self.use_barostat and (not self.implicit_solvent):
            system.addForce(MonteCarloBarostat(self.pressure_bar * bar, self.temperature * kelvin, self.barostat_frequency))

        # Flat-bottom ligand-pocket distance restraint (optional).
        _lig_pocket_restraint = bool(getattr(args, "ligand_pocket_restraint", False))
        if _lig_pocket_restraint:
            try:
                _lig_r0_nm = float(getattr(args, "ligand_pocket_restraint_radius_nm", 0.6))
                _lig_k = float(getattr(args, "ligand_pocket_restraint_k", 1000.0))
                _lig_seq_pos = int(getattr(args, "ligand_pocket_seq_pos", 170))
                _LIG_NAMES_R = {"UNK", "BEN", "LIG"}

                _md_top_r = md.Topology.from_openmm(modeller.topology)
                _prot_res_r = [r for r in _md_top_r.residues if r.is_protein]

                _lig_heavy_r = [a.index for a in _md_top_r.atoms
                                if a.residue.name in _LIG_NAMES_R and a.element.symbol != "H"]
                _pocket_resseq_r = _prot_res_r[_lig_seq_pos].resSeq if _lig_seq_pos < len(_prot_res_r) else None
                _pocket_heavy_r = [a.index for a in _md_top_r.atoms
                                   if _pocket_resseq_r is not None
                                   and a.residue.resSeq == _pocket_resseq_r
                                   and a.element.symbol != "H"] if _pocket_resseq_r else []

                if _lig_heavy_r and _pocket_heavy_r:
                    from openmm import CustomCentroidBondForce
                    _fb_force = CustomCentroidBondForce(
                        2, "steer_lambda_lig * step(distance(g1,g2)-r0) * 0.5 * k * (distance(g1,g2)-r0)^2"
                    )
                    _fb_force.addGlobalParameter("steer_lambda_lig", 1.0)
                    _fb_force.addGlobalParameter("r0", _lig_r0_nm)
                    _fb_force.addGlobalParameter("k", _lig_k)
                    _g1 = _fb_force.addGroup(_lig_heavy_r)
                    _g2 = _fb_force.addGroup(_pocket_heavy_r)
                    _fb_force.addBond([_g1, _g2])
                    _fb_force.setUsesPeriodicBoundaryConditions(True)
                    system.addForce(_fb_force)
                    self._lig_restraint_lambda_name = "steer_lambda_lig"
                    print(
                        f"[Ligand Restraint] Flat-bottom COM restraint (lambda-toggled): "
                        f"lig({len(_lig_heavy_r)} atoms) ↔ pocket({len(_pocket_heavy_r)} atoms, "
                        f"resSeq={_pocket_resseq_r}), r0={_lig_r0_nm:.2f} nm, k={_lig_k:.0f} kJ/mol/nm²",
                        flush=True,
                    )
                else:
                    print(f"[Ligand Restraint] Warning: could not find ligand ({len(_lig_heavy_r)}) "
                          f"or pocket ({len(_pocket_heavy_r)}) atoms; restraint NOT added.", flush=True)
            except Exception as _exc_lig_r:
                print(f"[Ligand Restraint] Warning: failed to add restraint: {_exc_lig_r}", flush=True)

        # Reference positions (nm) used for steering (and for explicit-solvent full-coordinate replay baseline).
        try:
            init_pos_nm = np.array(modeller.getPositions().value_in_unit(nanometer), dtype=np.float32)
            self._ref_positions_nm_full = init_pos_nm.reshape(-1, 3)
        except Exception:
            self._ref_positions_nm_full = None

        # Build mdtraj topology and CV definitions before creating a Context so we can attach
        # optional forces (e.g., replay steering) without requiring Context.reinitialize().
        self.md_topology = md.Topology.from_openmm(modeller.topology)
        self.cv_definitions = find_all_backbone_dihedrals(modeller.topology)
        self.n_cvs = len(self.cv_definitions)
        # Replay steering torsions: always include backbone φ/ψ; optionally include sidechain χ torsions.
        self._steer_cv_backbone_len = int(self.n_cvs)
        self._steer_chi_definitions = []
        try:
            self._steer_chi_definitions = find_all_sidechain_chi_dihedrals(modeller.topology, max_chi=4)
        except Exception:
            self._steer_chi_definitions = []
        self._steer_cv_definitions = list(self.cv_definitions) + list(self._steer_chi_definitions)

        # Add ligand rotatable bond dihedrals to steering CVs.
        # Prefer SMILES-based RDKit detection when ligand_smiles is available
        # (gives all RDKit-standard rotatable bonds; allows H as 4th atom).
        # Fall back to legacy heavy-atom-only detection if no SMILES.
        _lig_smiles = str(getattr(args, "ligand_smiles", "") or "")
        if _lig_smiles:
            from gespo.utils.geometry import find_ligand_rotatable_dihedrals_smiles
            try:
                self._steer_ligand_definitions = find_ligand_rotatable_dihedrals_smiles(
                    self.md_topology, _lig_smiles
                )
            except Exception as _exc_lig:
                print(f"[Env] Warning: SMILES rotatable bond detection failed ({_exc_lig}); falling back to heavy-atom-only.")
                from gespo.utils.geometry import find_ligand_rotatable_dihedrals
                self._steer_ligand_definitions = find_ligand_rotatable_dihedrals(self.md_topology)
        else:
            from gespo.utils.geometry import find_ligand_rotatable_dihedrals
            self._steer_ligand_definitions = find_ligand_rotatable_dihedrals(self.md_topology)
        if self._steer_ligand_definitions:
            self._steer_cv_definitions = self._steer_cv_definitions + list(self._steer_ligand_definitions)
            print(f"[Env] Added {len(self._steer_ligand_definitions)} ligand rotatable dihedral(s) to steering CVs")
            for _lbl, _atms in self._steer_ligand_definitions:
                print(f"  [Env]   {_lbl}: atoms {_atms}")

        # Cached targeted-MD steering forces for replay resets (optional).
        # Pre-install these forces into `system` before context creation to avoid CUDA stalls
        # from Context.reinitialize() or creating a second CUDA context mid-run.
        self._replay_steer_forces = None
        if self.steer_replay_starts and (
            self.implicit_solvent or (self.explicit_solvent and self.explicit_solvent_replay_full_coords)
        ):
            try:
                if self.steer_replay_verbose:
                    print("[TargetedMD] Preinstall steering forces into System...", flush=True)
                ref_nm = getattr(self, "_ref_positions_nm_full", None)
                if ref_nm is not None:
                    ref_nm = np.asarray(ref_nm, dtype=np.float64).reshape(-1, 3)
                if ref_nm is not None and int(system.getNumParticles()) == int(ref_nm.shape[0]):
                    n_particles = int(system.getNumParticles())
                    # CA indices ordered by residue slot (computed below); for now, build from topology.
                    ca_indices = self.md_topology.select("protein and name CA")
                    if ca_indices is None or len(ca_indices) == 0:
                        # Fallback: may be a nonstandard topology without 'protein' flag.
                        ca_indices = self.md_topology.select("name CA")
                    ca_idx = [int(i) for i in np.asarray(ca_indices, dtype=np.int64).reshape(-1) if int(i) >= 0]
                    e2e_pair = (int(ca_idx[0]), int(ca_idx[-1])) if len(ca_idx) >= 2 else None

                    # RMSD steering: prefer protein heavy atoms (non-H) so the RMSD steering forces act
                    # on all heavy atoms rather than only Cα.
                    rmsd_idx = None
                    rmsd_ref_nm = None
                    try:
                        heavy_sel_expr = None
                        heavy_idx: Optional[List[int]] = None
                        # MDTraj selection language can vary across versions; be permissive.
                        for expr in (
                            "protein and not element H",
                            "protein and not hydrogen",
                            "protein and not name H*",
                        ):
                            try:
                                sel = self.md_topology.select(expr)
                                idx = [int(i) for i in np.asarray(sel, dtype=np.int64).reshape(-1) if int(i) >= 0]
                                if len(idx) >= 3:
                                    heavy_sel_expr = expr
                                    heavy_idx = idx
                                    break
                            except Exception:
                                continue

                        # Fallback: if MDTraj doesn't classify residues as 'protein' (e.g., nonstandard names),
                        # approximate protein heavy atoms by excluding common solvent/ions.
                        if heavy_idx is None:
                            water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
                            ion_names = {
                                "NA",
                                "NA+",
                                "CL",
                                "CL-",
                                "K",
                                "K+",
                                "LI",
                                "LI+",
                                "RB",
                                "RB+",
                                "CS",
                                "CS+",
                                "MG",
                                "MG2+",
                                "CA",
                                "CA2+",
                                "ZN",
                                "ZN2+",
                            }
                            try:
                                from ..utils.selection import LIPID_RESNAMES

                                lipid_names = set(str(x).upper() for x in LIPID_RESNAMES)
                            except Exception:
                                lipid_names = {"POP", "POPC", "POPE", "POPG", "POPS", "DPPC", "CHOL"}
                            idx = []
                            for res in self.md_topology.residues:
                                rname = str(getattr(res, "name", "") or "").strip().upper()
                                if rname in water_names or rname in ion_names or rname in lipid_names:
                                    continue
                                for atom in res.atoms:
                                    aidx = int(atom.index)
                                    try:
                                        elem = getattr(atom, "element", None)
                                        if elem is not None and getattr(elem, "symbol", "") == "H":
                                            continue
                                    except Exception:
                                        pass
                                    aname = str(getattr(atom, "name", "") or "")
                                    if aname.startswith("H"):
                                        continue
                                    idx.append(aidx)
                            if len(idx) >= 3:
                                heavy_sel_expr = "manual(non-water/non-ion heavy)"
                                heavy_idx = idx

                        if heavy_idx is not None and len(heavy_idx) >= 3:
                            rmsd_idx = heavy_idx
                            if self.steer_replay_verbose and heavy_sel_expr:
                                print(
                                    f"[TargetedMD] RMSD subset selection: {heavy_sel_expr} (n={len(heavy_idx)})",
                                    flush=True,
                                )
                            try:
                                rmsd_ref_nm = ref_nm[np.asarray(rmsd_idx, dtype=np.int64)]
                            except Exception:
                                rmsd_ref_nm = None
                    except Exception:
                        rmsd_idx = None
                        rmsd_ref_nm = None

                    # Fallback: if heavy-atom selection fails, use Cα indices.
                    if rmsd_idx is None and ca_idx and len(ca_idx) >= 3:
                        rmsd_idx = list(ca_idx)
                        try:
                            rmsd_ref_nm = ref_nm[np.asarray(rmsd_idx, dtype=np.int64)]
                        except Exception:
                            rmsd_ref_nm = None
                        if self.steer_replay_verbose:
                            print(
                                f"[TargetedMD] RMSD subset selection: protein and name CA (n={len(rmsd_idx)})",
                                flush=True,
                            )

                    # Prefer the original endpoint-A PDB (args.pdb) as the RMSD reference when available,
                    # even if the Simulation is using a saved solvated topology PDB for replay.
                    try:
                        pdb_a_path = str(getattr(self, "_generator_args", {}).get("pdb", "") or "").strip()
                    except Exception:
                        pdb_a_path = ""
                    if pdb_a_path and rmsd_idx is not None and len(rmsd_idx) >= 3:
                        try:
                            tr_a = md.load(pdb_a_path)
                            if int(tr_a.n_frames) > 0:
                                xyz_a = np.asarray(tr_a.xyz[0], dtype=np.float32)  # nm
                                map_a = {
                                    (int(atom.residue.index), str(atom.name)): xyz_a[int(atom.index)]
                                    for atom in tr_a.topology.atoms
                                }
                                idx_keep = []
                                ref_a = []
                                for ai in rmsd_idx:
                                    try:
                                        atom = self.md_topology.atom(int(ai))
                                    except Exception:
                                        continue
                                    key = (int(atom.residue.index), str(atom.name))
                                    pa = map_a.get(key)
                                    if pa is None:
                                        continue
                                    idx_keep.append(int(ai))
                                    ref_a.append(pa)
                                if len(idx_keep) >= 3:
                                    rmsd_idx = idx_keep
                                    rmsd_ref_nm = np.asarray(ref_a, dtype=np.float64).reshape(-1, 3)
                                    if self.steer_replay_verbose:
                                        print(
                                            f"[TargetedMD] RMSD reference: endpoint-A PDB ({pdb_a_path})",
                                            flush=True,
                                        )
                        except Exception:
                            pass

                    k_rmsd = float(self.steer_k_rmsd)
                    k_rg = float(self.steer_k_rg)
                    k_dist = float(self.steer_k_dist)
                    k_theta = float(self.steer_k_theta)

                    # Optional extra distance/angle steering targets (system-specific).
                    # NOTE: must be defined before computing *_to_use.
                    extra_distance_pairs: Dict[str, Tuple[int, int]] = {}
                    extra_angle_triplets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

                    # --- ADK-specific extras ---
                    try:
                        adk_idx = build_adk_token_indices(self.md_topology, protein_indices=None)
                        if "dist_145_52_nm" in adk_idx.pairs:
                            extra_distance_pairs["dist_145_52_nm"] = adk_idx.pairs["dist_145_52_nm"]
                        if "dist_169_55_nm" in adk_idx.pairs:
                            extra_distance_pairs["dist_169_55_nm"] = adk_idx.pairs["dist_169_55_nm"]

                        g_core_lid = adk_idx.group_atoms.get("core_lid", np.zeros((0,), dtype=np.int64))
                        g_core1 = adk_idx.group_atoms.get("core1", np.zeros((0,), dtype=np.int64))
                        g_nmp = adk_idx.group_atoms.get("nmp", np.zeros((0,), dtype=np.int64))
                        g_core2 = adk_idx.group_atoms.get("core2", np.zeros((0,), dtype=np.int64))
                        g_lid = adk_idx.group_atoms.get("lid", np.zeros((0,), dtype=np.int64))
                        if g_core_lid.size and g_core1.size and g_nmp.size:
                            extra_angle_triplets["theta_nmp_rad"] = (g_core_lid, g_core1, g_nmp)
                        if g_core2.size and g_core_lid.size and g_lid.size:
                            extra_angle_triplets["theta_lid_rad"] = (g_core2, g_core_lid, g_lid)
                    except Exception:
                        pass

                    # --- Vinculin H5 extras ---
                    # Detect vinculin H5 tokens in the global token names.
                    _h5_token_names_in_use = set()
                    try:
                        from gespo.utils.vinculin_h5_metrics import VINCULIN_H5_TOKEN_NAMES
                        _h5_token_names_in_use = set(VINCULIN_H5_TOKEN_NAMES) & set(
                            getattr(self, "_global_token_names", []) or []
                        )
                    except Exception:
                        pass

                    if _h5_token_names_in_use:
                        try:
                            from gespo.utils.vinculin_h5_metrics import H4, H5, BUNDLE, H1
                            # Find protein CA atom indices in the simulation topology.
                            _ca_sim = []
                            for atom in self.md_topology.atoms:
                                if str(atom.name) == "CA" and hasattr(atom.residue, "is_protein"):
                                    if atom.residue.is_protein:
                                        _ca_sim.append(int(atom.index))
                            if not _ca_sim:
                                # Fallback: select by element name
                                _ca_sim = [int(a.index) for a in self.md_topology.atoms
                                           if str(a.name) == "CA" and str(a.element.symbol) == "C"]
                            _ca_sim = _ca_sim[:180]  # first 180 CA = vinculin Vt

                            if len(_ca_sim) >= 163:  # need at least up to H5 end (index 162)
                                # 1. H4-H5 linker e2e: CA distance between last H4 residue and first H5 residue
                                if "h4h5_linker_e2e_nm" in _h5_token_names_in_use:
                                    h4_last_ca = _ca_sim[H4[-1]]   # res index 123
                                    h5_first_ca = _ca_sim[H5[0]]   # res index 131
                                    extra_distance_pairs["h4h5_linker_e2e_nm"] = (h4_last_ca, h5_first_ca)
                                    if self.steer_replay_verbose:
                                        print(f"[TargetedMD] Vinculin H5: linker e2e atoms ({h4_last_ca}, {h5_first_ca})", flush=True)

                                # 2. H5-bundle COM distance: added directly as CustomCentroidBondForce
                                #    (cannot use extra_distance_pairs which is atom-atom only)
                                if "h5_bundle_dist_nm" in _h5_token_names_in_use and float(k_dist) > 0.0:
                                    h5_ca_atoms = [_ca_sim[i] for i in H5]
                                    bundle_ca_atoms = [_ca_sim[i] for i in BUNDLE if i < len(_ca_sim)]
                                    if h5_ca_atoms and bundle_ca_atoms:
                                        try:
                                            from openmm import CustomCentroidBondForce
                                            h5_com_bias = CustomCentroidBondForce(
                                                2,
                                                "0.5*steer_lambda_g*steer_k_h5bd*(distance(g1,g2) - steer_h5bd_r0)^2",
                                            )
                                            h5_com_bias.addGlobalParameter("steer_lambda_g", 0.0)
                                            h5_com_bias.addGlobalParameter("steer_k_h5bd", float(k_dist))
                                            h5_com_bias.addGlobalParameter("steer_h5bd_r0", 0.0)
                                            grp_h5 = h5_com_bias.addGroup(h5_ca_atoms)
                                            grp_bundle = h5_com_bias.addGroup(bundle_ca_atoms)
                                            h5_com_bias.addBond([grp_h5, grp_bundle], [])
                                            h5_com_bias.setForceGroup(2)  # STEER_GLOBAL_FORCE_GROUP_ID
                                            self._openmm_system.addForce(h5_com_bias)
                                            # Store for target updates in steer_replay_to_targets
                                            if not hasattr(self, "_h5_com_bias"):
                                                self._h5_com_bias = h5_com_bias
                                            if self.steer_replay_verbose:
                                                print(f"[TargetedMD] Vinculin H5: bundle COM distance force "
                                                      f"(H5={len(h5_ca_atoms)} atoms, bundle={len(bundle_ca_atoms)} atoms)", flush=True)
                                        except Exception as exc:
                                            if self.steer_replay_verbose:
                                                print(f"[TargetedMD] Vinculin H5: COM distance force failed: {exc}", flush=True)
                            else:
                                if self.steer_replay_verbose:
                                    print(f"[TargetedMD] Vinculin H5: insufficient CA atoms ({len(_ca_sim)} < 163), skipping H5 steering", flush=True)
                        except Exception as exc:
                            if self.steer_replay_verbose:
                                print(f"[TargetedMD] Vinculin H5 steering setup failed: {exc}", flush=True)

                    extra_distance_pairs_to_use = extra_distance_pairs if extra_distance_pairs else None
                    extra_angle_triplets_to_use = extra_angle_triplets if extra_angle_triplets else None

                    if (
                        (not self.implicit_solvent)
                        and self.explicit_solvent
                        and n_particles >= 20000
                        and (not self.steer_allow_large_explicit_extra_globals)
                    ):
                        # Keep RMSD steering enabled, but avoid compiling full-system RMSD references.
                        if k_rmsd > 0.0 and self.steer_replay_verbose:
                            try:
                                n_rmsd = len(rmsd_idx) if rmsd_idx is not None else 0
                            except Exception:
                                n_rmsd = 0
                            if n_rmsd >= 3:
                                print(
                                    f"[TargetedMD] RMSD steering (explicit large): using subset atoms (n={n_rmsd}) instead of full-system reference.",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[TargetedMD] RMSD steering (explicit large): subset selection unavailable; disabling RMSD (N={n_particles}).",
                                    flush=True,
                                )
                                k_rmsd = 0.0
                        # Rg + centroid/angle extras create larger/more complex kernels. Disable them for
                        # large explicit-solvent systems to avoid CUDA context creation stalls.
                        if k_rg > 0.0 and self.steer_replay_verbose:
                            print(
                                f"[TargetedMD] Disabling Rg steering for large explicit-solvent system (N={n_particles}).",
                                flush=True,
                            )
                        k_rg = 0.0
                        if (extra_distance_pairs_to_use is not None or extra_angle_triplets_to_use is not None) and self.steer_replay_verbose:
                            print(
                                "[TargetedMD] Disabling extra distance/angle steering (large explicit-solvent). "
                                "Override with --steer-allow-large-explicit-extra-globals.",
                                flush=True,
                            )
                        extra_distance_pairs_to_use = None
                        extra_angle_triplets_to_use = None
                        k_dist = 0.0
                        k_theta = 0.0

                    # Path-sampling globals may include RMSD-to-endpoint-A/B tokens (rmsd_a_nm, rmsd_b_nm).
                    # Precompute subset references (matching the RMSD steering subset) so targeted MD can steer both metrics.
                    rmsd_ref_a_nm = None
                    rmsd_ref_b_nm = None
                    try:
                        pdb_b_path = str(getattr(args, "pdb_b", "") or "").strip()
                    except Exception:
                        pdb_b_path = ""
                    if pdb_b_path:
                        try:
                            pdb_a_path = str(getattr(self, "_generator_args", {}).get("pdb", "") or "").strip()
                            if pdb_a_path and rmsd_idx is not None and len(rmsd_idx) >= 3:
                                tr_a = md.load(pdb_a_path)
                                tr_b = md.load(pdb_b_path)
                                if int(tr_a.n_frames) > 0 and int(tr_b.n_frames) > 0:
                                    xyz_a = np.asarray(tr_a.xyz[0], dtype=np.float32)  # nm
                                    xyz_b = np.asarray(tr_b.xyz[0], dtype=np.float32)  # nm
                                    map_a = {(atom.residue.index, str(atom.name)): xyz_a[int(atom.index)] for atom in tr_a.topology.atoms}
                                    map_b = {(atom.residue.index, str(atom.name)): xyz_b[int(atom.index)] for atom in tr_b.topology.atoms}

                                    idx_keep = []
                                    ref_a = []
                                    ref_b = []
                                    for ai in rmsd_idx:
                                        try:
                                            atom = self.md_topology.atom(int(ai))
                                        except Exception:
                                            continue
                                        key = (int(atom.residue.index), str(atom.name))
                                        pa = map_a.get(key)
                                        pb = map_b.get(key)
                                        if pa is None or pb is None:
                                            continue
                                        idx_keep.append(int(ai))
                                        ref_a.append(pa)
                                        ref_b.append(pb)

                                    if len(idx_keep) >= 3:
                                        rmsd_idx = idx_keep
                                        rmsd_ref_a_nm = np.asarray(ref_a, dtype=np.float64).reshape(-1, 3)
                                        rmsd_ref_b_nm = np.asarray(ref_b, dtype=np.float64).reshape(-1, 3)
                                        # Keep the legacy RMSD reference aligned with endpoint-A when available.
                                        rmsd_ref_nm = rmsd_ref_a_nm
                        except Exception:
                            rmsd_ref_a_nm = None
                            rmsd_ref_b_nm = None

                    self._replay_steer_forces = add_replay_steering_forces(
                        system,
                        ref_positions_nm=ref_nm,
                        ca_particle_indices=ca_idx,
                        rmsd_particle_indices=rmsd_idx,
                        rmsd_ref_positions_nm=rmsd_ref_nm,
                        rmsd_ref_positions_a_nm=rmsd_ref_a_nm,
                        rmsd_ref_positions_b_nm=rmsd_ref_b_nm,
                        e2e_atoms=e2e_pair,
                        cv_definitions=getattr(self, "_steer_cv_definitions", self.cv_definitions),
                        cv_targets_rad=[0.0] * int(len(getattr(self, "_steer_cv_definitions", self.cv_definitions))),
                        # Placeholder weights at preinstall; per-reset _update_replay_steering_torsion_targets
                        # overrides these with the correct chi/ligand weights from steer_weights.
                        cv_weights=([1.0] * int(len(self.cv_definitions)))
                        + ([0.0] * int(len(getattr(self, "_steer_chi_definitions", []))))
                        + ([1.0] * int(len(getattr(self, "_steer_ligand_definitions", [])))),
                        k_rmsd=float(k_rmsd),
                        k_e2e=float(self.steer_k_e2e),
                        k_torsion=float(self.steer_k_torsion),
                        k_rg=float(k_rg),
                        extra_distance_pairs=extra_distance_pairs_to_use,
                        k_dist=float(k_dist),
                        extra_angle_triplets=extra_angle_triplets_to_use,
                        k_theta=float(k_theta),
                    )
                if self.steer_replay_verbose:
                    print(
                        f"[TargetedMD] Preinstall steering forces: {'OK' if self._replay_steer_forces is not None else 'SKIPPED'}",
                        flush=True,
                    )
            except Exception as exc:
                if self.steer_replay_verbose:
                    print(f"[TargetedMD] Warning: failed to preinstall replay steering forces ({exc})", flush=True)
                self._replay_steer_forces = None
        integrator = LangevinIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            float(self.timestep_fs) * femtoseconds,
        )
        if self.steer_replay_verbose:
            try:
                print(
                    f"[OpenMM] Creating Simulation/Context (platform={self.platform_name}, n_particles={int(system.getNumParticles())})...",
                    flush=True,
                )
            except Exception:
                print(f"[OpenMM] Creating Simulation/Context (platform={self.platform_name})...", flush=True)
        self.sim, chosen_platform = _build_simulation_or_fallback(
            modeller.topology, system, integrator, self.platform_name, self.platform_device
        )
        if self.steer_replay_verbose:
            print(f"[OpenMM] Simulation created (platform={chosen_platform}).", flush=True)
        if chosen_platform != self.platform_name:
            self.platform_name = chosen_platform
        self.sim.context.setPositions(modeller.getPositions())
        try:
            if self._minimize_tolerance is not None:
                self.sim.minimizeEnergy(tolerance=self._minimize_tolerance, maxIterations=50)
            else:
                self.sim.minimizeEnergy(maxIterations=50)
        except TypeError:
            self.sim.minimizeEnergy(maxIterations=50)
        # Reference positions (nm) used for replay-reset steering.
        # (Precomputed above from modeller positions.)

        if self.topology_pdb_path is not None and (
            snapshot_topology_was_stale or (not self.topology_pdb_path.exists())
        ):
            try:
                with self.topology_pdb_path.open("w") as handle:
                    positions = self.sim.context.getState(getPositions=True).getPositions()
                    PDBFile.writeFile(self.sim.topology, positions, handle, keepIds=True)
                print(f"[INFO] Wrote simulation topology to {self.topology_pdb_path}")
            except Exception as exc:
                print(f"[INFO] Warning: failed to write simulation topology ({exc})")

        # CV definitions were built before context creation.
        # Build residue maps and CA indices up-front (used by multiple modes)
        residue_ids = sorted({int(name.split('_')[1]) for name, _ in self.cv_definitions})
        self.residue_ids = residue_ids
        self.n_residues = len(residue_ids)
        self._residue_index_map = {rid: idx for idx, rid in enumerate(residue_ids)}
        self.cv_is_phi = np.zeros(self.n_cvs, dtype=bool)
        self.cv_residue_slots = np.zeros(self.n_cvs, dtype=np.int64)
        for idx, (name, _) in enumerate(self.cv_definitions):
            kind, res_str = name.split('_')
            slot = self._residue_index_map[int(res_str)]
            self.cv_residue_slots[idx] = slot
            self.cv_is_phi[idx] = kind == 'phi'
        # Map residues (in residue_ids order) to their CA atom indices in the current topology
        self.ca_atom_indices = np.full((self.n_residues,), -1, dtype=np.int64)
        for res in self.md_topology.residues:
            rid1 = res.index + 1  # 1-based index to match residue_ids
            if rid1 not in self._residue_index_map:
                continue
            slot = self._residue_index_map[rid1]
            for atom in res.atoms:
                if atom.name == 'CA':
                    self.ca_atom_indices[slot] = atom.index
                    break
        # Convenience: end-to-end CA pair for the global E2E metric/steering.
        try:
            valid_ca = self.ca_atom_indices[self.ca_atom_indices >= 0]
            self._e2e_atom_pair = (int(valid_ca[0]), int(valid_ca[-1])) if valid_ca.size >= 2 else None
        except Exception:
            self._e2e_atom_pair = None
        # Determine state feature dimension based on policy mode and available pairwise metadata
        pair_dim = 0
        if self.generator is not None:
            try:
                if hasattr(self.generator, "pair_feature_dim") and isinstance(self.generator.pair_feature_dim, int):
                    pair_dim = max(0, int(self.generator.pair_feature_dim))
                if getattr(self.generator, "conditioning_mode", "torsion") == "hybrid":
                    # In hybrid mode, use the explicit hybrid pair dim if set
                    if hasattr(self.generator, "hybrid_pair_dim") and int(self.generator.hybrid_pair_dim) > 0:
                        pair_dim = int(self.generator.hybrid_pair_dim)
            except Exception:
                pair_dim = 0

        if self.policy_mode == "torsion":
            self.state_dim = self.n_cvs * 3
        elif self.policy_mode == "pairwise":
            if pair_dim <= 0:
                print("[WARN] Pairwise policy mode requested but dataset has no pairwise features; falling back to torsion mode.")
                self.policy_mode = "torsion"
                self.state_dim = self.n_cvs * 3
            else:
                self.state_dim = pair_dim
        elif self.policy_mode == "hybrid":
            if pair_dim <= 0:
                print("[WARN] Hybrid policy mode requested but dataset has no pairwise features; using torsion features only.")
                self.state_dim = self.n_cvs * 3
            else:
                self.state_dim = self.n_cvs * 3 + pair_dim
        elif self.policy_mode == "ca":
            # Flattened aligned Cα coordinates (nm)
            self.state_dim = self.n_residues * 3
        elif self.policy_mode == "ca_torsion":
            # Aligned Cα (nm) + torsion sin/cos
            self.state_dim = self.n_residues * 3 + 2 * self.n_cvs
        else:
            print(f"[WARN] Unknown policy mode '{self.policy_mode}'; defaulting to torsion.")
            self.policy_mode = "torsion"
            self.state_dim = self.n_cvs * 3
        # Capture reference CA coordinates (Å) ordered by residue slot for alignment
        try:
            state = self.sim.context.getState(getPositions=True)
            ref_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            ref_ca = []
            for idx in self.ca_atom_indices:
                if idx >= 0:
                    ref_ca.append(ref_nm[int(idx)] * 10.0)
                else:
                    ref_ca.append(np.zeros(3, dtype=np.float32))
            self._ref_ca_angstrom = np.asarray(ref_ca, dtype=np.float32)
            # Valid CA mask, reference center and scale for normalization
            self._valid_ca_mask = self.ca_atom_indices >= 0
            if np.any(self._valid_ca_mask):
                ref_valid = self._ref_ca_angstrom[self._valid_ca_mask]
                self._ref_ca_center = ref_valid.mean(axis=0).astype(np.float32)
                ref_centered = ref_valid - self._ref_ca_center
                # Scalar RMS radius to keep isotropic scaling
                self._ref_ca_scale = float(np.sqrt(np.mean(np.sum(ref_centered**2, axis=1))))
                if not np.isfinite(self._ref_ca_scale) or self._ref_ca_scale < 1e-6:
                    self._ref_ca_scale = 1.0
                # Also keep nm-space versions for CA (used by 'ca' policy mode)
                self._ref_ca_nm = (self._ref_ca_angstrom / 10.0).astype(np.float32)
                self._ref_ca_center_nm = (self._ref_ca_center / 10.0).astype(np.float32)
                self._ref_ca_scale_nm = float(self._ref_ca_scale / 10.0) if self._ref_ca_scale > 0 else 1.0
            else:
                self._valid_ca_mask = np.zeros((self.n_residues,), dtype=bool)
                self._ref_ca_center = np.zeros(3, dtype=np.float32)
                self._ref_ca_scale = 1.0
                self._ref_ca_nm = np.zeros_like(self._ref_ca_angstrom, dtype=np.float32)
                self._ref_ca_center_nm = np.zeros(3, dtype=np.float32)
                self._ref_ca_scale_nm = 1.0
        except Exception:
            self._ref_ca_angstrom = np.zeros((self.n_residues, 3), dtype=np.float32)
            self._valid_ca_mask = np.zeros((self.n_residues,), dtype=bool)
            self._ref_ca_center = np.zeros(3, dtype=np.float32)
            self._ref_ca_scale = 1.0
            self._ref_ca_nm = np.zeros((self.n_residues, 3), dtype=np.float32)
            self._ref_ca_center_nm = np.zeros(3, dtype=np.float32)
            self._ref_ca_scale_nm = 1.0

        # Feature dimension for the residue transformer policy input:
        # base torsion features [sinφ, cosφ, sinψ, cosψ] optionally followed by
        # per-residue sparse pairwise distances (max_partners per residue).
        self.include_pairwise_residue = str(getattr(args, "policy_conditioning_mode", "torsion")).lower() in ("pairwise", "hybrid")
        self.max_partners = 0
        if self.include_pairwise_residue and getattr(self.generator, "partner_residues", None) is not None:
            try:
                self.max_partners = int(np.asarray(self.generator.partner_residues).shape[1])
                print(f"[Env] Extracted max_partners={self.max_partners} from generator.partner_residues")
            except Exception as e:
                print(f"[Env] Failed to extract max_partners: {e}")
                self.max_partners = 0
        print(f"[Env] n_residues={self.n_residues}, max_partners={self.max_partners}, hybrid_pair_dim={getattr(self.generator, 'hybrid_pair_dim', 'N/A')}")
        
        # Per-residue feature dimension for transformer policy
        if self.policy_mode == "ca":
            self.per_residue_feature_dim = 3
        elif self.policy_mode == "ca_torsion":
            self.per_residue_feature_dim = 7
        else:
            self.per_residue_feature_dim = 4 + (self.max_partners if self.include_pairwise_residue else 0)
        
        print(f"[Env] per_residue_feature_dim={self.per_residue_feature_dim} (policy input dim per residue)")
        self._phi_exists = np.zeros(self.n_residues, dtype=bool)
        self._psi_exists = np.zeros(self.n_residues, dtype=bool)
        for idx in range(self.n_cvs):
            slot = self.cv_residue_slots[idx]
            if self.cv_is_phi[idx]:
                self._phi_exists[slot] = True
            else:
                self._psi_exists[slot] = True
        
        print(f"Found {self.n_cvs} total backbone dihedrals.")
        
        self.bias = None
        if self.use_bias:
            self.bias = NDimensionalGaussianBias(system, self.cv_definitions)
            print("[Bias] Metadynamics bias: ENABLED (deposition guarded by nonzero height)")
        else:
            print("[Bias] Metadynamics bias: DISABLED")
        # Reinitializing the CUDA/OpenCL context can be very slow or hang on some
        # systems. Only do it when we've actually added forces after context creation.
        if self.use_bias:
            self.sim.context.reinitialize(preserveState=True)
        self._pdb_xyz = self.sim.context.getState(getPositions=True).getPositions()
        self.cv_values = self.get_cvs()
        # Track last reset error for upstream logic (e.g., SVD failures)
        self.last_reset_error: Optional[Exception] = None

        # Replay steering forces are preinstalled above (if enabled/compatible).

    def _maybe_init_replay_steering_forces(self) -> None:
        # Steering forces must be installed before Context creation to avoid
        # Context.reinitialize() / extra CUDA context creation during reset.
        # If not present here, we simply disable steering.
        return

    def _update_replay_steering_torsion_targets(
        self,
        cv_targets_rad: np.ndarray,
        cv_weights: Optional[np.ndarray] = None,
    ) -> None:
        forces = self._replay_steer_forces
        if forces is None:
            return
        torsion_bias = getattr(forces, "torsion_bias", None)
        if torsion_bias is None:
            return
        try:
            targets = np.asarray(cv_targets_rad, dtype=np.float64).reshape(-1)
            steer_defs = getattr(self, "_steer_cv_definitions", self.cv_definitions)
            if int(targets.size) != int(len(steer_defs)):
                return
            if cv_weights is None:
                weights = np.ones_like(targets, dtype=np.float64)
            else:
                weights = np.asarray(cv_weights, dtype=np.float64).reshape(-1)
                if int(weights.size) != int(targets.size):
                    return
            n_t = int(torsion_bias.getNumTorsions())
            if n_t != int(targets.size):
                return
            for i in range(n_t):
                a, b, c, d, params = torsion_bias.getTorsionParameters(i)
                try:
                    n_params = int(len(params))
                except Exception:
                    n_params = 1
                if n_params >= 2:
                    torsion_bias.setTorsionParameters(i, a, b, c, d, [float(targets[i]), float(weights[i])])
                else:
                    torsion_bias.setTorsionParameters(i, a, b, c, d, [float(targets[i])])
            torsion_bias.updateParametersInContext(self.sim.context)
        except Exception:
            return

    @property
    def cv_names(self) -> List[str]:
        """Return list of CV names extracted from cv_definitions."""
        return [name for name, _ in self.cv_definitions]

    def get_cvs(self) -> np.ndarray:
        pos = self.sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        return np.array([_dihedral(pos, atoms) for _, atoms in self.cv_definitions])

    def per_residue_torsion_features(self) -> np.ndarray:
        mode = self.policy_mode
        # CA-based modes build aligned CA first
        if mode in ("ca", "ca_torsion"):
            try:
                state = self.sim.context.getState(getPositions=True)
                pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                cur_ca = np.zeros_like(self._ref_ca_nm, dtype=np.float32)
                for slot, idx in enumerate(self.ca_atom_indices):
                    if idx >= 0:
                        cur_ca[slot] = pos_nm[int(idx)]
                valid = self._valid_ca_mask
                cur_valid = cur_ca[valid]
                ref_valid = self._ref_ca_nm[valid]
                if cur_valid.shape[0] >= 3 and ref_valid.shape[0] >= 3:
                    aligned_valid = kabsch_align(cur_valid, ref_valid)
                    aligned_all = np.zeros_like(self._ref_ca_nm, dtype=np.float32)
                    aligned_all[valid] = aligned_valid
                else:
                    aligned_all = cur_ca
                center = getattr(self, '_ref_ca_center_nm', None)
                scale = float(getattr(self, '_ref_ca_scale_nm', 1.0) or 1.0)
                if center is None:
                    center = np.zeros(3, dtype=np.float32)
                if not np.isfinite(scale) or scale < 1e-6:
                    scale = 1.0
                ca_norm = (aligned_all - center) / scale
                if mode == "ca":
                    return ca_norm.astype(np.float32)
                # ca_torsion: concatenate per-residue torsion sin/cos
                feats = np.zeros((self.n_residues, 7), dtype=np.float32)
                feats[:, :3] = ca_norm
                # Fill sin/cos torsion slots [3:7] = [sinφ, cosφ, sinψ, cosψ]
                feats[:, 4] = 1.0  # cos(phi) default
                feats[:, 6] = 1.0  # cos(psi) default
                for cv_idx, angle in enumerate(self.cv_values):
                    slot = self.cv_residue_slots[cv_idx]
                    if self.cv_is_phi[cv_idx]:
                        feats[slot, 3] = math.sin(angle)
                        feats[slot, 4] = math.cos(angle)
                    else:
                        feats[slot, 5] = math.sin(angle)
                        feats[slot, 6] = math.cos(angle)
                return feats.astype(np.float32)
            except Exception:
                # Fallback to zeros
                return np.zeros((self.n_residues, self.per_residue_feature_dim), dtype=np.float32)

        # Default torsion (+ optional per-residue pairwise)
        feats = np.zeros((self.n_residues, self.per_residue_feature_dim), dtype=np.float32)
        # Initialize torsion part defaults
        feats[:, 1] = 1.0  # cos(phi) default
        feats[:, 3] = 1.0  # cos(psi) default
        for cv_idx, angle in enumerate(self.cv_values):
            slot = self.cv_residue_slots[cv_idx]
            if self.cv_is_phi[cv_idx]:
                feats[slot, 0] = math.sin(angle)
                feats[slot, 1] = math.cos(angle)
            else:
                feats[slot, 2] = math.sin(angle)
                feats[slot, 3] = math.cos(angle)
        # Append per-residue sparse pairwise controls if requested and available
        if self.include_pairwise_residue and self.max_partners > 0:
            try:
                pair_flat = self.current_pairwise_features()
                if pair_flat is not None and pair_flat.size == self.max_partners * self.n_residues:
                    # Normalize using dataset stats from the generator
                    if getattr(self.generator, "conditioning_mode", "torsion") == "hybrid" and getattr(self.generator, "hybrid_pair_dim", 0) > 0:
                        k = int(self.generator.hybrid_pair_dim)
                        mean = self.generator.cond_mean[-k:].detach().cpu().numpy().astype(np.float32)
                        std = self.generator.cond_std[-k:].detach().cpu().numpy().astype(np.float32)
                    else:
                        k = int(getattr(self.generator, "pair_feature_dim", pair_flat.size))
                        mean = self.generator.cond_mean.detach().cpu().numpy().astype(np.float32)
                        std = self.generator.cond_std.detach().cpu().numpy().astype(np.float32)
                    if mean.size != pair_flat.size:
                        # If stats length mismatch, fallback to unnormalized
                        pair_norm = pair_flat.astype(np.float32)
                    else:
                        std = np.where(std < 1e-6, 1.0, std)
                        pair_norm = ((pair_flat.astype(np.float32) - mean) / std).astype(np.float32)
                    pair_mat = pair_norm.reshape(self.n_residues, self.max_partners)
                    feats[:, 4 : 4 + self.max_partners] = pair_mat
            except Exception:
                pass
        return feats

    def residue_targets_to_cv(self, residue_phi_psi: np.ndarray) -> np.ndarray:
        arr = np.asarray(residue_phi_psi, dtype=np.float32)
        if arr.shape != (self.n_residues, 2):
            raise ValueError(f"Expected residue targets shape ({self.n_residues}, 2); got {arr.shape}")
        phi_mask = self.cv_is_phi
        psi_mask = ~phi_mask
        cv_targets = np.zeros((self.n_cvs,), dtype=np.float32)
        if phi_mask.any():
            phi_slots = self.cv_residue_slots[phi_mask]
            cv_targets[phi_mask] = arr[phi_slots, 0]
        if psi_mask.any():
            psi_slots = self.cv_residue_slots[psi_mask]
            cv_targets[psi_mask] = arr[psi_slots, 1]
        return cv_targets

    def residue_batch_to_cv(self, residue_batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(residue_batch, dtype=np.float32)
        if batch.ndim != 3 or batch.shape[1:] != (self.n_residues, 2):
            raise ValueError(f"Expected batch shape (B, {self.n_residues}, 2); got {batch.shape}")
        phi_mask = self.cv_is_phi
        psi_mask = ~phi_mask
        B = batch.shape[0]
        cv = np.zeros((B, self.n_cvs), dtype=np.float32)
        if phi_mask.any():
            phi_slots = self.cv_residue_slots[phi_mask]
            cv[:, phi_mask] = batch[:, phi_slots, 0]
        if psi_mask.any():
            psi_slots = self.cv_residue_slots[psi_mask]
            cv[:, psi_mask] = batch[:, psi_slots, 1]
        return cv

    def mask_residue_angles(self, residue_angles: torch.Tensor) -> torch.Tensor:
        """
        Zero-out phi/psi targets for residues that do not have valid torsions.
        Accepts tensors shaped (B, n_residues, 2) or (n_residues, 2).
        """
        if residue_angles.dim() == 2:
            residue_angles = residue_angles.unsqueeze(0)
            squeeze = True
        elif residue_angles.dim() == 3:
            squeeze = False
        else:
            raise ValueError("residue_angles must have shape (B, n_residues, 2) or (n_residues, 2)")
        device = residue_angles.device
        phi_mask = torch.from_numpy(self._phi_exists).to(device)
        psi_mask = torch.from_numpy(self._psi_exists).to(device)
        angles = residue_angles.clone()
        angles[:, ~phi_mask, 0] = 0.0
        angles[:, ~psi_mask, 1] = 0.0
        if squeeze:
            angles = angles.squeeze(0)
        return angles

    def current_pairwise_features(self) -> Optional[np.ndarray]:
        """
        Compute pairwise CA–CA distances using the exact residue-pair selection
        used during dataset preparation for hybrid/pairwise conditioning.

        Returns a flattened vector matching the dataset ordering
        (row-major over shape [n_residues, max_partners]). If pair metadata
        is unavailable or the generator wasn't trained with pairwise features,
        returns None.
        """
        gen = self.generator
        if gen is None:
            return None
        if getattr(gen, 'conditioning_mode', 'torsion') not in ('pairwise', 'hybrid'):
            return None
        if gen.partner_residues is None or gen.pair_mask is None:
            return None
        ca_idx = np.asarray(getattr(self, "ca_atom_indices", None), dtype=np.int64).reshape(-1)
        if ca_idx.size == 0 or np.any(ca_idx < 0):
            return None
        # Current coordinates (nm) → Å for consistency with dataset
        state = self.sim.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        ca_pos = pos_nm[ca_idx] * 10.0  # Å
        partners = np.asarray(gen.partner_residues, dtype=np.int64)
        mask = np.asarray(gen.pair_mask, dtype=bool)
        n_res, max_partners = partners.shape
        feats = np.zeros((n_res, max_partners), dtype=np.float32)
        for i in range(n_res):
            for j in range(max_partners):
                if not mask[i, j]:
                    continue
                k = partners[i, j]
                if k <= i or k >= n_res:
                    continue
                d = np.linalg.norm(ca_pos[i] - ca_pos[k]).astype(np.float32)
                feats[i, j] = d
        return feats.reshape(-1)

    def get_conformer_positions(self, conformer_index: int) -> Optional[np.ndarray]:
        """
        Get positions for a specific conformer from the multi-conformer PDB.

        Args:
            conformer_index: Index of the conformer to retrieve (0-based)

        Returns:
            Positions in nanometers as np.ndarray with shape (n_atoms, 3), or None if not available
        """
        if not self.has_multiple_conformers or not self.pdb_conformers:
            return None
        if conformer_index < 0 or conformer_index >= len(self.pdb_conformers):
            return None
        # Convert from Angstrom to nm
        return self.pdb_conformers[conformer_index] / 10.0

    def reset(
        self,
        cv_targets: List[float],
        pairwise_targets: Optional[np.ndarray] = None,
        force_pdb_coords: bool = False,
        explicit_coords_nm: Optional[np.ndarray] = None,
        validate_generator_geom: bool = False,
        anchor_ref_index: Optional[int] = None,
        target_globals_nm: Optional[np.ndarray] = None,
        target_global_names: Optional[List[str]] = None,
        conformer_index: Optional[int] = None,
        target_torsions_residue_rad: Optional[np.ndarray] = None,
    ):
        """
        Reset the simulation.

        Modes:
        1. Replay / Bypass (explicit_coords_nm is not None or force_pdb_coords):
           - Bypasses the conditional generator.
           - If implicit solvent: skip minimization and heating entirely.
           - If explicit solvent: run a brief minimization to relax new water; skip heating.

        2. Generator Mode (default):
           - Uses the conditional diffusion/flow model to generate a new conformation.
           - Full minimization and heating to relax generator artifacts.
        """
        # Turn ON ligand-pocket restraint during steering/reset.
        if hasattr(self, '_lig_restraint_lambda_name'):
            try:
                self.sim.context.setParameter(self._lig_restraint_lambda_name, 1.0)
            except Exception:
                pass

        cv_targets_arr = np.asarray(cv_targets, dtype=np.float32)
        phi_angles, psi_angles = cv_targets_to_phi_psi(cv_targets_arr, self.cv_definitions)
        phi_angles = round_and_wrap_angles(phi_angles, self.torsion_round_decimals)
        psi_angles = round_and_wrap_angles(psi_angles, self.torsion_round_decimals)

        # Steering torsion targets: backbone φ/ψ always follow `cv_targets_arr` order, and
        # sidechain χ torsions are enabled only when `target_torsions_residue_rad` provides them.
        steer_defs = getattr(self, "_steer_cv_definitions", self.cv_definitions)
        steer_backbone_len = int(getattr(self, "_steer_cv_backbone_len", len(self.cv_definitions)))
        steer_targets_rad = np.zeros((len(steer_defs),), dtype=np.float64)
        steer_weights = np.ones((len(steer_defs),), dtype=np.float64)
        if len(steer_defs) > steer_backbone_len:
            steer_weights[steer_backbone_len:] = 0.0

        try:
            cv_targets_wrapped = round_and_wrap_angles(cv_targets_arr, self.torsion_round_decimals).astype(np.float64, copy=False)
        except Exception:
            cv_targets_wrapped = np.asarray(cv_targets_arr, dtype=np.float64).reshape(-1)
        n0 = min(int(cv_targets_wrapped.size), int(steer_backbone_len), int(len(steer_defs)))
        if n0 > 0:
            steer_targets_rad[:n0] = cv_targets_wrapped[:n0]

        torsion_targets_res = None
        if target_torsions_residue_rad is not None:
            try:
                torsion_targets_res = np.asarray(target_torsions_residue_rad, dtype=np.float32)
                if torsion_targets_res.ndim == 2:
                    torsion_targets_res = round_and_wrap_angles(torsion_targets_res, self.torsion_round_decimals)
                else:
                    torsion_targets_res = None
            except Exception:
                torsion_targets_res = None

        if torsion_targets_res is not None and torsion_targets_res.ndim == 2 and len(steer_defs) > steer_backbone_len:
            n_chi_defs = int(len(getattr(self, "_steer_chi_definitions", [])))
            n_lig_defs = int(len(getattr(self, "_steer_ligand_definitions", [])))
            lig_start = int(steer_backbone_len) + n_chi_defs  # start index of ligand entries in steer_defs

            for j in range(int(steer_backbone_len), int(len(steer_defs))):
                name, _atoms = steer_defs[j]
                name_s = str(name)
                # ── Sidechain χ (existing) ──
                if name_s.startswith("chi"):
                    try:
                        kind, rid_s = name_s.split("_", 1)
                        chi_k = int(kind[3:])
                        rid = int(rid_s)
                        slot = self._residue_index_map.get(rid, None)
                        if slot is None:
                            continue
                        angle_idx = 2 + int(chi_k)  # [phi,psi,omega,chi1,...]
                        if int(slot) < 0 or int(slot) >= int(torsion_targets_res.shape[0]):
                            continue
                        if int(angle_idx) < 0 or int(angle_idx) >= int(torsion_targets_res.shape[1]):
                            continue
                        steer_targets_rad[j] = float(torsion_targets_res[int(slot), int(angle_idx)])
                        steer_weights[j] = float(getattr(self, "steer_chi_weight", 1.0) or 0.0)
                    except Exception:
                        continue
                # ── Ligand rotatable dihedrals ──
                # NEW LAYOUT: AR uses ONE position per rotatable bond. The d-th ligand
                # dihedral is stored at row (n_residues + d), column 0 (phi-like slot).
                # Steering definitions for ligand are appended after backbone+chi in
                # _steer_cv_definitions in the same order produced by find_ligand_*.
                elif n_lig_defs > 0 and j >= lig_start and j < lig_start + n_lig_defs:
                    try:
                        lig_dihedral_idx = int(j - lig_start)  # 0-based index of this ligand dihedral
                        lig_slot = int(self.n_residues) + lig_dihedral_idx  # AR row for this dihedral
                        if lig_slot < 0 or lig_slot >= int(torsion_targets_res.shape[0]):
                            continue
                        angle_col = 0  # phi-like slot
                        if angle_col < 0 or angle_col >= int(torsion_targets_res.shape[1]):
                            continue
                        steer_targets_rad[j] = float(torsion_targets_res[lig_slot, angle_col])
                        steer_weights[j] = float(getattr(self, "steer_ligand_weight", 1.0) or 0.0)
                    except Exception:
                        continue
        self.last_generation_record = None
        self.reset_failed = False
        self.last_reset_error = None
        self.last_reset_trace = None
        last_exception: Optional[Exception] = None

        # Default (generator) relaxation parameters
        current_min_iters = self.minimize_iterations
        current_heat_stages = self.heat_num_stages

        reset_stats = {
            'sample_batch': 0.0,
            'hydrogen_rebuild': 0.0,
            'energy_check': 0.0,
            'minimize': 0.0,
            'heat': 0.0,
            'steer': 0.0,
        }

        # Build conditioning for the generator (used only in generator mode)
        condition_vector = None
        if getattr(self.generator, 'conditioning_mode', 'torsion') == 'hybrid':
            k = int(getattr(self.generator, 'hybrid_pair_dim', 0))
            if k > 0:
                # Validate pairwise_targets shape and ensure 1D
                if pairwise_targets is not None and pairwise_targets.size == k:
                    condition_vector = pairwise_targets.flatten().astype(np.float32)
                else:
                    # Use dataset mean for neutral pairwise conditioning
                    condition_vector = (
                        self.generator.cond_mean[-k:]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

        printed_minimize = False
        use_replay = explicit_coords_nm is not None or force_pdb_coords
        if (not use_replay) and (self.generator is None):
            # AR/retrieval-only mode: allow running without diffusion generator.
            force_pdb_coords = True
            use_replay = True
        # Track whether this reset used an explicit replay frame (history/dcd).
        # Used to avoid advancing the system (e.g., solvent equilibration) before recording.
        self._last_reset_explicit_replay = bool(explicit_coords_nm is not None)
        provided_positions: Optional[np.ndarray] = None
        aligned_heavy_replay: Optional[np.ndarray] = None
        pre_min_full_angstrom_replay: Optional[np.ndarray] = None

        if use_replay:
            if explicit_coords_nm is not None:
                if force_pdb_coords:
                    print(
                        "[Reset] Walker explicitly bypassing generator. Using provided reference coords.",
                        flush=True,
                    )
                else:
                    print("[Reset] Using explicit replay coordinates (bypass generator).", flush=True)
                provided_positions = np.asarray(explicit_coords_nm, dtype=np.float32)
            else:
                # Check if we should use a specific conformer from multi-conformer PDB
                if conformer_index is not None:
                    conformer_pos = self.get_conformer_positions(conformer_index)
                    if conformer_pos is not None:
                        print(
                            f"[Reset] Walker using conformer {conformer_index} from multi-conformer PDB (bypassing generator).",
                            flush=True,
                        )
                        provided_positions = np.asarray(conformer_pos, dtype=np.float32)
                    else:
                        print(
                            f"[Reset] Warning: conformer {conformer_index} not found, falling back to default PDB coords.",
                            flush=True,
                        )
                        modeller = Modeller(self.clean_modeller.topology, self.clean_modeller.getPositions())
                        provided_positions = np.array(modeller.getPositions().value_in_unit(nanometer))
                else:
                    print(
                        "[Reset] Walker explicitly bypassing generator. Using PDB coords.",
                        flush=True,
                    )
                    modeller = Modeller(self.clean_modeller.topology, self.clean_modeller.getPositions())
                    provided_positions = np.array(modeller.getPositions().value_in_unit(nanometer))

            if provided_positions.ndim != 2 or provided_positions.shape[1] != 3:
                raise ValueError(
                    f"Replay/reset requires coordinates of shape (N, 3); got {provided_positions.shape}."
                )

            # Replay-mode relaxation settings
            # For the anchoring reference (force_pdb_coords), run a full
            # minimization so the native structure is well-relaxed.
            current_heat_stages = 0
            if force_pdb_coords:
                current_min_iters = self.minimize_iterations
            else:
                if self.implicit_solvent:
                    current_min_iters = 0
                else:
                    # For explicit-solvent *full-coordinate* replay (fixed solvated topology),
                    # a replay frame should already be equilibrated; avoid drifting it via
                    # extra minimization when we have full-system coordinates.
                    if (
                        self.explicit_solvent
                        and self.explicit_solvent_replay_full_coords
                        and explicit_coords_nm is not None
                    ):
                        n_full = None
                        try:
                            n_full = int(getattr(self.md_topology, "n_atoms", 0) or 0)
                        except Exception:
                            n_full = None
                        if n_full and provided_positions is not None and int(provided_positions.shape[0]) == n_full:
                            current_min_iters = 0
                        else:
                            # When the replay source only provides protein/solute coordinates (common for
                            # history DCDs that omit solvent), we transplant the solute into a fixed solvated
                            # topology and allow solvent to relax. A small minimization cap (100) is often
                            # insufficient and can lead to frequent energy rejects in replay mode.
                            current_min_iters = min(int(current_min_iters), 1000)
                    else:
                        current_min_iters = min(int(current_min_iters), 1000)

            aligned_heavy_replay = provided_positions * 10.0
            pre_min_full_angstrom_replay = provided_positions * 10.0

        # Parse optional global steering targets (named when possible).
        # - If `target_global_names` matches, steer RMSD/E2E by name and collect ADK extras.
        # - Otherwise, fall back to legacy positional interpretation:
        #     [RMSD_nm] or [RMSD_nm, E2E_nm] or [RMSD_nm, E2E_nm, Rg_nm]
        tgt_rmsd_nm = None
        tgt_rmsd_a_nm = None
        tgt_rmsd_b_nm = None
        tgt_e2e_nm = None
        tgt_rg_nm = None
        steer_extra_targets = {}
        if target_globals_nm is not None:
            try:
                tg = np.asarray(target_globals_nm, dtype=np.float32).reshape(-1)
                names = None
                try:
                    if target_global_names is not None and len(target_global_names) == int(tg.size):
                        names = [str(x) for x in target_global_names]
                except Exception:
                    names = None

                if names:
                    for i, name in enumerate(names):
                        try:
                            val = float(tg[int(i)])
                        except Exception:
                            continue
                        if not np.isfinite(val):
                            continue
                        key = str(name).strip()
                        key_l = key.lower()
                        if tgt_rmsd_a_nm is None and key_l.startswith(("rmsd_a", "rmsda")):
                            tgt_rmsd_a_nm = val
                        if tgt_rmsd_b_nm is None and key_l.startswith(("rmsd_b", "rmsdb")):
                            tgt_rmsd_b_nm = val
                        if tgt_rmsd_nm is None and key_l.startswith("rmsd"):
                            tgt_rmsd_nm = val
                        if tgt_e2e_nm is None and key_l.startswith("e2e"):
                            tgt_e2e_nm = val
                        if tgt_rg_nm is None and key_l.startswith("rg"):
                            tgt_rg_nm = val
                        # Any non-standard global token can be steered if a corresponding extra CV force exists.
                        # (Unknown keys are safely ignored by TargetedMD if no force was installed.)
                        if not (
                            key_l.startswith(("rmsd_a", "rmsda"))
                            or key_l.startswith(("rmsd_b", "rmsdb"))
                            or key_l.startswith("rmsd")
                            or key_l.startswith("e2e")
                            or key_l.startswith("rg")
                        ):
                            steer_extra_targets[key] = val
                else:
                    if tg.size >= 1:
                        tgt_rmsd_nm = float(tg[0])
                    if tg.size == 2:
                        tgt_e2e_nm = float(tg[1])
                    if tg.size >= 3:
                        tgt_rg_nm = float(tg[2])
            except Exception:
                tgt_rmsd_nm = None
                tgt_rmsd_a_nm = None
                tgt_rmsd_b_nm = None
                tgt_e2e_nm = None
                tgt_rg_nm = None
                steer_extra_targets = {}

        # Main generation / replay loop
        n_attempts = self.max_resample_attempts if not use_replay else 1
        printed_minimize = False
        for attempt in range(n_attempts):
            try:
                if use_replay:
                    ordered_positions = provided_positions.copy() if provided_positions is not None else None
                else:
                    # Generate a single conformation
                    if self.generator is None:
                        raise RuntimeError(
                            "No generator loaded and replay not enabled. "
                            "Enable replay resets (retrieval) so resets bypass generation."
                        )

                    sample_start = time.perf_counter()
                    with torch.no_grad():
                        generated_heavy = self.generator.sample(
                            phi_angles=phi_angles,
                            psi_angles=psi_angles,
                            condition_vector=condition_vector,
                        )
                    generated_angstrom = generated_heavy.astype(np.float32)  # Already single sample, not batch
                    reset_stats["sample_batch"] += time.perf_counter() - sample_start

                    aligned_angstrom = kabsch_align(generated_angstrom, self.reference_heavy_angstrom)
                    coords_nm = aligned_angstrom / 10.0

                    rebuild_start = time.perf_counter()
                    pos_nm = np.array(self.clean_modeller.getPositions().value_in_unit(nanometer))
                    pos_nm[self.heavy_atom_indices] = coords_nm

                    modeller = Modeller(self.clean_topology, pos_nm * nanometer)
                    use_fixed_solvated_topology = bool(
                        self.explicit_solvent and self.explicit_solvent_replay_full_coords
                    )
                    if not use_fixed_solvated_topology:
                        # Determine per-residue protonation variants (e.g., HIS tautomer) from the
                        # current topology before deleting hydrogens, so we can re-add the same names.
                        his_like = {"HIS", "HID", "HIE", "HIP"}
                        res_variant_by_index: Dict[int, str] = {}
                        for res in modeller.topology.residues():
                            if res.name in his_like:
                                # If explicit variant is present in name, honor it directly
                                if res.name in ("HID", "HIE", "HIP"):
                                    res_variant_by_index[res.index] = res.name
                                else:
                                    # Infer from existing hydrogens
                                    hnames = {atom.name for atom in res.atoms() if atom.element.symbol == "H"}
                                    if "HD1" in hnames and "HE2" in hnames:
                                        res_variant_by_index[res.index] = "HIP"
                                    elif "HD1" in hnames:
                                        res_variant_by_index[res.index] = "HID"
                                    elif "HE2" in hnames:
                                        res_variant_by_index[res.index] = "HIE"
                        # Remove all hydrogens and re-add with explicit variants to ensure names match
                        hydrogen_atoms = [atom for atom in modeller.topology.atoms() if atom.element.symbol == "H"]
                        if hydrogen_atoms:
                            modeller.delete(hydrogen_atoms)
                        # Build the variants list (length == number of residues) with None for unspecified residues
                        residues = list(modeller.topology.residues())
                        variants_list = [None] * len(residues)
                        for res in residues:
                            var = res_variant_by_index.get(res.index)
                            if var:
                                variants_list[res.index] = var
                        modeller.addHydrogens(self.forcefield, variants=variants_list)
                    else:
                        # Explicit-solvent full-coordinate replay uses a fixed solvated topology.
                        # Rebuilding hydrogens can change atom counts/order, causing replay mismatch.
                        # Keep the original atom ordering and rely on steering + equilibration.
                        if self.steer_replay_verbose:
                            print(
                                "[Reset] Skipping hydrogen rebuild (explicit-solvent full-replay topology).",
                                flush=True,
                            )

                    ordered_positions = np.array(modeller.getPositions().value_in_unit(nanometer))

                    reset_stats["hydrogen_rebuild"] += time.perf_counter() - rebuild_start

                if ordered_positions is None:
                    raise RuntimeError("Reset produced no coordinates.")

                if not np.isfinite(ordered_positions).all():
                    print(
                        f"[Geom Reject] Non-finite coordinates detected; "
                        f"{'replay' if use_replay else 'resampling'} "
                        f"({attempt+1}/{n_attempts})."
                    )
                    if use_replay:
                        break
                    continue

                use_fixed_solvated_topology = bool(self.explicit_solvent and self.explicit_solvent_replay_full_coords)
                full_positions_nm = None
                if use_fixed_solvated_topology:
                    # Full-coordinate replay mode (explicit solvent): keep a fixed solvated topology
                    # and avoid rebuilding the solvent box on every reset.
                    #
                    # Note: when the baseline PDB is already solvated, `clean_modeller.deleteWater()`
                    # leaves ions in place. In that case generator/PDB starts can naturally be
                    # "solute-only" (protein + ions) rather than protein-only. Support both.
                    try:
                        solute_indices = self.md_topology.select("not water")
                    except Exception:
                        solute_indices = None
                    if solute_indices is None or len(solute_indices) == 0:
                        solute_indices = np.arange(self.md_topology.n_atoms, dtype=int)
                    solute_indices = np.asarray(solute_indices, dtype=int)

                    protein_indices = self.md_topology.select("protein")
                    if protein_indices is None or len(protein_indices) == 0:
                        protein_indices = solute_indices
                    protein_indices = np.asarray(protein_indices, dtype=int)

                    n_full = int(self.md_topology.n_atoms)
                    n_solute = int(solute_indices.shape[0])
                    n_protein = int(protein_indices.shape[0])

                    # Reference full-system coordinates for explicit-solvent full-replay.
                    # Default to the initial solvated structure (fixed PDB/topology) for consistency
                    # across resets; fall back to the current context if unavailable.
                    base_full_nm = getattr(self, "_ref_positions_nm_full", None)
                    if base_full_nm is not None:
                        base_full_nm = np.asarray(base_full_nm, dtype=np.float32).reshape(-1, 3)
                    if base_full_nm is None or int(base_full_nm.shape[0]) != n_full:
                        state_full = self.sim.context.getState(getPositions=True)
                        base_full_nm = (
                            state_full.getPositions(asNumpy=True).value_in_unit(nanometer).astype(np.float32, copy=False)
                        )

                    ordered_n = int(ordered_positions.shape[0])
                    if ordered_n == n_full:
                        target_full_positions_nm = np.asarray(ordered_positions, dtype=np.float32)
                    elif ordered_n == n_solute:
                        target_full_positions_nm = np.asarray(base_full_nm, dtype=np.float32).copy()
                        target_full_positions_nm[solute_indices] = np.asarray(ordered_positions, dtype=np.float32)
                    elif ordered_n == n_protein:
                        target_full_positions_nm = np.asarray(base_full_nm, dtype=np.float32).copy()
                        target_full_positions_nm[protein_indices] = np.asarray(ordered_positions, dtype=np.float32)
                    else:
                        raise ValueError(
                            "Explicit-solvent full-coordinate replay expects either full-system coordinates "
                            f"(N={n_full}), solute-only coordinates (N={n_solute}, i.e. not-water), "
                            f"or protein-only coordinates (N={n_protein}); got N={ordered_n}."
                        )

                    # Generator resets in explicit-solvent full-replay mode: start steering from the
                    # same solvated reference (baseline) for consistency, then steer toward the
                    # generated conformation.
                    if (not use_replay) and (ordered_n != n_full):
                        full_positions_nm = np.asarray(base_full_nm, dtype=np.float32).copy()
                    else:
                        full_positions_nm = np.asarray(target_full_positions_nm, dtype=np.float32)

                    solv_modeller = Modeller(self.sim.topology, full_positions_nm * nanometer)
                else:
                    # Default behavior: rebuild the solvent box around the new peptide shape.
                    solv_modeller = Modeller(self.clean_topology, ordered_positions * nanometer)
                    if self.explicit_solvent:
                        solvent_kwargs = {
                            "ionicStrength": self.ionic_strength,
                            "positiveIon": self.solvent_positive_ion,
                            "negativeIon": self.solvent_negative_ion,
                            "neutralize": self.solvent_neutralize,
                        }
                        if self._box_size is not None:
                            solvent_kwargs["boxSize"] = self._box_size
                        else:
                            solvent_kwargs["padding"] = self.solvent_padding
                        solv_modeller.addSolvent(self.forcefield, **solvent_kwargs)

                # Targeted steering is only meaningful for replay resets (and implicit solvent,
                # or explicit-solvent full-coordinate replay mode).
                replay_steer_forces = None
                steer_rmsd_target_nm = tgt_rmsd_nm
                has_any_global_target = bool(
                    tgt_rmsd_nm is not None
                    or tgt_rmsd_a_nm is not None
                    or tgt_rmsd_b_nm is not None
                    or tgt_e2e_nm is not None
                    or tgt_rg_nm is not None
                    or bool(steer_extra_targets)
                )
                want_torsion_stage = bool(
                    float(getattr(self, "steer_k_torsion", 0.0) or 0.0) > 0.0
                    and int(getattr(self, "steer_torsion_steps", 0) or 0) > 0
                    and int(getattr(self, "steer_torsion_updates", 0) or 0) > 0
                    and len(getattr(self, "_steer_cv_definitions", self.cv_definitions)) > 0
                    and (np.asarray(steer_weights, dtype=np.float64).sum() > 0.0)
                )
                steer_enabled = bool(
                    use_replay
                    and (not force_pdb_coords)
                    and self.steer_replay_starts
                    and (has_any_global_target or want_torsion_stage)
                    and getattr(self, "_ref_positions_nm_full", None) is not None
                    and (
                        self.implicit_solvent
                        or (self.explicit_solvent and self.explicit_solvent_replay_full_coords)
                    )
                )
                if (
                    (not self.implicit_solvent)
                    and self.explicit_solvent
                    and (not self.explicit_solvent_replay_full_coords)
                    and self.steer_replay_starts
                    and (tgt_rmsd_nm is not None or tgt_rmsd_a_nm is not None or tgt_rmsd_b_nm is not None or tgt_e2e_nm is not None or bool(steer_extra_targets))
                ):
                    print(
                        "[TargetedMD] Note: steering is disabled in explicit-solvent resolvation mode "
                        "(--no-explicit-solvent-replay-full-coords). Full-coordinate replay (fixed solvated topology) "
                        "is required and is the default for explicit-solvent runs.",
                        flush=True,
                    )

                # Fast path: reuse the existing Simulation for replay resets.
                reuse_existing_sim = bool(
                    use_replay
                    and getattr(self, "sim", None) is not None
                    and (
                        (self.implicit_solvent)
                        or (self.explicit_solvent and self.explicit_solvent_replay_full_coords and use_fixed_solvated_topology)
                    )
                )

                if reuse_existing_sim:
                    positions_nm = ordered_positions
                    if use_fixed_solvated_topology and full_positions_nm is not None:
                        positions_nm = full_positions_nm
                    if int(positions_nm.shape[0]) != int(self.sim.system.getNumParticles()):
                        reuse_existing_sim = False

                if reuse_existing_sim:
                    if self._last_reset_explicit_replay and self.steer_replay_verbose:
                        _t_set = time.perf_counter()
                        print(f"[Reset][Replay] setPositions start (N={int(positions_nm.shape[0])})", flush=True)
                    self.sim.context.setPositions(np.asarray(positions_nm, dtype=np.float32) * nanometer)
                    if self._last_reset_explicit_replay and self.steer_replay_verbose:
                        print(f"[Reset][Replay] setPositions done ({time.perf_counter()-_t_set:.2f}s)", flush=True)
                        _t_state = time.perf_counter()
                    self._pdb_xyz = self.sim.context.getState(getPositions=True).getPositions()
                    if self._last_reset_explicit_replay and self.steer_replay_verbose:
                        print(f"[Reset][Replay] getState(pos) done ({time.perf_counter()-_t_state:.2f}s)", flush=True)
                    pre_min_angstrom = np.asarray(positions_nm, dtype=np.float32) * 10.0
                    if steer_enabled:
                        if self._last_reset_explicit_replay and self.steer_replay_verbose:
                            _t_steer_init = time.perf_counter()
                            print("[Reset][Replay] init steering forces...", flush=True)
                        self._maybe_init_replay_steering_forces()
                        if self._last_reset_explicit_replay and self.steer_replay_verbose:
                            print(f"[Reset][Replay] init steering forces done ({time.perf_counter()-_t_steer_init:.2f}s)", flush=True)
                        self._update_replay_steering_torsion_targets(steer_targets_rad, steer_weights)
                        replay_steer_forces = self._replay_steer_forces
                        if replay_steer_forces is None and self.steer_replay_verbose:
                            print(
                                "[TargetedMD] Warning: steering enabled for replay reset but steering forces are unavailable; "
                                "skipping steering. (Was preinstall disabled/failed?)",
                                flush=True,
                            )
                else:
                    # Slow path: rebuild the system/simulation (used for explicit-solvent resolvation mode
                    # and generator-mode starts that modify topology/hydrogens).
                    old_sim = self.sim if hasattr(self, 'sim') and self.sim is not None else None

                    system = self.forcefield.createSystem(solv_modeller.topology, **self.ff_args)
                    if self.use_barostat and (not self.implicit_solvent):
                        system.addForce(
                            MonteCarloBarostat(
                                self.pressure_bar * bar,
                                self.temperature * kelvin,
                                self.barostat_frequency,
                            )
                        )

                    if steer_enabled:
                        try:
                            ref_nm = np.asarray(self._ref_positions_nm_full, dtype=np.float64).reshape(-1, 3)
                            if int(system.getNumParticles()) == int(ref_nm.shape[0]):
                                md_top = md.Topology.from_openmm(solv_modeller.topology)

                                ca_indices = md_top.select("protein and name CA")
                                if ca_indices is None or len(ca_indices) == 0:
                                    ca_indices = md_top.select("name CA")
                                ca_idx = [int(i) for i in np.asarray(ca_indices, dtype=np.int64).reshape(-1) if int(i) >= 0]
                                e2e_pair = (int(ca_idx[0]), int(ca_idx[-1])) if (tgt_e2e_nm is not None and len(ca_idx) >= 2) else None

                                rmsd_idx = None
                                rmsd_ref_nm = None
                                try:
                                    heavy_sel_expr = None
                                    heavy_idx: Optional[List[int]] = None
                                    for expr in (
                                        "protein and not element H",
                                        "protein and not hydrogen",
                                        "protein and not name H*",
                                    ):
                                        try:
                                            sel = md_top.select(expr)
                                            idx = [int(i) for i in np.asarray(sel, dtype=np.int64).reshape(-1) if int(i) >= 0]
                                            if len(idx) >= 3:
                                                heavy_sel_expr = expr
                                                heavy_idx = idx
                                                break
                                        except Exception:
                                            continue
                                    if heavy_idx is not None and len(heavy_idx) >= 3:
                                        rmsd_idx = heavy_idx
                                        if self.steer_replay_verbose and heavy_sel_expr:
                                            print(
                                                f"[TargetedMD] RMSD subset selection: {heavy_sel_expr} (n={len(heavy_idx)})",
                                                flush=True,
                                            )
                                        try:
                                            rmsd_ref_nm = ref_nm[np.asarray(rmsd_idx, dtype=np.int64)]
                                        except Exception:
                                            rmsd_ref_nm = None
                                except Exception:
                                    rmsd_idx = None
                                    rmsd_ref_nm = None

                                if rmsd_idx is None and ca_idx and len(ca_idx) >= 3:
                                    rmsd_idx = list(ca_idx)
                                    try:
                                        rmsd_ref_nm = ref_nm[np.asarray(rmsd_idx, dtype=np.int64)]
                                    except Exception:
                                        rmsd_ref_nm = None

                                rmsd_ref_a_nm = None
                                rmsd_ref_b_nm = None
                                if str(getattr(self, "_endpoint_b_pdb", "") or "").strip():
                                    try:
                                        pdb_a_path = str(getattr(self, "_generator_args", {}).get("pdb", "") or "").strip()
                                        pdb_b_path = str(getattr(self, "_endpoint_b_pdb", "") or "").strip()
                                        if pdb_a_path and pdb_b_path and rmsd_idx is not None and len(rmsd_idx) >= 3:
                                            tr_a = md.load(pdb_a_path)
                                            tr_b = md.load(pdb_b_path)
                                            if int(tr_a.n_frames) > 0 and int(tr_b.n_frames) > 0:
                                                xyz_a = np.asarray(tr_a.xyz[0], dtype=np.float32)
                                                xyz_b = np.asarray(tr_b.xyz[0], dtype=np.float32)
                                                map_a = {(atom.residue.index, str(atom.name)): xyz_a[int(atom.index)] for atom in tr_a.topology.atoms}
                                                map_b = {(atom.residue.index, str(atom.name)): xyz_b[int(atom.index)] for atom in tr_b.topology.atoms}
                                                ref_a = []
                                                ref_b = []
                                                keep = []
                                                for ai in rmsd_idx:
                                                    atom = md_top.atom(int(ai))
                                                    key = (int(atom.residue.index), str(atom.name))
                                                    pa = map_a.get(key)
                                                    pb = map_b.get(key)
                                                    if pa is None or pb is None:
                                                        continue
                                                    keep.append(int(ai))
                                                    ref_a.append(pa)
                                                    ref_b.append(pb)
                                                if len(keep) >= 3:
                                                    rmsd_idx = keep
                                                    rmsd_ref_a_nm = np.asarray(ref_a, dtype=np.float64).reshape(-1, 3)
                                                    rmsd_ref_b_nm = np.asarray(ref_b, dtype=np.float64).reshape(-1, 3)
                                                    # Keep the legacy RMSD reference aligned with endpoint-A when available.
                                                    rmsd_ref_nm = rmsd_ref_a_nm
                                    except Exception:
                                        rmsd_ref_a_nm = None
                                        rmsd_ref_b_nm = None
                                replay_steer_forces = add_replay_steering_forces(
                                    system,
                                    ref_positions_nm=ref_nm,
                                    ca_particle_indices=ca_idx,
                                    rmsd_particle_indices=rmsd_idx,
                                    rmsd_ref_positions_nm=rmsd_ref_nm,
                                    rmsd_ref_positions_a_nm=rmsd_ref_a_nm,
                                    rmsd_ref_positions_b_nm=rmsd_ref_b_nm,
                                    e2e_atoms=e2e_pair,
                                    cv_definitions=getattr(self, "_steer_cv_definitions", self.cv_definitions),
                                    cv_targets_rad=steer_targets_rad.tolist(),
                                    cv_weights=steer_weights.tolist(),
                                    k_rmsd=float(self.steer_k_rmsd),
                                    k_e2e=float(self.steer_k_e2e),
                                    k_torsion=float(self.steer_k_torsion),
                                    k_rg=float(self.steer_k_rg),
                                    extra_distance_pairs=None,
                                    k_dist=float(self.steer_k_dist),
                                    extra_angle_triplets=None,
                                    k_theta=float(self.steer_k_theta),
                                )
                        except Exception as exc_steer_add:
                            if self.steer_replay_verbose:
                                print(f"[TargetedMD] Warning: failed to add steering forces ({exc_steer_add})", flush=True)
                            replay_steer_forces = None
                        if replay_steer_forces is None and self.steer_replay_verbose:
                            print(
                                "[TargetedMD] Warning: steering enabled for replay reset but steering forces could not be created; "
                                "skipping steering.",
                                flush=True,
                            )

                    integrator = LangevinIntegrator(
                        self.temperature * kelvin,
                        self.friction / picosecond,
                        float(self.timestep_fs) * femtoseconds,
                    )
                    self.sim, chosen_platform = _build_simulation_or_fallback(
                        solv_modeller.topology, system, integrator, self.platform_name, self.platform_device
                    )
                    if chosen_platform != self.platform_name:
                        self.platform_name = chosen_platform
                    self.sim.context.setPositions(solv_modeller.getPositions())

                    if old_sim is not None:
                        try:
                            del old_sim
                        except Exception:
                            pass

                    self._pdb_xyz = self.sim.context.getState(getPositions=True).getPositions()
                    self.md_topology = md.Topology.from_openmm(solv_modeller.topology)
                    if self.use_bias:
                        self.bias = NDimensionalGaussianBias(system, self.cv_definitions)
                        self.sim.context.reinitialize(preserveState=True)

                    pre_min_angstrom = np.array(solv_modeller.getPositions().value_in_unit(nanometer)) * 10.0

                # Single minimization pass: do it once, log it once, and use the
                # minimized state for energy checks and downstream stats.
                if current_min_iters > 0:
                    if not printed_minimize:
                        if attempt == 0:
                            print("Minimizing replay sample..." if use_replay else "Minimizing conditional sample...")
                        else:
                            print(
                                f"[Resample] Minimizing conditional sample (attempt {attempt+1}/{n_attempts})..."
                            )
                        printed_minimize = True
                    minimize_start = time.perf_counter()
                    try:
                        with _OPENMM_MINIMIZE_LOCK:
                            if self._minimize_tolerance is not None:
                                self.sim.minimizeEnergy(
                                    tolerance=self._minimize_tolerance,
                                    maxIterations=current_min_iters,
                                )
                            else:
                                self.sim.minimizeEnergy(maxIterations=current_min_iters)
                    except TypeError:
                        with _OPENMM_MINIMIZE_LOCK:
                            self.sim.minimizeEnergy(maxIterations=current_min_iters)
                    reset_stats["minimize"] += time.perf_counter() - minimize_start
                else:
                    reset_stats["minimize"] += 0.0

                self._pdb_xyz = self.sim.context.getState(getPositions=True).getPositions()

                energy_start = time.perf_counter()
                energy_state = self.sim.context.getState(getEnergy=True)
                energy_kj = energy_state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
                reset_stats["energy_check"] += time.perf_counter() - energy_start
                if not math.isfinite(energy_kj):
                    print(
                        f"[Energy Reject] Potential energy is non-finite; "
                        f"{'replay' if use_replay else 'resampling'} "
                        f"({attempt+1}/{n_attempts})."
                    )
                    if use_replay:
                        break
                    continue
                if self.energy_threshold_kj > 0 and energy_kj > self.energy_threshold_kj:
                    print(
                        f"[Energy Reject] {energy_kj:.1f} kJ/mol exceeds threshold {self.energy_threshold_kj:.1f}; "
                        f"{'replay' if use_replay else 'resampling'} "
                        f"({attempt+1}/{n_attempts})."
                    )
                    if use_replay:
                        break
                    continue

                # Geometric sanity check for generator samples (initial conformations).
                # Only used when explicitly enabled (e.g., anchoring runs).
                if validate_generator_geom and not use_replay:
                    try:
                        state_geom = self.sim.context.getState(getPositions=True)
                        coords_nm_full = state_geom.getPositions(asNumpy=True).value_in_unit(nanometer)
                        coords_ang_full = coords_nm_full * 10.0

                        ca_idx_arr = np.asarray(self.ca_atom_indices, dtype=int)
                        valid_ca = ca_idx_arr[ca_idx_arr >= 0]

                        end_ok = True
                        rmsd_ok = True
                        if valid_ca.size >= 2:
                            first_ca = int(valid_ca[0])
                            last_ca = int(valid_ca[-1])

                            ref_ca_ang = getattr(self, "_ref_ca_angstrom", None)
                            ref_ca_nm = getattr(self, "_ref_ca_nm", None)
                            valid_mask = getattr(self, "_valid_ca_mask", None)
                            if anchor_ref_index is not None and anchor_ref_index > 0:
                                idx = anchor_ref_index - 1
                                try:
                                    if 0 <= idx < len(self.secondary_ref_ca_angstrom):
                                        ref_ca_ang = self.secondary_ref_ca_angstrom[idx]
                                except Exception:
                                    pass
                                try:
                                    if 0 <= idx < len(self.secondary_ref_ca_nm):
                                        ref_ca_nm = self.secondary_ref_ca_nm[idx]
                                except Exception:
                                    pass

                            if ref_ca_ang is not None and ref_ca_ang.shape[0] >= 2:
                                ref_end = ref_ca_ang[[0, ref_ca_ang.shape[0] - 1]]
                                cur_end = coords_ang_full[[first_ca, last_ca]]
                                ref_d = float(np.linalg.norm(ref_end[0] - ref_end[1]))
                                cur_d = float(np.linalg.norm(cur_end[0] - cur_end[1]))
                                if np.isfinite(ref_d) and np.isfinite(cur_d):
                                    if abs(cur_d - ref_d) > 5.0:
                                        end_ok = False

                            if ref_ca_nm is not None and valid_mask is not None:
                                cur_ca_nm = np.zeros_like(ref_ca_nm, dtype=np.float32)
                                for slot, idx_ca in enumerate(ca_idx_arr):
                                    if idx_ca >= 0:
                                        cur_ca_nm[slot] = coords_nm_full[int(idx_ca)]
                                cur_valid = cur_ca_nm[valid_mask]
                                ref_valid = ref_ca_nm[valid_mask]
                                if cur_valid.size and ref_valid.size:
                                    cur_valid_3d = cur_valid.reshape(1, *cur_valid.shape)
                                    rmsd_vals = calculate_rmsd(cur_valid_3d, ref_valid, atom_indices=None)
                                    rmsd_val_nm = float(rmsd_vals[0])
                                    if rmsd_val_nm > 1.5:
                                        rmsd_ok = False

                        if not (end_ok and rmsd_ok):
                            print(
                                f"[Geom Reject] Generator sample failed end-to-end/CA RMSD check; "
                                f"resampling ({attempt+1}/{n_attempts})."
                            )
                            continue
                    except Exception as exc_geom:
                        print(f"[Geom Reject] Generator geometry check failed ({exc_geom}); resampling.", flush=True)
                        continue

                # Heating: enabled only in generator mode
                if current_heat_stages > 0 and self.heat_stage_steps > 0:
                    heat_start = time.perf_counter()
                    original_step = self.sim.integrator.getStepSize()
                    self.sim.context.setVelocitiesToTemperature(100 * kelvin)
                    self.sim.integrator.setStepSize(0.5 * femtoseconds)
                    temps = np.linspace(150, self.temperature, current_heat_stages)
                    for temp in temps:
                        self.sim.integrator.setTemperature(temp * kelvin)
                        self.sim.step(self.heat_stage_steps)
                    self.sim.integrator.setStepSize(original_step)
                    self.sim.integrator.setTemperature(self.temperature * kelvin)
                    reset_stats["heat"] += time.perf_counter() - heat_start
                else:
                    # Fast path: assign velocities only
                    self.sim.context.setVelocitiesToTemperature(self.temperature * kelvin)
                    reset_stats["heat"] += 0.0

                # Targeted MD steering (replay resets): match globals (RMSD/E2E/extra) first,
                # then torsions, and finally release. Supported for implicit solvent and
                # explicit-solvent full-coordinate replay mode.
                #
                # Pre-steering minimize: relax replay positions before applying steering forces.
                # High-T frames (400-450 K) placed into a 298 K context can have clashing
                # water/protein geometry that causes NaN in the very first steering step.
                # steer_refine_minimize_iters is re-used here as the pre-steering budget.
                if replay_steer_forces is not None and self.steer_refine_minimize_iters > 0:
                    _pre_min_iters = int(self.steer_refine_minimize_iters)
                    try:
                        if self.steer_replay_verbose:
                            print(f"[TargetedMD] Pre-steering minimize ({_pre_min_iters} iters)...", flush=True)
                        self.sim.minimizeEnergy(maxIterations=_pre_min_iters)
                        if self.steer_replay_verbose:
                            print("[TargetedMD] Pre-steering minimize done.", flush=True)
                    except Exception as _exc_pre_min:
                        if self.steer_replay_verbose:
                            print(f"[TargetedMD] Pre-steering minimize failed ({_exc_pre_min}); proceeding.", flush=True)
                if replay_steer_forces is not None:
                    if not self.steer_replay_verbose:
                        print("[TargetedMD] Steering start (set --steer-replay-verbose for diagnostics).", flush=True)
                    steer_start = time.perf_counter()
                    try:
                        steer_replay_to_targets(
                            self.sim,
                            replay_steer_forces,
                            temperature_k=float(self.temperature),
                            target_rmsd_nm=steer_rmsd_target_nm,
                            target_rmsd_a_nm=tgt_rmsd_a_nm,
                            target_rmsd_b_nm=tgt_rmsd_b_nm,
                            target_e2e_nm=tgt_e2e_nm,
                            target_rg_nm=tgt_rg_nm,
                            extra_targets=steer_extra_targets,
                            global_steps=int(self.steer_global_steps),
                            global_updates=int(self.steer_global_updates),
                            torsion_steps=int(self.steer_torsion_steps),
                            torsion_updates=int(self.steer_torsion_updates),
                            release_steps=int(self.steer_release_steps),
                            refine_minimize_iters=int(self.steer_refine_minimize_iters),
                            max_energy_kj_mol=float(self.steer_max_energy_kj_mol),
                            global_lambda_during_torsion=float(self.steer_global_lambda_during_torsion),
                            verbose=bool(self.steer_replay_verbose),
                        )
                    except Exception as exc_steer:
                        if self.steer_replay_verbose:
                            print(f"[TargetedMD] Warning: steering failed ({exc_steer}); disabling.", flush=True)
                        try:
                            self.sim.context.setParameter("steer_lambda_g", 0.0)
                            self.sim.context.setParameter("steer_lambda_t", 0.0)
                        except Exception:
                            pass
                    reset_stats["steer"] += time.perf_counter() - steer_start

                final_state = self.sim.context.getState(getPositions=True)
                final_positions_nm = final_state.getPositions(asNumpy=True).value_in_unit(nanometer)
                final_positions_angstrom = final_positions_nm * 10.0
                self._pdb_xyz = final_state.getPositions()

                aligned_heavy = None
                if use_replay and aligned_heavy_replay is not None:
                    aligned_heavy = aligned_heavy_replay
                else:
                    aligned_heavy = aligned_angstrom.copy()
                pre_min_full = (
                    pre_min_full_angstrom_replay.copy()
                    if use_replay and pre_min_full_angstrom_replay is not None
                    else pre_min_angstrom.copy()
                )

                self.last_generation_record = {
                    'phi_angles': phi_angles.copy(),
                    'psi_angles': psi_angles.copy(),
                    'generated_heavy_angstrom': aligned_heavy,
                    'pre_min_full_angstrom': pre_min_full,
                    'relaxed_full_angstrom': final_positions_angstrom.copy(),
                    'relaxed_heavy_angstrom': final_positions_angstrom[self.heavy_atom_indices].copy(),
                    'timing': reset_stats.copy(),
                }

                self.cv_values = self.get_cvs()
                if np.any(np.isnan(self.cv_values)):
                    print(
                        f"[State Reject] CVs contained NaNs after reset; "
                        f"{'replay' if use_replay else 'resampling'} "
                        f"({attempt+1}/{n_attempts})."
                    )
                    self.last_generation_record = None
                    if use_replay:
                        break
                    continue
                self.reset_failed = False
                return self.cv_values

            except Exception as e:
                last_exception = e
                self.last_reset_error = e
                self.last_reset_trace = traceback.format_exc()
                mode_label = "replay" if use_replay else "attempt"
                print(f"WARNING: Reset failed ({mode_label} {attempt+1}/{n_attempts}): {e}")
                if self.last_reset_trace:
                    print(self.last_reset_trace)
                try:
                    self.sim.context.setPositions(self._pdb_xyz)
                    self.sim.context.setVelocitiesToTemperature(self.temperature * kelvin)
                except Exception:
                    pass
                self.last_generation_record = None
                # Bail early on SVD failures to avoid repeated noisy retries; upstream will refresh policy.
                if "SVD did not converge" in str(e):
                    self.reset_failed = True
                    break

        if last_exception is not None:
            print(f"Resetting walker to original PDB structure after repeated failures. Last error: {last_exception}")
        # Mark this reset as failed so the training loop can skip this walker
        # for the current epoch (no rollout / rewards / diagnostics).
        self.reset_failed = True
        try:
            self.sim.context.setPositions(self._pdb_xyz)
            self.sim.context.setVelocitiesToTemperature(self.temperature * kelvin)
        except Exception:
            pass
        self.last_generation_record = None

        try:
            self.cv_values = self.get_cvs()
        except Exception:
            self.cv_values = np.full((len(self.cv_definitions),), np.nan, dtype=np.float32)

        if np.any(np.isnan(self.cv_values)):
            print("[Fallback] CVs still NaN; rebuilding simulation from reference topology.")
            try:
                if self.explicit_solvent and self.explicit_solvent_replay_full_coords:
                    if self.topology_pdb_path is not None and self.topology_pdb_path.exists():
                        pdb_ref = PDBFile(str(self.topology_pdb_path))
                        modeller = Modeller(pdb_ref.topology, pdb_ref.positions)
                    else:
                        modeller = Modeller(self.sim.topology, self._pdb_xyz)
                else:
                    modeller = Modeller(self.clean_topology, self.clean_modeller.getPositions())
                    if self.explicit_solvent:
                        solvent_kwargs = {
                            "ionicStrength": self.ionic_strength,
                            "positiveIon": self.solvent_positive_ion,
                            "negativeIon": self.solvent_negative_ion,
                            "neutralize": self.solvent_neutralize,
                        }
                        if self._box_size is not None:
                            solvent_kwargs["boxSize"] = self._box_size
                        else:
                            solvent_kwargs["padding"] = self.solvent_padding
                        modeller.addSolvent(self.forcefield, **solvent_kwargs)
                system = self.forcefield.createSystem(modeller.topology, **self.ff_args)
                if self.use_barostat and (not self.implicit_solvent):
                    system.addForce(
                        MonteCarloBarostat(
                            self.pressure_bar * bar,
                            self.temperature * kelvin,
                            self.barostat_frequency,
                        )
                    )
                integrator = LangevinIntegrator(
                    self.temperature * kelvin,
                    self.friction / picosecond,
                    float(self.timestep_fs) * femtoseconds,
                )
                self.sim, chosen_platform = _build_simulation_or_fallback(
                    modeller.topology, system, integrator, self.platform_name, self.platform_device
                )
                if chosen_platform != self.platform_name:
                    self.platform_name = chosen_platform
                self.sim.context.setPositions(modeller.getPositions())
                self.md_topology = md.Topology.from_openmm(modeller.topology)
                if self.use_bias:
                    self.bias = NDimensionalGaussianBias(system, self.cv_definitions)
                    self.sim.context.reinitialize(preserveState=True)
                self._pdb_xyz = self.sim.context.getState(getPositions=True).getPositions()
                self.cv_values = self.get_cvs()
            except Exception as exc:
                print(f"[Fallback] Rebuild failed: {exc}")

        if np.any(np.isnan(self.cv_values)):
            raise RuntimeError("Fallback state CVs contain NaNs.")
        return self.cv_values
    def state(self) -> np.ndarray:
        """Return policy input features according to configured policy mode.

        Modes:
          - torsion: [angles, cos(angles), sin(angles)] over all CVs
          - pairwise: flattened CA distance features matching dataset ordering, normalized
          - hybrid: torsion block concatenated with normalized pairwise block
          - ca: aligned flattened Cα coordinates (nm), centered/scaled by reference
        """
        if self.policy_mode == "ca":
            try:
                state = self.sim.context.getState(getPositions=True)
                pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                # Build CA array in residue-slot order to match reference arrays
                cur_ca = np.zeros_like(self._ref_ca_nm, dtype=np.float32)
                for slot, idx in enumerate(self.ca_atom_indices):
                    if idx >= 0:
                        cur_ca[slot] = pos_nm[int(idx)]
                cur_ca = cur_ca.astype(np.float32)
                valid = self._valid_ca_mask
                cur_valid = cur_ca[valid]
                ref_valid = self._ref_ca_nm[valid]
                if cur_valid.shape[0] >= 3 and ref_valid.shape[0] >= 3:
                    aligned_valid = kabsch_align(cur_valid, ref_valid)
                    aligned_all = np.zeros_like(self._ref_ca_nm, dtype=np.float32)
                    aligned_all[valid] = aligned_valid
                else:
                    aligned_all = cur_ca
                center = getattr(self, '_ref_ca_center_nm', None)
                scale = float(getattr(self, '_ref_ca_scale_nm', 1.0) or 1.0)
                if center is None:
                    center = np.zeros(3, dtype=np.float32)
                if not np.isfinite(scale) or scale < 1e-6:
                    scale = 1.0
                aligned_all = (aligned_all - center) / scale
                return aligned_all.reshape(-1).astype(np.float32)
            except Exception:
                return np.zeros((self.n_residues * 3,), dtype=np.float32)
        if self.policy_mode == "ca_torsion":
            try:
                state = self.sim.context.getState(getPositions=True)
                pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                # CA features (aligned, normalized)
                cur_ca = np.zeros_like(self._ref_ca_nm, dtype=np.float32)
                for slot, idx in enumerate(self.ca_atom_indices):
                    if idx >= 0:
                        cur_ca[slot] = pos_nm[int(idx)]
                cur_ca = cur_ca.astype(np.float32)
                valid = self._valid_ca_mask
                cur_valid = cur_ca[valid]
                ref_valid = self._ref_ca_nm[valid]
                if cur_valid.shape[0] >= 3 and ref_valid.shape[0] >= 3:
                    aligned_valid = kabsch_align(cur_valid, ref_valid)
                    aligned_all = np.zeros_like(self._ref_ca_nm, dtype=np.float32)
                    aligned_all[valid] = aligned_valid
                else:
                    aligned_all = cur_ca
                center = getattr(self, '_ref_ca_center_nm', None)
                scale = float(getattr(self, '_ref_ca_scale_nm', 1.0) or 1.0)
                if center is None:
                    center = np.zeros(3, dtype=np.float32)
                if not np.isfinite(scale) or scale < 1e-6:
                    scale = 1.0
                ca_feat = ((aligned_all - center) / scale).reshape(-1).astype(np.float32)
                # Torsion sin/cos block (φ/ψ for all CVs in order)
                sines = np.sin(self.cv_values).astype(np.float32)
                cosines = np.cos(self.cv_values).astype(np.float32)
                torsion_block = np.concatenate([sines, cosines], dtype=np.float32)
                return np.concatenate([ca_feat, torsion_block], dtype=np.float32)
            except Exception:
                return np.zeros((self.n_residues * 3 + 2 * self.n_cvs,), dtype=np.float32)
        torsion_vec = np.concatenate([
            self.cv_values,
            np.cos(self.cv_values),
            np.sin(self.cv_values),
        ], dtype=np.float32)

        if self.policy_mode == "torsion":
            return torsion_vec.astype(np.float32)

        # Build normalized pairwise block if available
        pair_flat = self.current_pairwise_features()
        if pair_flat is None:
            # Fallback to zeros with appropriate dimension if known
            pair_dim = 0
            try:
                pair_dim = int(getattr(self.generator, "pair_feature_dim", 0))
                if getattr(self.generator, "conditioning_mode", "torsion") == "hybrid":
                    pair_dim = int(getattr(self.generator, "hybrid_pair_dim", pair_dim))
            except Exception:
                pair_dim = 0
            pair_norm = np.zeros((pair_dim,), dtype=np.float32) if pair_dim > 0 else None
        else:
            # Normalize with dataset stats (use last k dims for pairwise in hybrid)
            try:
                if getattr(self.generator, "conditioning_mode", "torsion") == "hybrid":
                    k = int(self.generator.hybrid_pair_dim)
                    mean = self.generator.cond_mean[-k:].detach().cpu().numpy().astype(np.float32)
                    std = self.generator.cond_std[-k:].detach().cpu().numpy().astype(np.float32)
                else:
                    k = int(self.generator.pair_feature_dim)
                    mean = self.generator.cond_mean.detach().cpu().numpy().astype(np.float32)
                    std = self.generator.cond_std.detach().cpu().numpy().astype(np.float32)
                std = np.where(std < 1e-6, 1.0, std)
                pair_norm = ((pair_flat.astype(np.float32) - mean) / std).astype(np.float32)
            except Exception:
                pair_norm = pair_flat.astype(np.float32)

        if self.policy_mode == "pairwise":
            return pair_norm if pair_norm is not None else torsion_vec.astype(np.float32)
        # hybrid
        if pair_norm is None:
            return torsion_vec.astype(np.float32)
        return np.concatenate([torsion_vec.astype(np.float32), pair_norm], axis=0).astype(np.float32)

    def step(
        self,
        height_scales: List[float],
        sigmas: List[List[float]],
        record_frequency: int,
        output_dir: Optional[Path] = None,
        epoch_index: Optional[int] = None,
        walker_idx: Optional[int] = None,
        equilibration_steps: int = 5000,
        write_full_dcd: Optional[bool] = None,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], bool, Optional[Path]]:
        # Turn OFF ligand-pocket restraint during production rollout (unbiased dynamics).
        if hasattr(self, '_lig_restraint_lambda_name'):
            try:
                self.sim.context.setParameter(self._lig_restraint_lambda_name, 0.0)
            except Exception:
                pass

        if self.use_bias and self.bias is not None:
            bias_E = self.sim.context.getState(getEnergy=True, groups={BIAS_GROUP_ID}).getPotentialEnergy()
            kBT = (MOLAR_GAS_CONSTANT_R * (self.temperature * kelvin)).value_in_unit(kilojoule_per_mole)
            h_wt = H0 * math.exp(-bias_E.value_in_unit(kilojoule_per_mole) / (kBT * GAMMA))
            final_heights = [min(max(0.0, float(hs)) * h_wt, 2.0) for hs in height_scales]
            if sum(final_heights) > 1e-12:
                if self.cv_values is not None and hasattr(self.cv_values, "tolist"):
                    self.bias.add_mixture(self.cv_values.tolist(), final_heights, sigmas)
                else:
                    print("[Bias] Warning: Invalid cv_values, cannot add mixture")

        frames_nm: List[np.ndarray] = []
        unitcell_lengths_nm: List[np.ndarray] = []
        unitcell_angles_deg: List[np.ndarray] = []
        traj_data: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        dcd_path: Optional[Path] = None
        is_stable = True

        write_full_dcd_default = bool(self.explicit_solvent and self.explicit_solvent_replay_full_coords)
        if write_full_dcd is None:
            write_full_dcd = write_full_dcd_default
        else:
            write_full_dcd = bool(write_full_dcd)
        protein_indices = self.md_topology.select("protein")
        if protein_indices is None or len(protein_indices) == 0:
            try:
                from ..utils.selection import infer_protein_atom_indices

                protein_indices = infer_protein_atom_indices(self.md_topology, include_caps=True, allow_empty=True)
            except Exception:
                protein_indices = np.arange(self.md_topology.n_atoms, dtype=int)
        protein_indices = np.asarray(protein_indices, dtype=int)

        # Explicit solvent equilibration:
        # - Resolvation mode (rebuild solvent each reset): equilibrate solvent box before recording.
        # - Full-coordinate replay mode (fixed solvated topology): do NOT run a long unrestrained
        #   equilibration here, since it can drift the carefully steered reset off-policy.
        if self.explicit_solvent and (not self.explicit_solvent_replay_full_coords) and equilibration_steps > 0:
            # Discrete/replay starts: avoid advancing dynamics before recording. In resolvation mode
            # the solvent is rebuilt anyway, but long burn-in can still drift the intended start.
            if bool(getattr(self, "_last_reset_explicit_replay", False)):
                equilibration_steps = 0
            if equilibration_steps > 0:
                try:
                    if equilibration_steps >= 100:
                        print(
                            f"[Equilibrate] Running {int(equilibration_steps)} MD steps before recording.",
                            flush=True,
                        )
                    self.sim.step(int(equilibration_steps))
                except Exception as e:
                    print(f"Instability during solvent equilibration: {e}")
                    return [], False, None

        def _box_lengths_angles_from_state(state) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            if not bool(self.explicit_solvent):
                return None, None
            try:
                a, b, c = state.getPeriodicBoxVectors()
            except Exception:
                return None, None

            def _vec3_to_nm(v) -> Optional[np.ndarray]:
                try:
                    # Some OpenMM builds may return unit-bearing quantities; normalize to nm.
                    vv = v
                    if hasattr(v, "value_in_unit"):
                        vv = v.value_in_unit(nanometer)
                    return np.array([float(vv.x), float(vv.y), float(vv.z)], dtype=np.float64)
                except Exception:
                    return None

            va = _vec3_to_nm(a)
            vb = _vec3_to_nm(b)
            vc = _vec3_to_nm(c)
            if va is None or vb is None or vc is None:
                return None, None

            la = float(np.linalg.norm(va))
            lb = float(np.linalg.norm(vb))
            lc = float(np.linalg.norm(vc))
            if not (np.isfinite(la) and np.isfinite(lb) and np.isfinite(lc) and la > 0 and lb > 0 and lc > 0):
                return None, None

            def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
                du = float(np.linalg.norm(u))
                dv = float(np.linalg.norm(v))
                if du <= 0 or dv <= 0:
                    return float("nan")
                cos = float(np.dot(u, v) / (du * dv))
                cos = max(-1.0, min(1.0, cos))
                return float(np.degrees(np.arccos(cos)))

            alpha = _angle_deg(vb, vc)  # angle between b and c
            beta = _angle_deg(va, vc)   # angle between a and c
            gamma = _angle_deg(va, vb)  # angle between a and b
            if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(gamma)):
                return None, None

            return np.array([la, lb, lc], dtype=np.float32), np.array([alpha, beta, gamma], dtype=np.float32)

        try:
            total_iterations = int(self.n_md_steps) // int(record_frequency)
            for iter_idx in range(total_iterations):
                self.sim.step(record_frequency)
                # For explicit solvent, wrap positions into the primary periodic box for output.
                # This keeps waters from appearing to "diffuse away" in visualization tools
                # when unit cell imaging isn't applied downstream.
                state = self.sim.context.getState(
                    getEnergy=True,
                    getPositions=True,
                    enforcePeriodicBox=bool(self.explicit_solvent),
                )
                energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
                if np.isnan(energy):
                    raise ValueError("Energy is NaN.")

                current_coords = state.getPositions(asNumpy=True)
                coords_nm = current_coords.value_in_unit(nanometer)
                protein_coords = coords_nm[protein_indices]
                if bool(self.explicit_solvent):
                    lnm, adeg = _box_lengths_angles_from_state(state)
                    if lnm is not None and adeg is not None:
                        unitcell_lengths_nm.append(lnm)
                        unitcell_angles_deg.append(adeg)

                self.cv_values = self.get_cvs()
                if np.any(np.isnan(self.cv_values)):
                    raise ValueError("CV is NaN.")

                bias_energy = 0.0
                if self.use_bias and self.bias is not None:
                    try:
                        bias_energy = (
                            self.sim.context.getState(getEnergy=True, groups={BIAS_GROUP_ID})
                            .getPotentialEnergy()
                            .value_in_unit(kilojoule_per_mole)
                        )
                    except Exception:
                        bias_energy = 0.0

                frames_nm.append(coords_nm if write_full_dcd else protein_coords)

                # Compute ligand rotatable torsions for this frame (from full-system coords).
                lig_tors_frame = np.zeros((0,), dtype=np.float32)
                _lig_defs = getattr(self, "_steer_ligand_definitions", None)
                if _lig_defs:
                    try:
                        _vals = np.empty((len(_lig_defs),), dtype=np.float32)
                        for _i, (_lbl, _atms) in enumerate(_lig_defs):
                            _vals[_i] = float(_dihedral(coords_nm, _atms))
                        lig_tors_frame = _vals
                    except Exception:
                        lig_tors_frame = np.zeros((0,), dtype=np.float32)

                traj_data.append((self.cv_values.copy(), protein_coords.copy(), float(energy), float(bias_energy), lig_tors_frame))

                if total_iterations >= 20 and (iter_idx + 1) % (total_iterations // 10) == 0:
                    progress_pct = int(100 * (iter_idx + 1) / total_iterations)
                    print(f"    MD progress: {progress_pct}% ({iter_idx + 1}/{total_iterations} chunks)", flush=True)

        except Exception as e:
            print(f"Walker unstable: {e}. Resetting to last stable state.")
            is_stable = False
            try:
                self.sim.context.setPositions(self._pdb_xyz)
                self.sim.context.setVelocitiesToTemperature(self.temperature * kelvin)
            except Exception:
                pass
            try:
                self.cv_values = self.get_cvs()
            except Exception:
                pass
            frames_nm = []
            traj_data = []

        if frames_nm and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            w_idx = int(walker_idx) if walker_idx is not None else 0
            if epoch_index is not None:
                dcd_name = f"epoch_{int(epoch_index):04d}_walker_{w_idx:02d}.dcd"
            else:
                dcd_name = f"walker_{w_idx:02d}.dcd"
            dcd_path = output_dir / dcd_name
            try:
                if write_full_dcd:
                    traj = md.Trajectory(
                        np.asarray(frames_nm, dtype=np.float32),
                        self.md_topology,
                        unitcell_lengths=(np.asarray(unitcell_lengths_nm, dtype=np.float32) if unitcell_lengths_nm else None),
                        unitcell_angles=(np.asarray(unitcell_angles_deg, dtype=np.float32) if unitcell_angles_deg else None),
                    )
                else:
                    protein_topology = self.md_topology.subset(protein_indices)
                    traj = md.Trajectory(
                        np.asarray(frames_nm, dtype=np.float32),
                        protein_topology,
                        unitcell_lengths=(np.asarray(unitcell_lengths_nm, dtype=np.float32) if unitcell_lengths_nm else None),
                        unitcell_angles=(np.asarray(unitcell_angles_deg, dtype=np.float32) if unitcell_angles_deg else None),
                    )
                traj.save_dcd(str(dcd_path))
            except Exception as exc:
                warnings.warn(f"Failed to write DCD for walker {w_idx}: {exc}")
                dcd_path = None

        return traj_data, is_stable, dcd_path
