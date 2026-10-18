"""Shared OpenMM system-building for seed generation.

Faithful port of the helpers in ``metad_seeds_holo.py`` / ``metad_seeds_apo.py``
/ ``high_T_md.py``, generalized with an **explicit/implicit solvent switch** —
the one capability the one-off scripts lack (they are explicit-only). Explicit
runs re-solvate + PME + optional barostat; implicit runs use a GB model with no
box, no barostat, no re-solvation.

OpenMM and mdtraj are imported at module top (matching the scripts); this module
is only imported when a seed method actually runs, so `gespo seed --help` stays
cheap.

NOTE(consolidate): pocket/ligand detection and the solvent name set are also
present in ``gespo/sampling/env.py``. Those two builders should share one
implementation once the RL loop can be re-verified end to end (needs OpenMM).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openmm as mm
from openmm import app, unit

import mdtraj as _md

from ._common import LIGAND_NAMES, SOLVENT_NAMES, log

_NM = unit.nanometer
_K = unit.kelvin
_BAR = unit.bar
_PS = unit.picoseconds
_FS = unit.femtoseconds


def platform_and_props(device: str, platform_name: str = "CUDA"):
    """Resolve an OpenMM platform + property dict from a device string.

    ``device`` like ``cuda:2`` selects DeviceIndex 2 on the CUDA platform. For
    CPU/Reference platforms the device index is ignored.
    """
    plat = mm.Platform.getPlatformByName(platform_name)
    if platform_name.upper() == "CUDA":
        idx = device.split(":")[-1] if ":" in str(device) else "0"
        return plat, {"DeviceIndex": str(idx), "Precision": "mixed"}
    return plat, {}


def make_forcefield(
    *,
    explicit: bool,
    ligand_smiles: str = "",
    forcefield_protein: str = "amber14-all.xml",
    forcefield_water: str = "amber14/tip3p.xml",
    forcefield_implicit: str = "implicit/gbn2.xml",
) -> app.ForceField:
    """Build an OpenMM ForceField and register GAFF for a SMILES ligand if given."""
    if explicit:
        ff = app.ForceField(forcefield_protein, forcefield_water)
    else:
        ff = app.ForceField(forcefield_protein, forcefield_implicit)
    if ligand_smiles:
        from openmmforcefields.generators import GAFFTemplateGenerator
        from openff.toolkit import Molecule
        lig = Molecule.from_smiles(ligand_smiles, allow_undefined_stereo=True)
        gaff = GAFFTemplateGenerator(molecules=[lig], forcefield="gaff-2.11")
        ff.registerTemplateGenerator(gaff.generator)
        log(f"Registered GAFF-2.11: {ligand_smiles}")
    return ff


def prepare_topology(pdb_path: str, ff: app.ForceField, *, explicit: bool):
    """Return ``(topology, positions, is_periodic)`` ready for ``build_system``.

    Explicit: strip any existing solvent/ions and re-solvate a fresh TIP3P box in
    the original box vectors (avoids residue-numbering overflow artifacts that
    seed NaN forces). Implicit: strip solvent/ions and use the solute as-is (no
    box, no water).
    """
    pdb = app.PDBFile(pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    solv_res = [r for r in modeller.topology.residues()
                if r.name.upper() in SOLVENT_NAMES]
    if solv_res:
        modeller.delete(solv_res)
    if explicit:
        orig_box = pdb.topology.getPeriodicBoxVectors()
        modeller.addSolvent(ff, model="tip3p", boxVectors=orig_box, neutralize=True)
        log(f"  {modeller.topology.getNumAtoms()} atoms after re-solvation")
        return modeller.topology, modeller.positions, True
    log(f"  {modeller.topology.getNumAtoms()} atoms (implicit solvent, solute only)")
    return modeller.topology, modeller.positions, False


def write_simulation_topology(out_dir, topology, positions, box=None) -> Path:
    """Save the exact atom ordering and box used by a walker's DCD files."""
    if box is not None:
        topology.setPeriodicBoxVectors(box)
    path = Path(out_dir) / 'simulation_topology.pdb'
    with path.open('w') as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)
    return path


def build_system(
    topology,
    ff: app.ForceField,
    *,
    explicit: bool,
    with_barostat: bool = False,
    T_K: float = 298.0,
    nonbonded_cutoff_nm: float = 1.0,
) -> mm.System:
    """Create an OpenMM System for explicit (PME) or implicit (GB, no cutoff) solvent."""
    if explicit:
        system = ff.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=nonbonded_cutoff_nm * _NM,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=5e-4,
        )
        if with_barostat:
            system.addForce(mm.MonteCarloBarostat(1.0 * _BAR, T_K * _K, 25))
    else:
        system = ff.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
        )
    return system


def make_sim(topology, system, T_K: float, dt_fs: float, plat, props,
             positions, velocities=None, box=None) -> app.Simulation:
    """Construct a Langevin-middle Simulation with positions/velocities set."""
    integ = mm.LangevinMiddleIntegrator(T_K * _K, 1.0 / _PS, dt_fs * _FS)
    sim = app.Simulation(topology, system, integ, plat, props)
    sim.context.setPositions(positions)
    if velocities is not None:
        sim.context.setVelocities(velocities)
    else:
        sim.context.setVelocitiesToTemperature(T_K * _K)
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)
    return sim


def get_state(sim: app.Simulation):
    """Return ``(positions, velocities, box)`` from a Simulation context."""
    s = sim.context.getState(getPositions=True, getVelocities=True)
    box = sim.context.getState().getPeriodicBoxVectors()
    return s.getPositions(), s.getVelocities(), box


# ── topology introspection (mdtraj) ──────────────────────────────────────

def detect_pocket_and_ligand(ref_pdb: str, cutoff_nm: float = 0.5) -> Tuple[List[int], List[int]]:
    """From a holo reference PDB, return (pocket residue indices, ligand heavy idx).

    Pocket residues are protein residues with any heavy atom within ``cutoff_nm``
    of a ligand heavy atom. Residue indices are 0-based among protein residues.
    """
    ref = _md.load(ref_pdb)
    top = ref.topology
    lig_heavy = [a.index for a in top.atoms
                 if a.residue.name.upper() in LIGAND_NAMES and a.element.symbol != "H"]
    if not lig_heavy:
        raise ValueError(f"No ligand atoms in reference PDB: {ref_pdb}")
    prot_res = [r for r in top.residues if r.is_protein]
    lig_xyz = ref.xyz[0, lig_heavy]
    pocket_res_idx = set()
    for ri, r in enumerate(prot_res):
        r_heavy = [a.index for a in r.atoms if a.element.symbol != "H"]
        if not r_heavy:
            continue
        d = np.min(np.linalg.norm(
            ref.xyz[0, r_heavy, :][:, None, :] - lig_xyz[None, :, :], axis=-1))
        if d < cutoff_nm:
            pocket_res_idx.add(ri)
    log(f"[CVs] Identified {len(pocket_res_idx)} pocket residues, "
        f"{len(lig_heavy)} ligand heavy atoms from reference PDB")
    return sorted(pocket_res_idx), lig_heavy


def map_pocket_to_ca(topology, pocket_res_idx) -> List[int]:
    """Map pocket residue indices (0-based, protein) → Cα atom indices in ``topology``."""
    md_top = _md.Topology.from_openmm(topology)
    prot_res = [r for r in md_top.residues if r.is_protein]
    ca_idx = []
    for ri in pocket_res_idx:
        if ri < len(prot_res):
            for a in prot_res[ri].atoms:
                if a.name == "CA":
                    ca_idx.append(a.index)
                    break
    return ca_idx


def all_ca_indices(topology) -> List[int]:
    """All protein Cα atom indices in ``topology`` (for global-RMSD / e2e CVs)."""
    md_top = _md.Topology.from_openmm(topology)
    return [a.index for a in md_top.atoms if a.name == "CA" and a.residue.is_protein]


def find_ligand_heavy(topology) -> List[int]:
    """Ligand heavy-atom indices in a built ``topology`` (empty if no ligand)."""
    md_top = _md.Topology.from_openmm(topology)
    return [a.index for a in md_top.atoms
            if a.residue.name.upper() in LIGAND_NAMES and a.element.symbol != "H"]


def pocket_ca_for(
    topology,
    *,
    pocket_ref_pdb: Optional[str],
    self_pdb: Optional[str],
    cutoff_nm: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """Resolve (pocket Cα indices, ligand heavy indices) for the built system.

    Uses ``pocket_ref_pdb`` when given (apo case: pocket comes from a holo
    reference), otherwise ``self_pdb`` (holo case: the ligand is in ``--pdb``).
    """
    ref = pocket_ref_pdb or self_pdb
    pocket_res_idx, _ = detect_pocket_and_ligand(ref, cutoff_nm=cutoff_nm)
    pocket_ca = map_pocket_to_ca(topology, pocket_res_idx)
    if len(pocket_ca) < 2:
        raise ValueError(f"Pocket too small in built system: {len(pocket_ca)} Cα atoms")
    lig_heavy = find_ligand_heavy(topology)
    return pocket_ca, lig_heavy
