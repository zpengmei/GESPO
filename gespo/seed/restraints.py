"""Restraint forces for seed generation (faithful ports of the seed scripts).

- ``add_position_restraints``   — harmonic heavy-atom position restraint used
  during minimization (protein+ligand non-H; water/ions left free).
- ``add_lig_pocket_com_restraint`` — PBC-aware flat-bottom on the ligand↔pocket-Cα
  COM distance (metadynamics scripts; keeps the ligand bound).
- ``add_lig_pocket_seqpos_restraint`` — flat-bottom between the ligand and a single
  pocket residue chosen by sequence position (``high_T_md.py``).
"""
from __future__ import annotations

from typing import List

import openmm as mm

from ._common import LIGAND_NAMES, SOLVENT_NAMES, log


def add_position_restraints(system: mm.System, topology, positions, k_kj: float = 1000.0) -> int:
    """Harmonic position restraints on protein/ligand non-H atoms; returns count."""
    from openmm import unit
    f = mm.CustomExternalForce("0.5*k_restr*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    f.addGlobalParameter("k_restr", k_kj)
    for p in ("x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    n = 0
    for atom in topology.atoms():
        if (atom.element is not None
                and atom.element.symbol != "H"
                and atom.residue.name.upper() not in SOLVENT_NAMES):
            x, y, z = positions[atom.index].value_in_unit(unit.nanometer)
            f.addParticle(atom.index, [x, y, z])
            n += 1
    system.addForce(f)
    return n


def add_lig_pocket_com_restraint(system: mm.System, lig_heavy: List[int], pocket_ca: List[int],
                                 r0_nm: float = 1.0, k_kj: float = 1000.0) -> None:
    """PBC-aware flat-bottom on COM distance between ligand heavy atoms and pocket Cα."""
    fb = mm.CustomCentroidBondForce(
        2, "step(distance(g1,g2)-r0) * 0.5 * k * (distance(g1,g2)-r0)^2")
    fb.addGlobalParameter("r0", r0_nm)
    fb.addGlobalParameter("k", k_kj)
    fb.addGroup(lig_heavy)
    fb.addGroup(pocket_ca)
    fb.addBond([0, 1])
    fb.setUsesPeriodicBoundaryConditions(True)
    system.addForce(fb)
    log(f"[Restraint] lig-pocket flat-bottom: r0={r0_nm} nm, k={k_kj} kJ/mol/nm² "
        f"(lig={len(lig_heavy)} atoms, pocket={len(pocket_ca)} Cα), PBC-aware")


def add_lig_pocket_seqpos_restraint(system: mm.System, topology, seq_pos: int = 170,
                                    r0_nm: float = 1.5, k_kj: float = 1000.0) -> bool:
    """Flat-bottom COM restraint between the ligand and one pocket residue (by seq index).

    Returns True if added. Mirrors ``high_T_md._add_ligand_pocket_restraint``.
    """
    import mdtraj as _md
    md_top = _md.Topology.from_openmm(topology)
    prot_res = [r for r in md_top.residues if r.is_protein]
    if seq_pos >= len(prot_res):
        log(f"[Restraint] pocket seq_pos={seq_pos} out of range ({len(prot_res)} residues)")
        return False
    pocket_resseq = prot_res[seq_pos].resSeq
    lig_heavy = [a.index for a in md_top.atoms
                 if a.residue.name.upper() in LIGAND_NAMES and a.element.symbol != "H"]
    pocket_heavy = [a.index for a in md_top.atoms
                    if a.residue.resSeq == pocket_resseq and a.element.symbol != "H"]
    if not lig_heavy or not pocket_heavy:
        log(f"[Restraint] No atoms found: lig={len(lig_heavy)}, pocket={len(pocket_heavy)}")
        return False
    fb = mm.CustomCentroidBondForce(
        2, "step(distance(g1,g2)-r0) * 0.5 * k * (distance(g1,g2)-r0)^2")
    fb.addGlobalParameter("r0", r0_nm)
    fb.addGlobalParameter("k", k_kj)
    g1 = fb.addGroup(lig_heavy)
    g2 = fb.addGroup(pocket_heavy)
    fb.addBond([g1, g2])
    system.addForce(fb)
    log(f"[Restraint] Flat-bottom COM: lig({len(lig_heavy)} atoms) <-> "
        f"pocket(resSeq={pocket_resseq}, {len(pocket_heavy)} atoms), "
        f"r0={r0_nm:.2f} nm, k={k_kj:.0f} kJ/mol/nm²")
    return True
