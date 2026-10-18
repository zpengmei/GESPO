"""Well-tempered metadynamics seeding — one walker.

Faithful port of ``metad_seeds_holo.py`` / ``metad_seeds_apo.py``, generalized:
the CV set (``holo`` / ``apo`` / ``folding``) is selected by name, and explicit or
implicit solvent is supported. The MD protocol (minimize → equilibrate → metaD
production with the same γ / hill height / deposition period) is unchanged.

Output: ``<out_dir>/production.dcd`` (+ ``colvar.log``, ``equilibration.log``,
``production.log``). N-walker fan-out with a shared ``bias_dir`` is handled by
``gespo.seed.walkers``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from openmm import app, unit

from .. import cvs as _cvs
from .. import restraints as _restr
from .. import system as _sys
from .._common import ns_to_steps, log

_KJ = unit.kilojoule_per_mole
_K = unit.kelvin


def run_metad(
    *,
    pdb: str,
    out_dir: str,
    cv_set: str = "holo",
    ligand_smiles: str = "",
    pocket_ref_pdb: str = "",
    explicit: bool = True,
    device: str = "cuda:0",
    platform_name: str = "CUDA",
    bias_dir: Optional[str] = None,
    production_ns: float = 25.0,
    equilibration_ns: float = 2.0,
    save_ps: float = 10.0,
    dt_fs: float = 2.0,
    temperature: float = 298.0,
    bias_factor: float = 10.0,
    hill_height: float = 1.0,
    hill_freq: int = 500,
    pocket_cutoff_nm: float = 0.5,
    cv1_max_nm: Optional[float] = None,
    cv2_min_nm: Optional[float] = None,
    cv2_max_nm: Optional[float] = None,
    lig_restraint: bool = True,
    lig_restraint_r0_nm: float = 1.0,
    lig_restraint_k: float = 1000.0,
    forcefield_protein: str = "amber14-all.xml",
    forcefield_water: str = "amber14/tip3p.xml",
    forcefield_implicit: str = "implicit/gbn2.xml",
) -> str:
    """Run one metadynamics walker; returns the path to the production DCD."""
    out = Path(out_dir)
    if equilibration_ns < 0:
        raise ValueError('equilibration_ns must be nonnegative.')
    out.mkdir(parents=True, exist_ok=True)
    bias = Path(bias_dir) if bias_dir else (out / "bias")
    bias.mkdir(parents=True, exist_ok=True)

    prod_dcd = out / "production.dcd"
    equil_log = out / "equilibration.log"
    prod_log = out / "production.log"
    colvar_log = out / "colvar.log"

    T_K = float(temperature)
    dt = float(dt_fs)
    prod_n = ns_to_steps(production_ns, dt)
    save_stride = ns_to_steps(save_ps / 1000.0, dt)
    log_stride = ns_to_steps(0.1, dt)
    bias_save = ns_to_steps(0.1, dt)

    specs = _cvs.resolve_cv_set(
        cv_set, cv1_max_nm=cv1_max_nm, cv2_min_nm=cv2_min_nm, cv2_max_nm=cv2_max_nm)
    needs_pocket = _cvs.cv_set_needs_pocket(cv_set)
    needs_ligand = _cvs.cv_set_needs_ligand(cv_set)
    use_lig_restraint = bool(lig_restraint and needs_ligand)

    log(f"metad walker | cv_set={cv_set} | device={device} | "
        f"production={production_ns} ns | explicit={explicit} | bias_dir={bias}")

    ff = _sys.make_forcefield(
        explicit=explicit, ligand_smiles=ligand_smiles,
        forcefield_protein=forcefield_protein, forcefield_water=forcefield_water,
        forcefield_implicit=forcefield_implicit)

    log("Preparing topology …")
    top, pos0, is_periodic = _sys.prepare_topology(pdb, ff, explicit=explicit)

    pocket_ca, lig_heavy = [], []
    if needs_pocket:
        pocket_ca, lig_heavy = _sys.pocket_ca_for(
            top, pocket_ref_pdb=(pocket_ref_pdb or None), self_pdb=pdb,
            cutoff_nm=pocket_cutoff_nm)
        log(f"[CVs] pocket Cα={len(pocket_ca)} ligand heavy={len(lig_heavy)}")
    if needs_ligand and not lig_heavy:
        raise ValueError("cv_set needs a ligand but none was found in the built system.")
    all_ca = _sys.all_ca_indices(top)

    plat, props = _sys.platform_and_props(device, platform_name=platform_name)

    # ── Stage 1: minimise (heavy-atom restraints, then free) ────────────────
    log("Stage 1a — minimise 10 000 steps (heavy-atom restraints)")
    sys1 = _sys.build_system(top, ff, explicit=explicit, T_K=T_K)
    _restr.add_position_restraints(sys1, top, pos0)
    sim1 = _sys.make_sim(top, sys1, T_K, dt, plat, props, pos0)
    sim1.minimizeEnergy(maxIterations=10_000)
    pos1, _, _ = _sys.get_state(sim1)
    del sim1, sys1

    log("Stage 1b — minimise 5 000 steps (no restraints)")
    sys1b = _sys.build_system(top, ff, explicit=explicit, T_K=T_K)
    sim1b = _sys.make_sim(top, sys1b, T_K, dt, plat, props, pos1)
    sim1b.minimizeEnergy(maxIterations=5_000)
    pos_ref, _, _ = _sys.get_state(sim1b)
    del sim1b, sys1b
    log("  → post-minimization positions are the RMSD reference")

    # ── Stage 2: equilibration (barostat + lig restraint for explicit) ──────
    log(f"Stage 2 — {'NPT' if explicit else 'NVT'} {T_K:.0f} K, {equilibration_ns} ns equilibration")
    sys2 = _sys.build_system(top, ff, explicit=explicit, with_barostat=explicit, T_K=T_K)
    if use_lig_restraint:
        _restr.add_lig_pocket_com_restraint(
            sys2, lig_heavy, pocket_ca, r0_nm=lig_restraint_r0_nm, k_kj=lig_restraint_k)
    sim2 = _sys.make_sim(top, sys2, T_K, dt, plat, props, pos_ref)
    sim2.reporters.append(app.StateDataReporter(
        str(equil_log), log_stride, step=True, time=True,
        potentialEnergy=True, temperature=True, density=explicit))
    sim2.step(ns_to_steps(equilibration_ns, dt))
    pos2, vel2, box2 = _sys.get_state(sim2)
    del sim2, sys2
    log("  → equilibration done")

    # ── Stage 3: well-tempered metaD production ─────────────────────────────
    log(f"Stage 3 — well-tempered metaD, {production_ns} ns "
        f"(γ={bias_factor}, h={hill_height} kJ/mol, "
        f"Δt={hill_freq} steps = {hill_freq * dt / 1000:.1f} ps)")
    sys3 = _sys.build_system(top, ff, explicit=explicit, with_barostat=explicit, T_K=T_K)
    if use_lig_restraint:
        _restr.add_lig_pocket_com_restraint(
            sys3, lig_heavy, pocket_ca, r0_nm=lig_restraint_r0_nm, k_kj=lig_restraint_k)

    cv_variables = _cvs.build_bias_variables(
        specs, pos_ref=pos_ref, all_ca=all_ca, pocket_ca=pocket_ca,
        lig_heavy=lig_heavy, periodic_box=is_periodic)
    for spec in specs:
        log(f"[CVs] {spec.kind}: [{spec.min_nm}, {spec.max_nm}] nm, width={spec.width_nm} nm")

    meta = app.metadynamics.Metadynamics(
        system=sys3, variables=cv_variables, temperature=T_K * _K,
        biasFactor=bias_factor, height=hill_height * _KJ, frequency=int(hill_freq),
        saveFrequency=bias_save, biasDir=str(bias))

    sim3 = _sys.make_sim(top, sys3, T_K, dt, plat, props, pos2, vel2, box2)
    _sys.write_simulation_topology(out, top, pos2, box2 if explicit else None)
    sim3.reporters.append(app.DCDReporter(str(prod_dcd), save_stride))
    sim3.reporters.append(app.StateDataReporter(
        str(prod_log), log_stride, step=True, time=True, potentialEnergy=True,
        temperature=True, density=explicit, progress=True, remainingTime=True,
        totalSteps=prod_n))

    with open(str(colvar_log), "w") as colvar_f:
        colvar_f.write("# step  " + "  ".join(s.kind for s in specs) + "\n")
        colvar_f.flush()
        log(f"Starting metaD production ({prod_n:,} steps) …")
        steps_done = 0
        while steps_done < prod_n:
            chunk = min(save_stride, prod_n - steps_done)
            meta.step(sim3, chunk)
            steps_done += chunk
            cvs_now = meta.getCollectiveVariables(sim3)
            colvar_f.write(f"{steps_done}" + "".join(f"  {cv:.6f}" for cv in cvs_now) + "\n")
            colvar_f.flush()

    log(f"Complete — DCD: {prod_dcd}")
    return str(prod_dcd)
