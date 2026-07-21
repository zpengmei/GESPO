"""High-temperature MD seeding — one walker.

Faithful port of ``high_T_md.py`` (staged heat 298 K → T, then production),
generalized with an explicit/implicit solvent switch. Explicit runs use NPT +
barostat + box equilibration; implicit runs use NVT with no barostat/box.
Supports checkpoint/restart. Output: ``<out_dir>/production.dcd``.
"""
from __future__ import annotations

from pathlib import Path

import openmm as mm
from openmm import app, unit

from .. import restraints as _restr
from .. import system as _sys
from .._common import ns_to_steps, log

_K = unit.kelvin


def _set_temperature(sim: app.Simulation, system: mm.System, T_K: float) -> None:
    """Update integrator + (if present) barostat temperature, then reinitialize."""
    sim.integrator.setTemperature(T_K * _K)
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.MonteCarloBarostat):
            f.setDefaultTemperature(T_K * _K)
            break
    sim.context.reinitialize(preserveState=True)


def _attach_reporters(sim, dcd_path, log_path, ckpt_path,
                      save_stride, log_stride, ckpt_stride, total_steps,
                      *, append: bool, density: bool) -> None:
    sim.reporters.append(app.DCDReporter(str(dcd_path), save_stride, append=append))
    sim.reporters.append(app.CheckpointReporter(str(ckpt_path), ckpt_stride))
    sim.reporters.append(app.StateDataReporter(
        str(log_path), log_stride, append=append, step=True, time=True,
        potentialEnergy=True, kineticEnergy=True, temperature=True, density=density,
        progress=True, remainingTime=True, totalSteps=total_steps))


def run_high_temp(
    *,
    pdb: str,
    out_dir: str,
    temperature: float,
    production_ns: float = 50.0,
    save_ps: float = 10.0,
    ckpt_ns: float = 1.0,
    dt_fs: float = 2.0,
    ligand_smiles: str = "",
    explicit: bool = True,
    device: str = "cuda:0",
    platform_name: str = "CUDA",
    restart: bool = False,
    ligand_pocket_restraint: bool = False,
    pocket_radius_nm: float = 1.5,
    pocket_k: float = 1000.0,
    pocket_seq_pos: int = 170,
    forcefield_protein: str = "amber14-all.xml",
    forcefield_water: str = "amber14/tip3p.xml",
    forcefield_implicit: str = "implicit/gbn2.xml",
) -> str:
    """Run one high-temperature walker; returns the path to the production DCD."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "production.chk"
    prod_dcd = out / "production.dcd"
    equil_log = out / "equilibration.log"
    prod_log = out / "production.log"

    T_tgt = float(temperature)
    T_298 = 298.0
    dt = float(dt_fs)
    prod_n = ns_to_steps(production_ns, dt)
    save_stride = ns_to_steps(save_ps / 1000.0, dt)
    ckpt_stride = ns_to_steps(ckpt_ns, dt)
    log_stride = ns_to_steps(0.1, dt)
    use_restraint = bool(ligand_pocket_restraint)

    log(f"high-temp walker | T={T_tgt:.0f} K | device={device} | explicit={explicit} | out={out}")

    ff = _sys.make_forcefield(
        explicit=explicit, ligand_smiles=ligand_smiles,
        forcefield_protein=forcefield_protein, forcefield_water=forcefield_water,
        forcefield_implicit=forcefield_implicit)

    log("Preparing topology …")
    top, pos0, _is_periodic = _sys.prepare_topology(pdb, ff, explicit=explicit)
    plat, props = _sys.platform_and_props(device, platform_name=platform_name)

    def _maybe_restraint(system):
        if use_restraint:
            _restr.add_lig_pocket_seqpos_restraint(
                system, top, seq_pos=pocket_seq_pos, r0_nm=pocket_radius_nm, k_kj=pocket_k)

    # ── RESTART: resume production from checkpoint ──────────────────────────
    if restart and ckpt_path.exists():
        log("Restarting production from checkpoint …")
        sys_p = _sys.build_system(top, ff, explicit=explicit, with_barostat=explicit, T_K=T_tgt)
        _maybe_restraint(sys_p)
        sim = _sys.make_sim(top, sys_p, T_tgt, dt, plat, props, pos0)
        sim.loadCheckpoint(str(ckpt_path))
        _attach_reporters(sim, prod_dcd, prod_log, ckpt_path, save_stride, log_stride,
                          ckpt_stride, prod_n, append=True, density=explicit)
        sim.step(prod_n)
        log("Done.")
        return str(prod_dcd)

    # ── Stage 1: minimise (restraints, then free) ───────────────────────────
    log("Stage 1a — minimise 10 000 steps (heavy-atom restraints)")
    sys1 = _sys.build_system(top, ff, explicit=explicit, T_K=T_298)
    _restr.add_position_restraints(sys1, top, pos0, k_kj=1000.0)
    sim1 = _sys.make_sim(top, sys1, T_298, dt, plat, props, pos0)
    sim1.minimizeEnergy(maxIterations=10_000)
    pos1, _, _ = _sys.get_state(sim1)
    del sim1, sys1

    log("Stage 1b — minimise 5 000 steps (no restraints)")
    sys1b = _sys.build_system(top, ff, explicit=explicit, T_K=T_298)
    sim1b = _sys.make_sim(top, sys1b, T_298, dt, plat, props, pos1)
    sim1b.minimizeEnergy(maxIterations=5_000)
    pos1b, _, _ = _sys.get_state(sim1b)
    del sim1b, sys1b

    # ── Stage 2: 298 K equilibration ────────────────────────────────────────
    log(f"Stage 2 — {'NPT' if explicit else 'NVT'} 298 K, 2 ns equilibration")
    sys_eq = _sys.build_system(top, ff, explicit=explicit, with_barostat=explicit, T_K=T_298)
    _maybe_restraint(sys_eq)
    sim = _sys.make_sim(top, sys_eq, T_298, dt, plat, props, pos1b)
    sim.reporters.append(app.StateDataReporter(
        str(equil_log), log_stride, step=True, time=True,
        potentialEnergy=True, temperature=True, density=explicit))
    sim.step(ns_to_steps(2.0, dt))

    # ── Stage 3: gradual heat 298 K → T_tgt over 2 ns (10 × 200 ps) ─────────
    log(f"Stage 3 — gradual heat 298 K → {T_tgt:.0f} K (10 × 200 ps)")
    n_stages = 10
    for i in range(n_stages):
        T_i = T_298 + (T_tgt - T_298) * (i + 1) / n_stages
        _set_temperature(sim, sys_eq, T_i)
        sim.step(ns_to_steps(0.2, dt))
        log(f"  step {i + 1:2d}/{n_stages}  {T_i:.1f} K → done")

    # ── Stage 4: T_tgt stabilisation, 5 ns ──────────────────────────────────
    log(f"Stage 4 — {T_tgt:.0f} K, 5 ns stabilisation")
    sim.step(ns_to_steps(5.0, dt))
    pos5, vel5, box5 = _sys.get_state(sim)
    del sim, sys_eq

    # ── Stage 5: production ─────────────────────────────────────────────────
    log(f"Stage 5 — production {T_tgt:.0f} K, {production_ns:.0f} ns")
    sys_p = _sys.build_system(top, ff, explicit=explicit, with_barostat=explicit, T_K=T_tgt)
    _maybe_restraint(sys_p)
    sim_p = _sys.make_sim(top, sys_p, T_tgt, dt, plat, props, pos5, vel5, box5)
    _attach_reporters(sim_p, prod_dcd, prod_log, ckpt_path, save_stride, log_stride,
                      ckpt_stride, prod_n, append=False, density=explicit)
    sim_p.step(prod_n)
    log(f"Complete — DCD saved to {prod_dcd}")
    return str(prod_dcd)
