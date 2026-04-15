"""
Bootstrap utilities CLI (experimental).

Current focus:
- RMSD-biased short MD to generate a proposal trajectory (NOT for history).
- Optional extraction of a multi-model start-pool PDB from the biased trajectory.

Run:
  python -m gespo.bootstrap_cli rmsd-bias --help
  python -m gespo.bootstrap_cli rmsd-bias-ab --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


PROTOCOL_FAST = "fast"
PROTOCOL_PAPER_310K_NACL = "paper_310k_nacl"

# OpenMM force group used to isolate the RMSD-bias potential energy so we can cap it
# without needing to query/alter the full system energy.
RMSD_BIAS_FORCE_GROUP_ID: int = 4


def _require_openmm_mdtraj():
    try:
        import mdtraj as md  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "mdtraj is required for this tool.\n"
            "Install: pip/conda install mdtraj\n"
        ) from exc

    try:
        import openmm  # noqa: F401
        from openmm import unit  # noqa: F401
        from openmm.app import ForceField, PDBFile, Simulation  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "OpenMM is required for this tool.\n"
            "If you're on this repo's cluster setup, try:\n"
            "  /project/dinner/zpengmei/conda_envs/openmm/bin/python -m gespo.bootstrap_cli ...\n"
        ) from exc


def _flag_present(argv: list[str], flag: str) -> bool:
    if flag in argv:
        return True
    prefix = flag + "="
    return any(arg.startswith(prefix) for arg in argv)


def _apply_protocol_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    if getattr(args, "protocol", PROTOCOL_FAST) != PROTOCOL_PAPER_310K_NACL:
        return

    # Paper-like defaults (user can still override by passing explicit flags).
    if not _flag_present(argv, "--temperature-k"):
        args.temperature_k = 310.0
    if not _flag_present(argv, "--friction-inv-ps"):
        args.friction_inv_ps = 1.0

    if not _flag_present(argv, "--box-length-nm"):
        args.box_length_nm = 0.0
    if not _flag_present(argv, "--solvent-padding-nm"):
        args.solvent_padding_nm = 1.0
    if not _flag_present(argv, "--ionic-strength-m"):
        args.ionic_strength_m = 0.15
    if not _flag_present(argv, "--solvent-positive-ion"):
        args.solvent_positive_ion = "Na+"
    if not _flag_present(argv, "--solvent-negative-ion"):
        args.solvent_negative_ion = "Cl-"
    if not _flag_present(argv, "--no-solvent-neutralize"):
        args.solvent_neutralize = True

    if not _flag_present(argv, "--nonbonded-cutoff-nm"):
        args.nonbonded_cutoff_nm = 1.2
    if not _flag_present(argv, "--switch-distance-nm"):
        args.switch_distance_nm = 1.0
    if not _flag_present(argv, "--no-switching-function"):
        args.use_switching_function = True

    if not _flag_present(argv, "--no-dispersion-correction"):
        args.use_dispersion_correction = True
    if not _flag_present(argv, "--pme-ewald-error-tolerance"):
        args.pme_ewald_error_tolerance = 5e-4
    if not _flag_present(argv, "--pme-grid-spacing-angstrom"):
        args.pme_grid_spacing_angstrom = 1.0

    if not _flag_present(argv, "--minimize-restrained-iters"):
        args.minimize_restrained_iters = 3000
    if not _flag_present(argv, "--minimize-iters"):
        args.minimize_iters = 97000
    if not _flag_present(argv, "--minimize-restraint-selection"):
        args.minimize_restraint_selection = "protein and not element H"
    if not _flag_present(argv, "--minimize-restraint-k-kj-mol-nm2"):
        args.minimize_restraint_k_kj_mol_nm2 = 10000.0

    if not _flag_present(argv, "--heat-steps"):
        args.heat_steps = 20000
    if not _flag_present(argv, "--heat-start-k"):
        args.heat_start_k = 10.0
    if not _flag_present(argv, "--heat-friction-inv-ps"):
        args.heat_friction_inv_ps = 1.0
    if not _flag_present(argv, "--heat-chunks"):
        args.heat_chunks = 20

    if not _flag_present(argv, "--equilibrate-npt-steps"):
        args.equilibrate_npt_steps = 500000
    if not _flag_present(argv, "--equilibrate-pressure-bar"):
        args.equilibrate_pressure_bar = 1.0
    if not _flag_present(argv, "--equilibrate-barostat-frequency"):
        args.equilibrate_barostat_frequency = 25
    if not _flag_present(argv, "--equilibrate-friction-inv-ps"):
        args.equilibrate_friction_inv_ps = 20.0


@dataclass(frozen=True)
class RmsdBiasManifest:
    pdb: str
    out_dir: str
    protocol: str
    implicit_solvent: bool
    topology_pdb: str
    selection: str
    k_kj_mol_nm2: float
    bias_max_energy_kj_mol: float
    r0_schedule_nm: list[float]
    steps_per_r0: int
    temperature_k: float
    friction_inv_ps: float
    timestep_fs: float
    minimize_restrained_iters: int
    minimize_iters: int
    heat_steps: int
    equilibrate_npt_steps: int
    dcd_path: str
    log_path: str
    pme_ewald_error_tolerance: float | None = None
    pme_grid_spacing_angstrom: float | None = None
    use_dispersion_correction: bool | None = None
    start_pool_pdb: str | None = None


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [float(a)]
    out = []
    for i in range(n):
        t = float(i) / float(n - 1)
        out.append((1.0 - t) * float(a) + t * float(b))
    return out


def _select_start_pool_indices_stratified(
    rmsd_nm: "list[float]",
    *,
    n: int,
    rmsd_min: float,
    rmsd_max: float,
    bins: int,
) -> list[int]:
    import random

    if n <= 0:
        raise ValueError("--start-pool-n must be > 0")
    if bins <= 0:
        raise ValueError("--start-pool-bins must be > 0")
    if rmsd_max <= rmsd_min:
        raise ValueError("--start-pool-rmsd-range must satisfy max > min")

    edges = _linspace(float(rmsd_min), float(rmsd_max), bins + 1)
    by_bin: list[list[int]] = [[] for _ in range(bins)]
    for i, x in enumerate(rmsd_nm):
        if not math.isfinite(float(x)):
            continue
        if x < rmsd_min or x > rmsd_max:
            continue
        if x == rmsd_max:
            b = bins - 1
        else:
            # edges are increasing; find first edge greater than x.
            b = 0
            while b + 1 < len(edges) and x >= edges[b + 1]:
                b += 1
            b = min(max(0, b), bins - 1)
        by_bin[b].append(i)

    # Target ~equal per bin, then fill remainder.
    per_bin = max(1, int(math.ceil(float(n) / float(bins))))
    rng = random.Random(0)
    chosen: list[int] = []
    for b in range(bins):
        idx = by_bin[b]
        if not idx:
            continue
        rng.shuffle(idx)
        chosen.extend(idx[:per_bin])
        if len(chosen) >= n:
            break

    # Fill with any remaining frames in-range.
    if len(chosen) < n:
        rest: list[int] = []
        for b in range(bins):
            rest.extend(by_bin[b])
        rest = [i for i in rest if i not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: max(0, n - len(chosen))])

    chosen = sorted(set(chosen))[:n]
    if not chosen:
        raise RuntimeError("Selected 0 start-pool frames; check RMSD range/bins and trajectory.")
    return chosen


def _is_good_fft_size(n: int) -> bool:
    if n <= 0:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def _next_good_fft_size(n: int) -> int:
    n = max(1, int(n))
    while not _is_good_fft_size(n):
        n += 1
    return n


def _pme_alpha_from_tolerance(*, cutoff_nm: float, tol: float) -> float:
    cutoff_nm = float(cutoff_nm)
    tol = float(tol)
    if not (math.isfinite(cutoff_nm) and cutoff_nm > 0):
        raise ValueError("cutoff_nm must be > 0")
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError("tol must be > 0")

    def err(alpha_inv_nm: float) -> float:
        return math.erfc(alpha_inv_nm * cutoff_nm) / cutoff_nm

    lo = 0.0
    hi = 1.0 / cutoff_nm
    while err(hi) > tol:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Failed to bracket PME alpha; check cutoff/tolerance.")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if err(mid) > tol:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _run_rmsd_biased_md(args: argparse.Namespace) -> int:
    _require_openmm_mdtraj()
    import mdtraj as md
    import numpy as np
    import openmm
    from openmm import unit
    from openmm.app import (
        DCDReporter,
        ForceField,
        HBonds,
        NoCutoff,
        PME,
        PDBFile,
        Modeller,
        Simulation,
        StateDataReporter,
    )

    pdb_path = Path(args.pdb).expanduser().resolve()
    if not pdb_path.exists():
        print(f"ERROR: --pdb not found: {pdb_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dcd_path = out_dir / (args.dcd_name or "biased_rmsd.dcd")
    log_path = out_dir / (args.log_name or "biased_rmsd.log")
    manifest_path = out_dir / (args.manifest_name or "biased_rmsd_manifest.json")
    topology_pdb_path = out_dir / (args.topology_pdb_name or "simulation_topology.pdb")

    pdb = PDBFile(str(pdb_path))

    modeller = Modeller(pdb.topology, pdb.positions)
    if not bool(args.implicit_solvent):
        if bool(getattr(args, "box_length_nm", 0.0) or 0.0) > 0:
            L = float(args.box_length_nm)
            modeller.addSolvent(
                ForceField(str(args.forcefield_protein), str(args.forcefield_water)),
                model="tip3p",
                boxSize=openmm.Vec3(L, L, L) * unit.nanometer,
                ionicStrength=float(args.ionic_strength_m) * unit.molar,
                positiveIon=str(args.solvent_positive_ion),
                negativeIon=str(args.solvent_negative_ion),
                neutralize=bool(args.solvent_neutralize),
            )
        else:
            modeller.addSolvent(
                ForceField(str(args.forcefield_protein), str(args.forcefield_water)),
                model="tip3p",
                padding=float(args.solvent_padding_nm) * unit.nanometer,
                ionicStrength=float(args.ionic_strength_m) * unit.molar,
                positiveIon=str(args.solvent_positive_ion),
                negativeIon=str(args.solvent_negative_ion),
                neutralize=bool(args.solvent_neutralize),
            )

        # Save solvated topology for downstream MDtraj loading / full-coordinate replay compatibility.
        with topology_pdb_path.open("w") as handle:
            PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    else:
        # In implicit mode, topology is unchanged; still snapshot for consistency.
        with topology_pdb_path.open("w") as handle:
            PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    # Choose RMSD atom indices on the *final* topology (explicit includes solvent).
    top_traj = md.load(str(topology_pdb_path))
    try:
        atom_idx = top_traj.topology.select(str(args.rmsd_selection))
    except Exception as exc:
        raise ValueError(f"Invalid --rmsd-selection {args.rmsd_selection!r}: {exc}") from exc
    if atom_idx is None or len(atom_idx) == 0:
        raise ValueError(f"--rmsd-selection matched 0 atoms: {args.rmsd_selection!r}")
    atom_idx = np.asarray(atom_idx, dtype=int)

    restraint_atom_idx: Optional[np.ndarray] = None
    if int(getattr(args, "minimize_restrained_iters", 0) or 0) > 0:
        try:
            rest_idx = top_traj.topology.select(str(args.minimize_restraint_selection))
        except Exception as exc:
            raise ValueError(
                f"Invalid --minimize-restraint-selection {args.minimize_restraint_selection!r}: {exc}"
            ) from exc
        if rest_idx is None or len(rest_idx) == 0:
            raise ValueError(f"--minimize-restraint-selection matched 0 atoms: {args.minimize_restraint_selection!r}")
        restraint_atom_idx = np.asarray(rest_idx, dtype=int)

    # Build forcefield/system
    if bool(args.implicit_solvent):
        ff = ForceField(str(args.forcefield_protein), str(args.forcefield_implicit))
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )
    else:
        ff = ForceField(str(args.forcefield_protein), str(args.forcefield_water))
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=float(args.nonbonded_cutoff_nm) * unit.nanometer,
            ewaldErrorTolerance=float(getattr(args, "pme_ewald_error_tolerance", 5e-4) or 5e-4),
            constraints=HBonds,
        )
        if bool(args.use_switching_function) and float(args.switch_distance_nm) >= float(args.nonbonded_cutoff_nm):
            raise ValueError("--switch-distance-nm must be < --nonbonded-cutoff-nm")
        for f in system.getForces():
            if not isinstance(f, openmm.NonbondedForce):
                continue
            f.setUseSwitchingFunction(bool(args.use_switching_function))
            if bool(args.use_switching_function):
                f.setSwitchingDistance(float(args.switch_distance_nm) * unit.nanometer)
            f.setUseDispersionCorrection(bool(getattr(args, "use_dispersion_correction", True)))
            if getattr(args, "pme_ewald_error_tolerance", None) is not None:
                f.setEwaldErrorTolerance(float(args.pme_ewald_error_tolerance))

            spacing_a = float(getattr(args, "pme_grid_spacing_angstrom", 0.0) or 0.0)
            if spacing_a > 0:
                box_vecs = modeller.topology.getPeriodicBoxVectors()
                if box_vecs is None:
                    raise RuntimeError("PME grid sizing requested, but periodic box vectors are missing.")
                spacing_nm = 0.1 * spacing_a

                def norm_nm(vec) -> float:
                    return math.sqrt(sum(float((vec[i] / unit.nanometer)) ** 2 for i in range(3)))

                lx = norm_nm(box_vecs[0])
                ly = norm_nm(box_vecs[1])
                lz = norm_nm(box_vecs[2])
                nx = _next_good_fft_size(int(math.ceil(lx / spacing_nm)))
                ny = _next_good_fft_size(int(math.ceil(ly / spacing_nm)))
                nz = _next_good_fft_size(int(math.ceil(lz / spacing_nm)))
                alpha = _pme_alpha_from_tolerance(
                    cutoff_nm=float(args.nonbonded_cutoff_nm),
                    tol=float(getattr(args, "pme_ewald_error_tolerance", 5e-4) or 5e-4),
                )
                f.setPMEParameters(float(alpha), int(nx), int(ny), int(nz))

    # RMSD collective variable to the reference positions, on selected atoms.
    rmsd_cv = openmm.RMSDForce(modeller.positions, atom_idx.tolist())
    bias = openmm.CustomCVForce("0.5*k*(rmsd-r0)^2")
    bias.addGlobalParameter("k", float(args.k_kj_mol_nm2))
    bias.addGlobalParameter("r0", float(args.r0_nm_min))
    bias.addCollectiveVariable("rmsd", rmsd_cv)
    bias.setForceGroup(RMSD_BIAS_FORCE_GROUP_ID)
    system.addForce(bias)

    if restraint_atom_idx is not None:
        rest = openmm.CustomExternalForce("0.5*k_rest*periodicdistance(x,y,z,x0,y0,z0)^2")
        rest.addGlobalParameter("k_rest", float(args.minimize_restraint_k_kj_mol_nm2))
        rest.addPerParticleParameter("x0")
        rest.addPerParticleParameter("y0")
        rest.addPerParticleParameter("z0")
        for i in restraint_atom_idx.tolist():
            pos = modeller.positions[int(i)]
            rest.addParticle(int(i), [pos[0] / unit.nanometer, pos[1] / unit.nanometer, pos[2] / unit.nanometer])
        system.addForce(rest)

    barostat: Optional[openmm.MonteCarloBarostat] = None
    if (not bool(args.implicit_solvent)) and int(getattr(args, "equilibrate_npt_steps", 0) or 0) > 0:
        barostat = openmm.MonteCarloBarostat(
            float(getattr(args, "equilibrate_pressure_bar", 1.0)) * unit.bar,
            float(args.temperature_k) * unit.kelvin,
            0,
        )
        system.addForce(barostat)

    integrator = openmm.LangevinMiddleIntegrator(
        float(args.temperature_k) * unit.kelvin,
        float(args.friction_inv_ps) / unit.picosecond,
        float(args.timestep_fs) * unit.femtoseconds,
    )
    platform = openmm.Platform.getPlatformByName(str(args.platform))
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Disable RMSD bias during minimization/heating/equilibration; enable right before the r0 schedule.
    sim.context.setParameter("k", 0.0)
    if barostat is not None:
        barostat.setFrequency(0)

    if restraint_atom_idx is not None and int(getattr(args, "minimize_restrained_iters", 0) or 0) > 0:
        sim.context.setParameter("k_rest", float(args.minimize_restraint_k_kj_mol_nm2))
        sim.minimizeEnergy(maxIterations=int(args.minimize_restrained_iters))
        sim.context.setParameter("k_rest", 0.0)

    if int(args.minimize_iters) > 0:
        sim.minimizeEnergy(maxIterations=int(args.minimize_iters))

    # Reporters.
    sim.reporters.append(DCDReporter(str(dcd_path), int(args.report_stride)))
    sim.reporters.append(
        StateDataReporter(
            str(log_path),
            int(args.report_stride),
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            separator="\t",
        )
    )

    heat_steps = int(getattr(args, "heat_steps", 0) or 0)
    if heat_steps > 0:
        init_temp = float(getattr(args, "heat_start_k", 0.0))
        if not math.isfinite(init_temp) or init_temp < 0:
            raise ValueError("--heat-start-k must be >= 0")
    else:
        init_temp = float(args.temperature_k)
    sim.context.setVelocitiesToTemperature(init_temp * unit.kelvin)

    if heat_steps > 0:
        if barostat is not None:
            barostat.setFrequency(0)
        integrator.setFriction(float(getattr(args, "heat_friction_inv_ps", args.friction_inv_ps)) / unit.picosecond)

        chunks = int(getattr(args, "heat_chunks", 20) or 20)
        chunks = max(1, min(chunks, heat_steps))
        base = heat_steps // chunks
        rem = heat_steps % chunks
        t0 = float(getattr(args, "heat_start_k", 0.0))
        t1 = float(args.temperature_k)
        for i in range(chunks):
            nsteps = base + (1 if i < rem else 0)
            if nsteps <= 0:
                continue
            if chunks == 1:
                temp = t1
            else:
                temp = t0 + (t1 - t0) * (float(i) / float(chunks - 1))
            integrator.setTemperature(float(temp) * unit.kelvin)
            sim.step(int(nsteps))
        integrator.setTemperature(float(args.temperature_k) * unit.kelvin)

    npt_steps = int(getattr(args, "equilibrate_npt_steps", 0) or 0)
    if npt_steps > 0:
        if bool(args.implicit_solvent):
            raise ValueError("--equilibrate-npt-steps requires explicit solvent (do not pass --implicit-solvent).")
        if barostat is None:
            raise RuntimeError("Internal error: NPT equilibration requested but barostat is missing.")
        integrator.setTemperature(float(args.temperature_k) * unit.kelvin)
        integrator.setFriction(float(getattr(args, "equilibrate_friction_inv_ps", args.friction_inv_ps)) / unit.picosecond)
        barostat.setFrequency(int(getattr(args, "equilibrate_barostat_frequency", 25)))
        sim.step(int(npt_steps))
        barostat.setFrequency(0)

    # Production (RMSD-bias) stage: NVT + r0 schedule.
    integrator.setTemperature(float(args.temperature_k) * unit.kelvin)
    integrator.setFriction(float(args.friction_inv_ps) / unit.picosecond)
    base_k = float(args.k_kj_mol_nm2)
    max_bias_energy = float(getattr(args, "bias_max_energy_kj_mol", 0.0) or 0.0)
    if (not math.isfinite(max_bias_energy)) or max_bias_energy < 0.0:
        max_bias_energy = 0.0
    sim.context.setParameter("k", float(base_k))

    # Run schedule: sweep target RMSD r0 from min to max (nm), changing every segment.
    r0_schedule = _linspace(float(args.r0_nm_min), float(args.r0_nm_max), int(args.r0_steps))
    for r0 in r0_schedule:
        bias.setGlobalParameterDefaultValue(1, float(r0))  # parameter index 1 is r0
        sim.context.setParameter("r0", float(r0))
        # Optional: cap the RMSD-bias energy by reducing k (energy is linear in k).
        if max_bias_energy > 0.0:
            try:
                sim.context.setParameter("k", float(base_k))
                state = sim.context.getState(getEnergy=True, groups={RMSD_BIAS_FORCE_GROUP_ID})
                e_kj = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                if math.isfinite(e_kj) and e_kj > max_bias_energy and e_kj > 0.0:
                    k_eff = float(base_k) * float(max_bias_energy) / float(e_kj)
                    k_eff = max(0.0, min(float(base_k), float(k_eff)))
                    sim.context.setParameter("k", float(k_eff))
            except Exception:
                pass
        sim.step(int(args.steps_per_r0))

    # Optional: build a start pool by sampling biased trajectory frames.
    start_pool_pdb: Optional[str] = None
    if bool(args.make_start_pool):
        # Load written DCD and compute RMSD over CA (or user selection).
        t = md.load(str(dcd_path), top=str(topology_pdb_path))
        ref_for_rmsd = md.load(str(topology_pdb_path))

        if args.extract_selection:
            try:
                keep = t.topology.select(str(args.extract_selection))
            except Exception as exc:
                raise ValueError(f"Invalid --extract-selection {args.extract_selection!r}: {exc}") from exc
            if keep is None or len(keep) == 0:
                raise ValueError(f"--extract-selection matched 0 atoms: {args.extract_selection!r}")
            t = t.atom_slice(keep)
            # Apply the same extraction to the reference so atom ordering matches.
            try:
                keep_ref = ref_for_rmsd.topology.select(str(args.extract_selection))
            except Exception as exc:
                raise ValueError(f"Invalid --extract-selection for ref {args.extract_selection!r}: {exc}") from exc
            if keep_ref is None or len(keep_ref) == 0:
                raise ValueError(f"--extract-selection matched 0 atoms in ref: {args.extract_selection!r}")
            ref_for_rmsd = ref_for_rmsd.atom_slice(keep_ref)

        try:
            rmsd_idx = t.topology.select(str(args.rmsd_selection))
        except Exception as exc:
            raise ValueError(f"Invalid --rmsd-selection {args.rmsd_selection!r}: {exc}") from exc
        if rmsd_idx is None or len(rmsd_idx) == 0:
            raise ValueError(f"--rmsd-selection matched 0 atoms in output trajectory: {args.rmsd_selection!r}")
        rmsd_vals = md.rmsd(t, ref_for_rmsd, atom_indices=list(map(int, rmsd_idx))).astype(np.float64)
        chosen = _select_start_pool_indices_stratified(
            rmsd_vals.tolist(),
            n=int(args.start_pool_n),
            rmsd_min=float(args.start_pool_rmsd_min),
            rmsd_max=float(args.start_pool_rmsd_max),
            bins=int(args.start_pool_bins),
        )
        pool = t[chosen]
        out_pdb = out_dir / (args.start_pool_name or "start_pool.pdb")
        pool.save_pdb(str(out_pdb))
        start_pool_pdb = str(out_pdb)

    manifest = RmsdBiasManifest(
        pdb=str(pdb_path),
        out_dir=str(out_dir),
        protocol=str(getattr(args, "protocol", PROTOCOL_FAST)),
        implicit_solvent=bool(args.implicit_solvent),
        topology_pdb=str(topology_pdb_path),
        selection=str(args.rmsd_selection),
        k_kj_mol_nm2=float(args.k_kj_mol_nm2),
        bias_max_energy_kj_mol=float(max_bias_energy),
        r0_schedule_nm=[float(x) for x in r0_schedule],
        steps_per_r0=int(args.steps_per_r0),
        temperature_k=float(args.temperature_k),
        friction_inv_ps=float(args.friction_inv_ps),
        timestep_fs=float(args.timestep_fs),
        minimize_restrained_iters=int(getattr(args, "minimize_restrained_iters", 0) or 0),
        minimize_iters=int(args.minimize_iters),
        heat_steps=int(getattr(args, "heat_steps", 0) or 0),
        equilibrate_npt_steps=int(getattr(args, "equilibrate_npt_steps", 0) or 0),
        pme_ewald_error_tolerance=float(getattr(args, "pme_ewald_error_tolerance", 0.0) or 0.0)
        if (not bool(args.implicit_solvent))
        else None,
        pme_grid_spacing_angstrom=float(getattr(args, "pme_grid_spacing_angstrom", 0.0) or 0.0)
        if (not bool(args.implicit_solvent))
        else None,
        use_dispersion_correction=bool(getattr(args, "use_dispersion_correction", True))
        if (not bool(args.implicit_solvent))
        else None,
        dcd_path=str(dcd_path),
        log_path=str(log_path),
        start_pool_pdb=start_pool_pdb,
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")

    print(f"[OK] Wrote biased trajectory: {dcd_path}")
    print(f"[OK] Wrote log: {log_path}")
    print(f"[OK] Wrote topology PDB: {topology_pdb_path}")
    print(f"[OK] Wrote manifest: {manifest_path}")
    if start_pool_pdb:
        print(f"[OK] Wrote start pool PDB: {start_pool_pdb}")
        print("[NOTE] Use start_pool.pdb only as a proposal source; do not treat the biased DCD as history.")
    return 0


def _add_rmsd_bias_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--platform", default="CPU", help="OpenMM platform (CPU/CUDA).")
    p.add_argument(
        "--protocol",
        choices=[PROTOCOL_FAST, PROTOCOL_PAPER_310K_NACL],
        default=PROTOCOL_FAST,
        help="Preset protocol (paper_310k_nacl approximates the requested CHARMM36/310K/0.15M NaCl setup).",
    )

    # Forcefield / integrator.
    # Default forcefields match this repo's common scripts (TRP-cage etc).
    p.add_argument("--forcefield-protein", default="charmm36.xml", help="Protein forcefield XML.")
    p.add_argument("--forcefield-water", default="charmm36/water.xml", help="Water forcefield XML (explicit solvent).")
    p.add_argument("--forcefield-implicit", default="implicit/gbn2.xml", help="Implicit solvent forcefield XML.")
    p.add_argument("--temperature-k", type=float, default=300.0, help="Temperature (K).")
    p.add_argument("--friction-inv-ps", type=float, default=10.0, help="Langevin friction (1/ps).")
    p.add_argument("--timestep-fs", type=float, default=2.0, help="Timestep (fs).")
    p.add_argument("--minimize-iters", type=int, default=200, help="Energy minimization iterations.")
    p.add_argument(
        "--minimize-restrained-iters",
        type=int,
        default=0,
        help="If >0, run a first minimization stage with positional restraints on --minimize-restraint-selection.",
    )
    p.add_argument(
        "--minimize-restraint-selection",
        default="protein and not element H",
        help="MDTraj selection for atoms restrained during --minimize-restrained-iters (ignored if 0).",
    )
    p.add_argument(
        "--minimize-restraint-k-kj-mol-nm2",
        type=float,
        default=1000.0,
        help="Positional restraint force constant (kJ/mol/nm^2) used during --minimize-restrained-iters.",
    )

    # Solvent mode: default explicit so the biased proposal matches downstream runs (and stores solvent coords).
    p.add_argument(
        "--implicit-solvent",
        action="store_true",
        help="Use implicit solvent instead of explicit solvent (explicit is default).",
    )

    p.add_argument(
        "--box-length-nm",
        type=float,
        default=4.48,
        help="Explicit solvent: cubic box length (nm). Set <=0 to use --solvent-padding-nm instead.",
    )
    p.add_argument(
        "--solvent-padding-nm",
        type=float,
        default=1.2,
        help="Explicit solvent: padding (nm) used when --box-length-nm <= 0.",
    )
    p.add_argument("--ionic-strength-m", type=float, default=0.15, help="Explicit solvent: ionic strength (M).")
    p.add_argument("--solvent-positive-ion", default="Na+", help="Explicit solvent: positive ion species (e.g., Na+).")
    p.add_argument("--solvent-negative-ion", default="Cl-", help="Explicit solvent: negative ion species (e.g., Cl-).")
    p.add_argument("--no-solvent-neutralize", dest="solvent_neutralize", action="store_false", help="Do not neutralize system.")
    p.set_defaults(solvent_neutralize=True)

    p.add_argument("--nonbonded-cutoff-nm", type=float, default=1.2, help="Explicit solvent: PME cutoff (nm).")
    p.add_argument("--switch-distance-nm", type=float, default=1.0, help="Explicit solvent: switching distance (nm).")
    p.add_argument("--no-switching-function", dest="use_switching_function", action="store_false", help="Disable switching function.")
    p.set_defaults(use_switching_function=True)
    p.add_argument(
        "--pme-ewald-error-tolerance",
        type=float,
        default=5e-4,
        help="Explicit solvent: PME Ewald error tolerance.",
    )
    p.add_argument(
        "--pme-grid-spacing-angstrom",
        type=float,
        default=0.0,
        help="Explicit solvent: if >0, set PME grid sizes to ~this spacing (Å) using FFT-friendly dimensions.",
    )
    p.add_argument(
        "--no-dispersion-correction",
        dest="use_dispersion_correction",
        action="store_false",
        help="Explicit solvent: disable long-range LJ (dispersion) correction.",
    )
    p.set_defaults(use_dispersion_correction=True)

    p.add_argument(
        "--heat-steps",
        type=int,
        default=0,
        help="If >0, run an NVT heating ramp from --heat-start-k to --temperature-k for this many steps.",
    )
    p.add_argument("--heat-start-k", type=float, default=10.0, help="Heating ramp start temperature (K).")
    p.add_argument("--heat-friction-inv-ps", type=float, default=1.0, help="Heating ramp Langevin friction (1/ps).")
    p.add_argument("--heat-chunks", type=int, default=20, help="Number of temperature update chunks during heating.")

    p.add_argument(
        "--equilibrate-npt-steps",
        type=int,
        default=0,
        help="If >0 (explicit only), run NPT equilibration for this many steps before the RMSD-bias schedule.",
    )
    p.add_argument("--equilibrate-pressure-bar", type=float, default=1.0, help="NPT equilibration pressure (bar).")
    p.add_argument("--equilibrate-barostat-frequency", type=int, default=25, help="NPT barostat frequency (steps).")
    p.add_argument("--equilibrate-friction-inv-ps", type=float, default=20.0, help="NPT equilibration Langevin friction (1/ps).")

    # RMSD bias.
    p.add_argument("--rmsd-selection", default="name CA", help="MDTraj selection for RMSD atoms.")
    p.add_argument("--k-kj-mol-nm2", type=float, default=5000.0, help="Harmonic RMSD force constant (kJ/mol/nm^2).")
    p.add_argument(
        "--bias-max-energy-kj-mol",
        dest="bias_max_energy_kj_mol",
        type=float,
        default=0.0,
        help=(
            "Optional cap for the RMSD-bias potential energy (kJ/mol). When >0, "
            "k is reduced on-the-fly per r0 segment to keep bias energy under the cap."
        ),
    )
    p.add_argument("--r0-nm-min", type=float, default=0.0, help="Min target RMSD (nm).")
    p.add_argument("--r0-nm-max", type=float, default=2.0, help="Max target RMSD (nm).")
    p.add_argument("--r0-steps", type=int, default=10, help="Number of target RMSD values between min/max.")
    p.add_argument("--steps-per-r0", type=int, default=20000, help="MD steps per target RMSD value.")
    p.add_argument("--report-stride", type=int, default=250, help="Frames/steps stride for DCD/log output.")

    p.add_argument("--dcd-name", default="biased_rmsd.dcd", help="Output DCD filename.")
    p.add_argument("--log-name", default="biased_rmsd.log", help="Output log filename.")
    p.add_argument("--manifest-name", default="biased_rmsd_manifest.json", help="Output manifest filename.")
    p.add_argument("--topology-pdb-name", default="simulation_topology.pdb", help="Output solvated topology PDB filename.")

    # Optional: make a start pool directly (recommended usage is still: biased traj -> start pool -> unbiased anchoring).
    p.add_argument("--make-start-pool", action="store_true", help="Also write a multi-model start_pool.pdb by sampling the biased trajectory.")
    p.add_argument("--extract-selection", default="protein", help="Atom selection to keep in start pool PDB.")
    p.add_argument("--start-pool-name", default="start_pool.pdb", help="Start pool PDB filename.")
    p.add_argument("--start-pool-n", type=int, default=50, help="Number of models in start pool PDB.")
    p.add_argument("--start-pool-bins", type=int, default=10, help="RMSD bins for stratified sampling.")
    p.add_argument("--start-pool-rmsd-min", type=float, default=0.0, help="Min RMSD (nm) for start pool sampling.")
    p.add_argument("--start-pool-rmsd-max", type=float, default=2.0, help="Max RMSD (nm) for start pool sampling.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gespo-bootstrap",
        description="GESPO bootstrap utilities (experimental).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    rb = sub.add_parser(
        "rmsd-bias",
        help="Run short RMSD-biased MD (explicit solvent by default) to generate proposal conformations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rb.add_argument("--pdb", required=True, help="Input sanitized PDB (protein-only recommended).")
    rb.add_argument("--out-dir", required=True, help="Output directory (biased traj + logs + optional start_pool.pdb).")
    _add_rmsd_bias_common_args(rb)

    rb_ab = sub.add_parser(
        "rmsd-bias-ab",
        help="Run RMSD-biased MD twice (endpoint A and B) to generate proposal conformations in both basins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rb_ab.add_argument("--pdb-a", required=True, help="Endpoint A PDB (sanitized; protein-only recommended).")
    rb_ab.add_argument("--pdb-b", required=True, help="Endpoint B PDB (sanitized; protein-only recommended).")
    rb_ab.add_argument("--out-dir", required=True, help="Base output directory (writes subdirs for A and B).")
    rb_ab.add_argument("--out-subdir-a", default="A", help="Subdirectory name under --out-dir for endpoint A outputs.")
    rb_ab.add_argument("--out-subdir-b", default="B", help="Subdirectory name under --out-dir for endpoint B outputs.")
    _add_rmsd_bias_common_args(rb_ab)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv_list)
    _apply_protocol_defaults(args, argv_list)
    if args.cmd == "rmsd-bias":
        return _run_rmsd_biased_md(args)
    if args.cmd == "rmsd-bias-ab":
        base = Path(str(args.out_dir)).expanduser().resolve()
        out_a = base / str(getattr(args, "out_subdir_a", "A") or "A")
        out_b = base / str(getattr(args, "out_subdir_b", "B") or "B")
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)

        args_a = argparse.Namespace(**vars(args))
        args_a.cmd = "rmsd-bias"
        args_a.pdb = str(getattr(args, "pdb_a"))
        args_a.out_dir = str(out_a)
        rc = _run_rmsd_biased_md(args_a)
        if int(rc) != 0:
            return int(rc)

        args_b = argparse.Namespace(**vars(args))
        args_b.cmd = "rmsd-bias"
        args_b.pdb = str(getattr(args, "pdb_b"))
        args_b.out_dir = str(out_b)
        return _run_rmsd_biased_md(args_b)
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
