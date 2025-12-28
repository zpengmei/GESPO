"""Targeted MD steering utilities for replay resets.

This module provides a lightweight "bring the retrieved frame to the suggested
conformation" routine used when replay-starts bypass the generator.

Design goals:
- Fast replay-start correction for implicit solvent and explicit-solvent full-coordinate replay.
- Two-stage schedule:
    1) Match global metrics (CA RMSD-to-reference and end-to-end CA distance)
    2) Then match backbone torsions (phi/psi CV targets)
- Bias forces are disabled (lambda=0) before returning to normal rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

import openmm
from openmm import CustomBondForce, CustomCVForce, CustomTorsionForce, RMSDForce
from openmm import CustomCentroidBondForce
from openmm.app import Simulation
from openmm.unit import kelvin, kilojoule_per_mole, nanometer


STEER_FORCE_GROUP_ID: int = 2


@dataclass
class ReplaySteeringForces:
    rmsd_force: Optional[RMSDForce]
    rmsd_bias: Optional[CustomCVForce]
    e2e_bias: Optional[CustomBondForce]
    torsion_bias: Optional[CustomTorsionForce]
    e2e_atoms: Optional[Tuple[int, int]]
    ca_particle_indices: Tuple[int, ...] = ()
    ref_ca_nm: Optional[np.ndarray] = None
    extra_distance_atoms: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    extra_distance_target_param: Dict[str, str] = field(default_factory=dict)
    extra_angle_groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    extra_angle_target_param: Dict[str, str] = field(default_factory=dict)

    lambda_global: str = "steer_lambda_g"
    lambda_torsion: str = "steer_lambda_t"
    rmsd_target: str = "steer_rmsd_r0"
    e2e_target: str = "steer_e2e_r0"


def _get_context_parameter(sim: Simulation, name: str) -> Optional[float]:
    try:
        return float(sim.context.getParameter(str(name)))
    except Exception:
        return None


def _steering_energy_kj(sim: Simulation) -> Optional[float]:
    try:
        state = sim.context.getState(getEnergy=True, groups={STEER_FORCE_GROUP_ID})
        return float(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
    except Exception:
        return None


def _current_rmsd_cv_nm(sim: Simulation, forces: ReplaySteeringForces) -> Optional[float]:
    if forces.rmsd_force is None:
        return None
    try:
        val = forces.rmsd_force.getRMSD(sim.context)
        # OpenMM versions differ: may return a Quantity or a plain float (in nm).
        if hasattr(val, "value_in_unit"):
            return float(val.value_in_unit(nanometer))
        return float(val)
    except Exception:
        return None


def _stage_diag(sim: Simulation, forces: ReplaySteeringForces) -> str:
    parts = []
    lam_g = _get_context_parameter(sim, forces.lambda_global)
    lam_t = _get_context_parameter(sim, forces.lambda_torsion)
    r0 = _get_context_parameter(sim, forces.rmsd_target)
    d0 = _get_context_parameter(sim, forces.e2e_target)
    rmsd_cv = _current_rmsd_cv_nm(sim, forces)
    e_kj = _steering_energy_kj(sim)
    if lam_g is not None:
        parts.append(f"λg={lam_g:.2f}")
    if lam_t is not None:
        parts.append(f"λt={lam_t:.2f}")
    if r0 is not None:
        parts.append(f"r0={r0:.3f}")
    if d0 is not None:
        parts.append(f"d0={d0:.3f}")
    if rmsd_cv is not None:
        parts.append(f"rmsd_cv={rmsd_cv:.3f}")
    if e_kj is not None:
        parts.append(f"E_steer={e_kj:.1f}kJ/mol")
    return f" ({' '.join(parts)})" if parts else ""


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _sanitize_suffix(name: str) -> str:
    s = str(name)
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in s)


def add_replay_steering_forces(
    system: openmm.System,
    *,
    ref_positions_nm: np.ndarray,
    ca_particle_indices: Sequence[int],
    e2e_atoms: Optional[Tuple[int, int]],
    cv_definitions: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    cv_targets_rad: Sequence[float],
    k_rmsd: float,
    k_e2e: float,
    k_torsion: float,
    extra_distance_pairs: Optional[Dict[str, Tuple[int, int]]] = None,
    k_dist: float = 0.0,
    extra_angle_triplets: Optional[Dict[str, Tuple[Sequence[int], Sequence[int], Sequence[int]]]] = None,
    k_theta: float = 0.0,
) -> ReplaySteeringForces:
    """Attach steering forces to `system`.

    Forces are added with lambdas initialized to 0.0 (disabled) so that they
    do not affect upstream energy checks unless explicitly enabled later.
    """
    ref_positions_nm = np.asarray(ref_positions_nm, dtype=np.float64).reshape(-1, 3)
    n_particles = int(ref_positions_nm.shape[0])
    ca_idx = [int(i) for i in ca_particle_indices if 0 <= int(i) < n_particles]
    ref_ca_nm = None
    if ref_positions_nm.size and ca_idx:
        try:
            ref_ca_nm = ref_positions_nm[np.asarray(ca_idx, dtype=np.int64)]
        except Exception:
            ref_ca_nm = None

    rmsd_force = None
    rmsd_bias = None
    if ref_positions_nm.size and len(ca_idx) >= 3:
        rmsd_force = RMSDForce(ref_positions_nm * nanometer, ca_idx)
        rmsd_bias = CustomCVForce("0.5*steer_lambda_g*steer_k_rmsd*(rmsd - steer_rmsd_r0)^2")
        rmsd_bias.addCollectiveVariable("rmsd", rmsd_force)
        rmsd_bias.addGlobalParameter("steer_lambda_g", 0.0)
        rmsd_bias.addGlobalParameter("steer_k_rmsd", float(k_rmsd))
        rmsd_bias.addGlobalParameter("steer_rmsd_r0", 0.0)
        rmsd_bias.setForceGroup(STEER_FORCE_GROUP_ID)
        system.addForce(rmsd_bias)

    e2e_bias = None
    if e2e_atoms is not None:
        a, b = int(e2e_atoms[0]), int(e2e_atoms[1])
        if 0 <= a < n_particles and 0 <= b < n_particles and a != b:
            e2e_bias = CustomBondForce("0.5*steer_lambda_g*steer_k_e2e*(r - steer_e2e_r0)^2")
            e2e_bias.addGlobalParameter("steer_lambda_g", 0.0)
            e2e_bias.addGlobalParameter("steer_k_e2e", float(k_e2e))
            e2e_bias.addGlobalParameter("steer_e2e_r0", 0.0)
            e2e_bias.addBond(a, b, [])
            e2e_bias.setForceGroup(STEER_FORCE_GROUP_ID)
            system.addForce(e2e_bias)

    torsion_bias = None
    if cv_definitions and cv_targets_rad:
        targets = np.asarray(list(cv_targets_rad), dtype=np.float64).reshape(-1)
        if targets.size == len(cv_definitions):
            torsion_bias = CustomTorsionForce(
                "0.5*steer_lambda_t*steer_k_torsion*(atan2(sin(theta-theta0), cos(theta-theta0)))^2"
            )
            torsion_bias.addGlobalParameter("steer_lambda_t", 0.0)
            torsion_bias.addGlobalParameter("steer_k_torsion", float(k_torsion))
            torsion_bias.addPerTorsionParameter("theta0")
            for (name, atoms), theta0 in zip(cv_definitions, targets):
                a, b, c, d = (int(atoms[0]), int(atoms[1]), int(atoms[2]), int(atoms[3]))
                torsion_bias.addTorsion(a, b, c, d, [float(theta0)])
            torsion_bias.setForceGroup(STEER_FORCE_GROUP_ID)
            system.addForce(torsion_bias)

    extra_distance_atoms: Dict[str, Tuple[int, int]] = {}
    extra_distance_target_param: Dict[str, str] = {}
    if extra_distance_pairs and float(k_dist) > 0.0:
        for name, pair in dict(extra_distance_pairs).items():
            try:
                a, b = int(pair[0]), int(pair[1])
            except Exception:
                continue
            if not (0 <= a < n_particles and 0 <= b < n_particles and a != b):
                continue
            suffix = _sanitize_suffix(name)
            target_param = f"steer_{suffix}_r0"
            try:
                dist_bias = CustomBondForce(f"0.5*steer_lambda_g*steer_k_dist*(r - {target_param})^2")
                dist_bias.addGlobalParameter("steer_lambda_g", 0.0)
                dist_bias.addGlobalParameter("steer_k_dist", float(k_dist))
                dist_bias.addGlobalParameter(target_param, 0.0)
                dist_bias.addBond(a, b, [])
                dist_bias.setForceGroup(STEER_FORCE_GROUP_ID)
                system.addForce(dist_bias)
                extra_distance_atoms[str(name)] = (a, b)
                extra_distance_target_param[str(name)] = str(target_param)
            except Exception:
                continue

    extra_angle_groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    extra_angle_target_param: Dict[str, str] = {}
    if extra_angle_triplets and float(k_theta) > 0.0:
        for name, trip in dict(extra_angle_triplets).items():
            try:
                g1 = np.asarray(list(trip[0]), dtype=np.int64).reshape(-1)
                g2 = np.asarray(list(trip[1]), dtype=np.int64).reshape(-1)
                g3 = np.asarray(list(trip[2]), dtype=np.int64).reshape(-1)
            except Exception:
                continue
            if g1.size == 0 or g2.size == 0 or g3.size == 0:
                continue
            if np.any((g1 < 0) | (g1 >= n_particles)):
                continue
            if np.any((g2 < 0) | (g2 >= n_particles)):
                continue
            if np.any((g3 < 0) | (g3 >= n_particles)):
                continue
            suffix = _sanitize_suffix(name)
            target_param = f"steer_{suffix}_r0"
            try:
                ang_bias = CustomCentroidBondForce(
                    3,
                    f"0.5*steer_lambda_g*steer_k_theta*(angle(g1,g2,g3) - {target_param})^2",
                )
                ang_bias.addGlobalParameter("steer_lambda_g", 0.0)
                ang_bias.addGlobalParameter("steer_k_theta", float(k_theta))
                ang_bias.addGlobalParameter(target_param, 0.0)

                grp1 = ang_bias.addGroup([int(i) for i in g1.tolist()])
                grp2 = ang_bias.addGroup([int(i) for i in g2.tolist()])
                grp3 = ang_bias.addGroup([int(i) for i in g3.tolist()])
                ang_bias.addBond([grp1, grp2, grp3], [])
                ang_bias.setForceGroup(STEER_FORCE_GROUP_ID)
                system.addForce(ang_bias)

                extra_angle_groups[str(name)] = (g1, g2, g3)
                extra_angle_target_param[str(name)] = str(target_param)
            except Exception:
                continue

    return ReplaySteeringForces(
        rmsd_force=rmsd_force,
        rmsd_bias=rmsd_bias,
        e2e_bias=e2e_bias,
        torsion_bias=torsion_bias,
        e2e_atoms=e2e_atoms,
        ca_particle_indices=tuple(ca_idx),
        ref_ca_nm=ref_ca_nm.astype(np.float64, copy=False) if isinstance(ref_ca_nm, np.ndarray) else None,
        extra_distance_atoms=extra_distance_atoms,
        extra_distance_target_param=extra_distance_target_param,
        extra_angle_groups=extra_angle_groups,
        extra_angle_target_param=extra_angle_target_param,
    )


def _kabsch_rmsd_nm(cur_nm: np.ndarray, ref_nm: np.ndarray) -> Optional[float]:
    try:
        cur = np.asarray(cur_nm, dtype=np.float64).reshape(-1, 3)
        ref = np.asarray(ref_nm, dtype=np.float64).reshape(-1, 3)
        if cur.shape != ref.shape or cur.shape[0] < 3:
            return None
        cur_c = cur - cur.mean(axis=0, keepdims=True)
        ref_c = ref - ref.mean(axis=0, keepdims=True)
        h = cur_c.T @ ref_c
        u, _, vt = np.linalg.svd(h)
        # For H = A^T B, the optimal rotation that aligns A -> B is R = U V^T.
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        cur_a = cur_c @ r
        diff = cur_a - ref_c
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    except Exception:
        return None


def _current_ca_rmsd_nm(sim: Simulation, forces: ReplaySteeringForces) -> Optional[float]:
    if forces.ref_ca_nm is None or not forces.ca_particle_indices:
        return None
    try:
        idx = np.asarray(forces.ca_particle_indices, dtype=np.int64)
        state = sim.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        cur_ca = np.asarray(pos_nm, dtype=np.float64)[idx]
        return _kabsch_rmsd_nm(cur_ca, forces.ref_ca_nm)
    except Exception:
        return None


def _current_e2e_nm(sim: Simulation, e2e_atoms: Tuple[int, int]) -> Optional[float]:
    try:
        a, b = int(e2e_atoms[0]), int(e2e_atoms[1])
        state = sim.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        return float(np.linalg.norm(pos_nm[a] - pos_nm[b]))
    except Exception:
        return None


def _current_distance_nm(sim: Simulation, a: int, b: int) -> Optional[float]:
    try:
        state = sim.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        return float(np.linalg.norm(pos_nm[int(a)] - pos_nm[int(b)]))
    except Exception:
        return None


def _current_centroid_angle_rad(
    sim: Simulation,
    g1: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
) -> Optional[float]:
    try:
        g1 = np.asarray(g1, dtype=np.int64).reshape(-1)
        g2 = np.asarray(g2, dtype=np.int64).reshape(-1)
        g3 = np.asarray(g3, dtype=np.int64).reshape(-1)
        if g1.size == 0 or g2.size == 0 or g3.size == 0:
            return None
        state = sim.context.getState(getPositions=True)
        pos_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(nanometer), dtype=np.float64)
        c1 = pos_nm[g1].mean(axis=0)
        c2 = pos_nm[g2].mean(axis=0)
        c3 = pos_nm[g3].mean(axis=0)
        v1 = c1 - c2
        v2 = c3 - c2
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.arccos(cosang))
    except Exception:
        return None


def steer_replay_to_targets(
    sim: Simulation,
    forces: ReplaySteeringForces,
    *,
    temperature_k: float,
    target_rmsd_nm: Optional[float],
    target_e2e_nm: Optional[float],
    extra_targets: Optional[Dict[str, float]] = None,
    global_steps: int,
    global_updates: int,
    torsion_steps: int,
    torsion_updates: int,
    release_steps: int,
    refine_minimize_iters: int = 0,
    global_lambda_during_torsion: float,
    verbose: bool = False,
) -> None:
    """Run staged steering in-place on `sim`."""
    tgt_rmsd = _safe_float(target_rmsd_nm)
    tgt_e2e = _safe_float(target_e2e_nm)
    if tgt_rmsd is not None:
        tgt_rmsd = max(0.0, tgt_rmsd)
    if tgt_e2e is not None:
        tgt_e2e = max(0.0, tgt_e2e)

    extra_targets = dict(extra_targets) if extra_targets else {}
    has_extra = bool(bool(forces.extra_distance_atoms) or bool(forces.extra_angle_groups))
    has_globals = bool(forces.rmsd_force is not None or forces.e2e_atoms is not None or has_extra)
    has_torsions = bool(forces.torsion_bias is not None)
    if not has_globals and not has_torsions:
        return

    # Initialize velocities for stable short steering.
    try:
        sim.context.setVelocitiesToTemperature(float(temperature_k) * kelvin)
    except Exception:
        pass

    # Measure start values.
    rmsd_start = _current_ca_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
    e2e_start = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
    extra_start: Dict[str, Optional[float]] = {}
    for name, (a, b) in (forces.extra_distance_atoms or {}).items():
        extra_start[str(name)] = _current_distance_nm(sim, int(a), int(b))
    for name, groups in (forces.extra_angle_groups or {}).items():
        try:
            g1, g2, g3 = groups
        except Exception:
            continue
        extra_start[str(name)] = _current_centroid_angle_rad(sim, g1, g2, g3)

    extra_target: Dict[str, Optional[float]] = {}
    for name in set(list(extra_start.keys()) + list(extra_targets.keys())):
        v = _safe_float(extra_targets.get(name))
        if v is None:
            extra_target[name] = None
            continue
        if str(name).lower().endswith("_rad"):
            v = max(0.0, min(float(np.pi), float(v)))
        else:
            v = max(0.0, float(v))
        extra_target[name] = v

    if verbose:
        msg = "[TargetedMD] Steering"
        if rmsd_start is not None and tgt_rmsd is not None:
            msg += f" RMSD {rmsd_start:.3f}→{tgt_rmsd:.3f} nm"
        if e2e_start is not None and tgt_e2e is not None:
            msg += f" E2E {e2e_start:.3f}→{tgt_e2e:.3f} nm"
        try:
            for name in sorted(extra_start.keys()):
                s0 = extra_start.get(name)
                t0 = extra_target.get(name)
                if s0 is None or t0 is None:
                    continue
                unit = "rad" if name.lower().endswith("_rad") else "nm"
                msg += f" {name} {s0:.3f}→{t0:.3f} {unit}"
        except Exception:
            pass
        msg += f" steps(g={int(global_steps)}, t={int(torsion_steps)}, rel={int(release_steps)})"
        print(msg, flush=True)

    # Stage 0: ensure torsions are off
    try:
        sim.context.setParameter(forces.lambda_torsion, 0.0)
    except Exception:
        pass

    # Stage 1: global metrics (RMSD/E2E).
    if has_globals and global_steps > 0 and global_updates > 0:
        n_updates = max(1, int(global_updates))
        steps_per = max(1, int(global_steps) // n_updates)

        # Defaults for missing targets: keep the current value.
        if tgt_rmsd is None:
            tgt_rmsd = rmsd_start
        if tgt_e2e is None:
            tgt_e2e = e2e_start
        if rmsd_start is None:
            rmsd_start = tgt_rmsd
        if e2e_start is None:
            e2e_start = tgt_e2e

        for u in range(n_updates):
            frac = float(u + 1) / float(n_updates)
            lam = frac
            try:
                sim.context.setParameter(forces.lambda_global, lam)
            except Exception:
                pass
            if tgt_rmsd is not None and rmsd_start is not None:
                r0 = (1.0 - frac) * float(rmsd_start) + frac * float(tgt_rmsd)
                try:
                    sim.context.setParameter(forces.rmsd_target, float(r0))
                except Exception:
                    pass
            if tgt_e2e is not None and e2e_start is not None:
                d0 = (1.0 - frac) * float(e2e_start) + frac * float(tgt_e2e)
                try:
                    sim.context.setParameter(forces.e2e_target, float(d0))
                except Exception:
                    pass
            # Extra globals (distances/angles).
            for name, param in (forces.extra_distance_target_param or {}).items():
                s0 = extra_start.get(str(name))
                if s0 is None:
                    continue
                t0 = extra_target.get(str(name))
                if t0 is None:
                    t0 = s0
                v0 = (1.0 - frac) * float(s0) + frac * float(t0)
                try:
                    sim.context.setParameter(str(param), float(v0))
                except Exception:
                    pass
            for name, param in (forces.extra_angle_target_param or {}).items():
                s0 = extra_start.get(str(name))
                if s0 is None:
                    continue
                t0 = extra_target.get(str(name))
                if t0 is None:
                    t0 = s0
                v0 = (1.0 - frac) * float(s0) + frac * float(t0)
                try:
                    sim.context.setParameter(str(param), float(v0))
                except Exception:
                    pass
            sim.step(steps_per)

        rem = int(global_steps) - steps_per * n_updates
        if rem > 0:
            sim.step(rem)

        # Pin targets at final values for the torsion stage.
        try:
            sim.context.setParameter(forces.rmsd_target, float(tgt_rmsd) if tgt_rmsd is not None else 0.0)
        except Exception:
            pass
        try:
            sim.context.setParameter(forces.e2e_target, float(tgt_e2e) if tgt_e2e is not None else 0.0)
        except Exception:
            pass
        for name, param in (forces.extra_distance_target_param or {}).items():
            t0 = extra_target.get(str(name))
            if t0 is None:
                t0 = extra_start.get(str(name))
            if t0 is None:
                continue
            try:
                sim.context.setParameter(str(param), float(t0))
            except Exception:
                pass
        for name, param in (forces.extra_angle_target_param or {}).items():
            t0 = extra_target.get(str(name))
            if t0 is None:
                t0 = extra_start.get(str(name))
            if t0 is None:
                continue
            try:
                sim.context.setParameter(str(param), float(t0))
            except Exception:
                pass
        if verbose:
            rmsd_mid = _current_ca_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
            e2e_mid = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
            parts = []
            if rmsd_mid is not None:
                parts.append(f"RMSD={rmsd_mid:.3f} nm")
            if e2e_mid is not None:
                parts.append(f"E2E={e2e_mid:.3f} nm")
            try:
                for name, (a, b) in (forces.extra_distance_atoms or {}).items():
                    v = _current_distance_nm(sim, int(a), int(b))
                    if v is not None:
                        parts.append(f"{name}={v:.3f} nm")
                for name, groups in (forces.extra_angle_groups or {}).items():
                    try:
                        g1, g2, g3 = groups
                    except Exception:
                        continue
                    v = _current_centroid_angle_rad(sim, g1, g2, g3)
                    if v is not None:
                        parts.append(f"{name}={v:.3f} rad")
            except Exception:
                pass
            if parts:
                print(f"[TargetedMD] After globals: {' '.join(parts)}{_stage_diag(sim, forces)}", flush=True)

    # Stage 2: torsions.
    if has_torsions and torsion_steps > 0 and torsion_updates > 0:
        try:
            sim.context.setParameter(forces.lambda_global, float(global_lambda_during_torsion))
        except Exception:
            pass
        n_updates = max(1, int(torsion_updates))
        steps_per = max(1, int(torsion_steps) // n_updates)
        for u in range(n_updates):
            frac = float(u + 1) / float(n_updates)
            try:
                sim.context.setParameter(forces.lambda_torsion, frac)
            except Exception:
                pass
            sim.step(steps_per)
        rem = int(torsion_steps) - steps_per * n_updates
        if rem > 0:
            sim.step(rem)
        if verbose:
            rmsd_mid2 = _current_ca_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
            e2e_mid2 = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
            parts = []
            if rmsd_mid2 is not None:
                parts.append(f"RMSD={rmsd_mid2:.3f} nm")
            if e2e_mid2 is not None:
                parts.append(f"E2E={e2e_mid2:.3f} nm")
            try:
                for name, (a, b) in (forces.extra_distance_atoms or {}).items():
                    v = _current_distance_nm(sim, int(a), int(b))
                    if v is not None:
                        parts.append(f"{name}={v:.3f} nm")
                for name, groups in (forces.extra_angle_groups or {}).items():
                    try:
                        g1, g2, g3 = groups
                    except Exception:
                        continue
                    v = _current_centroid_angle_rad(sim, g1, g2, g3)
                    if v is not None:
                        parts.append(f"{name}={v:.3f} rad")
            except Exception:
                pass
            if parts:
                print(f"[TargetedMD] After torsions: {' '.join(parts)}{_stage_diag(sim, forces)}", flush=True)

    # Optional refinement: energy minimization under active steering.
    refine_iters = int(refine_minimize_iters) if int(refine_minimize_iters) > 0 else 0
    if refine_iters > 0:
        # Ensure targets are pinned at the final requested values.
        try:
            if tgt_rmsd is not None:
                sim.context.setParameter(forces.rmsd_target, float(tgt_rmsd))
        except Exception:
            pass
        try:
            if tgt_e2e is not None:
                sim.context.setParameter(forces.e2e_target, float(tgt_e2e))
        except Exception:
            pass
        for name, param in (forces.extra_distance_target_param or {}).items():
            t0 = extra_target.get(str(name))
            if t0 is None:
                t0 = extra_start.get(str(name))
            if t0 is None:
                continue
            try:
                sim.context.setParameter(str(param), float(t0))
            except Exception:
                pass
        for name, param in (forces.extra_angle_target_param or {}).items():
            t0 = extra_target.get(str(name))
            if t0 is None:
                t0 = extra_start.get(str(name))
            if t0 is None:
                continue
            try:
                sim.context.setParameter(str(param), float(t0))
            except Exception:
                pass

        # Turn steering on strongly for minimization.
        try:
            sim.context.setParameter(forces.lambda_global, 1.0)
        except Exception:
            pass
        try:
            sim.context.setParameter(forces.lambda_torsion, 1.0 if has_torsions else 0.0)
        except Exception:
            pass

        if verbose:
            print(f"[TargetedMD] Refining with minimizeEnergy(maxIterations={refine_iters})", flush=True)
        try:
            sim.minimizeEnergy(maxIterations=refine_iters)
        except Exception as exc_min:
            if verbose:
                print(f"[TargetedMD] Warning: minimizeEnergy refine failed ({exc_min})", flush=True)

        if verbose:
            rmsd_ref = _current_ca_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
            e2e_ref = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
            parts = []
            if rmsd_ref is not None:
                parts.append(f"RMSD={rmsd_ref:.3f} nm")
            if e2e_ref is not None:
                parts.append(f"E2E={e2e_ref:.3f} nm")
            try:
                for name, (a, b) in (forces.extra_distance_atoms or {}).items():
                    v = _current_distance_nm(sim, int(a), int(b))
                    if v is not None:
                        parts.append(f"{name}={v:.3f} nm")
                for name, groups in (forces.extra_angle_groups or {}).items():
                    try:
                        g1, g2, g3 = groups
                    except Exception:
                        continue
                    v = _current_centroid_angle_rad(sim, g1, g2, g3)
                    if v is not None:
                        parts.append(f"{name}={v:.3f} rad")
            except Exception:
                pass
            if parts:
                print(f"[TargetedMD] After refine: {' '.join(parts)}{_stage_diag(sim, forces)}", flush=True)

    # Release: disable steering and relax briefly.
    try:
        sim.context.setParameter(forces.lambda_global, 0.0)
    except Exception:
        pass
    try:
        sim.context.setParameter(forces.lambda_torsion, 0.0)
    except Exception:
        pass
    if int(release_steps) > 0:
        sim.step(int(release_steps))
    if verbose:
        rmsd_end = _current_ca_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
        e2e_end = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
        parts = []
        if rmsd_end is not None:
            parts.append(f"RMSD={rmsd_end:.3f} nm")
        if e2e_end is not None:
            parts.append(f"E2E={e2e_end:.3f} nm")
        try:
            for name, (a, b) in (forces.extra_distance_atoms or {}).items():
                v = _current_distance_nm(sim, int(a), int(b))
                if v is not None:
                    parts.append(f"{name}={v:.3f} nm")
            for name, groups in (forces.extra_angle_groups or {}).items():
                try:
                    g1, g2, g3 = groups
                except Exception:
                    continue
                v = _current_centroid_angle_rad(sim, g1, g2, g3)
                if v is not None:
                    parts.append(f"{name}={v:.3f} rad")
        except Exception:
            pass
        if parts:
            print(f"[TargetedMD] After release: {' '.join(parts)}{_stage_diag(sim, forces)}", flush=True)
