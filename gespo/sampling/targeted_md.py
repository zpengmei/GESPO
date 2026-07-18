"""Targeted MD steering utilities for replay resets.

This module provides a lightweight "bring the retrieved frame to the suggested
conformation" routine used when replay-starts bypass the generator.

Design goals:
- Fast replay-start correction for implicit solvent and explicit-solvent full-coordinate replay.
- Two-stage schedule:
    1) Match global metrics (CA RMSD-to-reference, end-to-end CA distance, optional CA Rg)
    2) Then match backbone torsions (phi/psi CV targets)
- Bias forces are disabled (lambda=0) before returning to normal rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import openmm
from openmm import CustomBondForce, CustomCVForce, CustomTorsionForce, RMSDForce
from openmm import CustomCentroidBondForce, CustomCompoundBondForce
from openmm.app import Simulation
from openmm.unit import kelvin, kilojoule_per_mole, nanometer

# Steering forces are assigned to dedicated OpenMM force groups so we can query/cap
# their potential energies without touching the rest of the system energy.
STEER_GLOBAL_FORCE_GROUP_ID: int = 2
STEER_TORSION_FORCE_GROUP_ID: int = 3

# Back-compat alias (historically all steering forces used group 2).
STEER_FORCE_GROUP_ID: int = STEER_GLOBAL_FORCE_GROUP_ID


@dataclass
class ReplaySteeringForces:
    rmsd_force: Optional[RMSDForce]
    rmsd_bias: Optional[CustomCVForce]
    # Optional: path sampling may provide separate RMSD-to-endpoint-A/B targets.
    rmsd_force_a: Optional[RMSDForce] = None
    rmsd_bias_a: Optional[CustomCVForce] = None
    rmsd_force_b: Optional[RMSDForce] = None
    rmsd_bias_b: Optional[CustomCVForce] = None
    e2e_bias: Optional[CustomBondForce] = None
    rg_bias: Optional[CustomCompoundBondForce] = None
    torsion_bias: Optional[CustomTorsionForce] = None
    e2e_atoms: Optional[Tuple[int, int]] = None
    ca_particle_indices: Tuple[int, ...] = ()
    ref_ca_nm: Optional[np.ndarray] = None
    rmsd_particle_indices: Tuple[int, ...] = ()
    # Reference positions (nm) for the particle subset used by the primary RMSD bias.
    # This is used for diagnostics because the RMSDForce CV may not be queryable when
    # embedded inside a CustomCVForce.
    rmsd_ref_nm: Optional[np.ndarray] = None
    rmsd_ref_a_nm: Optional[np.ndarray] = None
    rmsd_ref_b_nm: Optional[np.ndarray] = None
    extra_distance_atoms: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    extra_distance_target_param: Dict[str, str] = field(default_factory=dict)
    extra_angle_groups: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    extra_angle_target_param: Dict[str, str] = field(default_factory=dict)

    lambda_global: str = "steer_lambda_g"
    lambda_torsion: str = "steer_lambda_t"
    rmsd_target: str = "steer_rmsd_r0"
    rmsd_target_a: str = "steer_rmsd_a_r0"
    rmsd_target_b: str = "steer_rmsd_b_r0"
    e2e_target: str = "steer_e2e_r0"
    rg_target: str = "steer_rg_r0"


def _get_context_parameter(sim: Simulation, name: str) -> Optional[float]:
    try:
        return float(sim.context.getParameter(str(name)))
    except Exception:
        return None


def _steering_energy_kj(sim: Simulation, force_group_id: int) -> Optional[float]:
    try:
        state = sim.context.getState(getEnergy=True, groups={int(force_group_id)})
        return float(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
    except Exception:
        return None


def _current_rmsd_cv_nm(sim: Simulation, rmsd_force: Optional[RMSDForce]) -> Optional[float]:
    if rmsd_force is None:
        return None
    try:
        val = rmsd_force.getRMSD(sim.context)
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
    r0a = _get_context_parameter(sim, forces.rmsd_target_a)
    r0b = _get_context_parameter(sim, forces.rmsd_target_b)
    d0 = _get_context_parameter(sim, forces.e2e_target)
    g0 = _get_context_parameter(sim, forces.rg_target)
    rmsd_cv = _current_rmsd_cv_nm(sim, forces.rmsd_force)
    rmsd_a_cv = _current_rmsd_cv_nm(sim, forces.rmsd_force_a)
    rmsd_b_cv = _current_rmsd_cv_nm(sim, forces.rmsd_force_b)
    rmsd_sel = None
    if rmsd_cv is None and forces.rmsd_ref_nm is not None and forces.rmsd_particle_indices:
        rmsd_sel = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_nm)
    e_g_kj = _steering_energy_kj(sim, STEER_GLOBAL_FORCE_GROUP_ID)
    e_t_kj = _steering_energy_kj(sim, STEER_TORSION_FORCE_GROUP_ID)
    if lam_g is not None:
        parts.append(f"λg={lam_g:.2f}")
    if lam_t is not None:
        parts.append(f"λt={lam_t:.2f}")
    if r0 is not None:
        parts.append(f"r0={r0:.3f}")
    if r0a is not None:
        parts.append(f"r0a={r0a:.3f}")
    if r0b is not None:
        parts.append(f"r0b={r0b:.3f}")
    if d0 is not None:
        parts.append(f"d0={d0:.3f}")
    if g0 is not None:
        parts.append(f"g0={g0:.3f}")
    if rmsd_cv is not None:
        parts.append(f"rmsd_cv={rmsd_cv:.3f}")
    if rmsd_sel is not None:
        parts.append(f"rmsd_sel={rmsd_sel:.3f}")
    if rmsd_a_cv is not None:
        parts.append(f"rmsd_a_cv={rmsd_a_cv:.3f}")
    if rmsd_b_cv is not None:
        parts.append(f"rmsd_b_cv={rmsd_b_cv:.3f}")
    if e_g_kj is not None:
        parts.append(f"E_g={e_g_kj:.1f}kJ/mol")
    if e_t_kj is not None:
        parts.append(f"E_t={e_t_kj:.1f}kJ/mol")
    return f" ({' '.join(parts)})" if parts else ""


def _cap_lambda_by_energy(
    sim: Simulation,
    *,
    lambda_param: str,
    force_group_id: int,
    desired_lambda: float,
    max_energy_kj_mol: float,
) -> float:
    """Set `lambda_param` and (optionally) reduce it so steering energy <= cap.

    This works because each steering term is linear in its lambda (λ) parameter.
    """

    lam = float(desired_lambda)
    cap = float(max_energy_kj_mol)
    if not (np.isfinite(lam) and lam >= 0.0):
        lam = 0.0
    if not (np.isfinite(cap) and cap > 0.0):
        try:
            sim.context.setParameter(str(lambda_param), float(lam))
        except Exception:
            pass
        return lam

    try:
        sim.context.setParameter(str(lambda_param), float(lam))
    except Exception:
        return lam

    e_kj = _steering_energy_kj(sim, int(force_group_id))
    if e_kj is None or (not np.isfinite(e_kj)) or e_kj <= 0.0 or e_kj <= cap or lam <= 0.0:
        return lam

    lam_capped = lam * (cap / float(e_kj))
    # Numerical guard: keep in [0, desired_lambda].
    lam_capped = max(0.0, min(float(lam), float(lam_capped)))
    try:
        sim.context.setParameter(str(lambda_param), float(lam_capped))
    except Exception:
        return lam
    return lam_capped


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


def _to_vec3_quantity_nm(positions_nm: np.ndarray):
    arr = np.asarray(positions_nm, dtype=np.float64).reshape(-1, 3)
    # Using explicit Vec3 objects avoids ambiguous SWIG overload resolution with numpy arrays.
    vecs = [openmm.Vec3(float(x), float(y), float(z)) for x, y, z in arr]
    return vecs * nanometer


def add_replay_steering_forces(
    system: openmm.System,
    *,
    ref_positions_nm: np.ndarray,
    ca_particle_indices: Sequence[int],
    rmsd_particle_indices: Optional[Sequence[int]] = None,
    rmsd_ref_positions_nm: Optional[np.ndarray] = None,
    rmsd_ref_positions_a_nm: Optional[np.ndarray] = None,
    rmsd_ref_positions_b_nm: Optional[np.ndarray] = None,
    e2e_atoms: Optional[Tuple[int, int]],
    cv_definitions: Sequence[Tuple[str, Tuple[int, int, int, int]]],
    cv_targets_rad: Sequence[float],
    cv_weights: Optional[Sequence[float]] = None,
    k_rmsd: float,
    k_rmsd_a: Optional[float] = None,
    k_rmsd_b: Optional[float] = None,
    k_e2e: float,
    k_torsion: float,
    k_rg: float = 0.0,
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
    system_n = int(system.getNumParticles())
    ca_idx = [int(i) for i in ca_particle_indices if 0 <= int(i) < n_particles]

    rmsd_idx_src = rmsd_particle_indices if rmsd_particle_indices is not None else ca_idx
    rmsd_idx = [int(i) for i in rmsd_idx_src if 0 <= int(i) < n_particles]
    rmsd_ref_nm = None
    if rmsd_ref_positions_nm is not None:
        try:
            arr = np.asarray(rmsd_ref_positions_nm, dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] == len(rmsd_idx) and len(rmsd_idx) >= 3:
                rmsd_ref_nm = arr
        except Exception:
            rmsd_ref_nm = None

    rmsd_ref_a_nm = None
    if rmsd_ref_positions_a_nm is not None:
        try:
            arr = np.asarray(rmsd_ref_positions_a_nm, dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] == len(rmsd_idx) and len(rmsd_idx) >= 3:
                rmsd_ref_a_nm = arr
        except Exception:
            rmsd_ref_a_nm = None

    rmsd_ref_b_nm = None
    if rmsd_ref_positions_b_nm is not None:
        try:
            arr = np.asarray(rmsd_ref_positions_b_nm, dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] == len(rmsd_idx) and len(rmsd_idx) >= 3:
                rmsd_ref_b_nm = arr
        except Exception:
            rmsd_ref_b_nm = None

    ref_ca_nm = None
    if ref_positions_nm.size and ca_idx:
        try:
            ref_ca_nm = ref_positions_nm[np.asarray(ca_idx, dtype=np.int64)]
        except Exception:
            ref_ca_nm = None

    rmsd_force = None
    rmsd_bias = None
    rmsd_force_a = None
    rmsd_bias_a = None
    rmsd_force_b = None
    rmsd_bias_b = None
    if ref_positions_nm.size and len(rmsd_idx) >= 3 and float(k_rmsd) > 0.0 and system_n == n_particles:
        # IMPORTANT: OpenMM will throw at Context creation time if the number of reference
        # positions does not match the number of particles in the System.
        # The most robust approach across OpenMM versions is:
        #   - Always construct RMSDForce with a full-length reference (System-sized)
        #   - Optionally restrict to a subset via setParticles(), if available
        use_subset = len(rmsd_idx) != system_n

        def _full_ref_with_subset(subset_ref_nm: Optional[np.ndarray]) -> np.ndarray:
            full = np.array(ref_positions_nm, dtype=np.float64, copy=True)
            if subset_ref_nm is None:
                return full
            try:
                full[np.asarray(rmsd_idx, dtype=np.int64)] = np.asarray(subset_ref_nm, dtype=np.float64)
            except Exception:
                pass
            return full

        def _make_rmsd_force(full_ref_nm: np.ndarray) -> Optional[RMSDForce]:
            try:
                ref_q = _to_vec3_quantity_nm(full_ref_nm)
                f = RMSDForce(ref_q)
            except Exception:
                return None
            if use_subset:
                if hasattr(f, "setParticles"):
                    try:
                        f.setParticles([int(i) for i in rmsd_idx])
                    except Exception:
                        return None
                else:
                    # No subset API available; safest is to disable RMSD rather than
                    # accidentally steering solvent-heavy explicit systems.
                    return None
            return f

        def _add_rmsd_bias(
            *,
            cv_name: str,
            force: RMSDForce,
            k_param: str,
            r0_param: str,
            k_value: float,
        ) -> Optional[CustomCVForce]:
            expr = f"0.5*steer_lambda_g*{k_param}*({cv_name} - {r0_param})^2"
            try:
                bias = CustomCVForce(expr)
                bias.addCollectiveVariable(str(cv_name), force)
                bias.addGlobalParameter("steer_lambda_g", 0.0)
                bias.addGlobalParameter(str(k_param), float(k_value))
                bias.addGlobalParameter(str(r0_param), 0.0)
                bias.setForceGroup(STEER_GLOBAL_FORCE_GROUP_ID)
                system.addForce(bias)
                return bias
            except Exception:
                return None

        # Legacy single RMSD: allow a subset reference for the selected particles.
        full_ref = _full_ref_with_subset(rmsd_ref_nm)
        rmsd_force = _make_rmsd_force(full_ref)
        if rmsd_force is not None:
            rmsd_bias = _add_rmsd_bias(
                cv_name="rmsd",
                force=rmsd_force,
                k_param="steer_k_rmsd",
                r0_param="steer_rmsd_r0",
                k_value=float(k_rmsd),
            )
            if rmsd_bias is None:
                rmsd_force = None

        # Optional RMSD-to-endpoint-A/B biases (path globals).
        if rmsd_ref_a_nm is not None:
            rmsd_force_a = _make_rmsd_force(_full_ref_with_subset(rmsd_ref_a_nm))
            if rmsd_force_a is not None:
                rmsd_bias_a = _add_rmsd_bias(
                    cv_name="rmsd_a",
                    force=rmsd_force_a,
                    k_param="steer_k_rmsd_a",
                    r0_param="steer_rmsd_a_r0",
                    k_value=float(k_rmsd_a if k_rmsd_a is not None else k_rmsd),
                )
                if rmsd_bias_a is None:
                    rmsd_force_a = None
        if rmsd_ref_b_nm is not None:
            rmsd_force_b = _make_rmsd_force(_full_ref_with_subset(rmsd_ref_b_nm))
            if rmsd_force_b is not None:
                rmsd_bias_b = _add_rmsd_bias(
                    cv_name="rmsd_b",
                    force=rmsd_force_b,
                    k_param="steer_k_rmsd_b",
                    r0_param="steer_rmsd_b_r0",
                    k_value=float(k_rmsd_b if k_rmsd_b is not None else k_rmsd),
                )
                if rmsd_bias_b is None:
                    rmsd_force_b = None

    e2e_bias = None
    if e2e_atoms is not None and float(k_e2e) > 0.0:
        a, b = int(e2e_atoms[0]), int(e2e_atoms[1])
        if 0 <= a < n_particles and 0 <= b < n_particles and a != b:
            e2e_bias = CustomBondForce("0.5*steer_lambda_g*steer_k_e2e*(r - steer_e2e_r0)^2")
            e2e_bias.addGlobalParameter("steer_lambda_g", 0.0)
            e2e_bias.addGlobalParameter("steer_k_e2e", float(k_e2e))
            e2e_bias.addGlobalParameter("steer_e2e_r0", 0.0)
            e2e_bias.addBond(a, b, [])
            e2e_bias.setForceGroup(STEER_GLOBAL_FORCE_GROUP_ID)
            system.addForce(e2e_bias)
        else:
            e2e_atoms = None

    rg_bias = None
    if ca_idx and float(k_rg) > 0.0:
        n_ca = int(len(ca_idx))
        if n_ca >= 2:
            # NOTE: CustomCompoundBondForce does not support semicolon-separated
            # subexpressions on all platforms (e.g., CPU context creation fails),
            # so we build a semicolon-free expression by inlining COM terms.
            xsum = "+".join(f"x{i}" for i in range(1, n_ca + 1))
            ysum = "+".join(f"y{i}" for i in range(1, n_ca + 1))
            zsum = "+".join(f"z{i}" for i in range(1, n_ca + 1))
            xcom = f"(({xsum})/{n_ca})"
            ycom = f"(({ysum})/{n_ca})"
            zcom = f"(({zsum})/{n_ca})"
            sq_terms = [
                f"(x{i}-{xcom})^2+(y{i}-{ycom})^2+(z{i}-{zcom})^2" for i in range(1, n_ca + 1)
            ]
            sumsq = f"({'+'.join(sq_terms)})"
            rg = f"sqrt(({sumsq})/{n_ca})"
            rg_expr = f"0.5*steer_lambda_g*steer_k_rg*({rg} - steer_rg_r0)^2"
            try:
                rg_bias = CustomCompoundBondForce(n_ca, rg_expr)
                rg_bias.addGlobalParameter("steer_lambda_g", 0.0)
                rg_bias.addGlobalParameter("steer_k_rg", float(k_rg))
                rg_bias.addGlobalParameter("steer_rg_r0", 0.0)
                rg_bias.addBond([int(i) for i in ca_idx], [])
                rg_bias.setForceGroup(STEER_GLOBAL_FORCE_GROUP_ID)
                system.addForce(rg_bias)
            except Exception:
                rg_bias = None

    torsion_bias = None
    if cv_definitions and cv_targets_rad and float(k_torsion) > 0.0:
        targets = np.asarray(list(cv_targets_rad), dtype=np.float64).reshape(-1)
        if targets.size == len(cv_definitions):
            weights = None
            if cv_weights is not None:
                try:
                    w = np.asarray(list(cv_weights), dtype=np.float64).reshape(-1)
                    if w.size == targets.size:
                        weights = w
                except Exception:
                    weights = None
            if weights is None:
                weights = np.ones_like(targets, dtype=np.float64)

            torsion_bias = CustomTorsionForce(
                "0.5*steer_lambda_t*steer_k_torsion*w*(atan2(sin(theta-theta0), cos(theta-theta0)))^2"
            )
            torsion_bias.addGlobalParameter("steer_lambda_t", 0.0)
            torsion_bias.addGlobalParameter("steer_k_torsion", float(k_torsion))
            torsion_bias.addPerTorsionParameter("theta0")
            torsion_bias.addPerTorsionParameter("w")
            for (name, atoms), theta0, w in zip(cv_definitions, targets, weights):
                a, b, c, d = (int(atoms[0]), int(atoms[1]), int(atoms[2]), int(atoms[3]))
                torsion_bias.addTorsion(a, b, c, d, [float(theta0), float(w)])
            torsion_bias.setForceGroup(STEER_TORSION_FORCE_GROUP_ID)
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
                dist_bias.setForceGroup(STEER_GLOBAL_FORCE_GROUP_ID)
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
                ang_bias.setForceGroup(STEER_GLOBAL_FORCE_GROUP_ID)
                system.addForce(ang_bias)

                extra_angle_groups[str(name)] = (g1, g2, g3)
                extra_angle_target_param[str(name)] = str(target_param)
            except Exception:
                continue

    return ReplaySteeringForces(
        rmsd_force=rmsd_force,
        rmsd_bias=rmsd_bias,
        rmsd_force_a=rmsd_force_a,
        rmsd_bias_a=rmsd_bias_a,
        rmsd_force_b=rmsd_force_b,
        rmsd_bias_b=rmsd_bias_b,
        e2e_bias=e2e_bias,
        rg_bias=rg_bias,
        torsion_bias=torsion_bias,
        e2e_atoms=e2e_atoms,
        ca_particle_indices=tuple(ca_idx),
        ref_ca_nm=ref_ca_nm.astype(np.float64, copy=False) if isinstance(ref_ca_nm, np.ndarray) else None,
        rmsd_particle_indices=tuple(int(i) for i in rmsd_idx),
        rmsd_ref_nm=rmsd_ref_nm.astype(np.float64, copy=False) if isinstance(rmsd_ref_nm, np.ndarray) else None,
        rmsd_ref_a_nm=rmsd_ref_a_nm.astype(np.float64, copy=False) if isinstance(rmsd_ref_a_nm, np.ndarray) else None,
        rmsd_ref_b_nm=rmsd_ref_b_nm.astype(np.float64, copy=False) if isinstance(rmsd_ref_b_nm, np.ndarray) else None,
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


def _current_primary_rmsd_nm(sim: Simulation, forces: ReplaySteeringForces) -> Optional[float]:
    """Return an RMSD diagnostic consistent with the primary RMSD bias reference."""
    if forces.rmsd_ref_nm is not None and forces.rmsd_particle_indices:
        v = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_nm)
        if v is not None:
            return v
    return _current_ca_rmsd_nm(sim, forces)


def _current_subset_rmsd_nm(sim: Simulation, atom_indices: Sequence[int], ref_nm: np.ndarray) -> Optional[float]:
    try:
        idx = np.asarray(list(atom_indices), dtype=np.int64).reshape(-1)
        if idx.size < 3:
            return None
        ref = np.asarray(ref_nm, dtype=np.float64).reshape(-1, 3)
        if ref.shape[0] != int(idx.size):
            return None
        state = sim.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        cur = np.asarray(pos_nm, dtype=np.float64)[idx]
        return _kabsch_rmsd_nm(cur, ref)
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


def _current_rg_nm(sim: Simulation, forces: ReplaySteeringForces) -> Optional[float]:
    if not forces.ca_particle_indices:
        return None
    try:
        idx = np.asarray(forces.ca_particle_indices, dtype=np.int64)
        state = sim.context.getState(getPositions=True)
        pos_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(nanometer), dtype=np.float64)
        ca = pos_nm[idx]
        if ca.ndim != 2 or ca.shape[0] < 2:
            return None
        centered = ca - ca.mean(axis=0, keepdims=True)
        rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        return rg if np.isfinite(rg) else None
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
    target_rmsd_nm: Optional[float] = None,
    target_rmsd_a_nm: Optional[float] = None,
    target_rmsd_b_nm: Optional[float] = None,
    target_e2e_nm: Optional[float] = None,
    target_rg_nm: Optional[float] = None,
    extra_targets: Optional[Dict[str, float]] = None,
    global_steps: int,
    global_updates: int,
    torsion_steps: int,
    torsion_updates: int,
    release_steps: int,
    refine_minimize_iters: int = 0,
    max_energy_kj_mol: float = 0.0,
    global_lambda_during_torsion: float,
    verbose: bool = False,
) -> None:
    """Run staged steering in-place on `sim`."""

    tgt_rmsd = _safe_float(target_rmsd_nm)
    tgt_rmsd_a = _safe_float(target_rmsd_a_nm)
    tgt_rmsd_b = _safe_float(target_rmsd_b_nm)
    tgt_e2e = _safe_float(target_e2e_nm)
    tgt_rg = _safe_float(target_rg_nm)
    if tgt_rmsd is not None:
        tgt_rmsd = max(0.0, tgt_rmsd)
    if tgt_rmsd_a is not None:
        tgt_rmsd_a = max(0.0, tgt_rmsd_a)
    if tgt_rmsd_b is not None:
        tgt_rmsd_b = max(0.0, tgt_rmsd_b)
    if tgt_e2e is not None:
        tgt_e2e = max(0.0, tgt_e2e)
    if tgt_rg is not None:
        tgt_rg = max(0.0, tgt_rg)

    extra_targets = dict(extra_targets) if extra_targets else {}
    has_extra = bool(bool(forces.extra_distance_atoms) or bool(forces.extra_angle_groups))
    has_rmsd = bool(
        forces.rmsd_bias is not None
        or forces.rmsd_bias_a is not None
        or forces.rmsd_bias_b is not None
    )
    has_globals = bool(has_rmsd or forces.e2e_atoms is not None or forces.rg_bias is not None or has_extra)
    has_torsions = bool(forces.torsion_bias is not None)
    if not has_globals and not has_torsions:
        return

    try:
        sim.context.setVelocitiesToTemperature(float(temperature_k) * kelvin)
    except Exception:
        pass

    rmsd_start = _current_primary_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
    rmsd_a_start = None
    rmsd_b_start = None
    if forces.rmsd_bias_a is not None:
        rmsd_a_start = _current_rmsd_cv_nm(sim, forces.rmsd_force_a)
        if rmsd_a_start is None and forces.rmsd_ref_a_nm is not None and forces.rmsd_particle_indices:
            rmsd_a_start = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_a_nm)
    if forces.rmsd_bias_b is not None:
        rmsd_b_start = _current_rmsd_cv_nm(sim, forces.rmsd_force_b)
        if rmsd_b_start is None and forces.rmsd_ref_b_nm is not None and forces.rmsd_particle_indices:
            rmsd_b_start = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_b_nm)
    e2e_start = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
    rg_start = _current_rg_nm(sim, forces)

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
        if forces.rmsd_bias is not None and rmsd_start is not None and tgt_rmsd is not None:
            msg += f" RMSD {rmsd_start:.3f}→{tgt_rmsd:.3f} nm"
        if forces.rmsd_bias_a is not None and rmsd_a_start is not None and tgt_rmsd_a is not None:
            msg += f" RMSD_A {rmsd_a_start:.3f}→{tgt_rmsd_a:.3f} nm"
        if forces.rmsd_bias_b is not None and rmsd_b_start is not None and tgt_rmsd_b is not None:
            msg += f" RMSD_B {rmsd_b_start:.3f}→{tgt_rmsd_b:.3f} nm"
        if forces.e2e_atoms is not None and e2e_start is not None and tgt_e2e is not None:
            msg += f" E2E {e2e_start:.3f}→{tgt_e2e:.3f} nm"
        if forces.rg_bias is not None and rg_start is not None and tgt_rg is not None:
            msg += f" Rg {rg_start:.3f}→{tgt_rg:.3f} nm"
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
        msg += (
            f" steps(g={int(global_steps)}, t={int(torsion_steps)}, rel={int(release_steps)}) "
            f"updates(g={int(global_updates)}, t={int(torsion_updates)})"
        )
        print(msg, flush=True)

    refine_iters = max(0, int(refine_minimize_iters or 0))
    max_energy_kj_mol = float(max_energy_kj_mol or 0.0)
    if not np.isfinite(max_energy_kj_mol) or max_energy_kj_mol < 0.0:
        max_energy_kj_mol = 0.0

    try:
        sim.context.setParameter(forces.lambda_torsion, 0.0)
    except Exception:
        pass

    if has_globals and int(global_steps) > 0 and int(global_updates) > 0:
        n_updates = max(1, int(global_updates))
        total_steps = int(global_steps)
        hold_steps = 0

        if tgt_rmsd is None:
            tgt_rmsd = rmsd_start
        if tgt_rmsd_a is None:
            tgt_rmsd_a = rmsd_a_start
        if tgt_rmsd_b is None:
            tgt_rmsd_b = rmsd_b_start
        if tgt_e2e is None:
            tgt_e2e = e2e_start
        if tgt_rg is None:
            tgt_rg = rg_start
        if rmsd_start is None:
            rmsd_start = tgt_rmsd
        if rmsd_a_start is None:
            rmsd_a_start = tgt_rmsd_a
        if rmsd_b_start is None:
            rmsd_b_start = tgt_rmsd_b
        if e2e_start is None:
            e2e_start = tgt_e2e
        if rg_start is None:
            rg_start = tgt_rg

        # IMPORTANT: if we interpolate the target every update and only step a small number
        # of steps at the final target, large target moves often "lag" badly even with large
        # total_steps. Reserve a chunk of steps to hold at the final target (λg=1, r0=tgt).
        #
        # We make the hold fraction adaptive based on how far the targets are from the start.
        if total_steps >= 1000 and n_updates >= 5:
            deltas = []
            try:
                if tgt_rmsd is not None and rmsd_start is not None:
                    deltas.append(abs(float(tgt_rmsd) - float(rmsd_start)))
                if tgt_e2e is not None and e2e_start is not None:
                    deltas.append(abs(float(tgt_e2e) - float(e2e_start)))
                if tgt_rg is not None and rg_start is not None:
                    deltas.append(abs(float(tgt_rg) - float(rg_start)))
                for name in sorted(extra_target.keys()):
                    # Only treat distance-like extras as "nm"; ignore *_rad in hold sizing.
                    if str(name).lower().endswith("_rad"):
                        continue
                    s0 = extra_start.get(str(name))
                    t0 = extra_target.get(str(name))
                    if s0 is None or t0 is None:
                        continue
                    deltas.append(abs(float(t0) - float(s0)))
            except Exception:
                deltas = []
            delta_max = float(max(deltas)) if deltas else 0.0
            # Heuristic: for ~0.5 nm moves (or larger), spend up to 75% of steps holding at the final target.
            hold_frac = 0.25 + 0.5 * min(1.0, max(0.0, delta_max / 0.5))
            hold_frac = max(0.25, min(0.75, float(hold_frac)))
            hold_steps = int(round(float(total_steps) * float(hold_frac)))
            # Ensure the ramp still has enough steps to make progress across updates.
            hold_steps = int(min(max(0, hold_steps), max(0, total_steps - n_updates)))

        ramp_steps = max(0, total_steps - int(hold_steps))
        steps_per = max(1, int(ramp_steps) // n_updates)
        if verbose and int(total_steps) > 0:
            try:
                print(
                    f"[TargetedMD] Globals schedule: ramp={int(ramp_steps)} hold={int(hold_steps)} "
                    f"steps_per_update={int(steps_per)}",
                    flush=True,
                )
            except Exception:
                pass

        for u in range(n_updates):
            frac = float(u + 1) / float(n_updates)
            if forces.rmsd_bias is not None and tgt_rmsd is not None and rmsd_start is not None:
                r0 = (1.0 - frac) * float(rmsd_start) + frac * float(tgt_rmsd)
                try:
                    sim.context.setParameter(forces.rmsd_target, float(r0))
                except Exception:
                    pass
            if forces.rmsd_bias_a is not None and tgt_rmsd_a is not None and rmsd_a_start is not None:
                r0a = (1.0 - frac) * float(rmsd_a_start) + frac * float(tgt_rmsd_a)
                try:
                    sim.context.setParameter(forces.rmsd_target_a, float(r0a))
                except Exception:
                    pass
            if forces.rmsd_bias_b is not None and tgt_rmsd_b is not None and rmsd_b_start is not None:
                r0b = (1.0 - frac) * float(rmsd_b_start) + frac * float(tgt_rmsd_b)
                try:
                    sim.context.setParameter(forces.rmsd_target_b, float(r0b))
                except Exception:
                    pass
            if forces.e2e_atoms is not None and tgt_e2e is not None and e2e_start is not None:
                d0 = (1.0 - frac) * float(e2e_start) + frac * float(tgt_e2e)
                try:
                    sim.context.setParameter(forces.e2e_target, float(d0))
                except Exception:
                    pass
            if forces.rg_bias is not None and tgt_rg is not None and rg_start is not None:
                g0 = (1.0 - frac) * float(rg_start) + frac * float(tgt_rg)
                try:
                    sim.context.setParameter(forces.rg_target, float(g0))
                except Exception:
                    pass
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
            # After updating targets, cap the global steering energy by (optionally) reducing λg.
            _cap_lambda_by_energy(
                sim,
                lambda_param=forces.lambda_global,
                force_group_id=STEER_GLOBAL_FORCE_GROUP_ID,
                desired_lambda=frac,
                max_energy_kj_mol=max_energy_kj_mol,
            )
            sim.step(steps_per)

        rem = int(ramp_steps) - steps_per * n_updates
        if rem > 0:
            sim.step(rem)

        if forces.rmsd_bias is not None:
            try:
                sim.context.setParameter(forces.rmsd_target, float(tgt_rmsd) if tgt_rmsd is not None else 0.0)
            except Exception:
                pass
        if forces.rmsd_bias_a is not None:
            try:
                sim.context.setParameter(forces.rmsd_target_a, float(tgt_rmsd_a) if tgt_rmsd_a is not None else 0.0)
            except Exception:
                pass
        if forces.rmsd_bias_b is not None:
            try:
                sim.context.setParameter(forces.rmsd_target_b, float(tgt_rmsd_b) if tgt_rmsd_b is not None else 0.0)
            except Exception:
                pass
        try:
            sim.context.setParameter(forces.e2e_target, float(tgt_e2e) if tgt_e2e is not None else 0.0)
        except Exception:
            pass
        if forces.rg_bias is not None:
            try:
                sim.context.setParameter(forces.rg_target, float(tgt_rg) if tgt_rg is not None else 0.0)
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

        # Hold at the final target so large moves have time to converge.
        if int(hold_steps) > 0:
            try:
                if forces.rmsd_bias is not None and tgt_rmsd is not None:
                    sim.context.setParameter(forces.rmsd_target, float(tgt_rmsd))
            except Exception:
                pass
            try:
                if forces.rmsd_bias_a is not None and tgt_rmsd_a is not None:
                    sim.context.setParameter(forces.rmsd_target_a, float(tgt_rmsd_a))
            except Exception:
                pass
            try:
                if forces.rmsd_bias_b is not None and tgt_rmsd_b is not None:
                    sim.context.setParameter(forces.rmsd_target_b, float(tgt_rmsd_b))
            except Exception:
                pass
            try:
                if forces.e2e_atoms is not None and tgt_e2e is not None:
                    sim.context.setParameter(forces.e2e_target, float(tgt_e2e))
            except Exception:
                pass
            try:
                if forces.rg_bias is not None and tgt_rg is not None:
                    sim.context.setParameter(forces.rg_target, float(tgt_rg))
            except Exception:
                pass
            try:
                for name, param in (forces.extra_distance_target_param or {}).items():
                    t0 = extra_target.get(str(name))
                    if t0 is None:
                        continue
                    sim.context.setParameter(str(param), float(t0))
                for name, param in (forces.extra_angle_target_param or {}).items():
                    t0 = extra_target.get(str(name))
                    if t0 is None:
                        continue
                    sim.context.setParameter(str(param), float(t0))
            except Exception:
                pass
            _cap_lambda_by_energy(
                sim,
                lambda_param=forces.lambda_global,
                force_group_id=STEER_GLOBAL_FORCE_GROUP_ID,
                desired_lambda=1.0,
                max_energy_kj_mol=max_energy_kj_mol,
            )
            sim.step(int(hold_steps))

        if verbose:
            rmsd_mid = _current_primary_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
            rmsd_a_mid = None
            rmsd_b_mid = None
            if forces.rmsd_bias_a is not None:
                rmsd_a_mid = _current_rmsd_cv_nm(sim, forces.rmsd_force_a)
                if rmsd_a_mid is None and forces.rmsd_ref_a_nm is not None and forces.rmsd_particle_indices:
                    rmsd_a_mid = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_a_nm)
            if forces.rmsd_bias_b is not None:
                rmsd_b_mid = _current_rmsd_cv_nm(sim, forces.rmsd_force_b)
                if rmsd_b_mid is None and forces.rmsd_ref_b_nm is not None and forces.rmsd_particle_indices:
                    rmsd_b_mid = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_b_nm)
            e2e_mid = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
            rg_mid = _current_rg_nm(sim, forces)
            parts = []
            if rmsd_mid is not None:
                parts.append(f"RMSD={rmsd_mid:.3f} nm")
            if rmsd_a_mid is not None:
                parts.append(f"RMSD_A={rmsd_a_mid:.3f} nm")
            if rmsd_b_mid is not None:
                parts.append(f"RMSD_B={rmsd_b_mid:.3f} nm")
            if e2e_mid is not None:
                parts.append(f"E2E={e2e_mid:.3f} nm")
            if rg_mid is not None:
                parts.append(f"Rg={rg_mid:.3f} nm")
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
            # Add global bias energy
            e_global = _steering_energy_kj(sim, STEER_GLOBAL_FORCE_GROUP_ID)
            if e_global is not None:
                parts.append(f"GlobalE={e_global:.2f} kJ/mol")
            if parts:
                print(f"[TargetedMD] After globals: {' '.join(parts)}{_stage_diag(sim, forces)}", flush=True)

        # Optional: if globals are still far from the requested targets after MD steering,
        # do a short restrained minimization (globals only) before torsion steering.
        if refine_iters > 0 and has_globals:
            try:
                # Compute residuals (nm) for distance-like targets.
                rmsd_now = _current_primary_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
                e2e_now = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
                rg_now = _current_rg_nm(sim, forces)
                errs = []
                if rmsd_now is not None and tgt_rmsd is not None:
                    errs.append(abs(float(rmsd_now) - float(tgt_rmsd)))
                if e2e_now is not None and tgt_e2e is not None:
                    errs.append(abs(float(e2e_now) - float(tgt_e2e)))
                if rg_now is not None and tgt_rg is not None and forces.rg_bias is not None:
                    errs.append(abs(float(rg_now) - float(tgt_rg)))
                for name, (a, b) in (forces.extra_distance_atoms or {}).items():
                    t0 = extra_target.get(str(name))
                    if t0 is None:
                        continue
                    v = _current_distance_nm(sim, int(a), int(b))
                    if v is not None:
                        errs.append(abs(float(v) - float(t0)))

                max_err = float(max(errs)) if errs else 0.0
                # Only run this if we're clearly not converged (avoid needless minimization).
                if max_err >= 0.05:
                    try:
                        _cap_lambda_by_energy(
                            sim,
                            lambda_param=forces.lambda_global,
                            force_group_id=STEER_GLOBAL_FORCE_GROUP_ID,
                            desired_lambda=1.0,
                            max_energy_kj_mol=max_energy_kj_mol,
                        )
                    except Exception:
                        pass
                    try:
                        sim.context.setParameter(forces.lambda_torsion, 0.0)
                    except Exception:
                        pass
                    if verbose:
                        print(
                            f"[TargetedMD] Refining globals with minimizeEnergy(maxIterations={refine_iters}) "
                            f"(max_err≈{max_err:.3f} nm)",
                            flush=True,
                        )
                    try:
                        sim.minimizeEnergy(maxIterations=refine_iters)
                    except Exception as exc_min_g:
                        if verbose:
                            print(f"[TargetedMD] Warning: globals minimizeEnergy refine failed ({exc_min_g})", flush=True)
            except Exception:
                pass

    if has_torsions and int(torsion_steps) > 0 and int(torsion_updates) > 0:
        _cap_lambda_by_energy(
            sim,
            lambda_param=forces.lambda_global,
            force_group_id=STEER_GLOBAL_FORCE_GROUP_ID,
            desired_lambda=float(global_lambda_during_torsion),
            max_energy_kj_mol=max_energy_kj_mol,
        )
        n_updates = max(1, int(torsion_updates))
        steps_per = max(1, int(torsion_steps) // n_updates)
        for u in range(n_updates):
            frac = float(u + 1) / float(n_updates)
            _cap_lambda_by_energy(
                sim,
                lambda_param=forces.lambda_torsion,
                force_group_id=STEER_TORSION_FORCE_GROUP_ID,
                desired_lambda=frac,
                max_energy_kj_mol=max_energy_kj_mol,
            )
            sim.step(steps_per)
        rem = int(torsion_steps) - steps_per * n_updates
        if rem > 0:
            sim.step(rem)
        if verbose:
            rmsd_mid2 = _current_primary_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
            rmsd_a_mid2 = None
            rmsd_b_mid2 = None
            if forces.rmsd_bias_a is not None:
                rmsd_a_mid2 = _current_rmsd_cv_nm(sim, forces.rmsd_force_a)
                if rmsd_a_mid2 is None and forces.rmsd_ref_a_nm is not None and forces.rmsd_particle_indices:
                    rmsd_a_mid2 = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_a_nm)
            if forces.rmsd_bias_b is not None:
                rmsd_b_mid2 = _current_rmsd_cv_nm(sim, forces.rmsd_force_b)
                if rmsd_b_mid2 is None and forces.rmsd_ref_b_nm is not None and forces.rmsd_particle_indices:
                    rmsd_b_mid2 = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_b_nm)
            e2e_mid2 = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
            rg_mid2 = _current_rg_nm(sim, forces)
            parts = []
            if rmsd_mid2 is not None:
                parts.append(f"RMSD={rmsd_mid2:.3f} nm")
            if rmsd_a_mid2 is not None:
                parts.append(f"RMSD_A={rmsd_a_mid2:.3f} nm")
            if rmsd_b_mid2 is not None:
                parts.append(f"RMSD_B={rmsd_b_mid2:.3f} nm")
            if e2e_mid2 is not None:
                parts.append(f"E2E={e2e_mid2:.3f} nm")
            if rg_mid2 is not None:
                parts.append(f"Rg={rg_mid2:.3f} nm")
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
            # Add torsion bias energy as a proxy for |θ_current - θ_target|
            e_torsion = _steering_energy_kj(sim, STEER_TORSION_FORCE_GROUP_ID)
            if e_torsion is not None:
                parts.append(f"TorsionE={e_torsion:.2f} kJ/mol")
            if parts:
                print(f"[TargetedMD] After torsions: {' '.join(parts)}{_stage_diag(sim, forces)}", flush=True)

    if refine_iters > 0 and (has_globals or has_torsions):
        if forces.rmsd_bias is not None:
            try:
                sim.context.setParameter(forces.rmsd_target, float(tgt_rmsd) if tgt_rmsd is not None else 0.0)
            except Exception:
                pass
        if forces.rmsd_bias_a is not None:
            try:
                sim.context.setParameter(forces.rmsd_target_a, float(tgt_rmsd_a) if tgt_rmsd_a is not None else 0.0)
            except Exception:
                pass
        if forces.rmsd_bias_b is not None:
            try:
                sim.context.setParameter(forces.rmsd_target_b, float(tgt_rmsd_b) if tgt_rmsd_b is not None else 0.0)
            except Exception:
                pass
        if forces.e2e_atoms is not None:
            try:
                sim.context.setParameter(forces.e2e_target, float(tgt_e2e) if tgt_e2e is not None else 0.0)
            except Exception:
                pass
        if forces.rg_bias is not None:
            try:
                sim.context.setParameter(forces.rg_target, float(tgt_rg) if tgt_rg is not None else 0.0)
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

        _cap_lambda_by_energy(
            sim,
            lambda_param=forces.lambda_global,
            force_group_id=STEER_GLOBAL_FORCE_GROUP_ID,
            desired_lambda=1.0,
            max_energy_kj_mol=max_energy_kj_mol,
        )
        _cap_lambda_by_energy(
            sim,
            lambda_param=forces.lambda_torsion,
            force_group_id=STEER_TORSION_FORCE_GROUP_ID,
            desired_lambda=1.0 if has_torsions else 0.0,
            max_energy_kj_mol=max_energy_kj_mol,
        )

        if verbose:
            print(f"[TargetedMD] Refining with minimizeEnergy(maxIterations={refine_iters})", flush=True)
        try:
            sim.minimizeEnergy(maxIterations=refine_iters)
        except Exception as exc_min:
            if verbose:
                print(f"[TargetedMD] Warning: minimizeEnergy refine failed ({exc_min})", flush=True)

        if verbose:
            rmsd_ref = _current_primary_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
            rmsd_a_ref = None
            rmsd_b_ref = None
            if forces.rmsd_bias_a is not None:
                rmsd_a_ref = _current_rmsd_cv_nm(sim, forces.rmsd_force_a)
                if rmsd_a_ref is None and forces.rmsd_ref_a_nm is not None and forces.rmsd_particle_indices:
                    rmsd_a_ref = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_a_nm)
            if forces.rmsd_bias_b is not None:
                rmsd_b_ref = _current_rmsd_cv_nm(sim, forces.rmsd_force_b)
                if rmsd_b_ref is None and forces.rmsd_ref_b_nm is not None and forces.rmsd_particle_indices:
                    rmsd_b_ref = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_b_nm)
            e2e_ref = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
            rg_ref = _current_rg_nm(sim, forces)
            parts = []
            if rmsd_ref is not None:
                parts.append(f"RMSD={rmsd_ref:.3f} nm")
            if rmsd_a_ref is not None:
                parts.append(f"RMSD_A={rmsd_a_ref:.3f} nm")
            if rmsd_b_ref is not None:
                parts.append(f"RMSD_B={rmsd_b_ref:.3f} nm")
            if e2e_ref is not None:
                parts.append(f"E2E={e2e_ref:.3f} nm")
            if rg_ref is not None:
                parts.append(f"Rg={rg_ref:.3f} nm")
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
        rmsd_end = _current_primary_rmsd_nm(sim, forces) if forces.rmsd_bias is not None else None
        rmsd_a_end = None
        rmsd_b_end = None
        if forces.rmsd_bias_a is not None:
            rmsd_a_end = _current_rmsd_cv_nm(sim, forces.rmsd_force_a)
            if rmsd_a_end is None and forces.rmsd_ref_a_nm is not None and forces.rmsd_particle_indices:
                rmsd_a_end = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_a_nm)
        if forces.rmsd_bias_b is not None:
            rmsd_b_end = _current_rmsd_cv_nm(sim, forces.rmsd_force_b)
            if rmsd_b_end is None and forces.rmsd_ref_b_nm is not None and forces.rmsd_particle_indices:
                rmsd_b_end = _current_subset_rmsd_nm(sim, forces.rmsd_particle_indices, forces.rmsd_ref_b_nm)
        e2e_end = _current_e2e_nm(sim, forces.e2e_atoms) if forces.e2e_atoms is not None else None
        rg_end = _current_rg_nm(sim, forces)
        parts = []
        if rmsd_end is not None:
            parts.append(f"RMSD={rmsd_end:.3f} nm")
        if rmsd_a_end is not None:
            parts.append(f"RMSD_A={rmsd_a_end:.3f} nm")
        if rmsd_b_end is not None:
            parts.append(f"RMSD_B={rmsd_b_end:.3f} nm")
        if e2e_end is not None:
            parts.append(f"E2E={e2e_end:.3f} nm")
        if rg_end is not None:
            parts.append(f"Rg={rg_end:.3f} nm")
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
