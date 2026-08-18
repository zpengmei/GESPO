"""TMEM63-specific global metrics for global tokens.

This module resolves a set of TMEM63 collective variables (RMSD, membrane-
normal angles, hydrophobic-latch/neck packing distances, helix hinge angles,
an IL2-to-TM6b-loop distance) into GESPO global tokens, following the same
two-phase pattern as adk_metrics.py:

  1. build_tmem63_token_indices(topology, residue_config) resolves residue
     numbers -> atom indices *once*, from an MDTraj-like topology.
  2. compute_tmem63_global_values_from_xyz(xyz_nm, idx=...) computes the
     actual token values *many times*, from packed [F, N, 3] coordinate
     arrays and the indices built in step 1 -- no topology, no selection
     strings, in the hot path.

What makes this reusable across paralogs/constructs (TMEM63B, TMEM63A, ...):
all of the residue-number specifics live in a `TMEM63ResidueConfig` (see
below) that gets passed into build_tmem63_token_indices(). None of the
functions in this file are hardcoded to one paralog's numbering -- pointing
this at TMEM63A means building a new TMEM63ResidueConfig, not new code here.

The low-level math (centroids, pairwise distances, PCA helix axes, angles)
lives in gespo/utils/geometry.py, shared with adk_metrics.py and any future
system's metrics file. Residue-number -> atom-index resolution (including the
configurable backbone / backbone+CB / all-heavy / all-atom granularity) lives
in gespo/utils/selection.py, for the same reason.

All distances are in nm; angles are in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .geometry import (
    angle_between_vectors,
    angle_to_axis_from_xyz,
    axis_from_xyz_pca,
    calculate_rmsd,
    centroid_from_xyz,
    mean_pairwise_distance_from_xyz,
)
from .log import warn as _log_warn
from .selection import AtomSelectionMode, select_atoms_for_residues


TMEM63_GLOBAL_TOKEN_NAMES: Tuple[str, ...] = (
    "tm_rmsd_nm",
    "tm6_membrane_angle_rad",
    "tm4_membrane_angle_rad",
    "hydrophobic_latch_dist_nm",
    "neck_dist_nm",
    "tm6_hinge_angle_rad",
    "tm3_hinge_angle_rad",
    "tm5_hinge_angle_rad",
    "il2_tm6b_dist_nm",
)

# TMEM63 domain-only globals (no RMSD). Same idea as ADK_DOMAIN_TOKEN_NAMES.
TMEM63_DOMAIN_TOKEN_NAMES: Tuple[str, ...] = TMEM63_GLOBAL_TOKEN_NAMES[1:]


# ---------------------------------------------------------------------------
# Per-paralog residue configuration.
#
# This is the *only* thing that should need to change to point these metrics
# at a different construct (e.g. TMEM63A instead of TMEM63B): build a new
# TMEM63ResidueConfig with that construct's residue numbers and pass it to
# build_tmem63_token_indices(). Nothing else in this file is paralog-specific.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionSpec:
    """A named group of residues, plus how much of each residue's atoms to use.

    `residues` is always an explicit tuple of residue numbers, never a range
    -- build one from a contiguous range with selection.expand_residue_range().
    Keeping selection code to one input shape (a set of residue numbers)
    instead of "range or list" makes it simpler to validate and test.
    """

    residues: Tuple[int, ...]
    atom_mode: AtomSelectionMode = AtomSelectionMode.BACKBONE_CB


@dataclass(frozen=True)
class HingeSpec:
    """A pivot residue splitting a helix into two sub-segments for a hinge angle.

    The hinge angle is the angle between each sub-segment's fitted axis (see
    geometry.axis_from_xyz_pca) -- not a new primitive, just angle_between_vectors
    applied to two region axes. `pivot_resid` is documentation of where the
    split is anchored; only n_side/c_side are actually used to compute anything.
    """

    pivot_resid: int
    n_side: RegionSpec
    c_side: RegionSpec


@dataclass(frozen=True)
class TMEM63ResidueConfig:
    """Residue-number configuration for one TMEM63 paralog/construct.

    Every field is Optional. build_tmem63_token_indices() only requires the
    field(s) needed for whichever token columns you actually request, and
    raises NotImplementedError naming exactly which field is still unset if
    you request a column that needs it -- see that function's docstring.
    """

    label: str

    # tm_rmsd_nm: union of these regions' atoms, compared against a reference
    # structure's matching selection (see compute_*'s `ref_rmsd_nm` arg).
    rmsd_regions: Optional[Dict[str, RegionSpec]] = None

    # tm6_membrane_angle_rad / tm4_membrane_angle_rad: axis-fit regions,
    # angled against `membrane_axis` (default the simulation-box z-axis).
    tm6_membrane_region: Optional[RegionSpec] = None
    tm4_membrane_region: Optional[RegionSpec] = None

    # hydrophobic_latch_dist_nm / neck_dist_nm: mean pairwise distance within
    # one (possibly non-contiguous) set of residues.
    hydrophobic_latch: Optional[RegionSpec] = None
    neck: Optional[RegionSpec] = None

    # tm6_hinge_angle_rad / tm3_hinge_angle_rad / tm5_hinge_angle_rad
    tm6_hinge: Optional[HingeSpec] = None
    tm3_hinge: Optional[HingeSpec] = None
    tm5_hinge: Optional[HingeSpec] = None

    # il2_tm6b_dist_nm: centroid-to-centroid distance between these two regions.
    il2_hinge_region: Optional[RegionSpec] = None
    tm6b_loop_region: Optional[RegionSpec] = None


# --- TMEM63B: filled in from tmem63b_gespo.md where residue numbers are known. ---
#
# Most fields are still None on purpose. tmem63b_gespo.md doesn't yet give TM
# helix boundaries (needed for tm_rmsd_nm and the membrane-angle regions), TM3/
# TM5 hinge pivots, neck residues, or the IL2H2-vs-H3 / TM6b-loop ranges (the
# doc's own note: "Ask Malavika what residues she's using to measure opening").
# Only the hydrophobic latch residues are fully specified today. Requesting a
# token column that needs an unset field raises NotImplementedError naming
# that field, rather than silently guessing a residue range.
#
# TM6's pivot residue (G580, the pi-helix hinge point) *is* known, but a hinge
# angle also needs the N-/C-side sub-segment ranges around it, which aren't
# pinned down yet -- so tm6_hinge is left unset too, with the pivot noted here
# for when those ranges are filled in.
TMEM63B_RESIDUE_CONFIG = TMEM63ResidueConfig(
    label="TMEM63B",
    hydrophobic_latch=RegionSpec(
        # V44/L47 (TM0), I474/I475/F478/F479/L482/L483 (TM4), A575/I579 (TM6).
        residues=(44, 47, 474, 475, 478, 479, 482, 483, 575, 579),
        # These are the residues literally hypothesized to sense membrane
        # thickness via side-chain packing -- backbone-only would miss the
        # point of this metric, so use every heavy atom.
        atom_mode=AtomSelectionMode.ALL_HEAVY,
    ),
    # tm6_hinge pivot_resid=580 (G580) once n_side/c_side ranges are known.
)

# --- TMEM63A: same shape, entirely unfilled until its numbering is available. ---
TMEM63A_RESIDUE_CONFIG = TMEM63ResidueConfig(label="TMEM63A")


# ---------------------------------------------------------------------------
# Build phase: residue_config -> TMEM63TokenIndices (once, from a topology).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TMEM63TokenIndices:
    """Precomputed atom indices (local to the protein-only coordinate array)
    needed to compute TMEM63_GLOBAL_TOKEN_NAMES from raw coordinates."""

    rmsd_atom_idx: Optional[np.ndarray] = None
    tm6_membrane_atom_idx: Optional[np.ndarray] = None
    tm4_membrane_atom_idx: Optional[np.ndarray] = None
    hydrophobic_latch_atom_idx: Optional[np.ndarray] = None
    neck_atom_idx: Optional[np.ndarray] = None
    tm6_hinge_sides: Optional[Tuple[np.ndarray, np.ndarray]] = None
    tm3_hinge_sides: Optional[Tuple[np.ndarray, np.ndarray]] = None
    tm5_hinge_sides: Optional[Tuple[np.ndarray, np.ndarray]] = None
    il2_hinge_atom_idx: Optional[np.ndarray] = None
    tm6b_loop_atom_idx: Optional[np.ndarray] = None


class _ResolverState:
    """Threads the resSeq/0-based-index fallback flag through one build() call.

    Mirrors adk_metrics.py's original behavior: try resSeq first; if any
    region's selection comes back empty, switch to 0-based residue indices
    for the rest of the build (some topologies don't carry resSeq at all).
    """

    def __init__(self, use_resseq: bool):
        self.use_resseq = bool(use_resseq)


def _resolve_region_atoms(
    topology,
    spec: RegionSpec,
    protein_map: Dict[int, int],
    state: _ResolverState,
    *,
    label: str,
    field_name: str,
) -> np.ndarray:
    """Resolve one RegionSpec to local (protein-array) atom indices.

    Raises:
        ValueError: the region's residues/mode matched no atoms in the
            topology (under either numbering convention), or matched atoms
            outside the given protein_indices subset. This is a "the config
            value is set but wrong" error, distinct from NotImplementedError
            ("the config value isn't set at all") raised by build_*_token_indices.
    """
    global_idx = select_atoms_for_residues(
        topology, spec.residues, spec.atom_mode, use_resseq=state.use_resseq
    )
    if global_idx.size == 0 and state.use_resseq:
        state.use_resseq = False
        global_idx = select_atoms_for_residues(
            topology, spec.residues, spec.atom_mode, use_resseq=state.use_resseq
        )
    if global_idx.size == 0:
        raise ValueError(
            f"TMEM63ResidueConfig '{label}': region '{field_name}' (residues="
            f"{spec.residues}, mode={spec.atom_mode.value}) matched no atoms in "
            f"the topology under either resSeq or 0-based-index numbering -- "
            f"check the residue numbers."
        )
    local_idx = np.asarray(
        [protein_map[int(i)] for i in global_idx.tolist() if int(i) in protein_map],
        dtype=np.int64,
    )
    if local_idx.size == 0:
        raise ValueError(
            f"TMEM63ResidueConfig '{label}': region '{field_name}' matched atoms "
            f"in the topology, but none fall inside the given protein_indices "
            f"subset -- check that protein_indices actually covers this region."
        )
    return local_idx


def _require_field(value, *, label: str, field_name: str):
    if value is None:
        raise NotImplementedError(
            f"TMEM63ResidueConfig '{label}': required field '{field_name}' is not "
            f"set. Fill it in (see tmem63_metrics.py's TMEM63B_RESIDUE_CONFIG "
            f"comments for what's already known / still needed) before requesting "
            f"the token column(s) that depend on it."
        )
    return value


def build_tmem63_token_indices(
    topology,
    residue_config: TMEM63ResidueConfig,
    *,
    token_names: Sequence[str] = TMEM63_GLOBAL_TOKEN_NAMES,
    protein_indices: Optional[np.ndarray] = None,
    prefer_resseq: bool = True,
) -> TMEM63TokenIndices:
    """Build local atom indices for the requested TMEM63 token columns.

    Only resolves what `token_names` actually asks for -- e.g. passing
    TMEM63_DOMAIN_TOKEN_NAMES skips the RMSD region entirely, so a config
    missing only `rmsd_regions` can still build a usable index object for
    every other column. This also makes it practical to build indices for a
    single metric in isolation while developing/testing a config
    incrementally (e.g. token_names=("hydrophobic_latch_dist_nm",)).

    Args:
        topology: MDTraj-like Topology (must have .atoms and .residues).
        residue_config: which paralog/construct to build indices for (e.g.
            TMEM63B_RESIDUE_CONFIG or TMEM63A_RESIDUE_CONFIG).
        token_names: which TMEM63_GLOBAL_TOKEN_NAMES columns to build indices
            for; defaults to all of them.
        protein_indices: optional global atom indices defining the
            "protein-only" coordinate arrays (as used by env.step()); when
            provided, returned indices are mapped into this local indexing.
        prefer_resseq: interpret residue numbers as PDB resSeq when possible;
            fall back to 0-based residue index if no matches are found.

    Raises:
        NotImplementedError: residue_config is missing a field required by
            one of the requested `token_names` (the message names the field).
        ValueError: a residue_config field is set, but its residues/mode
            didn't resolve to any atom in this topology.
    """
    if protein_indices is None:
        protein_indices = np.arange(int(getattr(topology, "n_atoms", 0) or 0), dtype=np.int64)
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    protein_map = {int(a): int(i) for i, a in enumerate(protein_indices.tolist())}

    state = _ResolverState(use_resseq=prefer_resseq)
    label = residue_config.label
    names = set(token_names)

    def region(spec: Optional[RegionSpec], field_name: str) -> np.ndarray:
        spec = _require_field(spec, label=label, field_name=field_name)
        return _resolve_region_atoms(topology, spec, protein_map, state, label=label, field_name=field_name)

    def hinge_sides(spec: Optional[HingeSpec], field_name: str) -> Tuple[np.ndarray, np.ndarray]:
        spec = _require_field(spec, label=label, field_name=field_name)
        n_idx = _resolve_region_atoms(
            topology, spec.n_side, protein_map, state, label=label, field_name=f"{field_name}.n_side"
        )
        c_idx = _resolve_region_atoms(
            topology, spec.c_side, protein_map, state, label=label, field_name=f"{field_name}.c_side"
        )
        return (n_idx, c_idx)

    rmsd_atom_idx = None
    if "tm_rmsd_nm" in names:
        regions = _require_field(residue_config.rmsd_regions, label=label, field_name="rmsd_regions")
        parts = [
            _resolve_region_atoms(topology, spec, protein_map, state, label=label, field_name=f"rmsd_regions[{key}]")
            for key, spec in regions.items()
        ]
        rmsd_atom_idx = np.unique(np.concatenate(parts)) if parts else np.zeros((0,), dtype=np.int64)

    return TMEM63TokenIndices(
        rmsd_atom_idx=rmsd_atom_idx,
        tm6_membrane_atom_idx=(
            region(residue_config.tm6_membrane_region, "tm6_membrane_region")
            if "tm6_membrane_angle_rad" in names else None
        ),
        tm4_membrane_atom_idx=(
            region(residue_config.tm4_membrane_region, "tm4_membrane_region")
            if "tm4_membrane_angle_rad" in names else None
        ),
        hydrophobic_latch_atom_idx=(
            region(residue_config.hydrophobic_latch, "hydrophobic_latch")
            if "hydrophobic_latch_dist_nm" in names else None
        ),
        neck_atom_idx=region(residue_config.neck, "neck") if "neck_dist_nm" in names else None,
        tm6_hinge_sides=hinge_sides(residue_config.tm6_hinge, "tm6_hinge") if "tm6_hinge_angle_rad" in names else None,
        tm3_hinge_sides=hinge_sides(residue_config.tm3_hinge, "tm3_hinge") if "tm3_hinge_angle_rad" in names else None,
        tm5_hinge_sides=hinge_sides(residue_config.tm5_hinge, "tm5_hinge") if "tm5_hinge_angle_rad" in names else None,
        il2_hinge_atom_idx=(
            region(residue_config.il2_hinge_region, "il2_hinge_region")
            if "il2_tm6b_dist_nm" in names else None
        ),
        tm6b_loop_atom_idx=(
            region(residue_config.tm6b_loop_region, "tm6b_loop_region")
            if "il2_tm6b_dist_nm" in names else None
        ),
    )


# ---------------------------------------------------------------------------
# Compute phase: TMEM63TokenIndices + xyz -> token values (every frame).
#
# Each column has its own small computer function below, all sharing one
# signature, dispatched through _COLUMN_COMPUTERS. Adding a 10th metric later
# means writing one such function and one dict entry -- nothing else in this
# section changes.
# ---------------------------------------------------------------------------

_WARNED_KEYS: set = set()


def _warn_once(key: str, message: str) -> None:
    """Log a warning at most once per process for a given `key`.

    Column-fill failures below are individually wrapped so one broken metric
    doesn't take down the others or the whole training run -- but a silently
    swallowed exception is exactly how a broken residue mapping produced
    meaningless flat-zero tokens for an entire run in the past (see this
    file's module docstring / implementing_global_tokens.md). Logging once
    per distinct failure reason keeps that visible without flooding the log
    of a multi-day run with a per-frame repeat of the same message.
    """
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    _log_warn(f"[tmem63_metrics] {message}")


def _idx_present(atom_idx: Optional[np.ndarray], name: str, field_desc: str) -> bool:
    if atom_idx is None or atom_idx.size == 0:
        _warn_once(
            f"{name}:missing_index",
            f"'{name}' skipped: {field_desc} was not built (unset in the residue "
            f"config, or build_tmem63_token_indices was called without this "
            f"token name in `token_names`).",
        )
        return False
    return True


def _compute_rmsd(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis) -> Optional[np.ndarray]:
    name = "tm_rmsd_nm"
    if not _idx_present(idx.rmsd_atom_idx, name, "rmsd_atom_idx"):
        return None
    if ref_rmsd_nm is None:
        _warn_once(f"{name}:no_reference", f"'{name}' skipped: no ref_rmsd_nm reference structure was supplied.")
        return None
    ref = np.asarray(ref_rmsd_nm, dtype=np.float64).reshape(-1, 3)
    if ref.shape[0] != idx.rmsd_atom_idx.size:
        _warn_once(
            f"{name}:shape_mismatch",
            f"'{name}' skipped: ref_rmsd_nm has {ref.shape[0]} atoms but rmsd_atom_idx has {idx.rmsd_atom_idx.size}.",
        )
        return None
    coords = xyz_nm[:, idx.rmsd_atom_idx, :]
    return calculate_rmsd(coords, ref, atom_indices=None)


def _compute_membrane_angle(atom_idx, name, xyz_nm, membrane_axis) -> Optional[np.ndarray]:
    if not _idx_present(atom_idx, name, f"{name}'s atom index"):
        return None
    return angle_to_axis_from_xyz(xyz_nm, atom_idx, axis=membrane_axis)


def _compute_tm6_membrane_angle(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_membrane_angle(idx.tm6_membrane_atom_idx, "tm6_membrane_angle_rad", xyz_nm, membrane_axis)


def _compute_tm4_membrane_angle(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_membrane_angle(idx.tm4_membrane_atom_idx, "tm4_membrane_angle_rad", xyz_nm, membrane_axis)


def _compute_mean_pairwise(atom_idx, name, xyz_nm) -> Optional[np.ndarray]:
    if not _idx_present(atom_idx, name, f"{name}'s atom index"):
        return None
    return mean_pairwise_distance_from_xyz(xyz_nm, atom_idx)


def _compute_hydrophobic_latch(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_mean_pairwise(idx.hydrophobic_latch_atom_idx, "hydrophobic_latch_dist_nm", xyz_nm)


def _compute_neck(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_mean_pairwise(idx.neck_atom_idx, "neck_dist_nm", xyz_nm)


def _compute_hinge_angle(sides, name, xyz_nm) -> Optional[np.ndarray]:
    if sides is None:
        _warn_once(f"{name}:missing_index", f"'{name}' skipped: hinge sub-segment indices were not built.")
        return None
    n_side_idx, c_side_idx = sides
    if n_side_idx.size == 0 or c_side_idx.size == 0:
        _warn_once(f"{name}:empty_side", f"'{name}' skipped: a hinge sub-segment resolved to zero atoms.")
        return None
    axis_n = axis_from_xyz_pca(xyz_nm, n_side_idx)
    axis_c = axis_from_xyz_pca(xyz_nm, c_side_idx)
    return angle_between_vectors(axis_n, axis_c)


def _compute_tm6_hinge(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_hinge_angle(idx.tm6_hinge_sides, "tm6_hinge_angle_rad", xyz_nm)


def _compute_tm3_hinge(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_hinge_angle(idx.tm3_hinge_sides, "tm3_hinge_angle_rad", xyz_nm)


def _compute_tm5_hinge(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    return _compute_hinge_angle(idx.tm5_hinge_sides, "tm5_hinge_angle_rad", xyz_nm)


def _compute_il2_tm6b_distance(xyz_nm, idx: TMEM63TokenIndices, ref_rmsd_nm, membrane_axis):
    name = "il2_tm6b_dist_nm"
    if not _idx_present(idx.il2_hinge_atom_idx, name, "il2_hinge_atom_idx"):
        return None
    if not _idx_present(idx.tm6b_loop_atom_idx, name, "tm6b_loop_atom_idx"):
        return None
    c1 = centroid_from_xyz(xyz_nm, idx.il2_hinge_atom_idx)
    c2 = centroid_from_xyz(xyz_nm, idx.tm6b_loop_atom_idx)
    return np.linalg.norm(c1 - c2, axis=1).astype(np.float32, copy=False)


_ColumnComputer = Callable[[np.ndarray, TMEM63TokenIndices, Optional[np.ndarray], Tuple[float, float, float]], Optional[np.ndarray]]

_COLUMN_COMPUTERS: Dict[str, _ColumnComputer] = {
    "tm_rmsd_nm": _compute_rmsd,
    "tm6_membrane_angle_rad": _compute_tm6_membrane_angle,
    "tm4_membrane_angle_rad": _compute_tm4_membrane_angle,
    "hydrophobic_latch_dist_nm": _compute_hydrophobic_latch,
    "neck_dist_nm": _compute_neck,
    "tm6_hinge_angle_rad": _compute_tm6_hinge,
    "tm3_hinge_angle_rad": _compute_tm3_hinge,
    "tm5_hinge_angle_rad": _compute_tm5_hinge,
    "il2_tm6b_dist_nm": _compute_il2_tm6b_distance,
}
assert set(_COLUMN_COMPUTERS) == set(TMEM63_GLOBAL_TOKEN_NAMES), (
    "every name in TMEM63_GLOBAL_TOKEN_NAMES needs exactly one _COLUMN_COMPUTERS entry"
)


def compute_tmem63_global_values_from_xyz(
    xyz_nm: np.ndarray,
    *,
    idx: TMEM63TokenIndices,
    ref_rmsd_nm: Optional[np.ndarray] = None,
    membrane_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Compute TMEM63 global token values from a packed coordinate array (nm).

    Each column is computed in isolation: if its required indices weren't
    built (that part of the residue config is unset, or wasn't requested via
    `token_names` at build time) or its computation raises, that column
    stays 0.0 for every frame rather than aborting the whole call -- a single
    misconfigured metric shouldn't crash a training run. Each such case logs
    a rate-limited warning (see _warn_once) so a systematically broken
    mapping is visible in the run log instead of silently producing a
    flat-zero token for the whole run, which would otherwise be
    indistinguishable from a genuinely-zero value.

    Args:
        xyz_nm: [F, N, 3] packed coordinates (nm). N is indexed the same way
            as the atom indices inside `idx` (i.e. local to whatever
            `protein_indices` was passed to build_tmem63_token_indices).
        idx: indices built by build_tmem63_token_indices().
        ref_rmsd_nm: [n_rmsd_atoms, 3] reference structure coordinates for
            tm_rmsd_nm, matching idx.rmsd_atom_idx in order and count.
        membrane_axis: reference axis for the membrane-angle columns
            (default: the simulation-box z-axis).

    Returns:
        values: [F, len(TMEM63_GLOBAL_TOKEN_NAMES)] float32, columns matching
        TMEM63_GLOBAL_TOKEN_NAMES order.
    """
    xyz_nm = np.asarray(xyz_nm, dtype=np.float64)
    if xyz_nm.ndim != 3 or xyz_nm.shape[-1] != 3:
        raise ValueError(f"xyz_nm must be [F,N,3], got {xyz_nm.shape}")
    n_frames = int(xyz_nm.shape[0])
    out = np.zeros((n_frames, len(TMEM63_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
    if n_frames == 0:
        return out

    for col, name in enumerate(TMEM63_GLOBAL_TOKEN_NAMES):
        try:
            value = _COLUMN_COMPUTERS[name](xyz_nm, idx, ref_rmsd_nm, membrane_axis)
        except Exception as exc:
            _warn_once(f"{name}:exception:{type(exc).__name__}", f"'{name}' raised {type(exc).__name__}: {exc}")
            value = None
        if value is not None:
            out[:, col] = value

    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32, copy=False)
    out = np.clip(out, 0.0, None)  # every column here (RMSD/angle/distance) is nonnegative by construction
    return out


def compute_tmem63_global_values_from_frames(
    frames_nm: Sequence[np.ndarray],
    *,
    idx: TMEM63TokenIndices,
    ref_rmsd_nm: Optional[np.ndarray] = None,
    membrane_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Same as compute_tmem63_global_values_from_xyz, for a sequence of
    per-frame [N, 3] coordinate arrays instead of one packed [F, N, 3] array.

    A thin stack-and-delegate wrapper: there is exactly one computation path
    to keep correct here, not two independently-maintained implementations.
    """
    if len(frames_nm) == 0:
        return np.zeros((0, len(TMEM63_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
    xyz_nm = np.stack([np.asarray(f, dtype=np.float64) for f in frames_nm], axis=0)
    return compute_tmem63_global_values_from_xyz(
        xyz_nm, idx=idx, ref_rmsd_nm=ref_rmsd_nm, membrane_axis=membrane_axis
    )
