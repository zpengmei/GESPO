import mdtraj as md
import numpy as np

from gespo.utils.selection import (
    AtomSelectionMode,
    reindexed_topology_subset,
    select_atoms_by_residue,
    select_atoms_for_residues,
)
from gespo.utils.tmem63_metrics import (
    TMEM63_GLOBAL_TOKEN_NAMES,
    TMEM63A_RESIDUE_CONFIG,
    TMEM63B_RESIDUE_CONFIG,
    HelixCurvatureSpec,
    TMEM63ResidueConfig,
    build_tmem63_token_indices,
    compute_tmem63_global_values_from_xyz,
)

_LATCH_RESIDS = [44, 47, 474, 475, 478, 479, 482, 483, 575, 579]


def _build_synthetic_latch_topology():
    """A tiny synthetic topology covering just the TMEM63B hydrophobic-latch
    residues, one heavy sidechain atom (CG) beyond CB, plus one hydrogen (HA)
    per residue -- enough to distinguish all four AtomSelectionMode granularities.
    """
    top = md.Topology()
    chain = top.add_chain()
    for resid in _LATCH_RESIDS:
        res = top.add_residue("ALA", chain, resSeq=resid)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)
        top.add_atom("CB", md.element.carbon, res)
        top.add_atom("CG", md.element.carbon, res)
        top.add_atom("HA", md.element.hydrogen, res)
    return top


def _synthetic_latch_xyz(n_frames=2):
    top = _build_synthetic_latch_topology()
    n_atoms = top.n_atoms
    rng = np.random.default_rng(0)
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    # Spread each residue's 7-atom cluster far apart so pairwise distances
    # between residues dominate and are clearly nonzero/finite.
    for i, resid in enumerate(_LATCH_RESIDS):
        base = np.array([10.0 * i, 0.0, 0.0])
        xyz[:, i * 7:(i + 1) * 7, :] = base + rng.normal(scale=0.05, size=(n_frames, 7, 3))
    return top, xyz


def test_select_atoms_for_residues_modes_distinguish_atom_counts():
    top = _build_synthetic_latch_topology()

    backbone = select_atoms_for_residues(top, _LATCH_RESIDS, AtomSelectionMode.BACKBONE, use_resseq=True)
    backbone_cb = select_atoms_for_residues(top, _LATCH_RESIDS, AtomSelectionMode.BACKBONE_CB, use_resseq=True)
    all_heavy = select_atoms_for_residues(top, _LATCH_RESIDS, AtomSelectionMode.ALL_HEAVY, use_resseq=True)
    all_atom = select_atoms_for_residues(top, _LATCH_RESIDS, AtomSelectionMode.ALL_ATOM, use_resseq=True)

    n_res = len(_LATCH_RESIDS)
    assert backbone.size == 4 * n_res      # N, CA, C, O
    assert backbone_cb.size == 5 * n_res   # + CB
    assert all_heavy.size == 6 * n_res     # + CG, no H
    assert all_atom.size == 7 * n_res      # + HA


def test_select_atoms_for_residues_unmatched_residue_is_empty():
    top = _build_synthetic_latch_topology()
    out = select_atoms_for_residues(top, [99999], AtomSelectionMode.ALL_ATOM, use_resseq=True)
    assert out.size == 0


def test_build_tmem63_token_indices_hydrophobic_latch_only():
    top = _build_synthetic_latch_topology()
    idx = build_tmem63_token_indices(
        top, TMEM63B_RESIDUE_CONFIG, token_names=("hydrophobic_latch_dist_nm",)
    )
    assert idx.hydrophobic_latch_pairs is not None
    idx_a, idx_b = idx.hydrophobic_latch_pairs
    n_res = len(_LATCH_RESIDS)
    atoms_per_res = 6  # ALL_HEAVY per config: N, CA, C, O, CB, CG (no H)
    n_res_pairs = n_res * (n_res - 1) // 2
    assert idx_a.size == idx_b.size == n_res_pairs * atoms_per_res * atoms_per_res

    # Ground truth: which residue does each atom index belong to (cross-checked
    # against select_atoms_by_residue, not inferred from an assumed layout).
    grouped = select_atoms_by_residue(top, _LATCH_RESIDS, AtomSelectionMode.ALL_HEAVY, use_resseq=True)
    residue_of = {int(a): resnum for resnum, atoms in grouped.items() for a in atoms.tolist()}
    same_residue = [residue_of[int(a)] == residue_of[int(b)] for a, b in zip(idx_a.tolist(), idx_b.tolist())]
    assert not any(same_residue)

    assert idx.rmsd_atom_idx is None
    assert idx.neck_pairs is None


def test_build_tmem63_token_indices_raises_on_unset_required_field():
    # Deliberately-empty dummy config (not TMEM63B_RESIDUE_CONFIG, which is
    # now fully filled in) so this test stays meaningful regardless of which
    # fields TMEM63B/TMEM63A have filled in over time. _require_field raises
    # before ever touching the topology, so any topology works here.
    top = _build_synthetic_latch_topology()
    dummy_config = TMEM63ResidueConfig(label="Dummy")
    try:
        build_tmem63_token_indices(top, dummy_config, token_names=("neck_dist_nm",))
    except NotImplementedError as exc:
        assert "neck" in str(exc)
        assert "Dummy" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError for unset neck")


def test_build_tmem63_token_indices_tmem63a_unconfigured_raises_immediately():
    top = _build_synthetic_latch_topology()
    try:
        build_tmem63_token_indices(
            top, TMEM63A_RESIDUE_CONFIG, token_names=("hydrophobic_latch_dist_nm",)
        )
    except NotImplementedError as exc:
        assert "TMEM63A" in str(exc)
        assert "hydrophobic_latch" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError: TMEM63A_RESIDUE_CONFIG has no fields set yet")


def test_compute_tmem63_global_values_only_fills_requested_column():
    top, xyz = _synthetic_latch_xyz(n_frames=3)
    idx = build_tmem63_token_indices(
        top, TMEM63B_RESIDUE_CONFIG, token_names=("hydrophobic_latch_dist_nm",)
    )

    out = compute_tmem63_global_values_from_xyz(xyz, idx=idx)

    assert out.shape == (3, len(TMEM63_GLOBAL_TOKEN_NAMES))
    latch_col = TMEM63_GLOBAL_TOKEN_NAMES.index("hydrophobic_latch_dist_nm")
    assert np.all(out[:, latch_col] > 0.0)
    assert np.all(np.isfinite(out))
    other_cols = [c for c in range(out.shape[1]) if c != latch_col]
    np.testing.assert_array_equal(out[:, other_cols], np.zeros((3, len(other_cols)), dtype=np.float32))


def test_compute_tmem63_global_values_from_xyz_rejects_bad_shape():
    idx = build_tmem63_token_indices(
        _build_synthetic_latch_topology(), TMEM63B_RESIDUE_CONFIG, token_names=("hydrophobic_latch_dist_nm",)
    )
    try:
        compute_tmem63_global_values_from_xyz(np.zeros((3, 3)), idx=idx)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-[F,N,3] xyz array")


# ---------------------------------------------------------------------------
# tm3_hinge_angle_rad / tm5_hinge_angle_rad: HELANAL-based helix curvature.
# ---------------------------------------------------------------------------

_CURVATURE_RESIDS = tuple(range(100, 120))  # 20 residues, contiguous


def _build_curvature_topology():
    """A single-chain, CA-only, 20-residue topology (resSeq 100-119) with no
    other atoms -- HELANAL only ever looks at 'name CA' regardless, and the
    curvature build path doesn't touch atom_mode at all."""
    top = md.Topology()
    chain = top.add_chain()
    for resid in _CURVATURE_RESIDS:
        res = top.add_residue("ALA", chain, resSeq=resid)
        top.add_atom("CA", md.element.carbon, res)
    return top


def _kinked_helix_ca_angstrom(n_res, kink_at, kink_deg, radius=2.3, rise=1.5, twist_deg=100.0):
    """CA positions (Angstrom) along an idealized alpha helix whose central
    axis direction changes by kink_deg partway through (at residue index
    kink_at). kink_deg=0.0 gives a continuous, unkinked helix -- the
    "straight" control. Twist phase (theta = i * twist) runs continuously
    across the whole range regardless of the kink, so a kink is the *only*
    injected discontinuity, isolating what's actually being tested.
    """
    axis = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.cross(axis, u)
    twist = np.deg2rad(twist_deg)
    kink = np.deg2rad(kink_deg)
    center = np.zeros(3)
    pts = []
    for i in range(n_res):
        if i == kink_at and kink_deg != 0.0:
            perp = v
            K = np.array([[0, -perp[2], perp[1]], [perp[2], 0, -perp[0]], [-perp[1], perp[0], 0]])
            R = np.eye(3) + np.sin(kink) * K + (1 - np.cos(kink)) * (K @ K)
            axis = R @ axis
            u = R @ u
            v = np.cross(axis, u)
        theta = i * twist
        pts.append(center + radius * np.cos(theta) * u + radius * np.sin(theta) * v)
        center = center + rise * axis
    return np.asarray(pts)


def test_helix_curvature_detects_injected_kink():
    top = _build_curvature_topology()
    straight_ang = _kinked_helix_ca_angstrom(20, kink_at=10, kink_deg=0.0)
    kinked_ang = _kinked_helix_ca_angstrom(20, kink_at=10, kink_deg=45.0)
    # tmem63_metrics expects nm (it converts to Angstrom internally for MDAnalysis).
    xyz_nm = np.stack([straight_ang, kinked_ang], axis=0) / 10.0

    config = TMEM63ResidueConfig(
        label="Synthetic",
        tm3_curvature_region=HelixCurvatureSpec(residues=_CURVATURE_RESIDS),
    )
    idx = build_tmem63_token_indices(top, config, token_names=("tm3_hinge_angle_rad",))
    out = compute_tmem63_global_values_from_xyz(xyz_nm, idx=idx)

    col = TMEM63_GLOBAL_TOKEN_NAMES.index("tm3_hinge_angle_rad")
    straight_bend_rad, kinked_bend_rad = out[0, col], out[1, col]
    assert straight_bend_rad < np.deg2rad(1.0)  # ideal continuous helix -> ~0
    assert kinked_bend_rad > np.deg2rad(30.0)  # clearly detects the injected 45-degree kink
    assert kinked_bend_rad > straight_bend_rad


def test_helix_curvature_too_few_residues_raises():
    top = _build_curvature_topology()
    config = TMEM63ResidueConfig(
        label="Synthetic",
        tm3_curvature_region=HelixCurvatureSpec(residues=_CURVATURE_RESIDS[:5]),  # < _HELANAL_MIN_RESIDUES
    )
    try:
        build_tmem63_token_indices(top, config, token_names=("tm3_hinge_angle_rad",))
    except ValueError as exc:
        assert "at least" in str(exc)
    else:
        raise AssertionError("expected ValueError for too few curvature residues")


def test_reindexed_topology_subset_preserves_requested_order():
    top = md.Topology()
    chain = top.add_chain()
    res1 = top.add_residue("ALA", chain, resSeq=10)
    top.add_atom("N", md.element.nitrogen, res1)
    top.add_atom("CA", md.element.carbon, res1)
    res2 = top.add_residue("GLY", chain, resSeq=500)
    top.add_atom("N", md.element.nitrogen, res2)
    top.add_atom("CA", md.element.carbon, res2)

    unsorted = [3, 0, 2, 1]  # CA(500), N(10), N(500), CA(10)
    sub = reindexed_topology_subset(top, unsorted)

    assert sub.n_atoms == 4
    got = [(sub.atom(i).name, sub.atom(i).residue.resSeq) for i in range(4)]
    assert got == [("CA", 500), ("N", 10), ("N", 500), ("CA", 10)]


# ---------------------------------------------------------------------------
# Full TMEM63B_RESIDUE_CONFIG integration test: exercises every field filled
# in from tmem63b_gespo.md's "Exact TMEM63B Metric Residues" section end to
# end, against a synthetic topology spanning all the real residue ranges --
# catches transcription mistakes (wrong range, wrong dict key, off-by-one)
# that individual field-level tests above wouldn't.
# ---------------------------------------------------------------------------

_TM_RANGES = {
    "TM0": (40, 66), "TM1": (146, 178), "TM2": (203, 229), "TM3": (428, 458),
    "TM4": (474, 502), "TM5": (507, 549), "TM6": (566, 587), "TM7": (620, 638),
    "TM8": (643, 667), "TM9": (674, 702), "TM10": (707, 730),
}
_IL2_RANGE = (285, 305)
_TM6B_RANGE = (589, 614)
_NECK_RESIDS = (434, 462, 515, 549, 618)  # ILE434, THR462, LEU515, PHE549, TYR618


def _build_full_tmem63b_topology():
    """Every residue TMEM63B_RESIDUE_CONFIG actually references (all TM0-TM10
    ranges, IL2, TM6b, neck -- the hydrophobic latch residues fall inside
    TM0/TM4/TM6 and don't need separate entries), each with N/CA/C/O/CB
    backbone+CB atoms. Two neck residues (462, 618) fall outside every TM/IL2/
    TM6b range and would otherwise be silently missing from this fixture.
    """
    all_resids = sorted(set().union(
        *[range(a, b + 1) for a, b in _TM_RANGES.values()],
        range(*(_IL2_RANGE[0], _IL2_RANGE[1] + 1)),
        range(*(_TM6B_RANGE[0], _TM6B_RANGE[1] + 1)),
        _NECK_RESIDS,
    ))
    top = md.Topology()
    chain = top.add_chain()
    for resid in all_resids:
        res = top.add_residue("ALA", chain, resSeq=resid)
        top.add_atom("N", md.element.nitrogen, res)
        top.add_atom("CA", md.element.carbon, res)
        top.add_atom("C", md.element.carbon, res)
        top.add_atom("O", md.element.oxygen, res)
        top.add_atom("CB", md.element.carbon, res)
    return top, all_resids


def _full_tmem63b_xyz(topology, all_resids):
    """One frame of coordinates: every residue gets a well-separated 5-atom
    cluster EXCEPT TM3/TM5 (428-458, 507-549), which get real idealized helix
    geometry (via the same generator used for the kink-detection test) so
    tm3_hinge_angle_rad/tm5_hinge_angle_rad get a physically sane, non-degenerate
    trace to fit rather than an arbitrary scattered cloud.
    """
    rng = np.random.default_rng(0)
    resid_to_local = {r: i for i, r in enumerate(all_resids)}
    n_atoms = topology.n_atoms
    xyz = np.zeros((1, n_atoms, 3), dtype=np.float64)

    helix_resids = set(range(*(_TM_RANGES["TM3"][0], _TM_RANGES["TM3"][1] + 1))) | set(
        range(*(_TM_RANGES["TM5"][0], _TM_RANGES["TM5"][1] + 1))
    )
    helix_ca_by_resid = {}
    for name, (start, end) in [("TM3", _TM_RANGES["TM3"]), ("TM5", _TM_RANGES["TM5"])]:
        ca_ang = _kinked_helix_ca_angstrom(end - start + 1, kink_at=(end - start) // 2, kink_deg=0.0)
        base = 10_000.0 * (0 if name == "TM3" else 1)  # keep the two helices far apart
        for i, resid in enumerate(range(start, end + 1)):
            helix_ca_by_resid[resid] = base + ca_ang[i] / 10.0  # Angstrom -> nm

    for atom in topology.atoms:
        resid = atom.residue.resSeq
        res_local = resid_to_local[resid]
        base_pos = np.array([res_local * 5.0, 0.0, 0.0])  # well-separated per-residue clusters
        if resid in helix_resids and atom.name == "CA":
            xyz[0, atom.index] = helix_ca_by_resid[resid]
        else:
            offset = {"N": [0, 0, 0], "CA": [0.15, 0, 0], "C": [0.3, 0, 0], "O": [0.3, 0.1, 0], "CB": [0.15, 0.15, 0]}[atom.name]
            xyz[0, atom.index] = base_pos + np.asarray(offset)
    return xyz


def test_tmem63b_residue_config_end_to_end_against_real_ranges():
    top, all_resids = _build_full_tmem63b_topology()
    xyz = _full_tmem63b_xyz(top, all_resids)

    # Every field in TMEM63B_RESIDUE_CONFIG is now configured (neck included).
    token_names = TMEM63_GLOBAL_TOKEN_NAMES
    idx = build_tmem63_token_indices(top, TMEM63B_RESIDUE_CONFIG, token_names=token_names)

    ref_rmsd_nm = xyz[0, idx.rmsd_atom_idx, :]  # RMSD-to-self -> should come out ~0
    out = compute_tmem63_global_values_from_xyz(xyz, idx=idx, ref_rmsd_nm=ref_rmsd_nm)

    assert np.all(np.isfinite(out))

    # TM3/TM5's synthetic coordinates are a perfectly straight helix
    # (kink_deg=0.0), so ~0 max local bend is the *correct* answer here, not
    # a sign of misconfiguration -- same as the "straight" control in
    # test_helix_curvature_detects_injected_kink.
    zero_is_expected = {"tm_rmsd_nm", "tm3_hinge_angle_rad", "tm5_hinge_angle_rad"}
    for name in token_names:
        col = TMEM63_GLOBAL_TOKEN_NAMES.index(name)
        if name in zero_is_expected:
            assert out[0, col] < 1e-3, f"'{name}' expected to be ~0 for this synthetic setup"
        else:
            assert out[0, col] != 0.0, f"'{name}' resolved but computed as exactly 0.0 -- likely a config mistake"
