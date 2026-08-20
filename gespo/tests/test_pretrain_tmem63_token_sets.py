import mdtraj as md
import numpy as np

from gespo.training.pretrain_torsion_ar import _build_tmem63_ref_rmsd_nm
from gespo.utils.tmem63_metrics import (
    TMEM63_GLOBAL_TOKEN_NAMES,
    TMEM63_TOKEN_SET_COLUMNS,
    TMEM63B_RESIDUE_CONFIG,
    build_tmem63_token_indices,
    compute_tmem63_global_values_from_xyz,
)
from gespo.tests.test_tmem63_metrics import (
    _build_full_tmem63b_topology,
    _full_tmem63b_xyz,
)


def test_token_sets_registered_match_comments_in_token_sets_py():
    # gespo/training/token_sets.py's KNOWN_GLOBAL_TOKEN_SETS comments define
    # exactly what each name should contain; this is the transcription of
    # those comments into TMEM63_TOKEN_SET_COLUMNS.
    expected = {
        "tmem63": TMEM63_GLOBAL_TOKEN_NAMES,
        "tmem63_dists": ("tm_rmsd_nm", "hydrophobic_latch_dist_nm", "neck_dist_nm", "il2_tm6b_dist_nm"),
        "tmem63_dists_only": ("hydrophobic_latch_dist_nm", "neck_dist_nm", "il2_tm6b_dist_nm"),
        "tmem63_hel_hinges": ("tm_rmsd_nm", "tm6_hinge_angle_rad", "tm3_hinge_angle_rad", "tm5_hinge_angle_rad"),
        "tmem63_hel_hinges_only": ("tm6_hinge_angle_rad", "tm3_hinge_angle_rad", "tm5_hinge_angle_rad"),
        "tmem63_hel_tilts": ("tm_rmsd_nm", "tm6_membrane_angle_rad", "tm4_membrane_angle_rad"),
        "tmem63_hel_tilts_only": ("tm6_membrane_angle_rad", "tm4_membrane_angle_rad"),
    }
    for name, cols in expected.items():
        assert TMEM63_TOKEN_SET_COLUMNS[name] == cols, name


def test_token_set_column_counts_match_KNOWN_GLOBAL_TOKEN_SETS():
    # Every name TMEM63_TOKEN_SET_COLUMNS claims to support must actually be
    # registered in token_sets.py's KNOWN_GLOBAL_TOKEN_SETS, or
    # --global-token-set <name> would fail validation before ever reaching
    # this dispatch table -- catches the two files drifting apart.
    from gespo.training.token_sets import KNOWN_GLOBAL_TOKEN_SETS
    for name in TMEM63_TOKEN_SET_COLUMNS:
        assert name in KNOWN_GLOBAL_TOKEN_SETS, f"'{name}' not registered in KNOWN_GLOBAL_TOKEN_SETS"


def _permuted_topology_same_residues(top):
    """A topology with the same atoms/residues as `top` but a different global
    atom ordering -- used to confirm resSeq+name matching doesn't rely on
    positional alignment between reference and trajectory topologies."""
    all_atoms = list(top.atoms)
    rng = np.random.default_rng(1)
    order = rng.permutation(len(all_atoms))
    new_top = md.Topology()
    chain = new_top.add_chain()
    residue_by_id = {}
    for i in order:
        atom = all_atoms[i]
        res = atom.residue
        key = int(res.index)
        new_res = residue_by_id.get(key)
        if new_res is None:
            new_res = new_top.add_residue(res.name, chain, resSeq=int(res.resSeq))
            residue_by_id[key] = new_res
        new_top.add_atom(atom.name, atom.element, new_res)
    return new_top, order


def test_build_tmem63_ref_rmsd_nm_matches_by_resseq_and_name_not_position():
    top, all_resids = _build_full_tmem63b_topology()
    xyz = _full_tmem63b_xyz(top, all_resids)  # [1, N, 3] nm

    ref_top, order = _permuted_topology_same_residues(top)
    ref_xyz0 = xyz[0][order]  # same physical positions, permuted to match ref_top's atom order

    idx = build_tmem63_token_indices(top, TMEM63B_RESIDUE_CONFIG, token_names=("tm_rmsd_nm",))
    aligned = _build_tmem63_ref_rmsd_nm(ref_top, ref_xyz0, top, idx.rmsd_atom_idx)

    assert aligned is not None
    assert aligned.shape == (idx.rmsd_atom_idx.size, 3)
    # Reference (permuted) and trajectory should agree on every RMSD atom's
    # position once correctly realigned -- this is the actual correctness bar,
    # not just "returned something the right shape."
    np.testing.assert_allclose(aligned, xyz[0, idx.rmsd_atom_idx, :], atol=1e-6)


def test_build_tmem63_ref_rmsd_nm_returns_none_when_reference_incomplete():
    top, all_resids = _build_full_tmem63b_topology()
    xyz = _full_tmem63b_xyz(top, all_resids)
    idx = build_tmem63_token_indices(top, TMEM63B_RESIDUE_CONFIG, token_names=("tm_rmsd_nm",))

    # A reference missing most residues can't possibly cover every RMSD atom.
    stub_top = md.Topology()
    chain = stub_top.add_chain()
    res = stub_top.add_residue("ALA", chain, resSeq=all_resids[0])
    stub_top.add_atom("CA", md.element.carbon, res)
    stub_xyz0 = np.zeros((1, 3))

    aligned = _build_tmem63_ref_rmsd_nm(stub_top, stub_xyz0, top, idx.rmsd_atom_idx)
    assert aligned is None


def test_tmem63_token_set_columns_end_to_end_for_all_schemes():
    top, all_resids = _build_full_tmem63b_topology()
    xyz = _full_tmem63b_xyz(top, all_resids)

    # Every field in TMEM63B_RESIDUE_CONFIG is now configured (neck included),
    # so every registered scheme should build and compute successfully.
    for scheme in TMEM63_TOKEN_SET_COLUMNS:
        columns = TMEM63_TOKEN_SET_COLUMNS[scheme]
        idx = build_tmem63_token_indices(top, TMEM63B_RESIDUE_CONFIG, token_names=columns)

        ref_rmsd_nm = None
        if "tm_rmsd_nm" in columns:
            ref_rmsd_nm = _build_tmem63_ref_rmsd_nm(top, xyz[0], top, idx.rmsd_atom_idx)
            assert ref_rmsd_nm is not None

        full9 = compute_tmem63_global_values_from_xyz(xyz, idx=idx, ref_rmsd_nm=ref_rmsd_nm)
        col_of = {name: i for i, name in enumerate(TMEM63_GLOBAL_TOKEN_NAMES)}
        glob_block = np.stack([full9[:, col_of[name]] for name in columns], axis=1)

        assert glob_block.shape == (1, len(columns))
        assert np.all(np.isfinite(glob_block))
        # tm_rmsd_nm should be ~0 (self-RMSD); tm3/tm5_hinge_angle_rad should
        # also be ~0 here since _full_tmem63b_xyz's TM3/TM5 are a perfectly
        # straight synthetic helix (kink_deg=0.0) -- same as
        # test_tmem63b_residue_config_end_to_end_against_real_ranges.
        zero_is_expected = {"tm_rmsd_nm", "tm3_hinge_angle_rad", "tm5_hinge_angle_rad"}
        for j, name in enumerate(columns):
            if name in zero_is_expected:
                assert glob_block[0, j] < 1e-3, f"'{name}' expected to be ~0 for this synthetic setup"
            else:
                assert glob_block[0, j] != 0.0, f"'{name}' computed as exactly 0.0 in scheme '{scheme}'"


def test_active_tmem63_residue_config_defaults_to_tmem63b():
    from gespo.utils.tmem63_metrics import ACTIVE_TMEM63_RESIDUE_CONFIG
    assert ACTIVE_TMEM63_RESIDUE_CONFIG is TMEM63B_RESIDUE_CONFIG


def test_build_tmem63_ref_rmsd_nm_uses_active_config_by_default_but_is_overridable():
    from gespo.utils.tmem63_metrics import ACTIVE_TMEM63_RESIDUE_CONFIG, TMEM63A_RESIDUE_CONFIG

    top, all_resids = _build_full_tmem63b_topology()
    xyz = _full_tmem63b_xyz(top, all_resids)
    idx = build_tmem63_token_indices(top, TMEM63B_RESIDUE_CONFIG, token_names=("tm_rmsd_nm",))

    # Default (residue_config unspecified) resolves ACTIVE_TMEM63_RESIDUE_CONFIG
    # at call time -- currently TMEM63B, so this should succeed against a
    # TMEM63B-shaped topology.
    default_result = _build_tmem63_ref_rmsd_nm(top, xyz[0], top, idx.rmsd_atom_idx)
    assert default_result is not None

    # Explicitly overriding to TMEM63A_RESIDUE_CONFIG (entirely unconfigured)
    # must fail cleanly (None), not silently fall back to the active default.
    overridden_result = _build_tmem63_ref_rmsd_nm(
        top, xyz[0], top, idx.rmsd_atom_idx, residue_config=TMEM63A_RESIDUE_CONFIG
    )
    assert overridden_result is None
