import mdtraj as md
import numpy as np

from gespo.utils.selection import AtomSelectionMode, select_atoms_for_residues
from gespo.utils.tmem63_metrics import (
    TMEM63_GLOBAL_TOKEN_NAMES,
    TMEM63A_RESIDUE_CONFIG,
    TMEM63B_RESIDUE_CONFIG,
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
    assert idx.hydrophobic_latch_atom_idx is not None
    assert idx.hydrophobic_latch_atom_idx.size == 6 * len(_LATCH_RESIDS)  # ALL_HEAVY per config
    assert idx.rmsd_atom_idx is None
    assert idx.neck_atom_idx is None


def test_build_tmem63_token_indices_raises_on_unset_required_field():
    top = _build_synthetic_latch_topology()
    try:
        build_tmem63_token_indices(top, TMEM63B_RESIDUE_CONFIG)  # default: all 9 columns
    except NotImplementedError as exc:
        assert "rmsd_regions" in str(exc)
        assert "TMEM63B" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError for unset rmsd_regions")


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
