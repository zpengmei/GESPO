import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Ensure repo root is importable as a package root (pytest may not add it).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.config import TrainingConfig
from gespo.utils.endpoints import validate_endpoint_pdbs


def _ensure_real_mdtraj():
    # Some existing tests may mock mdtraj; ensure we can import the real module here.
    if isinstance(sys.modules.get("mdtraj"), MagicMock):
        sys.modules.pop("mdtraj", None)
    import mdtraj as md  # noqa: F401


def _build_two_residue_topology():
    import mdtraj as md

    top = md.Topology()
    chain = top.add_chain()
    res0 = top.add_residue("ALA", chain)
    res1 = top.add_residue("GLY", chain)

    # Residue 0 atoms (include a hydrogen so strict validation includes it)
    top.add_atom("N", md.element.nitrogen, res0)
    top.add_atom("CA", md.element.carbon, res0)
    top.add_atom("C", md.element.carbon, res0)
    top.add_atom("O", md.element.oxygen, res0)
    top.add_atom("H", md.element.hydrogen, res0)

    # Residue 1 atoms
    top.add_atom("N", md.element.nitrogen, res1)
    top.add_atom("CA", md.element.carbon, res1)
    top.add_atom("C", md.element.carbon, res1)
    top.add_atom("O", md.element.oxygen, res1)
    top.add_atom("H", md.element.hydrogen, res1)
    return top


def test_training_config_parses_pdb_a_alias_and_pdb_b():
    parser = __import__("argparse").ArgumentParser()
    TrainingConfig.populate_parser(parser)
    args = parser.parse_args(["--pdb-a", "A.pdb", "--pdb-b", "B.pdb", "--path-sampling"])
    cfg = TrainingConfig.from_args(args)

    assert cfg.pdb == "A.pdb"
    assert cfg.pdb_a == "A.pdb"
    assert cfg.pdb_b == "B.pdb"
    assert cfg.path_sampling is True


def test_validate_endpoint_pdbs_strict_passes_for_matching_topology():
    _ensure_real_mdtraj()
    import mdtraj as md

    top = _build_two_residue_topology()
    xyz_nm = np.random.RandomState(0).randn(1, top.n_atoms, 3).astype(np.float32) * 0.1
    traj = md.Trajectory(xyz_nm, top)

    with tempfile.TemporaryDirectory() as tmp:
        pA = Path(tmp) / "A.pdb"
        pB = Path(tmp) / "B.pdb"
        traj.save_pdb(str(pA))
        # Same topology, different coordinates
        traj2 = md.Trajectory(xyz_nm + 0.01, top)
        traj2.save_pdb(str(pB))

        res = validate_endpoint_pdbs(str(pA), str(pB))
        assert res.n_solute_residues == 2
        assert res.n_solute_atoms == int(top.n_atoms)


def test_validate_endpoint_pdbs_strict_fails_for_atom_mismatch():
    _ensure_real_mdtraj()
    import mdtraj as md

    top_a = _build_two_residue_topology()
    xyz_nm = np.random.RandomState(1).randn(1, top_a.n_atoms, 3).astype(np.float32) * 0.1
    traj_a = md.Trajectory(xyz_nm, top_a)

    # Build B topology missing one hydrogen atom (strict mismatch).
    top_b = md.Topology()
    chain = top_b.add_chain()
    res0 = top_b.add_residue("ALA", chain)
    res1 = top_b.add_residue("GLY", chain)
    top_b.add_atom("N", md.element.nitrogen, res0)
    top_b.add_atom("CA", md.element.carbon, res0)
    top_b.add_atom("C", md.element.carbon, res0)
    top_b.add_atom("O", md.element.oxygen, res0)
    top_b.add_atom("H", md.element.hydrogen, res0)
    top_b.add_atom("N", md.element.nitrogen, res1)
    top_b.add_atom("CA", md.element.carbon, res1)
    top_b.add_atom("C", md.element.carbon, res1)
    top_b.add_atom("O", md.element.oxygen, res1)
    # Intentionally omit the GLY hydrogen

    xyz_b_nm = np.random.RandomState(2).randn(1, top_b.n_atoms, 3).astype(np.float32) * 0.1
    traj_b = md.Trajectory(xyz_b_nm, top_b)

    with tempfile.TemporaryDirectory() as tmp:
        pA = Path(tmp) / "A.pdb"
        pB = Path(tmp) / "B.pdb"
        traj_a.save_pdb(str(pA))
        traj_b.save_pdb(str(pB))

        try:
            validate_endpoint_pdbs(str(pA), str(pB))
        except ValueError as exc:
            assert "Endpoint atom lists differ" in str(exc)
        else:
            raise AssertionError("Expected validate_endpoint_pdbs() to fail for mismatched solute atoms.")
