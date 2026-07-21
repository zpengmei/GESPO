"""WS-1 seed module: CV registry, walker layout, CLI parsing, generate_seeds
dispatch. All OpenMM-free — the MD execution paths are verified by the user on a
GPU/OpenMM machine.
"""
import argparse
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.seed import _common, cvs, generate_seeds, SEED_METHODS, WALKER_METHODS
from gespo.seed.cli import build_seed_subparser


# ── CV registry (pure) ───────────────────────────────────────────────────

def test_cv_sets_exist():
    assert set(cvs.CV_SETS) == {"holo", "apo", "folding"}


def test_resolve_holo():
    specs = cvs.resolve_cv_set("holo")
    assert [s.kind for s in specs] == ["pocket_rmsd", "lig_pocket_dist"]


def test_resolve_overrides_apply():
    specs = cvs.resolve_cv_set("apo", cv1_max_nm=0.6, cv2_min_nm=0.0, cv2_max_nm=0.9)
    assert specs[0].max_nm == 0.6           # CV1 upper bound
    assert specs[1].min_nm == 0.0 and specs[1].max_nm == 0.9


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        cvs.resolve_cv_set("nope")


def test_needs_pocket_and_ligand():
    assert cvs.cv_set_needs_pocket("holo") and cvs.cv_set_needs_ligand("holo")
    assert cvs.cv_set_needs_pocket("apo") and not cvs.cv_set_needs_ligand("apo")
    assert not cvs.cv_set_needs_pocket("folding")


# ── _common helpers ──────────────────────────────────────────────────────

def test_ns_to_steps():
    assert _common.ns_to_steps(25.0, 2.0) == 12_500_000
    assert _common.ns_to_steps(0.1, 2.0) == 50_000


def test_walker_layout(tmp_path):
    dirs = _common.walker_dirs(tmp_path, 3)
    assert [d.name for d in dirs] == ["walker_1", "walker_2", "walker_3"]
    assert _common.production_dcd(dirs[0]).name == "production.dcd"


def test_resolve_devices_cycles():
    assert _common.resolve_devices("cuda:0", 3) == ["cuda:0", "cuda:0", "cuda:0"]
    assert _common.resolve_devices("cuda:0,cuda:1", 4) == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


# ── CLI parsing (OpenMM-free) ────────────────────────────────────────────

def _seed_parser():
    p = argparse.ArgumentParser(prog="gespo")
    sub = p.add_subparsers(dest="command")
    build_seed_subparser(sub)
    return p


def test_parse_metad():
    a = _seed_parser().parse_args([
        "seed", "metad", "--pdb", "h.pdb", "--out-dir", "s/",
        "--cv-set", "holo", "--ligand-smiles", "CCO", "--n-walkers", "4",
        "--implicit-solvent"])
    assert a.seed_method == "metad" and a.cv_set == "holo"
    assert a.n_walkers == 4 and a.implicit_solvent is True and a.lig_restraint is True


def test_parse_metad_no_lig_restraint():
    a = _seed_parser().parse_args(
        ["seed", "metad", "--pdb", "h.pdb", "--out-dir", "s/", "--no-lig-restraint"])
    assert a.lig_restraint is False


def test_parse_high_temp():
    a = _seed_parser().parse_args(
        ["seed", "high-temp", "--pdb", "x.pdb", "--out-dir", "s/", "--temperature", "400"])
    assert a.seed_method == "high-temp" and a.temperature == 400.0


def test_parse_rmsd_bias_reuses_bootstrap_args():
    a = _seed_parser().parse_args(["seed", "rmsd-bias", "--pdb", "x.pdb", "--out-dir", "s/"])
    assert a.seed_method == "rmsd-bias" and hasattr(a, "protocol")


def test_bad_cv_set_rejected():
    with pytest.raises(SystemExit):
        _seed_parser().parse_args(
            ["seed", "metad", "--pdb", "h.pdb", "--out-dir", "s/", "--cv-set", "bogus"])


# ── generate_seeds dispatch (no MD) ──────────────────────────────────────

def test_generate_seeds_rejects_rmsd_bias():
    assert set(WALKER_METHODS) <= set(SEED_METHODS)
    with pytest.raises(ValueError):
        generate_seeds("rmsd-bias", out_dir="s/", pdb="x.pdb")


def test_generate_seeds_rejects_unknown():
    with pytest.raises(ValueError):
        generate_seeds("nope", out_dir="s/")
