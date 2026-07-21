"""Tests for the M3 CLI-clarity changes: essential/advanced help and the
first-class --mode / --solvent shortcuts. Config-layer only; no OpenMM/torch.
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.cli import _apply_mode_solvent_shortcuts
from gespo.config import TrainingConfig
from gespo.config.cli_gen import ESSENTIAL_FIELDS


def _parser(show_advanced=True):
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="")
    p.add_argument("--mode", choices=["generative", "path-sampling", "standard"], default=None)
    p.add_argument("--solvent", choices=["explicit", "implicit", "gbsa", "gbn2"], default=None)
    TrainingConfig.populate_parser(p, show_advanced=show_advanced)
    return p


def _cfg(argv, show_advanced=True):
    ns = _parser(show_advanced).parse_args(argv)
    _apply_mode_solvent_shortcuts(ns)
    return TrainingConfig.from_args(ns).to_dict()


# ── M3.1: essential vs advanced help ────────────────────────────────────

def test_advanced_flag_hidden_but_still_parses():
    # show_advanced=False suppresses help of a non-essential flag, but it parses.
    c = _cfg(["--pdb", "x.pdb", "--committor-dga-seed", "7"], show_advanced=False)
    assert c["committor_dga_seed"] == 7


def test_essential_fields_all_real():
    valid = set(TrainingConfig().to_dict().keys())
    assert ESSENTIAL_FIELDS <= valid


# ── M3.2: --mode / --solvent shortcuts ──────────────────────────────────

def test_mode_generative_equivalent():
    a = _cfg(["--pdb", "x.pdb", "--mode", "generative"])
    b = _cfg(["--pdb", "x.pdb", "--generative-loop"])
    assert a["generative_loop"] == b["generative_loop"] is True
    assert a["path_sampling"] == b["path_sampling"]


def test_mode_path_sampling_equivalent():
    a = _cfg(["--pdb", "x.pdb", "--mode", "path-sampling"])
    b = _cfg(["--pdb", "x.pdb", "--generative-loop", "--path-sampling"])
    assert a["generative_loop"] is True and a["path_sampling"] is True
    assert a["generative_loop"] == b["generative_loop"]
    assert a["path_sampling"] == b["path_sampling"]


def test_solvent_implicit_equivalent():
    a = _cfg(["--pdb", "x.pdb", "--solvent", "implicit"])
    b = _cfg(["--pdb", "x.pdb", "--implicit-solvent"])
    assert a["implicit_solvent"] is True and a["explicit_solvent"] is False
    assert a["implicit_solvent"] == b["implicit_solvent"]


def test_explicit_flag_overrides_mode():
    c = _cfg(["--pdb", "x.pdb", "--mode", "standard", "--generative-loop"])
    assert c["generative_loop"] is True  # explicit flag wins over --mode standard


def test_mode_solvent_attrs_do_not_leak():
    c = _cfg(["--pdb", "x.pdb", "--mode", "generative", "--solvent", "explicit"])
    assert "mode" not in c and "solvent" not in c
    assert c["explicit_solvent"] is True and c["generative_loop"] is True
