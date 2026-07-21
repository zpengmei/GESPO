"""WS-4 correctness fixes: --torsion-mode canonicalization and --global-token-set
validation. These are pure config/validation checks — no OpenMM, no MD run.
"""
import argparse
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import from the dependency-light module so these tests run without OpenMM.
from gespo.training.token_sets import (
    COMMON_GLOBAL_TOKEN_SETS,
    KNOWN_GLOBAL_TOKEN_SETS,
    normalize_torsion_mode,
    validate_global_token_set,
)
from gespo.cli import _COMMON_TOKEN_SETS


# ── --torsion-mode canonicalization ─────────────────────────────────────

def test_torsion_mode_backbone_passthrough():
    assert normalize_torsion_mode("backbone") == "backbone"


def test_torsion_mode_sidechain_passthrough():
    assert normalize_torsion_mode("backbone+sidechain") == "backbone+sidechain"


def test_torsion_mode_legacy_alias_maps():
    # backbone_chi used to parse then crash deep in pretraining.
    assert normalize_torsion_mode("backbone_chi") == "backbone+sidechain"


def test_torsion_mode_none_defaults_backbone():
    assert normalize_torsion_mode(None) == "backbone"
    assert normalize_torsion_mode("") == "backbone"


# ── --global-token-set validation ───────────────────────────────────────

def test_token_set_default_ok():
    assert validate_global_token_set("rmsd_e2e") == "rmsd_e2e"
    assert validate_global_token_set(None) == "rmsd_e2e"


def test_token_set_pocket_ok():
    assert validate_global_token_set("rmsd_e2e_pocket_full") == "rmsd_e2e_pocket_full"
    # normalization: upper/whitespace tolerated
    assert validate_global_token_set("  RMSD_E2E_POCKET3 ") == "rmsd_e2e_pocket3"


def test_token_set_unknown_raises_with_list():
    with pytest.raises(ValueError) as exc:
        validate_global_token_set("rmsd_e2e_poket_full")  # typo
    msg = str(exc.value)
    assert "Unknown --global-token-set" in msg
    assert "rmsd_e2e_pocket_full" in msg  # the valid list is shown


def test_common_sets_are_all_valid():
    # The help-listed common sets (both copies) must be real token sets.
    assert set(COMMON_GLOBAL_TOKEN_SETS) <= KNOWN_GLOBAL_TOKEN_SETS
    # cli.py keeps a literal copy to avoid importing torch at help-time; keep in sync.
    assert set(_COMMON_TOKEN_SETS) == set(COMMON_GLOBAL_TOKEN_SETS)


# ── the pretrain parser accepts the new + legacy torsion-mode spellings ──

def _pretrain_parser():
    # Rebuild a parser mirroring the pretrain subparser's torsion-mode choices.
    p = argparse.ArgumentParser()
    p.add_argument(
        "--torsion-mode", default="backbone",
        choices=["backbone", "backbone+sidechain", "backbone_chi"],
    )
    return p


@pytest.mark.parametrize("value", ["backbone", "backbone+sidechain", "backbone_chi"])
def test_pretrain_torsion_mode_choices_parse(value):
    ns = _pretrain_parser().parse_args(["--torsion-mode", value])
    assert ns.torsion_mode == value
    # …and each canonicalizes to a tokenizer-accepted value.
    assert normalize_torsion_mode(ns.torsion_mode) in ("backbone", "backbone+sidechain")
