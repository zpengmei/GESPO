"""Regression tests for the M2 config-correctness changes.

Covers:
- default-drift reconciliation (dataclass = single source of truth),
- the None-sentinel fix so `train --config` respects YAML instead of clobbering
  it with CLI defaults,
- YAML footgun validation (warn on unknown keys/sections; --strict raises).

These exercise only the config layer, so they need no OpenMM/torch.
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

# Ensure repo root is importable as a package root (pytest may not add it).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.cli import _load_config_from_args
from gespo.config import TrainingConfig
from gespo.config.loader import load_config


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="")
    TrainingConfig.populate_parser(p)
    return p


def _effective(argv):
    return TrainingConfig.from_args(_parser().parse_args(argv)).to_dict()


def _write(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


# ── M2.1: reconciled defaults, dataclass = single source of truth ───────

def test_from_dict_empty_no_longer_raises_and_has_canonical_defaults():
    d = TrainingConfig.from_dict({}).to_dict()
    assert d["grp_size"] == 16
    assert d["episodes"] == 20
    assert d["pdb"] == ""
    assert d["retrieval_candidates"] == 4096
    assert d["anchoring_steps"] == 100000
    assert d["msm_ig_dim"] == 2
    assert d["geom2vec_label_mode"] == "linear_kmeans"
    assert d["steer_k_dist"] == 5000.0
    assert d["steer_k_theta"] == 200.0


def test_cli_and_from_dict_defaults_agree():
    cli_defaults = _effective([])
    api_defaults = TrainingConfig.from_dict({}).to_dict()
    for key in ("grp_size", "episodes", "retrieval_candidates", "anchoring_steps",
                "msm_ig_dim", "geom2vec_label_mode", "steer_k_dist", "steer_k_theta"):
        assert cli_defaults[key] == api_defaults[key], key


# ── M2.2: YAML survives `train --config`; CLI overrides YAML ─────────────

def test_yaml_values_survive_config_load():
    yp = _write(
        "endpoints:\n  pdb_a: from_yaml.pdb\n"
        "training:\n  episodes: 99\n  grp_size: 3\n"
        "io:\n  snapshot_dir: runs/from_yaml\n"
        "geom2vec:\n  lagtime: 77\n"
        "use_boundary_replay: false\n"      # bool whose dataclass default is True
        "save_epoch_coords: false\n"
    )
    args = _parser().parse_args(["--config", yp])  # only --config, no other flags
    cfg = TrainingConfig.from_args(_load_config_from_args(args)).to_dict()
    assert cfg["pdb"] == "from_yaml.pdb"
    assert cfg["episodes"] == 99
    assert cfg["grp_size"] == 3
    assert cfg["snapshot_dir"] == "runs/from_yaml"
    assert cfg["geom2vec_lagtime"] == 77
    # The clobber bug: these were forced back to True by the CLI default.
    assert cfg["use_boundary_replay"] is False
    assert cfg["save_epoch_coords"] is False


def test_cli_flag_overrides_yaml():
    yp = _write("training:\n  episodes: 99\nuse_boundary_replay: false\n")
    args = _parser().parse_args(["--config", yp, "--episodes", "5", "--boundary-replay"])
    cfg = TrainingConfig.from_args(_load_config_from_args(args)).to_dict()
    assert cfg["episodes"] == 5
    assert cfg["use_boundary_replay"] is True


# ── M2.3: YAML footgun validation ───────────────────────────────────────

def test_unknown_section_warns_not_dropped():
    yp = _write("training:\n  episodes: 5\nrewardz:\n  foo: 1\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_config(yp)
    assert any("rewardz" in str(x.message) for x in w)


def test_unknown_flat_key_warns():
    yp = _write("episods: 20\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_config(yp)
    assert any("episods" in str(x.message) for x in w)


def test_strict_raises_on_unknown():
    yp = _write("rewardz:\n  foo: 1\n")
    try:
        load_config(yp, strict=True)
    except ValueError as e:
        assert "rewardz" in str(e)
    else:
        raise AssertionError("expected ValueError in strict mode")


def test_known_dict_field_is_exempt():
    yp = _write("global_token_ranges:\n  rmsd: [0, 1]\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_config(yp)
    assert not any("global_token_ranges" in str(x.message) for x in w)
