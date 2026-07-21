"""Auto-detection of the geom2vec backbone checkpoint. Pure filesystem logic —
no OpenMM/torch, no geom2vec install required.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gespo.utils.geom2vec_paths as g2p
from gespo.utils.geom2vec_paths import (
    DEFAULT_CHECKPOINT_NAME,
    resolve_geom2vec_checkpoint,
)


def _mk_geom2vec(tmp_path, names, monkeypatch=None):
    """Create <tmp>/geom2vec/{src,checkpoints}; return (src_dir, ckpt_dir).

    When ``monkeypatch`` is given, the resolver's candidate dirs are pinned to just
    this checkpoints dir so real system checkpoints can't leak into the assertion.
    """
    src = tmp_path / "geom2vec" / "src"
    ckpts = tmp_path / "geom2vec" / "checkpoints"
    src.mkdir(parents=True)
    ckpts.mkdir(parents=True)
    for n in names:
        (ckpts / n).write_bytes(b"stub")
    if monkeypatch is not None:
        monkeypatch.setattr(g2p, "candidate_checkpoint_dirs", lambda: [ckpts])
    return src, ckpts


def test_explicit_existing_path_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOM2VEC_CKPT", raising=False)
    monkeypatch.delenv("GEOM2VEC_SRC", raising=False)
    f = tmp_path / "my.pth"
    f.write_bytes(b"x")
    assert resolve_geom2vec_checkpoint(str(f)) == str(f)


def test_env_ckpt_used_when_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOM2VEC_SRC", raising=False)
    f = tmp_path / "backbone.pth"
    f.write_bytes(b"x")
    monkeypatch.setenv("GEOM2VEC_CKPT", str(f))
    assert resolve_geom2vec_checkpoint("") == str(f)


def test_autodetect_prefers_default_name(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOM2VEC_CKPT", raising=False)
    _, ckpts = _mk_geom2vec(
        tmp_path, ["et_l6_h64_rbf64_r75.pth", DEFAULT_CHECKPOINT_NAME], monkeypatch)
    assert resolve_geom2vec_checkpoint("") == str(ckpts / DEFAULT_CHECKPOINT_NAME)


def test_autodetect_falls_back_to_tensornet(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOM2VEC_CKPT", raising=False)
    _, ckpts = _mk_geom2vec(tmp_path, ["tensornet_l2_h200_rbf32_r5.pth"], monkeypatch)
    assert resolve_geom2vec_checkpoint("") == str(ckpts / "tensornet_l2_h200_rbf32_r5.pth")


def test_bare_filename_resolved_to_full_path(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOM2VEC_CKPT", raising=False)
    _, ckpts = _mk_geom2vec(tmp_path, [DEFAULT_CHECKPOINT_NAME], monkeypatch)
    # user passed just the filename, no directory
    assert resolve_geom2vec_checkpoint(DEFAULT_CHECKPOINT_NAME) == str(ckpts / DEFAULT_CHECKPOINT_NAME)


def test_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("GEOM2VEC_CKPT", raising=False)
    _mk_geom2vec(tmp_path, [], monkeypatch)  # empty checkpoints dir, pinned
    assert resolve_geom2vec_checkpoint("") == ""
