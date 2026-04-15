import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is importable as a package root (pytest may not add it).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gespo.training.rl import TorsionARPolicy
from gespo.training.torsion_retrieval import build_residue_torsions_from_cv_and_coords, embed_sincos_from_torsions_np


def test_torsion_ar_policy_supports_variable_angle_counts():
    torch.manual_seed(0)
    B, L, A = 4, 6, 7
    policy = TorsionARPolicy(
        num_residues=L,
        residue_vocab_size=21,
        d_model=64,
        nhead=4,
        num_layers=2,
        kmix=3,
        n_angles=A,
        n_global_tokens=0,
    )
    torsions = torch.randn(B, L, A)
    residue_types = torch.randint(0, 21, (B, L))
    angle_mask = torch.ones((B, L, A), dtype=torch.bool)
    angle_mask[:, 0, 0] = False  # missing phi at N-terminus
    angle_mask[:, -1, 1] = False  # missing psi at C-terminus
    angle_mask[:, :, 5:] = False  # pretend chi3/chi4 missing everywhere

    lp = policy.log_prob(torsions, residue_types=residue_types, angle_mask=angle_mask)
    assert tuple(lp.shape) == (B,)

    mode = policy.mode_torsions(torsions, residue_types=residue_types, angle_mask=angle_mask)
    assert tuple(mode.shape) == (B, L, A)

    ent = policy.mean_token_entropy(torsions, residue_types=residue_types, angle_mask=angle_mask, include_omega=False)
    assert tuple(ent.shape) == (B,)

    samp = policy.sample(batch_size=B, residue_types=residue_types, angle_mask=angle_mask)
    assert tuple(samp.shape) == (B, L, A)


def test_embed_sincos_from_torsions_np_masks_angles():
    tors = np.zeros((1, 2, 2), dtype=np.float32)
    angle_exists = np.array([[True, False], [True, True]], dtype=bool)
    emb = embed_sincos_from_torsions_np(tors, angle_exists)
    assert emb.shape == (1, 8)
    # res0 angle0: sin=0 cos=1; res0 angle1 masked -> 0; res1 angle0: 0/1; res1 angle1: 0/1
    expected = np.array([0, 1, 0, 0, 0, 1, 0, 1], dtype=np.float32)
    assert np.allclose(emb[0], expected, atol=1e-6)


def test_build_residue_torsions_from_cv_and_coords_places_phi_psi():
    # Two CVs: phi at slot 0, psi at slot 2.
    cv_batch = np.array([[0.1, -0.2], [-1.0, 2.5]], dtype=np.float32)
    cv_residue_slots = np.array([0, 2], dtype=np.int64)
    cv_is_phi = np.array([True, False], dtype=bool)
    tors = build_residue_torsions_from_cv_and_coords(
        cv_batch,
        coords_batch_nm=None,
        cv_residue_slots=cv_residue_slots,
        cv_is_phi=cv_is_phi,
        n_residues=3,
        n_angles=7,
    )
    assert tors.shape == (2, 3, 7)
    assert np.allclose(tors[:, 0, 0], cv_batch[:, 0])
    assert np.allclose(tors[:, 2, 1], ((cv_batch[:, 1] + np.pi) % (2.0 * np.pi)) - np.pi)
    # Unset positions remain zero
    assert np.allclose(tors[:, 1, 0], 0.0)
    assert np.allclose(tors[:, 1, 1], 0.0)
