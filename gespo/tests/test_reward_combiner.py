import types

import numpy as np
import torch

from gespo.rewards.combiner import RewardCombiner, RewardTerms


def test_reward_combiner_scales_aux_terms_from_numpy_loo_array():
    config = types.SimpleNamespace(
        loo_vamp_reward_weight=1.0,
        start_cluster_entropy_coef=0.0,
        vamp_bin_entropy_coef=1.0,
        msm_ig_reward_coef=0.0,
        off_action_penalty_coef=0.0,
        committor_progress_reward_weight=0.0,
        committor_bin_entropy_reward_weight=0.0,
        committor_tube_diversity_reward_weight=0.0,
        committor_tube_edge_novelty_reward_weight=0.0,
        committor_connectivity_reward_weight=0.0,
        committor_start_basin_entropy_reward_weight=0.0,
        ent_coef=1e-3,
        vamp_bin_norm_ema_beta=0.0,
        msm_ig_norm_ema_beta=0.0,
        committor_norm_ema_beta=0.0,
    )
    combiner = RewardCombiner(config, torch.device("cpu"))

    terms = RewardTerms(
        loo_vamp=np.array([0.0, 0.0, 10.0], dtype=np.float32),
        vamp_bin=np.array([-1.0, 1.0, 0.0], dtype=np.float32),
    )
    active_mask = torch.tensor([True, True, False])
    combined, _ = combiner.combine(terms, torch.zeros(3), active_mask)

    loo_std = float(np.std(terms.loo_vamp))
    raw = terms.vamp_bin
    expected_aux0 = ((raw[0] - float(raw.mean())) / float(raw.std())) * loo_std
    assert np.isclose(float(combined[0]), expected_aux0, atol=1e-5)