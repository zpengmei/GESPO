"""MSM information-gain reward configuration."""
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MSMIGConfig:
    """Configuration for the MSM information-gain exploration reward.

    Only active when ``msm_ig_reward_coef > 0``.
    """

    msm_ig_reward_coef: float = 0.0
    msm_ig_state_mode: str = "kmeans"  # "kmeans" or "vamp"
    msm_ig_n_states: int = 50
    msm_ig_dim: int = 2
    msm_ig_prior_alpha: float = 1.0
    msm_ig_max_fit_samples: int = 50000
    msm_ig_seed: int = 42
    msm_ig_norm_ema_beta: float = 0.9
    # ── Start-distance (PCCA macrostate-centroid) reward ─────────────
    start_distance_reward_coef: float = 0.0
    pcca_n_macrostates: int = 7
    start_distance_norm_ema_beta: float = 0.9

    CLI_META: ClassVar[dict] = {
        "msm_ig_reward_coef": (
            "Coefficient for MSM information-gain reward (0=disabled)."
        ),
        "msm_ig_state_mode": {
            "help": "Discrete states for MSM-IG: 'kmeans' (microstates) or 'vamp' (argmax).",
            "choices": ["kmeans", "vamp"],
        },
        "msm_ig_n_states": "Number of k-means microstates (K) for MSM-IG.",
        "msm_ig_dim": "Linear VAMP coordinate dimension for MSM k-means.",
        "msm_ig_prior_alpha": "Symmetric Dirichlet prior pseudo-count per transition entry.",
        "msm_ig_max_fit_samples": "Max history frames for MSM k-means fitting.",
        "msm_ig_seed": "RNG seed for MSM k-means fitting.",
        "msm_ig_norm_ema_beta": "EMA beta for MSM-IG z-score normalization (0=disabled).",
        "start_distance_reward_coef": (
            "Coefficient for per-trajectory start-distance reward: L2 distance "
            "(in normalized linear-VAMP space) from each walker's starting frame "
            "to the nearest PCCA macrostate centroid. Rewards proposing starts "
            "that escape known basins. 0=disabled."
        ),
        "pcca_n_macrostates": "Number of PCCA macrostates fit on the MSM-IG microstate transition matrix.",
        "start_distance_norm_ema_beta": "EMA beta for start-distance reward z-score normalization.",
    }
