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
    msm_ig_dim: int = 5
    msm_ig_prior_alpha: float = 1.0
    msm_ig_max_fit_samples: int = 50000
    msm_ig_seed: int = 42
    msm_ig_norm_ema_beta: float = 0.9

    CLI_META: ClassVar[dict] = {
        "msm_ig_reward_coef": (
            "Coefficient for MSM information-gain reward (0=disabled)."
        ),
        "msm_ig_state_mode": {
            "help": "Discrete states for MSM-IG: 'kmeans' (microstates) or 'vamp' (argmax).",
            "choices": ["kmeans", "vamp"],
        },
        "msm_ig_n_states": "Number of k-means microstates (K) for MSM-IG.",
        "msm_ig_dim": {
            "help": "Linear VAMP coordinate dimension for MSM k-means.",
            "default_override": 2,
        },
        "msm_ig_prior_alpha": "Symmetric Dirichlet prior pseudo-count per transition entry.",
        "msm_ig_max_fit_samples": "Max history frames for MSM k-means fitting.",
        "msm_ig_seed": "RNG seed for MSM k-means fitting.",
        "msm_ig_norm_ema_beta": "EMA beta for MSM-IG z-score normalization (0=disabled).",
    }
