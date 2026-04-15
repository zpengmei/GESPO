"""Committor-related configuration fields (path-sampling mode only)."""
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class CommittorConfig:
    """Configuration for committor estimation and committor-based rewards.

    These fields are only used when ``path_sampling=True`` (in
    ``loop_generative_path.py``).
    """

    # Progress reward: Q = max(0, max_t q+(x_t) - q+(x_0)), EMA z-score normalized.
    committor_progress_reward_weight: float = 0.0

    # Bin-entropy reward: spread probability across committor bins, w_hat=4q(1-q).
    committor_bin_entropy_reward_weight: float = 0.0

    # State mode for committor estimation: "dga" or "vamp".
    committor_state_mode: str = "dga"

    # If True, always use RMSD-to-endpoints proxy committor.
    committor_use_rmsd_proxy: bool = False

    # Tube/edge microstate mode: "model", "q_only", or "rmsd_2d".
    committor_tube_state_mode: str = "model"

    # RMSD-2D discretization settings.
    committor_rmsd_state_bins: int = 10
    committor_rmsd_state_max_nm: float = 2.0

    # Tube-weighted diversity reward.
    committor_tube_diversity_reward_weight: float = 0.0
    committor_tube_diversity_q_bins: int = 20
    committor_tube_diversity_alpha: float = 1.0
    committor_tube_diversity_log_eps: float = 1e-12

    # Tube-weighted edge novelty reward.
    committor_tube_edge_novelty_reward_weight: float = 0.0
    committor_tube_edge_novelty_lag: int = 1
    committor_tube_edge_novelty_undirected: bool = True
    committor_tube_edge_novelty_bin_change_only: bool = False

    # Start-basin entropy bonus.
    committor_start_basin_entropy_reward_weight: float = 0.0

    # Path connectivity reward.
    committor_connectivity_reward_weight: float = 0.0
    committor_connectivity_connect_bonus: float = 1.0
    committor_connectivity_shortest_path_weight: float = 1.0
    committor_connectivity_flow_weight: float = 0.0
    committor_connectivity_flow_cap: int = 10
    committor_connectivity_component_merge_weight: float = 0.0
    committor_connectivity_gap_weight: float = 0.0
    committor_connectivity_auto_tune: bool = False

    # EMA normalization beta for committor rewards.
    committor_norm_ema_beta: float = 0.9

    # DGA committor backend and fitting settings.
    committor_dga_backend: str = "auto"
    committor_dga_n_clusters: int = 200
    committor_dga_dim: int = 0  # 0 => auto
    committor_dga_lagtime: int = 0  # 0 => use geom2vec_lagtime
    committor_dga_max_fit_samples: int = 50000
    committor_dga_seed: int = 42
    committor_dga_include_global_tokens: bool = False

    # Takens-style delay embedding for DGA committor.
    committor_dga_delay_n_embed: int = 0
    committor_dga_delay_delta: int = 0  # <= 0 => defaults to committor_dga_lagtime

    CLI_META: ClassVar[dict] = {
        "committor_progress_reward_weight": "Weight for committor-progress reward term (path mode).",
        "committor_bin_entropy_reward_weight": (
            "Weight for committor-bin entropy reward (path mode): spread mass "
            "across committor bins weighted by 4q(1-q)."
        ),
        "committor_state_mode": {
            "help": "Discrete states for committor estimation: 'dga' or 'vamp'.",
            "choices": ["dga", "vamp"],
        },
        "committor_use_rmsd_proxy": {
            "help": "Force RMSD-to-endpoints proxy committor.",
            "flag_only": True,
        },
        "committor_tube_state_mode": {
            "help": "Tube/edge bookkeeping state mode.",
            "choices": ["model", "q_only", "rmsd_2d"],
        },
        "committor_rmsd_state_bins": "Bins per axis for (rmsd_a, rmsd_b) discretization.",
        "committor_rmsd_state_max_nm": "RMSD max (nm) mapped to the last bin.",
        "committor_tube_diversity_reward_weight": "Weight for committor-tube diversity reward (path mode).",
        "committor_tube_diversity_q_bins": "Coarse committor bins (M) for tube-diversity.",
        "committor_tube_diversity_alpha": "Pseudocount alpha for tube-cell probabilities.",
        "committor_tube_diversity_log_eps": "Epsilon inside -log(p+eps) for tube-diversity.",
        "committor_tube_edge_novelty_reward_weight": "Weight for tube-weighted edge novelty reward (path mode).",
        "committor_tube_edge_novelty_lag": "Edge lag delta (in committor-frame units).",
        "committor_tube_edge_novelty_undirected": {
            "help": "Use undirected tube edges for edge novelty (default).",
            "no_flag": "--committor-tube-edge-novelty-directed",
            "no_help": "Use directed tube edges for edge novelty.",
        },
        "committor_tube_edge_novelty_bin_change_only": {
            "help": "Only reward tube edges that change committor bin m.",
            "flag_only": True,
        },
        "committor_start_basin_entropy_reward_weight": (
            "Weight for start-basin entropy bonus (path mode): reward starts in rarer basin."
        ),
        "committor_connectivity_reward_weight": (
            "Weight for path connectivity reward (path mode): A\u2192B connectivity."
        ),
        "committor_connectivity_connect_bonus": "Bonus when new edges connect A and B.",
        "committor_connectivity_shortest_path_weight": "Weight for reducing A\u2192B shortest-path length.",
        "committor_connectivity_flow_weight": "Weight for A\u2194B edge-disjoint path count.",
        "committor_connectivity_flow_cap": "Cap for max-flow computation (integer \u22651).",
        "committor_connectivity_component_merge_weight": "Weight for merging tube-graph components.",
        "committor_connectivity_gap_weight": "Weight for reducing A/B component committor-bin gap.",
        "committor_connectivity_auto_tune": {
            "help": "Auto-balance exploration vs bridging by scaling connectivity sub-weights.",
            "flag_only": True,
        },
        "committor_norm_ema_beta": "EMA beta for committor-progress z-score normalization.",
        "committor_dga_backend": {
            "help": "Backend for DGA committor solve.",
            "choices": ["auto", "extq", "numpy"],
        },
        "committor_dga_n_clusters": "k-means microstates for DGA committor basis (VAC coords).",
        "committor_dga_dim": "VAC coordinate dimension for DGA committor (0=auto).",
        "committor_dga_lagtime": "DGA committor lag time in saved frames (0=use geom2vec_lagtime).",
        "committor_dga_max_fit_samples": "Max history frames for fitting DGA k-means basis.",
        "committor_dga_seed": "RNG seed for DGA k-means fitting.",
        "committor_dga_include_global_tokens": {
            "help": "Concatenate policy global tokens to DGA committor features.",
            "flag_only": True,
        },
        "committor_dga_delay_n_embed": "Takens-style delay embedding steps for DGA committor (0=disabled).",
        "committor_dga_delay_delta": (
            "Delay-embedding spacing (saved frames). "
            "\u22640 defaults to committor_dga_lagtime."
        ),
    }
