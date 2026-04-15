"""VAMP-bin exploration bonus configuration."""
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class VampBinConfig:
    """Configuration for the VAMP-bin exploration bonus.

    Only active when ``vamp_bin_entropy_coef != 0``.
    """

    vamp_bin_entropy_coef: float = -1.0  # < 0 ties to ent_coef
    vamp_bin_norm_ema_beta: float = 0.9
    vamp_bin_bins: int = 50
    vamp_bin_clip: float = 3.0
    vamp_bin_max_samples: int = 20000

    CLI_META: ClassVar[dict] = {
        "vamp_bin_entropy_coef": (
            "Coefficient for 2D VAMP-bin count exploration bonus "
            "(-1=tie to ent-coef, 0=disabled)."
        ),
        "vamp_bin_norm_ema_beta": "EMA beta for VAMP-bin bonus normalization (0=off).",
        "vamp_bin_bins": "Bins per axis for the 2D VAMP-bin exploration bonus.",
        "vamp_bin_clip": "Z-score clip range for VAMP-bin bonus.",
        "vamp_bin_max_samples": "Max samples for VAMP-bin mean/std estimation at refresh.",
    }
