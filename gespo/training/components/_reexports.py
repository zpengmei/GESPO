"""Consolidated re-exports for backward-compatible shim imports.

Each section below corresponds to one of the former single-file shims that
lived in this package.  The canonical modules are in gespo.rewards.*,
gespo.analysis.*, and gespo.projections.*.
"""

# --- loo_vamp_rewards (was gespo.training.components.loo_vamp_rewards) -------
from gespo.rewards.loo_vamp import *  # noqa: F401,F403
from gespo.rewards.loo_vamp import LOOVAMPRewards  # noqa: F811

# --- global_metrics (was gespo.training.components.global_metrics) -----------
from gespo.analysis.metrics import *  # noqa: F401,F403
from gespo.analysis.metrics import compute_globals_from_coords  # noqa: F811

# --- vamp_bin_bonus (was gespo.training.components.vamp_bin_bonus) -----------
from gespo.rewards.vamp_bin import *  # noqa: F401,F403
from gespo.rewards.vamp_bin import VampBinBonus  # noqa: F811

# --- msm_ig_bonus (was gespo.training.components.msm_ig_bonus) --------------
from gespo.rewards.msm_ig import *  # noqa: F401,F403
from gespo.rewards.msm_ig import MSMIGBonus  # noqa: F811

# --- graph_algorithms (was gespo.training.components.graph_algorithms) ------
from gespo.rewards.connectivity import *  # noqa: F401,F403

# --- vamp_refresh (was gespo.training.components.vamp_refresh) --------------
from gespo.projections.vamp_refresh import *  # noqa: F401,F403

# --- lobe_projector (was gespo.training.components.lobe_projector) ----------
from gespo.projections.lobe_projector import *  # noqa: F401,F403
