"""Path-sampling configuration fields."""
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class PathSamplingConfig:
    """Configuration for path-sampling mode (A↔B transitions).

    These fields control how endpoint boundaries are defined and how
    path-sampling is enabled. Only active when ``path_sampling=True``.
    """

    # Enable path-sampling mode (requires pdb_b).
    path_sampling: bool = False

    # RMSD thresholds (nm) to endpoints A/B for boundary definition.
    path_boundary_cutoff_a_nm: float = 0.20
    path_boundary_cutoff_b_nm: float = 0.20

    # Boundary mode: "rmsd_cutoff", "vac_kmeans_rmsd_percent",
    # "vac_kmeans_rmsd_cluster_percent", or "vamp_state".
    path_boundary_mode: str = "rmsd_cutoff"

    # VAC-kmeans boundary parameters.
    path_boundary_vac_kmeans_n_clusters: int = 50
    path_boundary_vac_kmeans_dim: int = 3
    path_boundary_vac_kmeans_percent_a: float = 10.0
    path_boundary_vac_kmeans_percent_b: float = 10.0
    path_boundary_vac_kmeans_max_fit_samples: int = 50000
    path_boundary_vac_kmeans_seed: int = 42

    # Optional user hook: "path/to/plugin.py:callable_name".
    path_boundary_plugin: str = ""

    CLI_META: ClassVar[dict] = {
        "path_sampling": {
            "help": "Enable path-sampling mode (requires --pdb-b).",
            "flag_only": True,
        },
        "path_boundary_cutoff_a_nm": "Path-sampling boundary A cutoff (CA RMSD to endpoint A, nm).",
        "path_boundary_cutoff_b_nm": "Path-sampling boundary B cutoff (CA RMSD to endpoint B, nm).",
        "path_boundary_mode": {
            "help": (
                "Path boundary definition mode. "
                "'rmsd_cutoff' uses CA-RMSD cutoffs. "
                "'vamp_state' auto-detects from VAMP state assignments. "
                "'vac_kmeans_rmsd_percent' covers approx percent of frames. "
                "'vac_kmeans_rmsd_cluster_percent' selects percent of populated clusters."
            ),
            "choices": [
                "vamp_state", "rmsd_cutoff",
                "vac_kmeans_rmsd_percent", "vac_kmeans_rmsd_cluster_percent",
            ],
        },
        "path_boundary_vac_kmeans_n_clusters": "k-means clusters for VAC-space boundary definition.",
        "path_boundary_vac_kmeans_dim": "VAC coordinate dimension for boundary k-means.",
        "path_boundary_vac_kmeans_percent_a": (
            "Boundary A percent selector for VAC-kmeans modes."
        ),
        "path_boundary_vac_kmeans_percent_b": (
            "Boundary B percent selector for VAC-kmeans modes."
        ),
        "path_boundary_vac_kmeans_max_fit_samples": "Max history frames for VAC-kmeans boundary fitting.",
        "path_boundary_vac_kmeans_seed": "RNG seed for VAC-kmeans boundary fitting.",
        "path_boundary_plugin": "Optional Python hook for custom A/B boundaries: '/path/file.py:callable'.",
    }
