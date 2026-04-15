"""Targeted MD steering configuration for replay resets."""
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SteeringConfig:
    """Configuration for targeted MD steering during replay resets.

    Controls the multi-stage steering protocol: global (RMSD/E2E) first,
    then torsion refinement, then release.
    """

    steer_replay_starts: bool = True
    steer_replay_verbose: bool = False
    steer_global_steps: int = 2000
    steer_global_updates: int = 25
    steer_torsion_steps: int = 1000
    steer_torsion_updates: int = 25
    steer_release_steps: int = 200
    steer_refine_minimize_iters: int = 0

    # Force constants.
    steer_k_rmsd: float = 5000.0
    steer_k_e2e: float = 5000.0
    steer_k_rg: float = 0.0
    steer_k_torsion: float = 200.0
    steer_k_dist: float = 0.0  # defaults to steer_k_e2e at runtime if 0
    steer_k_theta: float = 0.0  # defaults to steer_k_torsion at runtime if 0

    # Sidechain chi torsion weight (0 disables chi steering).
    steer_chi_weight: float = 0.1

    # Global-restraint lambda during torsion stage.
    steer_global_lambda_during_torsion: float = 0.2

    # Cap steering potential energy (kJ/mol); 0 disables.
    steer_max_energy_kj_mol: float = 0.0

    # Allow extra global steering targets for large explicit-solvent systems.
    steer_allow_large_explicit_extra_globals: bool = False

    CLI_META: ClassVar[dict] = {
        "steer_replay_starts": {
            "help": "Enable targeted MD steering for replay resets.",
            "no_help": "Disable targeted MD steering for replay resets.",
        },
        "steer_replay_verbose": {
            "help": "Print per-reset steering diagnostics.",
            "flag_only": True,
        },
        "steer_global_steps": "Replay steering: steps for global (RMSD/E2E/Rg) stage.",
        "steer_global_updates": "Replay steering: schedule updates for global stage.",
        "steer_torsion_steps": "Replay steering: steps for torsion stage.",
        "steer_torsion_updates": "Replay steering: schedule updates for torsion stage.",
        "steer_release_steps": "Replay steering: unbiased relaxation steps after steering.",
        "steer_refine_minimize_iters": "Replay steering: optional minimization under active steering (0=disabled).",
        "steer_k_rmsd": "Replay steering: force constant for CA RMSD restraint (kJ/mol/nm^2).",
        "steer_k_e2e": "Replay steering: force constant for E2E distance restraint (kJ/mol/nm^2).",
        "steer_k_rg": "Replay steering: force constant for CA Rg restraint (kJ/mol/nm^2, 0=disabled).",
        "steer_k_torsion": "Replay steering: force constant for torsion restraints (kJ/mol/rad^2).",
        "steer_k_dist": {
            "help": "Replay steering: force constant for extra distance global tokens (kJ/mol/nm^2).",
            "default_override": 5000.0,
        },
        "steer_k_theta": {
            "help": "Replay steering: force constant for extra angle global tokens (kJ/mol/rad^2).",
            "default_override": 200.0,
        },
        "steer_chi_weight": (
            "Replay steering: per-torsion weight for sidechain \u03c7 torsions (0=disabled)."
        ),
        "steer_global_lambda_during_torsion": (
            "Replay steering: global lambda during torsion stage."
        ),
        "steer_max_energy_kj_mol": (
            "Replay steering: cap for steering potential energy (kJ/mol, 0=disabled)."
        ),
        "steer_allow_large_explicit_extra_globals": {
            "help": "Allow extra distance/angle steering in large explicit-solvent systems.",
            "flag_only": True,
        },
    }
