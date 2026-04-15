"""Molecular dynamics simulation configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional


@dataclass
class MDConfig:
    """Configuration for OpenMM MD simulation parameters."""

    forcefield_protein: str = "amber14-all.xml"
    forcefield_water: str = "amber14/tip3pfb.xml"
    forcefield_implicit: str = "implicit/gbn2.xml"
    temperature: float = 300.0
    friction: float = 1.0
    timestep_fs: float = 2.0
    minimize_iters: int = 200
    minimize_tolerance_kj_mol_nm: Optional[float] = None
    heat_stages: int = 4
    heat_steps: int = 100
    equilibration_steps: int = 5000

    # Solvent mode.
    implicit_solvent: bool = False
    explicit_solvent: bool = True
    explicit_solvent_replay_full_coords: bool = True

    # Explicit-solvent box + ions.
    solvent_padding_nm: float = 1.0
    box_length_nm: Optional[float] = None
    ionic_strength_m: float = 0.1
    solvent_positive_ion: str = "Na+"
    solvent_negative_ion: str = "Cl-"
    solvent_neutralize: bool = True

    # Nonbonded settings.
    nonbonded_cutoff_nm: Optional[float] = None
    use_switching_function: bool = False
    switch_distance_nm: Optional[float] = None
    ewald_error_tolerance: Optional[float] = None
    pme_grid_spacing_a: Optional[float] = None

    # Barostat (NPT); disabled by default (NVT).
    use_barostat: bool = False
    pressure_bar: float = 1.0
    barostat_frequency: int = 25

    # Dynamics.
    n_md_steps: int = 10000
    energy_threshold: float = 1e10
    max_resample_attempts: int = 5
    torsion_decimals: int = 3
    device: int = 0
    record_stride: int = 10
    sigma_min: float = 0.1
    use_replay_resets: bool = True

    CLI_META: ClassVar[dict] = {
        "forcefield_protein": "Protein force field XML file.",
        "forcefield_water": "Water model XML file for explicit solvent.",
        "forcefield_implicit": "Implicit solvent model XML file.",
        "temperature": "Simulation temperature (Kelvin).",
        "friction": "Langevin friction (1/ps).",
        "timestep_fs": "Integration timestep (fs).",
        "minimize_iters": "Energy minimization iterations.",
        "minimize_tolerance_kj_mol_nm": "Energy minimization tolerance (kJ/mol/nm). Unset=OpenMM default.",
        "heat_stages": "Number of heating stages.",
        "heat_steps": "Steps per heating stage.",
        "equilibration_steps": "Solvent equilibration steps before recording (explicit solvent).",
        "implicit_solvent": {"help": "Use implicit solvent.", "flag_only": True},
        "explicit_solvent": {
            "help": "Enable explicit solvent (default unless --implicit-solvent).",
            "flag_only": True,
        },
        "explicit_solvent_replay_full_coords": {
            "help": (
                "Restore full-system (protein+solvent) coordinates from saved DCD "
                "frames for replay resets."
            ),
            "no_help": "Rebuild solvent box each reset.",
        },
        "solvent_padding_nm": {
            "help": "Explicit solvent padding (nm). Ignored when --box-length-nm is set.",
            "mutex": "box_spec",
        },
        "box_length_nm": {
            "help": "Explicit solvent cubic box edge length (nm). Overrides --solvent-padding-nm.",
            "mutex": "box_spec",
        },
        "ionic_strength_m": "Explicit solvent ionic strength (M).",
        "solvent_positive_ion": "Positive ion type for ion placement (e.g. Na+, K+).",
        "solvent_negative_ion": "Negative ion type for ion placement (e.g. Cl-).",
        "solvent_neutralize": {
            "help": "Neutralize net charge during ion placement (default).",
            "no_help": "Do not auto-neutralize.",
        },
        "nonbonded_cutoff_nm": "Nonbonded cutoff (nm) for explicit solvent. Unset=OpenMM default.",
        "use_switching_function": {
            "help": "Enable Lennard-Jones switching function.",
            "no_help": "Disable LJ switching (default).",
        },
        "switch_distance_nm": "Switching distance (nm) when LJ switching is enabled.",
        "ewald_error_tolerance": "PME Ewald error tolerance (unset=OpenMM default).",
        "pme_grid_spacing_a": {
            "help": "Approximate PME grid spacing (Angstrom).",
            "aliases": ["--pme-grid-spacing-angstrom"],
        },
        "use_barostat": {
            "help": "Enable MonteCarloBarostat (NPT).",
            "flag": "--barostat",
            "no_flag": "--no-barostat",
            "no_help": "Disable barostat (NVT, default).",
        },
        "pressure_bar": "Target pressure for barostat (bar).",
        "barostat_frequency": "Barostat trial frequency in integration steps.",
        "n_md_steps": "MD steps per trajectory.",
        "energy_threshold": "Energy threshold for resampling.",
        "max_resample_attempts": "Max attempts to resample structures.",
        "torsion_decimals": "Decimal places for torsion rounding.",
        "device": "OpenMM device index.",
        "record_stride": "MD steps between recorded frames.",
        "sigma_min": "Minimum sigma for policy.",
        "use_replay_resets": {
            "help": "Use history/anchoring-based replay resets (default).",
            "flag": "--replay-resets",
            "no_flag": "--no-replay-resets",
            "no_help": "Disable replay resets; always use generator.",
        },
    }
