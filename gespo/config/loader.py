"""YAML config loader and saver for GESPO TrainingConfig.

Supports loading a YAML file with optional CLI overrides,
and saving the effective config for reproducibility.
"""
from __future__ import annotations

from dataclasses import fields, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# YAML section → flat field name prefix mapping.
# Nested YAML keys are flattened by joining with underscore where needed.
_YAML_SECTIONS = {
    "endpoints": {"pdb_a": "pdb", "pdb_b": "pdb_b"},
    "training": {
        "mode": "_mode",  # special: maps to generative_loop / path_sampling flags
        "episodes": "episodes",
        "grp_size": "grp_size",
        "platform": "platform",
        "gen_device": "gen_device",
    },
    "md": {
        "n_md_steps": "n_md_steps",
        "record_stride": "record_stride",
        "temperature": "temperature",
        "forcefield_protein": "forcefield_protein",
        "forcefield_water": "forcefield_water",
        "forcefield_implicit": "forcefield_implicit",
        "solvent": "_solvent_type",  # special: maps to implicit_solvent/explicit_solvent flags
        "ionic_strength_m": "ionic_strength_m",
        "solvent_padding_nm": "solvent_padding_nm",
        "friction": "friction",
        "timestep_fs": "timestep_fs",
        "minimize_iters": "minimize_iters",
        "equilibration_steps": "equilibration_steps",
    },
    "policy": {
        "torsion_ar_mode": "torsion_ar_mode",
        "torsion_ar_checkpoint": "torsion_ar_checkpoint",
        "lr": "lr",
    },
    "io": {
        "snapshot_dir": "snapshot_dir",
        "checkpoint_dir": "checkpoint_dir",
        "checkpoint_interval": "checkpoint_interval",
        "resume": "resume",
        "history_dcd_dir": "history_dcd_dir",
    },
}


def _flatten_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested YAML dict to flat config field names.

    Unknown keys are passed through as-is (allows setting any flat field name
    directly in the YAML without needing the nested structure).
    """
    flat: Dict[str, Any] = {}

    for section, mapping in _YAML_SECTIONS.items():
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for yaml_key, flat_key in mapping.items():
            if yaml_key in section_data:
                flat[flat_key] = section_data[yaml_key]

    # Pass-through: any key in top-level sections like "geom2vec", "rewards",
    # "path_sampling", "committor", "steering" gets prefixed with the section name
    # if the key doesn't already have it.
    prefix_sections = {
        "geom2vec": "geom2vec_",
        "rewards": "",  # reward fields have varied prefixes
        "path_sampling": "path_boundary_",
        "committor": "committor_",
        "steering": "steer_",
    }
    for section, prefix in prefix_sections.items():
        section_data = data.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for key, value in section_data.items():
            flat_key = key if key.startswith(prefix) else (prefix + key)
            flat[flat_key] = value

    # Any remaining top-level keys not in a known section are passed through directly.
    known_sections = set(_YAML_SECTIONS.keys()) | set(prefix_sections.keys())
    for key, value in data.items():
        if key not in known_sections and not isinstance(value, dict):
            flat[key] = value

    return flat


def _resolve_mode(flat: Dict[str, Any]) -> None:
    """Resolve the 'mode' shortcut to concrete flags."""
    mode = str(flat.pop("_mode", "")).strip().lower()
    if not mode:
        return
    if mode == "generative":
        flat.setdefault("generative_loop", True)
    elif mode in ("path-sampling", "path_sampling", "path"):
        flat.setdefault("generative_loop", True)
        flat.setdefault("path_sampling", True)
    elif mode == "standard":
        flat.setdefault("generative_loop", False)
    elif mode == "pretrain":
        # Pretrain mode is handled by the pretrain subcommand, not a flag.
        pass


def _resolve_solvent(flat: Dict[str, Any]) -> None:
    """Resolve the 'solvent' shortcut to concrete flags."""
    solvent = str(flat.pop("_solvent_type", "")).strip().lower()
    if not solvent:
        return
    if solvent == "explicit":
        flat["explicit_solvent"] = True
        flat["implicit_solvent"] = False
    elif solvent in ("implicit", "gbsa", "gbn2"):
        flat["implicit_solvent"] = True
        flat["explicit_solvent"] = False


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None):
    """Load a TrainingConfig from a YAML file with optional overrides.

    Parameters
    ----------
    path : str
        Path to the YAML config file.
    overrides : dict, optional
        CLI or programmatic overrides (flat field names).

    Returns
    -------
    TrainingConfig
        The resolved configuration.
    """
    from .training import TrainingConfig

    if yaml is None:
        raise ImportError("PyYAML is required for YAML config support. Install with: pip install pyyaml")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    flat = _flatten_yaml(data)
    _resolve_mode(flat)
    _resolve_solvent(flat)

    if overrides:
        # Only merge non-None overrides (argparse sets unspecified args to None)
        for k, v in overrides.items():
            if v is not None:
                flat[k] = v

    return TrainingConfig.from_dict(flat)


def save_config(config, path: str) -> None:
    """Save a TrainingConfig to YAML.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to save.
    path : str
        Output YAML file path.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for YAML config support. Install with: pip install pyyaml")
    d = config.to_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def generate_template(mode: str = "generative") -> str:
    """Generate a commented YAML config template.

    Parameters
    ----------
    mode : str
        Training mode: "generative", "path-sampling", or "standard".

    Returns
    -------
    str
        YAML template string.
    """
    lines = [
        "# GESPO training configuration",
        f"# Mode: {mode}",
        "",
        "endpoints:",
        "  pdb_a: endpoint_A.pdb",
    ]
    if mode in ("path-sampling", "path"):
        lines.append("  pdb_b: endpoint_B.pdb")
    else:
        lines.append("  # pdb_b: endpoint_B.pdb  # uncomment for path-sampling")

    lines += [
        "",
        "training:",
        f"  mode: {mode}",
        "  episodes: 20",
        "  grp_size: 8",
        "  platform: CUDA",
        '  gen_device: "cuda:0"',
        "",
        "md:",
        "  n_md_steps: 10000",
        "  record_stride: 10",
        "  temperature: 300.0",
        "  solvent: explicit",
        "  ionic_strength_m: 0.15",
        "",
        "geom2vec:",
        "  checkpoint: path/to/geom2vec.pt",
        "  lagtime: 150",
        "  dim: 10",
        "  vamp_epochs: 10",
        "",
        "policy:",
        "  torsion_ar_mode: backbone",
        "  # torsion_ar_checkpoint: ''  # resume from checkpoint",
        "  lr: 1.0e-4",
        "",
        "rewards:",
        "  loo_vamp_reward_weight: 1.0",
        "  vamp_bin_entropy_coef: -1.0  # < 0 ties to ent_coef",
        "  msm_ig_reward_coef: 0.0",
        "",
        "io:",
        "  snapshot_dir: runs/my_experiment",
        "  checkpoint_interval: 5",
        "  resume: false",
    ]

    if mode in ("path-sampling", "path"):
        lines += [
            "",
            "committor:",
            "  progress_reward_weight: 1.0",
            "  tube_diversity_reward_weight: 0.0",
            "  connectivity_reward_weight: 0.0",
            "  dga_n_clusters: 200",
        ]

    return "\n".join(lines) + "\n"
