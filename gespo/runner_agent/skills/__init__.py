"""Skill plugins for the GESPO runner agent."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..runtime.sandbox import Sandbox
from ..runtime.tool_registry import ToolRegistry
from .base import Skill, ToolRequest, ToolResult
from .shell import ShellSkill
from .file_ops import FileReadSkill, FileWriteSkill
from .slurm import SlurmSubmitSkill, SlurmWatchSkill, SlurmDefaults
from .python_script import PythonScriptSkill
from .traj_analysis import TrajectoryAnalysisSkill
from .dihedrals import DihedralSkill
from .plotting import RamachandranPlotSkill, FreeEnergySurfaceSkill
from .vamp_projection import VAMPProjectionSkill
from .committor_analysis import CommittorAnalysisSkill


def register_default_skills(
    registry: ToolRegistry,
    *,
    enabled: list[str],
    sandbox: Sandbox,
    shell_allowlist: list[str],
    timeout_s: float = 20.0,
    slurm_defaults: SlurmDefaults | None = None,
    allow_slurm_submit: bool = False,
    python_bin: str = "",
    script_root: str = "",
) -> None:
    """Register enabled skills into the registry."""

    builders = {
        # ── Core tools ──────────────────────────────────────────────
        "shell": lambda: ShellSkill(sandbox, shell_allowlist, timeout_s),
        "read_file": lambda: FileReadSkill(sandbox),
        "write_file": lambda: FileWriteSkill(sandbox),
        "slurm_submit": lambda: SlurmSubmitSkill(
            sandbox, slurm_defaults or SlurmDefaults(), allow_slurm_submit, timeout_s,
        ),
        "slurm_watch": lambda: SlurmWatchSkill(sandbox, timeout_s),

        # ── Analysis / science tools ────────────────────────────────
        "traj_analyze": lambda: TrajectoryAnalysisSkill(),
        "compute_dihedrals": lambda: DihedralSkill(),
        "plot_ramachandran": lambda: RamachandranPlotSkill(),
        "plot_fes": lambda: FreeEnergySurfaceSkill(),
        "vamp_project": lambda: VAMPProjectionSkill(),
        "committor_analyze": lambda: CommittorAnalysisSkill(),
    }

    # Python script tools (legacy; point to external scripts)
    script_defs = {
        "pdb_sanitize": ("pdb_sanitize", "Sanitize/validate PDB files", "tools/pdb_sanitize.py"),
        "traj_to_multimodel_pdb": ("traj_to_multimodel_pdb", "Convert trajectory to multi-model PDB", "tools/traj_to_multimodel_pdb.py"),
        "analyze_corner_pmf": ("analyze_corner_pmf", "Plot free energy surface from trajectory", "tools/analyze_corner_pmf.py"),
        "dga_committor": ("dga_committor", "Compute DGA committor probabilities", "tools/dga_committor_rmsd.py"),
    }
    for tool_name, (name, desc, rel_path) in script_defs.items():
        script_path = Path(script_root) / rel_path if script_root else Path(rel_path)
        builders[tool_name] = lambda n=name, d=desc, p=script_path: PythonScriptSkill(
            name=n, description=d, script_path=str(p),
            sandbox=sandbox, python_bin=python_bin or sys.executable,
            timeout=max(60.0, timeout_s),
        )

    for name in enabled:
        builder = builders.get(name)
        if builder:
            try:
                registry.register(builder())
            except Exception as e:
                print(f"[Skills] Warning: failed to register {name}: {e}", file=sys.stderr)


__all__ = [
    "Skill", "ToolRequest", "ToolResult",
    "ShellSkill", "FileReadSkill", "FileWriteSkill",
    "SlurmSubmitSkill", "SlurmWatchSkill",
    "PythonScriptSkill",
    "TrajectoryAnalysisSkill", "DihedralSkill",
    "RamachandranPlotSkill", "FreeEnergySurfaceSkill",
    "VAMPProjectionSkill", "CommittorAnalysisSkill",
    "register_default_skills",
]
