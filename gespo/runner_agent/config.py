"""Top-level agent configuration with YAML loading."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .skills.slurm import SlurmDefaults


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = ""
    api_key: str = ""
    api_key_file: str = ""
    timeout_s: float = 120.0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


@dataclass
class ToolsConfig:
    enabled: List[str] = field(default_factory=lambda: [
        "shell", "read_file", "write_file",
        "traj_analyze", "compute_dihedrals",
        "plot_ramachandran", "plot_fes",
        "vamp_project", "committor_analyze",
    ])
    approval: str = "prompt"
    shell_allowlist: List[str] = field(
        default_factory=lambda: ["ls", "cat", "head", "tail", "wc", "rg", "sed", "pwd", "squeue", "sacct"]
    )
    timeout_s: float = 20.0
    allow_slurm_submit: bool = False
    slurm: SlurmDefaults = field(default_factory=SlurmDefaults)
    python_bin: str = ""
    script_root: str = ""


@dataclass
class SandboxConfig:
    workdir: str = "."
    write_root: Optional[str] = None


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    docs_root: str = ""
    max_context_chars: int = 60000
    max_tool_steps: int = 25


def load_agent_config(
    path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AgentConfig:
    """Load agent config from YAML file with optional overrides.

    If no path is provided, returns default config with overrides applied.
    """
    config = AgentConfig()

    if path:
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            # Apply nested YAML sections to config
            if "llm" in data and isinstance(data["llm"], dict):
                for k, v in data["llm"].items():
                    if hasattr(config.llm, k):
                        setattr(config.llm, k, v)
            if "tools" in data and isinstance(data["tools"], dict):
                for k, v in data["tools"].items():
                    if k == "slurm" and isinstance(v, dict):
                        for sk, sv in v.items():
                            if hasattr(config.tools.slurm, sk):
                                setattr(config.tools.slurm, sk, sv)
                    elif hasattr(config.tools, k):
                        setattr(config.tools, k, v)
            if "sandbox" in data and isinstance(data["sandbox"], dict):
                for k, v in data["sandbox"].items():
                    if hasattr(config.sandbox, k):
                        setattr(config.sandbox, k, v)
            for k in ("docs_root", "max_context_chars", "max_tool_steps"):
                if k in data:
                    setattr(config, k, data[k])
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. Install: pip install pyyaml")

    # Apply CLI overrides
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if k == "provider":
                config.llm.provider = v
            elif k == "model":
                config.llm.model = v
            elif k == "stream":
                config.llm.stream = bool(v)
            elif k == "workdir":
                config.sandbox.workdir = str(v)
            elif k == "approve":
                config.tools.approval = str(v)
            elif k == "docs_root":
                config.docs_root = str(v)

    return config
