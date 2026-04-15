"""Runtime configuration (LLM, limits)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Configuration for the agent runtime (independent of specific skills)."""
    max_tool_steps: int = 25
    stream: bool = False
