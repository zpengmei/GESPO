"""Skill base class and result types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolRequest:
    """A parsed tool invocation request from the LLM."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    name: str
    success: bool
    output: str = ""
    error: str = ""

    @classmethod
    def ok(cls, name: str, output: str) -> "ToolResult":
        return cls(name=name, success=True, output=output)

    @classmethod
    def error(cls, name: str, error: str) -> "ToolResult":
        return cls(name=name, success=False, error=error)

    def to_message(self) -> str:
        """Format for injection into conversation as a system message."""
        if self.success:
            return f"TOOL_RESULT({self.name}):\n{self.output}"
        return f"TOOL_ERROR({self.name}):\n{self.error}"


class Skill(ABC):
    """Abstract base for all agent skills (tools).

    Each skill self-describes with a name, description, and JSON schema
    so the registry can auto-generate prompt sections.
    """

    name: str = ""
    description: str = ""
    schema: Dict[str, str] = {}  # param_name -> type description for the LLM

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the skill with the given parameters."""
        ...
