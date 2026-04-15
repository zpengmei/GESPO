"""Tool/skill registry with auto-generated prompt sections."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


class ToolRegistry:
    """Registry of available skills/tools."""

    def __init__(self):
        self._tools: Dict[str, Any] = {}  # name -> Skill instance

    def register(self, skill) -> None:
        """Register a skill. Raises if name already taken."""
        if skill.name in self._tools:
            raise ValueError(f"Skill {skill.name!r} already registered")
        self._tools[skill.name] = skill

    def get(self, name: str):
        """Look up a skill by name. Returns None if not found."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return sorted list of registered skill names."""
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def build_tool_prompt(self) -> str:
        """Auto-generate a tool description block for the system prompt.

        Each registered skill contributes its name, description, and JSON schema.
        """
        if not self._tools:
            return "No tools available."
        lines = [
            "Available tools — invoke by responding with a JSON object:",
            "",
        ]
        for i, skill in enumerate(self._tools.values(), 1):
            lines.append(f"{i}. **{skill.name}**: {skill.description}")
            schema = getattr(skill, "schema", {})
            if schema:
                lines.append(f"   Request format: {json.dumps(dict(tool=skill.name, **schema))}")
            lines.append("")
        lines.append("Rules:")
        lines.append("- Respond with ONLY the JSON object when invoking a tool (no other text).")
        lines.append("- Wait for TOOL_RESULT before proceeding.")
        lines.append("- Do not invent or simulate tool output.")
        return "\n".join(lines)
