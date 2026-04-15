"""File read/write skills."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..runtime.sandbox import Sandbox
from .base import Skill, ToolResult


class FileReadSkill(Skill):
    """Read file contents within the sandbox."""

    name = "read_file"
    description = "Read a file's contents (text files)"
    schema = {"path": "string (required)", "max_lines": "int (optional, default 200)"}

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        path_str = str(params.get("path", "")).strip()
        if not path_str:
            return ToolResult.error(self.name, "No path provided")

        max_lines = int(params.get("max_lines", 200) or 200)
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = self.sandbox.workdir / path

        try:
            self.sandbox.check_read(path)
        except PermissionError as e:
            return ToolResult.error(self.name, str(e))

        if not path.exists():
            return ToolResult.error(self.name, f"File not found: {path}")
        if not path.is_file():
            return ToolResult.error(self.name, f"Not a file: {path}")

        try:
            lines = path.read_text(errors="replace").splitlines()
            if len(lines) > max_lines:
                output = "\n".join(lines[:max_lines])
                output += f"\n... ({len(lines)} total lines, showing first {max_lines})"
            else:
                output = "\n".join(lines)
            return ToolResult.ok(self.name, output)
        except Exception as e:
            return ToolResult.error(self.name, f"Read error: {e}")


class FileWriteSkill(Skill):
    """Write content to a file within the sandbox."""

    name = "write_file"
    description = "Write content to a file (creates parent directories)"
    schema = {"path": "string (required)", "content": "string (required)"}

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        path_str = str(params.get("path", "")).strip()
        content = str(params.get("content", ""))
        if not path_str:
            return ToolResult.error(self.name, "No path provided")

        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = self.sandbox.workdir / path

        try:
            self.sandbox.check_write(path)
        except PermissionError as e:
            return ToolResult.error(self.name, str(e))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return ToolResult.ok(self.name, f"Written {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult.error(self.name, f"Write error: {e}")
