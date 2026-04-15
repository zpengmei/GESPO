"""Generic Python script execution skill."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..runtime.sandbox import Sandbox
from .base import Skill, ToolResult


class PythonScriptSkill(Skill):
    """Execute a specific Python script with arguments."""

    def __init__(
        self,
        name: str,
        description: str,
        script_path: str,
        sandbox: Sandbox,
        python_bin: str = "",
        timeout: float = 60.0,
    ):
        self.name = name
        self.description = description
        self.schema = {"argv": "list[string] (command-line arguments)", "cwd": "string (optional)"}
        self.script_path = Path(script_path)
        self.sandbox = sandbox
        self.python_bin = python_bin or sys.executable
        self.timeout = timeout

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        argv: List[str] = list(params.get("argv", []))
        cwd_param = params.get("cwd")

        if not self.script_path.exists():
            return ToolResult.error(self.name, f"Script not found: {self.script_path}")

        try:
            cwd = self.sandbox.resolve_cwd(cwd_param)
        except PermissionError as e:
            return ToolResult.error(self.name, str(e))

        cmd = [self.python_bin, str(self.script_path)] + argv
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, cwd=str(cwd),
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            max_chars = 8000
            if len(output) > max_chars:
                output = output[:max_chars] + f"\n... (truncated)"
            return ToolResult.ok(self.name, output)
        except subprocess.TimeoutExpired:
            return ToolResult.error(self.name, f"Script timed out after {self.timeout}s")
        except Exception as e:
            return ToolResult.error(self.name, f"Execution error: {e}")
