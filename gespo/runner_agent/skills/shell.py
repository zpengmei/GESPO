"""Shell command execution skill (allowlisted, read-only)."""
from __future__ import annotations

import shlex
import subprocess
from typing import Any, Dict, List

from ..runtime.sandbox import Sandbox
from .base import Skill, ToolResult


class ShellSkill(Skill):
    """Execute allowlisted shell commands within a sandbox."""

    name = "shell"
    description = "Run allowlisted read-only shell commands (ls, cat, head, rg, etc.)"
    schema = {"cmd": "string (required)", "cwd": "string (optional, defaults to workdir)"}

    def __init__(self, sandbox: Sandbox, allowlist: List[str], timeout: float = 20.0):
        self.sandbox = sandbox
        self.allowlist = set(allowlist)
        self.timeout = timeout

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        cmd = str(params.get("cmd", "")).strip()
        if not cmd:
            return ToolResult.error(self.name, "No command provided")

        # Check allowlist
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return ToolResult.error(self.name, f"Invalid command syntax: {e}")

        if not parts:
            return ToolResult.error(self.name, "Empty command")

        binary = parts[0].split("/")[-1]  # handle /usr/bin/ls -> ls
        if binary not in self.allowlist:
            return ToolResult.error(
                self.name,
                f"Command '{binary}' is not in the allowlist. "
                f"Allowed: {sorted(self.allowlist)}",
            )

        # Resolve working directory
        cwd_param = params.get("cwd")
        try:
            cwd = self.sandbox.resolve_cwd(cwd_param)
        except PermissionError as e:
            return ToolResult.error(self.name, str(e))

        # Execute
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(cwd),
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            # Truncate long output
            max_chars = 8000
            if len(output) > max_chars:
                output = output[:max_chars] + f"\n... (truncated, {len(output)} chars total)"
            return ToolResult.ok(self.name, output)
        except subprocess.TimeoutExpired:
            return ToolResult.error(self.name, f"Command timed out after {self.timeout}s")
        except Exception as e:
            return ToolResult.error(self.name, f"Execution error: {e}")
