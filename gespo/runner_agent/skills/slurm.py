"""SLURM job submission and monitoring skills."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..runtime.sandbox import Sandbox
from .base import Skill, ToolResult


@dataclass
class SlurmDefaults:
    """Default SLURM sbatch directives."""
    partition: str = "beagle3"
    account: str = "pi-dinner"
    qos: str = "beagle3"
    time: str = "24:00:00"
    nodes: int = 1
    ntasks: int = 8
    gres: str = "gpu:1"
    output: str = "slurm-%x-%j.out"
    error: str = "slurm-%x-%j.err"


class SlurmSubmitSkill(Skill):
    """Generate and optionally submit SLURM sbatch scripts."""

    name = "slurm_submit"
    description = "Generate a SLURM sbatch script and optionally submit it"
    schema = {
        "cmd": "string (required, the command to run)",
        "job_name": "string (required)",
        "submit": "bool (default false, set true to actually submit)",
        "cwd": "string (optional, working directory for the job)",
    }

    def __init__(self, sandbox: Sandbox, defaults: SlurmDefaults, allow_submit: bool = False, timeout: float = 20.0):
        self.sandbox = sandbox
        self.defaults = defaults
        self.allow_submit = allow_submit
        self.timeout = timeout

    def _build_script(self, cmd: str, job_name: str, cwd: Optional[str] = None) -> str:
        d = self.defaults
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={d.partition}",
            f"#SBATCH --account={d.account}",
            f"#SBATCH --qos={d.qos}",
            f"#SBATCH --time={d.time}",
            f"#SBATCH --nodes={d.nodes}",
            f"#SBATCH --ntasks={d.ntasks}",
            f"#SBATCH --gres={d.gres}",
            f"#SBATCH --output={d.output}",
            f"#SBATCH --error={d.error}",
            "",
            f"cd {cwd or str(self.sandbox.workdir)}",
            "",
            cmd,
        ]
        return "\n".join(lines) + "\n"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        cmd = str(params.get("cmd", "")).strip()
        job_name = str(params.get("job_name", "gespo_job")).strip()
        submit = bool(params.get("submit", False))
        cwd = params.get("cwd")

        if not cmd:
            return ToolResult.error(self.name, "No command provided")

        try:
            resolved_cwd = self.sandbox.resolve_cwd(cwd)
        except PermissionError as e:
            return ToolResult.error(self.name, str(e))

        script = self._build_script(cmd, job_name, str(resolved_cwd))

        # Write script
        script_path = resolved_cwd / f"{job_name}.sbatch"
        try:
            self.sandbox.check_write(script_path)
            script_path.write_text(script)
        except (PermissionError, Exception) as e:
            return ToolResult.error(self.name, f"Failed to write script: {e}")

        output = f"Script written to: {script_path}\n\n{script}"

        if submit:
            if not self.allow_submit:
                output += "\n\n[Submission blocked: allow_slurm_submit=false in config]"
                return ToolResult.ok(self.name, output)
            try:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True, text=True,
                    timeout=self.timeout, cwd=str(resolved_cwd),
                )
                if result.returncode == 0:
                    output += f"\n\nSubmitted: {result.stdout.strip()}"
                else:
                    output += f"\n\nSubmission failed: {result.stderr.strip()}"
                return ToolResult.ok(self.name, output)
            except Exception as e:
                return ToolResult.error(self.name, f"Submission failed: {e}")

        return ToolResult.ok(self.name, output)


class SlurmWatchSkill(Skill):
    """Poll a SLURM job's status and tail its output logs."""

    name = "slurm_watch"
    description = "Check a SLURM job's status and show recent log output"
    schema = {
        "job_id": "string (required)",
        "max_wait_s": "int (optional, default 60, max time to poll)",
        "tail_lines": "int (optional, default 30)",
    }

    def __init__(self, sandbox: Sandbox, timeout: float = 10.0):
        self.sandbox = sandbox
        self.timeout = timeout

    def _get_job_state(self, job_id: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True, text=True, timeout=self.timeout,
            )
            state = result.stdout.strip()
            return state if state else None
        except Exception:
            return None

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        job_id = str(params.get("job_id", "")).strip()
        if not job_id:
            return ToolResult.error(self.name, "No job_id provided")

        max_wait = int(params.get("max_wait_s", 60) or 60)
        tail_lines = int(params.get("tail_lines", 30) or 30)

        start = time.time()
        last_state = None

        while time.time() - start < max_wait:
            state = self._get_job_state(job_id)
            if state is None:
                # Job no longer in queue — check sacct
                try:
                    result = subprocess.run(
                        ["sacct", "-j", job_id, "--format=State", "-n", "-P"],
                        capture_output=True, text=True, timeout=self.timeout,
                    )
                    sacct_state = result.stdout.strip().split("\n")[0] if result.stdout.strip() else "UNKNOWN"
                    output = f"Job {job_id}: {sacct_state} (no longer in queue)"
                    return ToolResult.ok(self.name, output)
                except Exception:
                    return ToolResult.ok(self.name, f"Job {job_id}: completed or not found")

            last_state = state
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                break
            time.sleep(min(10, max(1, max_wait // 10)))

        output = f"Job {job_id}: {last_state or 'UNKNOWN'}"
        return ToolResult.ok(self.name, output)
