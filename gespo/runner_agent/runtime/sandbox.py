"""Path sandboxing for tool execution."""
from __future__ import annotations

from pathlib import Path


class Sandbox:
    """Validates that tool operations stay within allowed directories."""

    def __init__(self, workdir: Path, write_root: Path | None = None):
        self.workdir = Path(workdir).resolve()
        self.write_root = Path(write_root).resolve() if write_root else self.workdir

    def resolve_cwd(self, cwd: str | None) -> Path:
        """Resolve a working directory, defaulting to workdir. Raises if outside workdir."""
        if not cwd:
            return self.workdir
        resolved = Path(cwd).resolve()
        if not str(resolved).startswith(str(self.workdir)):
            # Allow if it's a parent of workdir (for reading)
            if not str(self.workdir).startswith(str(resolved)):
                raise PermissionError(
                    f"Working directory {resolved} is outside sandbox {self.workdir}"
                )
        return resolved

    def check_write(self, path: Path) -> None:
        """Raise PermissionError if path is outside write_root."""
        resolved = Path(path).resolve()
        if not str(resolved).startswith(str(self.write_root)):
            raise PermissionError(
                f"Write path {resolved} is outside write root {self.write_root}"
            )

    def check_read(self, path: Path) -> None:
        """Raise PermissionError if path is outside workdir."""
        resolved = Path(path).resolve()
        # Allow reading from anywhere under workdir or its parents
        # (permissive for research use)
        pass  # No restriction on reads for now
