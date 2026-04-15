"""Centralized tool approval gate with session memory."""
from __future__ import annotations

import sys
from typing import Set


class ApprovalGate:
    """Checks whether a tool execution should proceed.

    In ``"prompt"`` mode the user is asked interactively.  They can reply:

    - **y / yes** — approve this one invocation.
    - **a / always** — approve this tool for the rest of the session.
    - **n / no / Enter** — reject this invocation.

    Approved-always tools are remembered for the session so the user is not
    asked again for the same tool name.
    """

    def __init__(self, mode: str = "prompt"):
        if mode not in ("always", "never", "prompt"):
            raise ValueError(f"Invalid approval mode: {mode!r}")
        self.mode = mode
        self._always_approved: Set[str] = set()

    def check(self, request) -> bool:
        """Return True if the tool execution is approved."""
        if self.mode == "always":
            return True
        if self.mode == "never":
            return False

        # Session memory: already approved for this tool?
        if request.name in self._always_approved:
            return True

        # Interactive prompt
        from .. import terminal as t
        line = t.approval_prompt(request.name, request.params)
        print(line, file=sys.stderr, flush=True)
        try:
            hint = t.styled("[y/a/N]", t.DIM)
            answer = input(f"    approve? {hint} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return False

        if answer in ("a", "always"):
            self._always_approved.add(request.name)
            t.info(f"auto-approving '{request.name}' for this session")
            return True
        return answer in ("y", "yes")
