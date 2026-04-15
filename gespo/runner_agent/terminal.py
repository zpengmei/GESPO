"""Terminal output helpers — colors, formatting, spinners.

No external dependencies; uses ANSI escape codes with tty detection.
"""
from __future__ import annotations

import os
import sys
import time
import threading
from contextlib import contextmanager
from typing import IO

# ── ANSI color codes ────────────────────────────────────────────────────

_NO_COLOR = bool(os.environ.get("NO_COLOR"))


def _supports_color(stream: IO = sys.stderr) -> bool:
    if _NO_COLOR:
        return False
    if not hasattr(stream, "isatty"):
        return False
    return stream.isatty()


_COLOR = _supports_color()


def _ansi(code: str) -> str:
    return f"\033[{code}m" if _COLOR else ""


# Style constants.
RESET = _ansi("0")
BOLD = _ansi("1")
DIM = _ansi("2")
ITALIC = _ansi("3")

# Foreground colors.
RED = _ansi("31")
GREEN = _ansi("32")
YELLOW = _ansi("33")
BLUE = _ansi("34")
MAGENTA = _ansi("35")
CYAN = _ansi("36")
WHITE = _ansi("37")
GRAY = _ansi("90")


# ── Semantic styling ────────────────────────────────────────────────────

def styled(text: str, *codes: str) -> str:
    """Wrap *text* with ANSI codes and a reset suffix."""
    if not _COLOR:
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{RESET}"


def tool_name(name: str) -> str:
    return styled(name, BOLD, CYAN)


def tool_result_ok(text: str) -> str:
    return styled(text, GREEN)


def tool_result_err(text: str) -> str:
    return styled(text, RED)


def system_msg(text: str) -> str:
    return styled(text, DIM, GRAY)


def agent_label() -> str:
    return styled("agent", BOLD, MAGENTA)


def step_label(step: int, total: int) -> str:
    return styled(f"[{step}/{total}]", DIM)


def elapsed_str(seconds: float) -> str:
    if seconds < 1:
        return styled(f"{seconds*1000:.0f}ms", DIM)
    return styled(f"{seconds:.1f}s", DIM)


# ── Output helpers ──────────────────────────────────────────────────────

def info(msg: str, file: IO = sys.stderr) -> None:
    """Print an info-level message."""
    print(f"  {system_msg(msg)}", file=file, flush=True)


def warn(msg: str, file: IO = sys.stderr) -> None:
    """Print a warning."""
    print(f"  {styled('⚠', YELLOW)} {msg}", file=file, flush=True)


def error(msg: str, file: IO = sys.stderr) -> None:
    """Print an error."""
    print(f"  {styled('✗', RED)} {styled(msg, RED)}", file=file, flush=True)


def success(msg: str, file: IO = sys.stderr) -> None:
    """Print a success message."""
    print(f"  {styled('✓', GREEN)} {msg}", file=file, flush=True)


def tool_header(name: str, step: int, total: int, file: IO = sys.stderr) -> None:
    """Print a tool invocation header line."""
    print(
        f"\n  {step_label(step, total)} {styled('▶', CYAN)} {tool_name(name)}",
        file=file, flush=True,
    )


def tool_params_compact(params: dict, max_len: int = 120, file: IO = sys.stderr) -> None:
    """Print tool params on one line (compact)."""
    if not params:
        return
    parts = []
    for k, v in params.items():
        s = repr(v) if isinstance(v, str) else str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{styled(k, DIM)}={s}")
    line = "    " + ", ".join(parts)
    if len(line) > max_len:
        line = line[:max_len - 3] + "..."
    print(line, file=file, flush=True)


def tool_output(result, max_lines: int = 15, file: IO = sys.stderr) -> None:
    """Print a ToolResult to the user (truncated)."""
    if result.success:
        text = result.output
        tag = styled("✓", GREEN)
    else:
        text = result.error
        tag = styled("✗", RED)

    lines = text.splitlines() if text else []
    if not lines:
        print(f"    {tag} {system_msg('(no output)')}", file=file, flush=True)
        return

    shown = lines[:max_lines]
    prefix = "    │ "
    color = GREEN if result.success else RED
    for line in shown:
        print(f"{prefix}{styled(line, DIM)}", file=file, flush=True)
    if len(lines) > max_lines:
        remaining = len(lines) - max_lines
        print(
            f"{prefix}{system_msg(f'... ({remaining} more lines)')}",
            file=file, flush=True,
        )


# ── Spinner ─────────────────────────────────────────────────────────────

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def spinner(label: str = "working", file: IO = sys.stderr):
    """Show a spinner while a block executes.

    Usage::

        with spinner("running shell"):
            result = skill.execute(params)
    """
    if not _COLOR or not hasattr(file, "isatty") or not file.isatty():
        yield
        return

    stop = threading.Event()

    def _spin():
        i = 0
        while not stop.is_set():
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            print(f"\r    {styled(frame, CYAN)} {system_msg(label)}", end="", file=file, flush=True)
            i += 1
            stop.wait(0.08)
        # Clear spinner line
        print(f"\r{' ' * (len(label) + 10)}\r", end="", file=file, flush=True)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1)


# ── Prompt ──────────────────────────────────────────────────────────────

def prompt_str() -> str:
    """The REPL prompt string."""
    if _COLOR:
        return f"{BOLD}{GREEN}❯{RESET} "
    return ">> "


def approval_prompt(name: str, params: dict, file: IO = sys.stderr) -> str:
    """Format a compact approval prompt. Returns the formatted string."""
    header = f"  {styled('?', YELLOW)} {tool_name(name)}"
    parts = []
    for k, v in params.items():
        s = repr(v) if isinstance(v, str) else str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{styled(k, DIM)}={s}")
    detail = ", ".join(parts)
    if len(detail) > 140:
        detail = detail[:137] + "..."
    return f"{header}  {detail}"
