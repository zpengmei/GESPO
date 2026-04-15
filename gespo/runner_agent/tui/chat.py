"""Chat message and tool-call widgets for the GESPO TUI."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Collapsible, Markdown, Label, LoadingIndicator
from textual.widget import Widget
from rich.text import Text
from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Group


class MessageWidget(Widget):
    """A single chat message (user, assistant, or system)."""

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
        width: 100%;
    }
    """

    def __init__(
        self,
        role: str,
        content: str,
        *,
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        role_styles = {
            "user": ("bold green", "▸ You"),
            "assistant": ("bold magenta", "◆ Agent"),
            "system": ("dim", "⚙ System"),
        }
        style, label = role_styles.get(self.role, ("dim", self.role))
        yield Static(Text(label, style=style), classes="message-role")
        yield Static(
            Markdown(self.content, id=None),
            classes="message-body",
        )


class ToolCallWidget(Widget):
    """Collapsible panel showing a tool invocation and its result."""

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        margin: 1 0 0 2;
        padding: 0;
        width: 100%;
    }
    ToolCallWidget .tool-panel {
        height: auto;
        max-height: 24;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        tool_name: str,
        params: dict,
        step: int,
        total: int,
        *,
        result_text: str = "",
        success: bool = True,
        elapsed_ms: float = 0,
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.tool_name = tool_name
        self.params = params
        self.step = step
        self.total = total
        self.result_text = result_text
        self._success = success
        self.elapsed_ms = elapsed_ms

    def compose(self) -> ComposeResult:
        # Build header
        icon = "✓" if self._success else "✗"
        color = "green" if self._success else "red"
        timing = f" ({self.elapsed_ms:.0f}ms)" if self.elapsed_ms else ""

        title = Text.assemble(
            (f"[{self.step}/{self.total}] ", "dim"),
            (f"▶ {self.tool_name}", f"bold cyan"),
            (timing, "dim"),
            (" ", ""),
            (icon, color),
        )

        # Build params line
        parts = []
        for k, v in self.params.items():
            s = repr(v) if isinstance(v, str) else str(v)
            if len(s) > 60:
                s = s[:57] + "…"
            parts.append(f"[dim]{k}[/]={s}")
        params_line = ", ".join(parts)
        if len(params_line) > 200:
            params_line = params_line[:197] + "…"

        # Build result content
        content_parts = [
            Text(params_line.replace("[dim]", "").replace("[/]", ""), style="dim"),
        ]
        if self.result_text:
            lines = self.result_text.splitlines()
            shown = lines[:20]
            result_str = "\n".join(shown)
            if len(lines) > 20:
                result_str += f"\n… ({len(lines) - 20} more lines)"
            result_style = "" if self._success else "red"
            content_parts.append(Text(result_str, style=result_style))

        yield Collapsible(
            Static(Group(*content_parts), classes="tool-panel"),
            title=title,
            collapsed=self._success and len(self.result_text.splitlines()) > 5,
        )


class ThinkingIndicator(Widget):
    """Animated indicator shown while the agent is thinking."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 3;
        margin: 1 0 0 2;
        padding: 0 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Horizontal(
            LoadingIndicator(),
            Static(
                Text(" Agent is thinking…", style="italic dim"),
            ),
        )


class WelcomeWidget(Widget):
    """Workflow-oriented welcome screen for GESPO."""

    DEFAULT_CSS = """
    WelcomeWidget {
        align: center middle;
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    WelcomeWidget .title {
        text-align: center;
        text-style: bold;
        color: $primary-lighten-1;
        margin: 0 0 1 0;
        width: 100%;
    }
    WelcomeWidget .subtitle {
        text-align: center;
        color: $text-muted;
        width: 100%;
    }
    WelcomeWidget .workflow-item {
        text-align: center;
        color: $accent;
        width: 100%;
    }
    """

    def __init__(self, tools: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tools = tools or []

    def compose(self) -> ComposeResult:
        yield Static("GESPO Agent", classes="title")
        yield Static("Enhanced sampling workflow assistant", classes="subtitle")
        yield Static("", classes="subtitle")
        yield Static("Try asking:", classes="subtitle")
        yield Static("", classes="subtitle")

        examples = [
            "\"Analyze the trajectory in runs/traj.dcd using topology chignolin.pdb\"",
            "\"Compute backbone dihedrals for my protein and plot Ramachandran\"",
            "\"Plot the free energy surface from my simulation\"",
            "\"Project my trajectory through the VAMP lobe and show state populations\"",
            "\"Compute committor probabilities between endpoints A and B\"",
            "\"List the PDB files in my working directory\"",
        ]
        for ex in examples:
            yield Static(f"  → {ex}", classes="workflow-item")

        yield Static("", classes="subtitle")
        yield Static("ctrl+q quit  ·  ctrl+l clear  ·  ctrl+t tools", classes="subtitle")
