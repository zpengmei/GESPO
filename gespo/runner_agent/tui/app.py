"""OpenCode-style Textual TUI for the GESPO runner agent."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.widgets import Static, Input, Footer, Header
from textual.worker import Worker
from rich.text import Text

from .chat import MessageWidget, ToolCallWidget, ThinkingIndicator, WelcomeWidget


class GESPOApp(App):
    """Full-screen TUI for the GESPO runner agent."""

    CSS_PATH = "styles.tcss"

    TITLE = "GESPO Agent"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear", priority=True),
        Binding("ctrl+t", "show_tools", "Tools", priority=True),
    ]

    def __init__(
        self,
        agent: Any,
        config: Any,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.agent_config = config
        self._thinking = False
        self._msg_counter = 0
        # Patch the agent to use our TUI for approval
        self.agent.approval = _TUIApprovalBridge(self)

    def compose(self) -> ComposeResult:
        # Header bar
        model = getattr(self.agent_config, "llm", None)
        model_name = f"{model.provider}/{model.model}" if model else "unknown"
        yield Header()
        yield Static(
            Text.assemble(
                ("  GESPO Agent  ", "bold magenta"),
                ("│ ", "dim"),
                (f"model: {model_name}", "dim cyan"),
                (" │ ", "dim"),
                (f"tools: {len(self.agent.registry.names())}", "dim"),
            ),
            id="header",
        )

        # Chat area
        with VerticalScroll(id="chat-scroll"):
            yield WelcomeWidget(
                tools=self.agent.registry.names(),
                id="welcome",
            )

        # Input area
        with Vertical(id="input-container"):
            yield Input(
                placeholder="Type a message… (ctrl+q to quit)",
                id="prompt-input",
            )

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()
        self.sub_title = f"{self.agent_config.llm.provider}/{self.agent_config.llm.model}"

    # ── Input handling ──────────────────────────────────────────────

    @on(Input.Submitted, "#prompt-input")
    def handle_input(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        # Remove welcome widget if present
        welcome = self.query("#welcome")
        if welcome:
            welcome.remove()

        # Add user message
        self._add_message("user", text)

        # Disable input while processing
        event.input.disabled = True

        # Run agent in background worker
        self._run_agent(text)

    @work(thread=True)
    def _run_agent(self, user_input: str) -> str:
        """Run the agent turn in a background thread."""
        self.call_from_thread(self._show_thinking)

        # Monkey-patch the agent to capture tool calls for the TUI
        original_dispatch = self.agent._dispatch
        original_parse = self.agent._parse_tool_request
        agent = self.agent
        app = self
        max_steps = agent.config.max_tool_steps

        step_counter = [0]

        def patched_run_turn(user_input: str) -> str:
            """Custom run_turn that reports tool calls to the TUI."""
            agent.conversation.add_user(user_input)

            for step in range(1, max_steps + 1):
                step_counter[0] = step

                t0 = time.monotonic()
                reply = agent.client.chat(agent.conversation.to_api())
                agent.conversation.add_assistant(reply)

                request = original_parse(reply)
                if request is None:
                    return reply

                # Show tool call in TUI
                app.call_from_thread(app._hide_thinking)
                t0_tool = time.monotonic()
                result = original_dispatch(request)
                elapsed_ms = (time.monotonic() - t0_tool) * 1000

                app.call_from_thread(
                    app._add_tool_call,
                    request.name,
                    request.params,
                    step,
                    max_steps,
                    result.output if result.success else result.error,
                    result.success,
                    elapsed_ms,
                )
                app.call_from_thread(app._show_thinking)

                agent.conversation.add_tool_result(result)

            # Budget exhausted
            agent.conversation.add_system(
                f"[Agent] Tool step budget ({max_steps}) exhausted. "
                "Please provide a final answer without using tools."
            )
            reply = agent.client.chat(agent.conversation.to_api())
            agent.conversation.add_assistant(reply)
            return reply

        try:
            reply = patched_run_turn(user_input)
        except Exception as exc:
            reply = f"**Error:** {type(exc).__name__}: {exc}"

        self.call_from_thread(self._hide_thinking)
        self.call_from_thread(self._add_message, "assistant", reply)
        self.call_from_thread(self._enable_input)
        return reply

    # ── UI update methods (called from main thread) ─────────────────

    def _add_message(self, role: str, content: str) -> None:
        self._msg_counter += 1
        widget = MessageWidget(
            role, content, id=f"msg-{self._msg_counter}"
        )
        chat = self.query_one("#chat-scroll", VerticalScroll)
        chat.mount(widget)
        chat.scroll_end(animate=False)

    def _add_tool_call(
        self,
        tool_name: str,
        params: dict,
        step: int,
        total: int,
        result_text: str,
        success: bool,
        elapsed_ms: float,
    ) -> None:
        self._msg_counter += 1
        widget = ToolCallWidget(
            tool_name=tool_name,
            params=params,
            step=step,
            total=total,
            result_text=result_text,
            success=success,
            elapsed_ms=elapsed_ms,
            id=f"tool-{self._msg_counter}",
        )
        chat = self.query_one("#chat-scroll", VerticalScroll)
        chat.mount(widget)
        chat.scroll_end(animate=False)

    def _show_thinking(self) -> None:
        if not self._thinking:
            self._thinking = True
            chat = self.query_one("#chat-scroll", VerticalScroll)
            chat.mount(ThinkingIndicator(id="thinking-indicator"))
            chat.scroll_end(animate=False)

    def _hide_thinking(self) -> None:
        if self._thinking:
            self._thinking = False
            indicators = self.query("#thinking-indicator")
            if indicators:
                indicators.remove()

    def _enable_input(self) -> None:
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = False
        inp.focus()

    # ── Actions ─────────────────────────────────────────────────────

    async def action_clear_chat(self) -> None:
        """Clear chat messages, keep system prompt."""
        chat = self.query_one("#chat-scroll", VerticalScroll)
        children = list(chat.children)
        for child in children:
            await child.remove()
        # Reset conversation but keep system messages
        system_msgs = [
            m for m in self.agent.conversation.messages
            if m.role == "system"
        ]
        self.agent.conversation.messages.clear()
        self.agent.conversation.messages.extend(system_msgs)
        self._msg_counter += 1
        await chat.mount(WelcomeWidget(
            tools=self.agent.registry.names(),
            id=f"welcome-{self._msg_counter}",
        ))
        self.notify("Chat cleared", severity="information")

    def action_show_tools(self) -> None:
        """Show available tools."""
        names = self.agent.registry.names()
        if not names:
            self.notify("No tools registered", severity="warning")
            return
        lines = []
        for name in names:
            skill = self.agent.registry.get(name)
            desc = skill.description if skill else ""
            lines.append(f"• {name}: {desc}")
        self.notify("\n".join(lines), title="Tools", severity="information")


# ── TUI Approval Bridge ────────────────────────────────────────────────

class _TUIApprovalBridge:
    """Bridges the agent's ApprovalGate to a TUI dialog.

    In "always" mode, auto-approves. Otherwise shows an inline
    notification and auto-approves (for simplicity in the TUI).
    In a more advanced version, this would show a modal dialog.
    """

    def __init__(self, app: GESPOApp):
        self.app = app
        self.mode = "always"  # TUI defaults to auto-approve
        self._always_approved: set = set()

    def check(self, request) -> bool:
        if self.mode == "always":
            return True
        if self.mode == "never":
            return False
        # For TUI, auto-approve and notify
        if request.name not in self._always_approved:
            self.app.call_from_thread(
                self.app.notify,
                f"Auto-approved: {request.name}",
                severity="information",
            )
            self._always_approved.add(request.name)
        return True
