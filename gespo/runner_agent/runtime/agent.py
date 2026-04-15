"""Core agent loop: LLM → parse tool request → dispatch → repeat."""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Optional

from .conversation import Conversation
from .llm.base import LLMClient, Message
from .tool_registry import ToolRegistry
from .approval import ApprovalGate
from .config import RuntimeConfig


class Agent:
    """Framework-agnostic agent with a tool loop.

    The agent sends messages to an LLM, parses tool requests from the response,
    dispatches them through the registry, and feeds results back until the LLM
    produces a final text response (no tool call).
    """

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        approval: ApprovalGate,
        config: RuntimeConfig,
    ):
        self.client = client
        self.registry = registry
        self.approval = approval
        self.config = config
        self.conversation = Conversation()
        self._turn_count = 0

    def setup(self, system_messages: list[Message]) -> None:
        """Initialize conversation with system messages (rules, tools, docs)."""
        for msg in system_messages:
            self.conversation.add(msg)

    def run_turn(self, user_input: str) -> str:
        """Execute a full turn: user message → [tool loop] → final response."""
        from .. import terminal as t

        self._turn_count += 1
        self.conversation.add_user(user_input)
        max_steps = self.config.max_tool_steps

        for step in range(1, max_steps + 1):
            # ── Get LLM response ──────────────────────────────────
            t0 = time.monotonic()
            if self.config.stream:
                chunks = []
                for chunk in self.client.chat_stream(self.conversation.to_api()):
                    print(chunk, end="", flush=True, file=sys.stderr)
                    chunks.append(chunk)
                reply = "".join(chunks)
                print(file=sys.stderr)
            else:
                with t.spinner("thinking"):
                    reply = self.client.chat(self.conversation.to_api())
            llm_elapsed = time.monotonic() - t0

            self.conversation.add_assistant(reply)

            # ── Parse for tool request ────────────────────────────
            request = self._parse_tool_request(reply)
            if request is None:
                return reply  # No tool call — done

            # ── Dispatch tool ─────────────────────────────────────
            t.tool_header(request.name, step, max_steps)
            t.tool_params_compact(request.params)

            t0 = time.monotonic()
            result = self._dispatch(request)
            tool_elapsed = time.monotonic() - t0

            # Show result to user
            t.tool_output(result)
            timing = t.elapsed_str(tool_elapsed)
            if result.success:
                t.success(f"{request.name} {timing}")
            else:
                t.error(f"{request.name} failed {timing}")

            self.conversation.add_tool_result(result)

        # ── Budget exhausted ──────────────────────────────────────
        t.warn(f"Tool step budget ({max_steps}) exhausted — requesting final answer")
        self.conversation.add_system(
            f"[Agent] Tool step budget ({max_steps}) exhausted. "
            "Please provide a final answer without using tools."
        )
        reply = self.client.chat(self.conversation.to_api())
        self.conversation.add_assistant(reply)
        return reply

    def _parse_tool_request(self, text: str) -> Optional[Any]:
        """Extract a JSON tool request from LLM output.

        Handles:
        - Pure JSON: {"tool": "shell", "cmd": "ls"}
        - JSON in markdown fence: ```json {...} ```
        - JSON embedded in prose (first { ... } block)
        """
        text = text.strip()

        # Try to find JSON in code fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        # Find the first JSON object
        brace_start = text.find("{")
        if brace_start < 0:
            return None

        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text, brace_start)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(obj, dict) or "tool" not in obj:
            return None

        tool_name = str(obj["tool"])
        if tool_name not in self.registry:
            return None

        from ..skills.base import ToolRequest
        params = {k: v for k, v in obj.items() if k != "tool"}
        return ToolRequest(name=tool_name, params=params)

    def _dispatch(self, request) -> Any:
        """Look up skill, check approval, execute."""
        from ..skills.base import ToolResult
        from .. import terminal as t

        skill = self.registry.get(request.name)
        if skill is None:
            return ToolResult.error(request.name, f"Unknown tool: {request.name}")

        if not self.approval.check(request):
            return ToolResult.error(request.name, "User declined execution")

        try:
            with t.spinner(f"running {request.name}"):
                return skill.execute(request.params)
        except Exception as exc:
            return ToolResult.error(request.name, f"Execution failed: {exc}")
