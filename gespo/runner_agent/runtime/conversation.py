"""Conversation state management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from .llm.base import Message


@dataclass
class Conversation:
    """Ordered list of messages forming a conversation."""

    messages: List[Message] = field(default_factory=list)

    def add(self, msg: Message) -> None:
        self.messages.append(msg)

    def add_system(self, content: str) -> None:
        self.messages.append(Message("system", content))

    def add_user(self, content: str) -> None:
        self.messages.append(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message("assistant", content))

    def add_tool_result(self, result: Any) -> None:
        """Add a tool result as a system message."""
        self.messages.append(Message("system", result.to_message()))

    def to_api(self) -> list[dict]:
        """Convert to the list-of-dicts format LLM APIs expect."""
        return [m.to_dict() for m in self.messages]

    def token_estimate(self) -> int:
        """Rough token estimate (chars / 4)."""
        return sum(len(m.content) for m in self.messages) // 4

    def clear(self) -> None:
        """Remove all messages."""
        self.messages.clear()
