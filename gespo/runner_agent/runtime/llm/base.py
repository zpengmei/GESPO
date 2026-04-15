"""LLM client protocol and message types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable


@dataclass
class Message:
    """A single conversation message."""
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM backends. Any client implementing chat() works."""

    def chat(self, messages: list[dict]) -> str:
        """Send messages and return the assistant's response text."""
        ...

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """Send messages and yield response text chunks."""
        ...


def create_client(provider: str, **kwargs) -> LLMClient:
    """Factory: create an LLM client from a provider name."""
    if provider in ("deepseek", "openai", "openai_compatible"):
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
