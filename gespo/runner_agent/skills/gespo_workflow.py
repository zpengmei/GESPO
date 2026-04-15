"""GESPO workflow knowledge provider (not a tool executor).

Loads GESPO-specific workflow documentation and provides system prompt context.
"""
from __future__ import annotations

from typing import List

from ..runtime.llm.base import Message


class GESPOWorkflowProvider:
    """Loads GESPO workflow docs and provides system prompt context."""

    def __init__(self, docs_root: str = "", max_chars: int = 60000):
        self.docs_root = docs_root
        self.max_chars = max_chars

    def system_messages(self) -> List[Message]:
        """Return system messages with GESPO workflow documentation."""
        if not self.docs_root:
            return []

        from ..docs import load_docs
        docs_text = load_docs(self.docs_root, self.max_chars)
        if not docs_text.strip():
            return []

        return [
            Message(
                "system",
                "GESPO workflow reference documentation (use as context, do not quote verbatim):\n\n"
                + docs_text,
            )
        ]
