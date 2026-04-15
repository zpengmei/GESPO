"""OpenAI-compatible LLM client (DeepSeek, OpenAI, local endpoints)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Optional

import requests

_DEFAULT_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com",
}

_ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class OpenAICompatClient:
    """Client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        base_url: str = "",
        api_key: str = "",
        api_key_file: str = "",
        timeout_s: float = 60.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **_extra,
    ):
        self.model = model
        self.base_url = (base_url.rstrip("/") if base_url else _DEFAULT_URLS.get(provider, "")).rstrip("/")
        self.timeout = float(timeout_s)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve API key: explicit > file > env var
        self.api_key = api_key
        if not self.api_key and api_key_file:
            try:
                self.api_key = open(os.path.expanduser(api_key_file)).read().strip()
            except Exception:
                pass
        if not self.api_key:
            env_key = _ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
            self.api_key = os.environ.get(env_key, "")

    def _build_payload(self, messages: list[dict], stream: bool = False) -> dict:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages: list[dict]) -> str:
        """Blocking chat completion. Returns assistant response text."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(messages, stream=False)
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """Streaming chat completion. Yields text chunks."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(messages, stream=True)
        with requests.post(
            url, headers=self._headers(), json=payload, timeout=self.timeout, stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
