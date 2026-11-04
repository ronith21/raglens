"""Ollama chat-completion provider."""

from typing import Any

import httpx

from raglens.llm.base import BaseLLM, GenerationRequest, LLMError, with_retries


class OllamaLLM(BaseLLM):
    """Calls a local Ollama server's ``/api/chat`` endpoint (non-streaming)."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    def generate(self, request: GenerationRequest) -> str:
        """Send the prompt to Ollama and return the assistant message content."""
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
        }

        def call() -> str:
            response = self._client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Ollama response 'message.content' is not a string")
            return content.strip()

        try:
            return with_retries(call, max_retries=self.max_retries, provider="Ollama")
        except LLMError as exc:
            raise LLMError(f"{exc} (is Ollama running at {self.base_url}?)") from exc
