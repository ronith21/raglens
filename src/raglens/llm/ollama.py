"""Ollama chat provider (local models such as llama3)."""

import httpx

from raglens.llm.base import BaseLLM, GenerationRequest, LLMError, post_with_retries


class OllamaLLM(BaseLLM):
    """Calls ``POST {base_url}/api/chat`` on a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        timeout: float = 60.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout)

    def generate(self, request: GenerationRequest) -> str:
        """Send the system and user turns with temperature 0 for reproducibility."""
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt()},
            ],
        }
        data = post_with_retries(
            self.client, f"{self.base_url}/api/chat", payload, max_retries=self.max_retries
        )
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"Ollama returned no message content: {str(data)[:200]}")
        return content.strip()
