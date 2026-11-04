"""OpenAI-compatible chat-completions provider (OpenAI, vLLM, LM Studio, gateways)."""

from typing import Any

import httpx

from raglens.llm.base import BaseLLM, GenerationRequest, LLMError, with_retries


class OpenAICompatLLM(BaseLLM):
    """Calls ``{base_url}/chat/completions`` with a bearer API key."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("OpenAICompatLLM requires an API key (set OPENAI_API_KEY)")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def generate(self, request: GenerationRequest) -> str:
        """Send the prompt and return the first choice's message content."""
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
        }

        def call() -> str:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("response 'choices[0].message.content' is not a string")
            return content.strip()

        return with_retries(call, max_retries=self.max_retries, provider="OpenAI-compatible API")
