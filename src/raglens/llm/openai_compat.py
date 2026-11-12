"""Provider for any OpenAI-compatible ``/chat/completions`` endpoint."""

import httpx

from raglens.llm.base import BaseLLM, GenerationRequest, LLMError, post_with_retries


class OpenAICompatLLM(BaseLLM):
    """Works with OpenAI, Azure-style gateways, vLLM, LM Studio and similar servers."""

    name = "openai"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout)

    def generate(self, request: GenerationRequest) -> str:
        """Send a chat completion request with temperature 0."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt()},
            ],
        }
        data = post_with_retries(
            self.client,
            f"{self.base_url}/chat/completions",
            payload,
            headers=headers,
            max_retries=self.max_retries,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected completion payload: {str(data)[:200]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Completion contained no text")
        return content.strip()
