"""Common interface, prompt template and HTTP retry logic for LLM providers."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from raglens.textutil import REFUSAL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite every claim with the passage number in square brackets, for example [1] or [2]. "
    f'If the passages do not contain the answer, reply exactly: "{REFUSAL}" '
    "Do not use outside knowledge."
)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce an answer (after retries)."""


@dataclass
class GenerationRequest:
    """Everything a provider needs: the question and the numbered context passages."""

    question: str
    contexts: list[str]
    system_prompt: str = SYSTEM_PROMPT
    extra: dict[str, Any] = field(default_factory=dict)

    def render_context(self) -> str:
        """Context block with 1-based passage numbers matching the citation markers."""
        return "\n\n".join(f"[{i}] {text.strip()}" for i, text in enumerate(self.contexts, 1))

    def user_prompt(self) -> str:
        """The user turn sent to chat-style models."""
        return f"Context passages:\n\n{self.render_context()}\n\nQuestion: {self.question}\nAnswer:"


class BaseLLM(ABC):
    """A text generator that answers a :class:`GenerationRequest`."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Return the answer text (may contain [n] citation markers)."""


def post_with_retries(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST JSON with exponential backoff on timeouts, connection errors, 429 and 5xx.

    4xx errors other than 429 are not retried because repeating the request cannot help.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if response.status_code >= 400:
                raise LLMError(f"{url} returned {response.status_code}: {response.text[:200]}")
            data: dict[str, Any] = response.json()
            return data
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            delay = backoff_seconds * (2**attempt)
            logger.warning("LLM call failed (%s); retry %d in %.2fs", exc, attempt + 1, delay)
            sleep(delay)
    raise LLMError(f"LLM request to {url} failed after {max_retries + 1} attempts: {last_error}")
