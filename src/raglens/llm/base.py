"""LLM provider interface, request model, errors and retry helper."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REFUSAL_TEXT = "I don't know based on the provided documents."

T = TypeVar("T")


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot produce a completion."""


class GenerationRequest(BaseModel):
    """Everything a provider needs to answer one question."""

    system: str = Field(..., description="System instructions")
    prompt: str = Field(..., description="Full user prompt including numbered context")
    question: str = Field(..., description="The raw user question")
    contexts: list[str] = Field(
        default_factory=list, description="Context passages in citation order ([1] is index 0)"
    )


class BaseLLM(ABC):
    """Abstract LLM provider."""

    name: str = "base"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Return the model's answer text for ``request``."""


def _is_retryable(exc: Exception) -> bool:
    """Transport errors, timeouts, 429 and 5xx are worth retrying; other 4xx are not."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def with_retries(
    func: Callable[[], T],
    *,
    max_retries: int,
    base_delay_s: float = 0.5,
    provider: str = "llm",
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Call ``func`` with exponential backoff (0.5s, 1s, 2s, ...) on retryable HTTP failures.

    Non-retryable errors and exhaustion of ``max_retries`` raise :class:`LLMError` with a
    message that names the provider and the underlying cause.
    """
    attempt = 0
    while True:
        try:
            return func()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            if not _is_retryable(exc) or attempt >= max_retries:
                detail = exc
                if isinstance(exc, httpx.HTTPStatusError):
                    detail = httpx.HTTPStatusError(
                        f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                        request=exc.request,
                        response=exc.response,
                    )
                raise LLMError(
                    f"{provider} request failed after {attempt + 1} attempt(s): {detail}"
                ) from exc
            delay = base_delay_s * (2**attempt)
            logger.warning(
                "%s request failed (%s); retrying in %.1fs (%d/%d)",
                provider,
                exc,
                delay,
                attempt + 1,
                max_retries,
            )
            (sleep or time.sleep)(delay)
            attempt += 1
