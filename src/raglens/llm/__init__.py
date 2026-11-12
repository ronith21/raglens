"""LLM providers behind one interface."""

from raglens.config import Settings
from raglens.llm.base import SYSTEM_PROMPT, BaseLLM, GenerationRequest, LLMError
from raglens.llm.fake import FakeLLM
from raglens.llm.ollama import OllamaLLM
from raglens.llm.openai_compat import OpenAICompatLLM


def build_llm(settings: Settings) -> BaseLLM:
    """Create the provider selected by ``settings.llm_provider``."""
    provider = settings.llm_provider.lower()
    if provider == "fake":
        return FakeLLM()
    if provider == "ollama":
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if provider == "openai":
        return OpenAICompatLLM(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"Unknown llm_provider {settings.llm_provider!r}")


__all__ = [
    "SYSTEM_PROMPT",
    "BaseLLM",
    "FakeLLM",
    "GenerationRequest",
    "LLMError",
    "OllamaLLM",
    "OpenAICompatLLM",
    "build_llm",
]
