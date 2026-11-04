"""LLM providers behind one interface, plus a settings-driven factory."""

from raglens.config import Settings
from raglens.llm.base import REFUSAL_TEXT, BaseLLM, GenerationRequest, LLMError, with_retries
from raglens.llm.fake import FakeLLM
from raglens.llm.ollama import OllamaLLM
from raglens.llm.openai_compat import OpenAICompatLLM

PROVIDERS = ("fake", "ollama", "openai")


def create_llm(settings: Settings, provider: str | None = None) -> BaseLLM:
    """Instantiate the provider named by ``provider`` (or ``settings.llm_provider``)."""
    name = (provider or settings.llm_provider).lower()
    if name == "fake":
        return FakeLLM()
    if name == "ollama":
        return OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    if name == "openai":
        return OpenAICompatLLM(
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"Unknown LLM provider {name!r}; expected one of {PROVIDERS}")


__all__ = [
    "PROVIDERS",
    "REFUSAL_TEXT",
    "BaseLLM",
    "FakeLLM",
    "GenerationRequest",
    "LLMError",
    "OllamaLLM",
    "OpenAICompatLLM",
    "create_llm",
    "with_retries",
]
