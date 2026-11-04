"""Tests for LLM providers, retries and the factory. No network: httpx.MockTransport only."""

import json

import httpx
import pytest

from raglens.config import Settings
from raglens.llm import (
    REFUSAL_TEXT,
    FakeLLM,
    GenerationRequest,
    LLMError,
    OllamaLLM,
    OpenAICompatLLM,
    create_llm,
    with_retries,
)

CONTEXTS = [
    "# Gradient Descent\n\nGradient descent minimizes a loss. If the learning rate is too large, "
    "updates overshoot minima and may diverge.",
    "Learning rate warmup gradually scales the learning rate from zero during early iterations.",
]


def _request(question: str, contexts: list[str] | None = None) -> GenerationRequest:
    return GenerationRequest(
        system="sys",
        prompt="prompt",
        question=question,
        contexts=CONTEXTS if contexts is None else contexts,
    )


def test_fake_llm_is_extractive_cited_and_deterministic() -> None:
    """FakeLLM copies the most specific sentences and cites their passages."""
    llm = FakeLLM()
    answer = llm.generate(_request("What happens when the learning rate is too large?"))
    assert "overshoot" in answer
    assert "[1]" in answer
    assert answer == llm.generate(_request("What happens when the learning rate is too large?"))


def test_fake_llm_prefers_rare_terms() -> None:
    """A rare question word ('warmup') outweighs words every sentence shares."""
    answer = FakeLLM(max_sentences=1).generate(_request("Why use learning rate warmup?"))
    assert answer.startswith("Learning rate warmup")
    assert answer.endswith("[2].")


def test_fake_llm_refuses_without_overlap() -> None:
    """No overlapping words, no contexts, or a stopword-only question all refuse."""
    llm = FakeLLM()
    assert llm.generate(_request("Who won the FIFA World Cup?")) == REFUSAL_TEXT
    assert llm.generate(_request("learning rate", contexts=[])) == REFUSAL_TEXT
    assert llm.generate(_request("what is it")) == REFUSAL_TEXT


def test_fake_llm_validates_arguments() -> None:
    """Non-positive limits are rejected."""
    with pytest.raises(ValueError):
        FakeLLM(max_sentences=0)


def test_ollama_success_sends_chat_payload() -> None:
    """OllamaLLM posts system+user messages and returns the message content."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "  An answer [1].  "}})

    llm = OllamaLLM(
        model="m", base_url="http://ollama:11434/", transport=httpx.MockTransport(handler)
    )
    assert llm.generate(_request("q")) == "An answer [1]."
    assert seen["url"] == "http://ollama:11434/api/chat"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_ollama_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 then a 200 succeeds after one backoff sleep."""
    sleeps: list[float] = []
    monkeypatch.setattr("raglens.llm.base.time.sleep", sleeps.append)
    responses = iter(
        [httpx.Response(503, text="busy"), httpx.Response(200, json={"message": {"content": "ok"}})]
    )
    llm = OllamaLLM(transport=httpx.MockTransport(lambda _: next(responses)), max_retries=2)
    assert llm.generate(_request("q")) == "ok"
    assert sleeps == [0.5]


def test_ollama_connection_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhausted retries raise LLMError naming the server URL."""
    monkeypatch.setattr("raglens.llm.base.time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm = OllamaLLM(
        base_url="http://nowhere:1", transport=httpx.MockTransport(handler), max_retries=1
    )
    with pytest.raises(LLMError, match=r"2 attempt.*is Ollama running at http://nowhere:1"):
        llm.generate(_request("q"))


def test_openai_compat_success_and_auth_header() -> None:
    """The bearer key is sent and the first choice is returned."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"
        assert str(request.url) == "http://gw/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi [1]."}}]})

    llm = OpenAICompatLLM(
        base_url="http://gw/v1", api_key="sk-test", transport=httpx.MockTransport(handler)
    )
    assert llm.generate(_request("q")) == "Hi [1]."


def test_openai_compat_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 fails immediately with the status in the message."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    llm = OpenAICompatLLM(api_key="x", transport=httpx.MockTransport(handler), max_retries=3)
    with pytest.raises(LLMError, match="HTTP 401: bad key"):
        llm.generate(_request("q"))
    assert len(calls) == 1


def test_openai_compat_rejects_malformed_response() -> None:
    """Missing fields become an LLMError rather than a KeyError."""
    llm = OpenAICompatLLM(
        api_key="x",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
        max_retries=0,
    )
    with pytest.raises(LLMError):
        llm.generate(_request("q"))


def test_openai_compat_requires_key() -> None:
    """Constructing without a key fails fast with guidance."""
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAICompatLLM(api_key="")


def test_with_retries_backoff_schedule() -> None:
    """Delays double each attempt and the final error is wrapped."""
    sleeps: list[float] = []
    request = httpx.Request("GET", "http://x")

    def always_429() -> str:
        raise httpx.HTTPStatusError(
            "429", request=request, response=httpx.Response(429, request=request)
        )

    with pytest.raises(LLMError, match="after 4 attempt"):
        with_retries(always_429, max_retries=3, sleep=sleeps.append)
    assert sleeps == [0.5, 1.0, 2.0]


def test_create_llm_factory() -> None:
    """The factory maps provider names to classes and rejects unknown ones."""
    base = Settings(_env_file=None, openai_api_key="k")
    assert isinstance(create_llm(base, "fake"), FakeLLM)
    assert isinstance(create_llm(base, "ollama"), OllamaLLM)
    assert isinstance(create_llm(base, "openai"), OpenAICompatLLM)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm(base, "gpt-magic")
