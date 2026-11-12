"""Tests for the LLM providers: extractive FakeLLM, HTTP providers, retries and errors."""

import json

import httpx
import pytest

from raglens.config import Settings
from raglens.llm import FakeLLM, GenerationRequest, LLMError, OllamaLLM, OpenAICompatLLM, build_llm
from raglens.llm.base import post_with_retries
from raglens.textutil import REFUSAL

CTX = [
    "BM25 ranks passages by term frequency. It saturates repeated terms with k1.",
    "Reciprocal rank fusion adds 1 / (k + rank) for every retriever. The constant k is 60.",
]


def test_fake_llm_extracts_and_cites() -> None:
    """Answers are copied sentences tagged with the passage they came from."""
    answer = FakeLLM().generate(
        GenerationRequest("How does reciprocal rank fusion add ranks?", CTX)
    )
    assert "[2]" in answer
    assert "Reciprocal rank fusion adds" in answer


def test_fake_llm_does_not_pad_with_weak_sentences() -> None:
    """A second sentence sharing one incidental word with the question is dropped."""
    ctx = [
        "The zorblax protocol rotates quantum widgets every ninety seconds.",
        "Cross-encoders are optional components that rotate nothing at all here.",
    ]
    answer = FakeLLM().generate(
        GenerationRequest("How often does the zorblax protocol rotate widgets?", ctx)
    )
    assert answer == "The zorblax protocol rotates quantum widgets every ninety seconds. [1]"


def test_fake_llm_refuses_without_overlap_and_dedupes() -> None:
    """No shared content word means abstention; duplicate sentences appear once."""
    assert FakeLLM().generate(GenerationRequest("Capital of Peru?", CTX)) == REFUSAL
    assert FakeLLM().generate(GenerationRequest("the of and", CTX)) == REFUSAL
    dup = ["The constant k is sixty in reciprocal rank fusion."] * 2
    answer = FakeLLM(max_sentences=3).generate(
        GenerationRequest("reciprocal rank fusion constant", dup)
    )
    assert answer.count("constant k") == 1
    with pytest.raises(ValueError):
        FakeLLM(max_sentences=0)


def test_prompt_numbers_passages() -> None:
    """Passage numbers in the prompt match citation numbers."""
    prompt = GenerationRequest("q?", CTX).user_prompt()
    assert prompt.index("[1] BM25") < prompt.index("[2] Reciprocal")
    assert "Question: q?" in prompt


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_openai_compat_success_and_auth_header() -> None:
    """Sends a bearer token and parses the first choice."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " ok [1] "}}]})

    llm = OpenAICompatLLM(
        base_url="http://gw/v1/", api_key="k", client=_client(httpx.MockTransport(handler))
    )
    assert llm.generate(GenerationRequest("q", CTX)) == "ok [1]"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["temperature"] == 0  # type: ignore[index]


def test_openai_compat_bad_payloads() -> None:
    """Malformed or empty completions raise LLMError."""
    for payload in ({"choices": []}, {"choices": [{"message": {"content": "  "}}]}):
        transport = httpx.MockTransport(lambda r, p=payload: httpx.Response(200, json=p))
        with pytest.raises(LLMError):
            OpenAICompatLLM(client=_client(transport)).generate(GenerationRequest("q", CTX))


def test_ollama_success_and_empty() -> None:
    """Parses message.content and rejects empty replies."""
    ok = httpx.MockTransport(lambda r: httpx.Response(200, json={"message": {"content": "hi [1]"}}))
    assert OllamaLLM(client=_client(ok)).generate(GenerationRequest("q", CTX)) == "hi [1]"
    empty = httpx.MockTransport(lambda r: httpx.Response(200, json={"message": {}}))
    with pytest.raises(LLMError, match="no message"):
        OllamaLLM(client=_client(empty)).generate(GenerationRequest("q", CTX))


def test_retries_with_exponential_backoff_then_succeeds() -> None:
    """5xx and timeouts are retried with doubling delays."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("slow")
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"x": 1})

    delays: list[float] = []
    data = post_with_retries(
        _client(httpx.MockTransport(handler)),
        "http://x",
        {},
        max_retries=3,
        backoff_seconds=0.1,
        sleep=delays.append,
    )
    assert data == {"x": 1}
    assert delays == [0.1, 0.2]


def test_client_errors_are_not_retried_and_exhaustion_raises() -> None:
    """A 401 fails immediately; persistent 500s fail after max_retries."""
    calls = {"n": 0}

    def unauthorized(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="401"):
        post_with_retries(
            _client(httpx.MockTransport(unauthorized)), "http://x", {}, sleep=lambda s: None
        )
    assert calls["n"] == 1

    always_500 = httpx.MockTransport(lambda r: httpx.Response(500))
    with pytest.raises(LLMError, match="3 attempts"):
        post_with_retries(_client(always_500), "http://x", {}, max_retries=2, sleep=lambda s: None)


def test_build_llm_by_provider() -> None:
    """Settings select the provider."""
    assert isinstance(build_llm(Settings(llm_provider="fake")), FakeLLM)
    assert isinstance(build_llm(Settings(llm_provider="ollama")), OllamaLLM)
    assert isinstance(build_llm(Settings(llm_provider="openai")), OpenAICompatLLM)
    with pytest.raises(ValueError):
        build_llm(Settings(llm_provider="nope"))
