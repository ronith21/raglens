"""Small text helpers shared by the generator, the refusal gate and the evaluation metrics."""

import re

REFUSAL = "I don't know based on the provided documents."

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each few
    for from further had has have having he her here hers him his how i if in into is it its
    itself just me more most my no nor not now of off on once only or other our out over own
    same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why
    will with would you your yours explain describe define give list tell show please using
    use used uses way ways work works role main key""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[*`])")
_CITATION = re.compile(r"\[(\d{1,3})\]")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (keeps hyphenated terms such as ``cross-encoder`` together)."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and one-character tokens removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, dropping headings, math blocks and empty lines."""
    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("$$"):
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
        line = line.replace("**", "")
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    return sentences


def extract_citations(answer: str) -> list[int]:
    """Return the distinct citation numbers in order of first appearance."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def is_refusal(answer: str) -> bool:
    """True when the answer is the standard abstention string."""
    return answer.strip().lower().startswith("i don't know")


_SUFFIXES = ("ization", "ations", "ation", "ings", "ing", "ies", "ers", "er", "es", "ed", "s")


def stem(token: str) -> str:
    """Very light suffix stripping so "embeddings" matches "embedding" and "tokenizers" "tokenizer".

    Deliberately conservative: it only needs to make the refusal gate and the groundedness
    metric robust to plural and verb forms, not to be a linguistic stemmer.
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def stemmed_content(text: str) -> set[str]:
    """Set of stemmed content tokens."""
    return {stem(t) for t in content_tokens(text)}
