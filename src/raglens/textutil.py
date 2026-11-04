"""Small, dependency-free text helpers shared by the pipeline, FakeLLM and evaluation."""

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*])")
_LATEX_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$[^$\n]+\$")
_SUBSCRIPT_RE = re.compile(r"\b([a-z])_\{?(\d+)\}?")
_LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_MD_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*]\s+|\d+\.\s+)")
_MD_EMPHASIS_RE = re.compile(r"\*\*|__|`")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself
    yourselves explain describe tell give used use using does work works mean means
    difference differ different compared compare versus happens happen typically
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (letters, digits, inner hyphens/apostrophes).

    LaTeX-style subscripts are joined first, so ``k_1`` and ``L_{2}`` become ``k1`` and ``l2``.
    """
    return _WORD_RE.findall(_SUBSCRIPT_RE.sub(r"\1\2", text.lower()))


def _is_content(token: str) -> bool:
    """Keep non-stopwords of 3+ chars, short alphanumerics (``l1``, ``k1``) and single letters.

    Single letters matter in technical text ("the b parameter", "probability p"); the only
    common English one-letter words, "a" and "i", are stopwords.
    """
    if token in STOPWORDS:
        return False
    if len(token) == 1:
        return token.isalpha()
    return len(token) > 2 or any(c.isdigit() for c in token)


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and short non-alphanumeric fragments removed."""
    return [t for t in tokenize(text) if _is_content(t)]


_SUFFIXES = ("ization", "isation", "ations", "ation", "izing", "ized", "izes", "ize", "ing", "ed")


def stem(token: str) -> str:
    """A deliberately light stemmer: plural ``s``, a few derivational suffixes, 6-char prefix.

    ``parallelize``/``parallelized`` -> ``parall``, ``rates``/``rate`` -> ``rate``,
    ``difference``/``different`` -> ``differ``. Good enough to match question words to passage
    words; not meant to be linguistically correct.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[: -len(suffix)]
            break
    return token[:6]


def content_stems(text: str) -> list[str]:
    """Stemmed content tokens."""
    return [stem(t) for t in content_tokens(text)]


def _inline_math_to_text(match: re.Match[str]) -> str:
    """Render ``$\\eta > 0$`` as ``eta > 0`` so sentences stay readable."""
    body = _LATEX_CMD_RE.sub(r"\1", match.group(0).strip("$"))
    return body.replace("{", "").replace("}", "").replace("\\", "").strip()


def strip_markdown(text: str) -> str:
    """Remove LaTeX math, markdown list/heading markers and emphasis from text."""
    text = _LATEX_BLOCK_RE.sub(" ", text)
    text = _LATEX_INLINE_RE.sub(_inline_math_to_text, text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            # Headings are titles, not answer material.
            lines.append("")
            continue
        lines.append(_MD_EMPHASIS_RE.sub("", _MD_PREFIX_RE.sub("", stripped)))
    return "\n".join(lines)


def split_sentences(text: str) -> list[str]:
    """Split plain text into sentences, treating blank lines as hard boundaries."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip())
    return sentences


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of the sentences in ``text``, trimmed of surrounding whitespace.

    Unlike :func:`split_sentences` the offsets refer to the original text, so a span can be
    highlighted in place. Lines are hard boundaries, as in markdown lists.
    """
    spans: list[tuple[int, int]] = []
    for line in re.finditer(r"[^\n]+", text):
        cursor = line.start()
        for boundary in _SENTENCE_RE.finditer(line.group(0)):
            spans.append((cursor, line.start() + boundary.start()))
            cursor = line.start() + boundary.end()
        spans.append((cursor, line.end()))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((start, end))
    return trimmed


def query_coverage(query: str, text: str) -> float:
    """Fraction of the query's distinct (stemmed) content tokens that appear in ``text``.

    Returns 0.0 when the query has no content tokens.
    """
    q_stems = set(content_stems(query))
    if not q_stems:
        return 0.0
    t_stems = {stem(t) for t in tokenize(text)}
    return len(q_stems & t_stems) / len(q_stems)
