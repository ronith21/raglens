"""Multi-query retrieval: rewrite the question into variants and fuse their rankings with RRF.

The rewriter is deterministic and needs no LLM. It mines an acronym glossary from the indexed
chunks ("Reciprocal Rank Fusion (RRF)" -> ``rrf`` <-> ``reciprocal rank fusion``) and produces:

1. the original question;
2. a keyword-only form (stopwords removed), which sharpens dense search;
3. an acronym-expanded form, where each known acronym is replaced by its long form and each
   known long form gets its acronym appended, so "What is RRF?" can match passages that only
   spell the name out (and vice versa).

Each variant is retrieved independently and the lists are fused with Reciprocal Rank Fusion, so a
chunk that several phrasings agree on rises to the top.
"""

import re
from collections import Counter

from raglens.ingest.models import Chunk
from raglens.retrieve.base import BaseRetriever, SearchResult
from raglens.retrieve.hybrid import reciprocal_rank_fusion
from raglens.textutil import content_tokens, tokenize

_ACRONYM_RE = re.compile(r"\(([A-Z][A-Za-z0-9]{1,9})\)")
# "SMOTE (Synthetic Minority Over-sampling Technique)": acronym first, long form in brackets.
_REVERSED_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\s+\(([A-Z][A-Za-z\- ]{3,80})\)")
_MAX_PHRASE_WORDS = 8


def _spells(words: list[str], letters: str) -> bool:
    """True if the words' initials spell ``letters``.

    A hyphenated word may contribute either its first letter ("Over-sampling" -> o) or the first
    letter of every part ("Byte-Pair" -> bp).
    """
    options = [{w[0].lower(), "".join(p[0] for p in w.split("-") if p).lower()} for w in words if w]

    def match(i: int, j: int) -> bool:
        if i == len(options):
            return j == len(letters)
        return any(letters.startswith(opt, j) and match(i + 1, j + len(opt)) for opt in options[i])

    return bool(options) and match(0, 0)


def mine_acronyms(chunks: list[Chunk]) -> dict[str, str]:
    """Find ``Long Form (LF)`` and ``LF (Long Form)`` definitions whose initials spell ``LF``.

    Returns a mapping of lowercase acronym to lowercase long form; when an acronym is defined
    in several ways, the most frequent definition wins.
    """
    found: dict[str, Counter[str]] = {}
    for chunk in chunks:
        text = chunk.text
        for match in _ACRONYM_RE.finditer(text):
            acronym = match.group(1)
            letters = re.sub(r"[^a-z]", "", acronym.lower())
            if len(letters) < 2:
                continue
            words = re.findall(
                r"[A-Za-z][A-Za-z\-]*", text[max(0, match.start() - 120) : match.start()]
            )
            words = words[-_MAX_PHRASE_WORDS:]
            for size in range(1, len(words) + 1):
                candidate = words[-size:]
                if _spells(candidate, letters):
                    long_form = " ".join(candidate).lower()
                    found.setdefault(acronym.lower(), Counter())[long_form] += 1
                    break
        for match in _REVERSED_RE.finditer(text):
            acronym, long_form = match.group(1), match.group(2).strip()
            letters = re.sub(r"[^a-z]", "", acronym.lower())
            if len(letters) >= 2 and _spells(long_form.split(), letters):
                found.setdefault(acronym.lower(), Counter())[long_form.lower()] += 1
    return {acr: forms.most_common(1)[0][0] for acr, forms in found.items()}


class QueryRewriter:
    """Produces up to ``max_variants`` distinct rewrites of a query (the original first)."""

    def __init__(self, acronyms: dict[str, str] | None = None, max_variants: int = 3) -> None:
        if max_variants < 1:
            raise ValueError("max_variants must be at least 1")
        self.acronyms = dict(acronyms or {})
        self.max_variants = max_variants

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], max_variants: int = 3) -> "QueryRewriter":
        """Build a rewriter whose glossary is mined from ``chunks``."""
        return cls(mine_acronyms(chunks), max_variants)

    def expand(self, query: str) -> str:
        """Swap known acronyms for long forms and annotate long forms with their acronym."""
        lowered = " ".join(tokenize(query))
        additions: list[str] = []
        for acronym, long_form in self.acronyms.items():
            has_acronym = re.search(rf"\b{re.escape(acronym)}\b", lowered) is not None
            has_long = long_form in lowered
            if has_acronym and not has_long:
                lowered = re.sub(rf"\b{re.escape(acronym)}\b", long_form, lowered)
            elif has_long and not has_acronym:
                additions.append(acronym)
        return " ".join([lowered, *additions]).strip()

    def variants(self, query: str) -> list[str]:
        """Original query, keyword form and acronym-expanded form, deduplicated in order."""
        candidates = [query.strip(), " ".join(content_tokens(query)), self.expand(query)]
        out: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = " ".join(tokenize(candidate))
            if candidate and key and key not in seen:
                seen.add(key)
                out.append(candidate)
        return out[: self.max_variants]


class MultiQueryRetriever(BaseRetriever):
    """Retrieves every query variant with ``base`` and fuses the rankings with RRF."""

    def __init__(self, base: BaseRetriever, rewriter: QueryRewriter, rrf_k: int = 60) -> None:
        self.base = base
        self.rewriter = rewriter
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Fuse the base retriever's results for each variant; records the variants used."""
        if top_k <= 0:
            return []
        variants = self.rewriter.variants(query)
        pool = max(top_k * 2, 20)
        ranked_lists = []
        for i, variant in enumerate(variants):
            results = self.base.retrieve(variant, top_k=pool)
            ranked_lists.append([r.model_copy(update={"retriever_name": f"q{i}"}) for r in results])
        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, top_k=top_k)
        for result in fused:
            result.retriever_name = "multi-query"
            result.metadata["query_variants"] = variants
        return fused
