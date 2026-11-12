"""Pretrained word-vector embedder (GloVe-style vectors shipped with spaCy ``en_core_web_lg``).

A sentence vector is the IDF-weighted average of its words' pretrained 300-d vectors with the
corpus's dominant common direction removed (the "SIF" recipe of Arora et al., 2017), then
L2-normalised. Without that removal every averaged vector shares a large common component and
all chunks look similar to every query (cosines around 0.6 for unrelated text).

The vectors were trained on Common Crawl, so they know general-English synonymy
("doctor" ~ "physician", cosine 0.79) that neither BM25 nor corpus-only LSA can learn from 20
documents. IDF weights are fitted on the indexed corpus so frequent filler words count less.

Install the optional model with ``pip install -e .[vectors]`` (it is downloaded from the spaCy
GitHub releases, not from Hugging Face).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from raglens.embed.base import FittableEmbedder
from raglens.textutil import content_tokens

_STATE_FILE = "word_vector_idf.json"
_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)


def _words(text: str) -> list[str]:
    """Content words with LaTeX math removed (``\\mathbb``, ``frac`` carry no topic signal)."""
    return content_tokens(_MATH.sub(" ", text))


class WordVectorEmbedder(FittableEmbedder):
    """IDF-weighted mean of pretrained word vectors."""

    def __init__(self, model_name: str = "en_core_web_lg") -> None:
        self.model_name = model_name
        self._vocab: Any = None
        self._dim: int | None = None
        self._idf: dict[str, float] | None = None
        self._default_idf = 1.0
        self._common: npt.NDArray[np.float32] | None = None

    def _load(self) -> None:
        if self._vocab is not None:
            return
        import spacy

        nlp = spacy.load(
            self.model_name,
            exclude=[
                "tagger",
                "parser",
                "ner",
                "lemmatizer",
                "attribute_ruler",
                "senter",
                "tok2vec",
            ],
        )
        if nlp.vocab.vectors.shape[0] == 0:
            raise RuntimeError(f"spaCy model {self.model_name} has no word vectors")
        self._vocab = nlp.vocab
        self._dim = int(nlp.vocab.vectors.shape[1])

    @property
    def dimension(self) -> int:
        """Word-vector width (300 for ``en_core_web_lg``)."""
        self._load()
        return self._dim or 300

    @property
    def is_fitted(self) -> bool:
        """Whether IDF weights have been learned."""
        return self._idf is not None

    def fit(self, texts: list[str]) -> WordVectorEmbedder:
        """Learn IDF weights and the common component to remove, from the corpus chunks."""
        n = len(texts)
        doc_freq: dict[str, int] = {}
        for text in texts:
            for token in set(_words(text)):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self._idf = {t: math.log((1 + n) / (1 + df)) + 1 for t, df in doc_freq.items()}
        # Words never seen in the corpus are treated as rare (maximally informative).
        self._default_idf = math.log(1 + n) + 1
        self._common = None
        raw = self._weighted_means(texts)
        if n >= 2:
            # First right-singular vector of the (uncentred) matrix = shared direction.
            _, _, vt = np.linalg.svd(raw, full_matrices=False)
            self._common = np.asarray(vt[0], dtype=np.float32)
        return self

    def _weighted_means(self, texts: list[str]) -> npt.NDArray[np.float32]:
        self._load()
        dim = self.dimension
        out = np.zeros((len(texts), dim), dtype=np.float32)
        idf = self._idf or {}
        for row, text in enumerate(texts):
            total = 0.0
            for token in _words(text):
                lex = self._vocab[token]
                if not lex.has_vector:
                    continue
                weight = idf.get(token, self._default_idf)
                out[row] += weight * np.asarray(lex.vector, dtype=np.float32)
                total += weight
            if total > 0:
                out[row] /= total
        return out

    def embed_texts(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed texts; text without any known word maps to the zero vector."""
        vectors = self._weighted_means(texts)
        if self._common is not None:
            vectors = vectors - np.outer(vectors @ self._common, self._common)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def save_state(self, directory: str | Path) -> None:
        """Persist IDF weights and the common component as JSON."""
        if self._idf is None:
            raise RuntimeError("Cannot save an unfitted WordVectorEmbedder")
        Path(directory).mkdir(parents=True, exist_ok=True)
        state = {
            "model_name": self.model_name,
            "idf": self._idf,
            "default_idf": self._default_idf,
            "common": None if self._common is None else self._common.tolist(),
        }
        (Path(directory) / _STATE_FILE).write_text(json.dumps(state), encoding="utf-8")

    def load_state(self, directory: str | Path) -> None:
        """Restore IDF weights written by :meth:`save_state`."""
        state = json.loads((Path(directory) / _STATE_FILE).read_text(encoding="utf-8"))
        self._idf = {str(k): float(v) for k, v in state["idf"].items()}
        self._default_idf = float(state["default_idf"])
        common = state.get("common")
        self._common = None if common is None else np.asarray(common, dtype=np.float32)

    def has_saved_state(self, directory: str | Path) -> bool:
        """Whether IDF weights were saved in ``directory``."""
        return (Path(directory) / _STATE_FILE).exists()
