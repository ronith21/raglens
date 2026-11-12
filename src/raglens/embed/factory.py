"""Build the embedder named in the settings, with an honest, logged fallback chain."""

import logging

from raglens.embed.base import BaseEmbedder
from raglens.embed.hash import HashEmbedder
from raglens.embed.lsa import LSAEmbedder
from raglens.embed.sentence_transformer import SentenceTransformerEmbedder
from raglens.embed.word_vectors import WordVectorEmbedder

logger = logging.getLogger(__name__)

EMBEDDER_CHOICES = ("minilm", "glove", "lsa", "hash")


def build_embedder(
    name: str,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    hash_dim: int = 384,
    lsa_components: int = 128,
) -> BaseEmbedder:
    """Return an embedder by short name.

    ``minilm`` tries the SentenceTransformers model; if it cannot be loaded (no network, package
    missing) it falls back to pretrained GloVe-style word vectors, and then to corpus-trained LSA.
    It never silently falls back to random hashing, which carries no meaning at all.
    """
    key = name.lower()
    if key == "hash":
        return HashEmbedder(dimension=hash_dim)
    if key == "lsa":
        return LSAEmbedder(n_components=lsa_components)
    if key == "glove":
        wv = WordVectorEmbedder()
        _ = wv.dimension  # load now so a missing model fails here, not mid-indexing
        return wv
    if key == "minilm":
        st = SentenceTransformerEmbedder(
            model_name=model_name, device=device, fallback_to_hash=False
        )
        try:
            _ = st.dimension
            return st
        except Exception as exc:
            logger.warning("Could not load %s (%s); trying GloVe word vectors", model_name, exc)
        try:
            return build_embedder("glove")
        except Exception as exc:
            logger.warning("Word vectors unavailable (%s); falling back to LSA", exc)
            return LSAEmbedder(n_components=lsa_components)
    raise ValueError(f"Unknown embedder {name!r}; choose one of {EMBEDDER_CHOICES}")


def embedder_label(embedder: BaseEmbedder) -> str:
    """Short human-readable name for reports and API responses."""
    if isinstance(embedder, LSAEmbedder):
        return "lsa"
    if isinstance(embedder, HashEmbedder):
        return "hash"
    if isinstance(embedder, WordVectorEmbedder):
        return "glove"
    if isinstance(embedder, SentenceTransformerEmbedder):
        return "hash-fallback" if embedder._fallback_embedder is not None else embedder.model_name
    return type(embedder).__name__
