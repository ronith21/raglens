"""Application engine: owns the index, retrievers, reranker and LLM used by CLI, API and UI."""

import logging
import threading
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.config import Settings
from raglens.embed.base import BaseEmbedder
from raglens.embed.hash import HashEmbedder
from raglens.embed.sentence_transformer import SentenceTransformerEmbedder
from raglens.ingest.models import Chunk
from raglens.ingest.parser import SUPPORTED_EXTENSIONS, ingest_directory, ingest_file
from raglens.llm import BaseLLM, create_llm
from raglens.pipeline.gate import LearnedGate, RefusalGate, RuleGate
from raglens.pipeline.rag import RAGAnswer, RAGPipeline
from raglens.rerank.cross_encoder import CrossEncoderReranker
from raglens.retrieve.base import BaseRetriever
from raglens.retrieve.hybrid import HybridRetriever
from raglens.retrieve.multi_query import MultiQueryRetriever, QueryRewriter
from raglens.retrieve.store import IndexStore

logger = logging.getLogger(__name__)

RETRIEVERS = ("bm25", "dense", "hybrid")


class DocumentInfo(BaseModel):
    """Summary of one indexed document."""

    doc_id: str
    title: str
    source_path: str
    num_chunks: int


class IngestReport(BaseModel):
    """Outcome of an ingest call."""

    documents_added: int
    documents_skipped: int
    chunks_added: int
    total_documents: int
    total_chunks: int


def load_gate(settings: Settings, embedder_name: str) -> RefusalGate:
    """Build the refusal gate selected by ``settings.refusal_gate``.

    ``learned`` requires a compatible model file; ``auto`` falls back to the rule gate (with a
    warning) when the file is missing or was trained with a different embedder.
    """
    mode = settings.refusal_gate.lower()
    if mode not in ("auto", "rule", "learned"):
        raise ValueError(f"REFUSAL_GATE must be auto, rule or learned, not {mode!r}")
    rule = RuleGate(settings.min_query_coverage)
    if mode == "rule":
        return rule
    path = Path(settings.gate_model_path)
    if not path.exists():
        if mode == "learned":
            raise FileNotFoundError(f"Gate model not found at {path}; run `make gate`")
        return rule
    gate = LearnedGate.load(path)
    if not gate.compatible_with(embedder_name):
        message = (
            f"Gate model was trained with {gate.model.embedder!r} but the embedder is "
            f"{embedder_name!r}; retrain it with `make gate`"
        )
        if mode == "learned":
            raise ValueError(message)
        logger.warning("%s. Using the rule gate.", message)
        return rule
    return gate


def create_embedder(settings: Settings) -> BaseEmbedder:
    """Return the configured embedder (hash embedder when requested or as a fallback)."""
    if settings.use_hash_embedder:
        return HashEmbedder(dimension=settings.hash_embed_dim)
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model, device=settings.embedding_device
    )


class RAGEngine:
    """Thread-safe facade over indexing and question answering."""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or create_embedder(settings)
        self._llm = llm
        self._reranker: CrossEncoderReranker | None = None
        self._rewriter: QueryRewriter | None = None
        self._gate: RefusalGate | None = None
        self._rewriter_chunks = -1
        self._lock = threading.RLock()
        self.chunker = RecursiveCharacterChunker(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.hybrid = HybridRetriever(
            embedder=self.embedder,
            rrf_k=settings.rrf_k,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
        )

    # ------------------------------------------------------------------ index lifecycle
    @property
    def chunks(self) -> list[Chunk]:
        """All indexed chunks."""
        return self.hybrid.chunks

    def load_or_build(self) -> None:
        """Load the persisted index, or build it from ``settings.corpus_dir`` and save it."""
        index_dir = Path(self.settings.index_dir)
        with self._lock:
            if IndexStore.exists(index_dir):
                self.hybrid = IndexStore.load(index_dir, embedder=self.embedder)
                logger.info("Loaded index with %d chunks from %s", len(self.chunks), index_dir)
                return
            corpus = Path(self.settings.corpus_dir)
            if corpus.is_dir():
                self.ingest_paths([corpus])
            else:
                logger.warning("Corpus directory %s not found; starting with empty index", corpus)

    def ingest_paths(self, paths: list[Path], rebuild: bool = False) -> IngestReport:
        """Ingest files and/or directories, re-index, and persist to ``settings.index_dir``.

        Documents whose ``doc_id`` is already indexed are skipped, so re-ingesting is idempotent.
        ``rebuild=True`` discards the existing index first.
        """
        documents = []
        for path in paths:
            if path.is_dir():
                documents.extend(ingest_directory(path))
            elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(ingest_file(path))
            else:
                raise ValueError(f"Unsupported path for ingestion: {path}")

        with self._lock:
            existing = [] if rebuild else list(self.chunks)
            known = {c.doc_id for c in existing}
            new_docs = [d for d in documents if d.doc_id not in known]
            new_chunks = self.chunker.chunk_documents(new_docs)
            all_chunks = existing + new_chunks
            if new_chunks or rebuild:
                self.hybrid.index_chunks(all_chunks)
                IndexStore.save(self.hybrid, self.settings.index_dir)
            return IngestReport(
                documents_added=len(new_docs),
                documents_skipped=len(documents) - len(new_docs),
                chunks_added=len(new_chunks),
                total_documents=len({c.doc_id for c in all_chunks}),
                total_chunks=len(all_chunks),
            )

    def documents(self) -> list[DocumentInfo]:
        """List indexed documents with their chunk counts, sorted by source path."""
        counts = Counter(c.doc_id for c in self.chunks)
        first: dict[str, Chunk] = {}
        for chunk in self.chunks:
            first.setdefault(chunk.doc_id, chunk)
        infos = [
            DocumentInfo(
                doc_id=doc_id,
                title=str(chunk.metadata.get("title", doc_id)),
                source_path=str(chunk.metadata.get("source_path", "")),
                num_chunks=counts[doc_id],
            )
            for doc_id, chunk in first.items()
        ]
        return sorted(infos, key=lambda d: d.source_path)

    # ------------------------------------------------------------------ components
    def retriever(self, name: str) -> BaseRetriever:
        """Return the named retriever (bm25, dense or hybrid) over the shared index."""
        mapping: dict[str, BaseRetriever] = {
            "bm25": self.hybrid.bm25_retriever,
            "dense": self.hybrid.dense_retriever,
            "hybrid": self.hybrid,
        }
        if name not in mapping:
            raise ValueError(f"Unknown retriever {name!r}; expected one of {RETRIEVERS}")
        return mapping[name]

    @property
    def rewriter(self) -> QueryRewriter:
        """Query rewriter whose acronym glossary is mined from the current index."""
        if self._rewriter is None or self._rewriter_chunks != len(self.chunks):
            self._rewriter = QueryRewriter.from_chunks(self.chunks)
            self._rewriter_chunks = len(self.chunks)
        return self._rewriter

    @property
    def gate(self) -> RefusalGate:
        """The configured refusal gate (see ``REFUSAL_GATE``), created on first use."""
        if self._gate is None:
            self._gate = load_gate(self.settings, self.embedder.name)
        return self._gate

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Lazily created cross-encoder reranker (pass-through if the model is unavailable)."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=self.settings.reranker_model)
        return self._reranker

    def llm(self, provider: str | None = None) -> BaseLLM:
        """The default LLM, or a fresh one for an explicitly requested provider."""
        if provider is None or (self._llm is not None and provider == self._llm.name):
            if self._llm is None:
                self._llm = create_llm(self.settings)
            return self._llm
        return create_llm(self.settings, provider)

    def pipeline(
        self,
        retriever: str = "hybrid",
        rerank: bool | None = None,
        top_k: int | None = None,
        llm_provider: str | None = None,
        multi_query: bool = False,
    ) -> RAGPipeline:
        """Assemble a pipeline for the requested configuration."""
        use_rerank = self.settings.enable_reranker if rerank is None else rerank
        base = self.retriever(retriever)
        if multi_query:
            base = MultiQueryRetriever(base, self.rewriter, rrf_k=self.settings.rrf_k)
        name = retriever + ("+mq" if multi_query else "") + ("+rerank" if use_rerank else "")
        return RAGPipeline(
            retriever=base,
            llm=self.llm(llm_provider),
            reranker=self.reranker if use_rerank else None,
            retriever_name=name,
            top_k=top_k or self.settings.default_top_k,
            rerank_candidates=self.settings.rerank_candidates,
            min_score=self.settings.min_relevance_threshold,
            min_query_coverage=self.settings.min_query_coverage,
            gate=self.gate,
            embedder=self.embedder,
        )

    def ask(
        self,
        question: str,
        retriever: str = "hybrid",
        rerank: bool | None = None,
        top_k: int | None = None,
        llm_provider: str | None = None,
        multi_query: bool = False,
    ) -> RAGAnswer:
        """Answer a question with the requested configuration."""
        with self._lock:
            pipe = self.pipeline(retriever, rerank, top_k, llm_provider, multi_query)
            return pipe.ask(question)
