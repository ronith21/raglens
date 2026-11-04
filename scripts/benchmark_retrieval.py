"""Benchmark retrieval latency across BM25, Dense, and Hybrid retrievers.

Usage: python scripts/benchmark_retrieval.py  (writes docs/proof/retrieval_bench.txt)
"""

import platform
import time
from pathlib import Path

import numpy as np

from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.embed.hash import HashEmbedder
from raglens.ingest.parser import ingest_directory
from raglens.retrieve.base import BaseRetriever
from raglens.retrieve.hybrid import HybridRetriever

TEST_QUERIES = [
    "What is the mathematical formulation of gradient descent?",
    "How does L2 regularization differ from L1 lasso regularization?",
    "Explain the bias-variance tradeoff in terms of expected mean squared error.",
    "How do decision trees and random forests prevent overfitting?",
    "What are the core components of the Transformer architecture?",
    "How does multi-head self-attention compute query key value projections?",
    "Explain Byte-Pair Encoding and subword tokenization algorithms.",
    "What is the difference between bi-encoder and cross-encoder architectures?",
    "How does Reciprocal Rank Fusion combine BM25 and vector search rankings?",
    "What are the metrics Hit@k, Recall@k, and nDCG for evaluation?",
    "How do Isolation Forests detect point anomalies in high-dimensional data?",
    "What are the principles of MLOps for model monitoring and concept drift?",
    "Explain Hierarchical Navigable Small World graphs for vector search.",
    "What strategies mitigate extrinsic hallucination in RAG pipelines?",
    "How does model quantization to INT8 or INT4 preserve memory?",
]
REPETITIONS = 50


def _time_retriever(retriever: BaseRetriever) -> list[float]:
    """Latency in ms for every (repetition, query) pair."""
    latencies: list[float] = []
    for _ in range(REPETITIONS):
        for query in TEST_QUERIES:
            t0 = time.perf_counter()
            retriever.retrieve(query, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def run_benchmark(corpus_dir: Path = Path("data/sample_corpus")) -> str:
    """Ingest the sample corpus, chunk, index, and measure retrieval latencies."""
    docs = ingest_directory(corpus_dir)
    chunks = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=64).chunk_documents(docs)
    hybrid = HybridRetriever(embedder=HashEmbedder(dimension=384), chunks=chunks, rrf_k=60)
    retrievers: dict[str, BaseRetriever] = {
        "BM25": hybrid.bm25_retriever,
        "Dense (FAISS)": hybrid.dense_retriever,
        "Hybrid (RRF)": hybrid,
    }

    for retriever in retrievers.values():  # warm-up
        for query in TEST_QUERIES[:3]:
            retriever.retrieve(query, top_k=5)

    columns = ["Mean (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)"]
    header = f"{'Retriever':<15} | " + " | ".join(f"{c:<9}" for c in columns)
    rows = []
    for name, retriever in retrievers.items():
        arr = np.array(_time_retriever(retriever))
        rows.append(
            f"{name:<15} | {arr.mean():<9.3f} | {np.median(arr):<9.3f} | "
            f"{np.percentile(arr, 95):<9.3f} | {np.percentile(arr, 99):<9.3f}"
        )

    lines = [
        "=" * 67,
        "RAGLens Retrieval Latency Benchmark",
        "=" * 67,
        f"Platform         : Python {platform.python_version()} on {platform.machine()}",
        "Embedder         : HashEmbedder (dim=384)",
        f"Corpus documents : {len(docs)}",
        f"Total chunks     : {len(chunks)}",
        f"Queries tested   : {len(TEST_QUERIES)}",
        f"Repetitions      : {REPETITIONS}",
        f"Total calls/mode : {len(TEST_QUERIES) * REPETITIONS}",
        "-" * 67,
        header,
        "-" * 67,
        *rows,
        "=" * 67,
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    report_text = run_benchmark()
    print(report_text)
    out_path = Path("docs/proof/retrieval_bench.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Benchmark results written to {out_path}")
