"""Benchmark script for measuring retrieval latency across BM25, Dense, and Hybrid retrievers."""

import time
from pathlib import Path

import numpy as np

from raglens.chunking.splitter import RecursiveCharacterChunker
from raglens.embed.hash import HashEmbedder
from raglens.ingest.parser import ingest_directory
from raglens.retrieve.bm25 import BM25Retriever
from raglens.retrieve.dense import DenseRetriever
from raglens.retrieve.hybrid import HybridRetriever


def run_benchmark() -> str:
    """Ingest sample corpus, chunk, index, and measure retrieval latencies."""
    corpus_dir = Path("data/sample_corpus")
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    docs = ingest_directory(corpus_dir)
    chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(docs)

    embedder = HashEmbedder(dimension=384)

    bm25 = BM25Retriever(chunks=chunks)
    dense = DenseRetriever(embedder=embedder, chunks=chunks)
    hybrid = HybridRetriever(embedder=embedder, chunks=chunks, rrf_k=60)

    test_queries = [
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

    repetitions = 50
    bm25_latencies: list[float] = []
    dense_latencies: list[float] = []
    hybrid_latencies: list[float] = []

    # Warmup
    for q in test_queries[:3]:
        bm25.retrieve(q, top_k=5)
        dense.retrieve(q, top_k=5)
        hybrid.retrieve(q, top_k=5)

    # Benchmark BM25
    for _ in range(repetitions):
        for q in test_queries:
            t0 = time.perf_counter()
            bm25.retrieve(q, top_k=5)
            bm25_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark Dense
    for _ in range(repetitions):
        for q in test_queries:
            t0 = time.perf_counter()
            dense.retrieve(q, top_k=5)
            dense_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark Hybrid
    for _ in range(repetitions):
        for q in test_queries:
            t0 = time.perf_counter()
            hybrid.retrieve(q, top_k=5)
            hybrid_latencies.append((time.perf_counter() - t0) * 1000.0)

    def stats(latencies: list[float]) -> tuple[float, float, float, float]:
        arr = np.array(latencies)
        return (
            float(np.mean(arr)),
            float(np.median(arr)),
            float(np.percentile(arr, 95)),
            float(np.percentile(arr, 99)),
        )

    b_mean, b_p50, b_p95, b_p99 = stats(bm25_latencies)
    d_mean, d_p50, d_p95, d_p99 = stats(dense_latencies)
    h_mean, h_p50, h_p95, h_p99 = stats(hybrid_latencies)

    header = (
        f"{'Retriever':<15} | {'Mean (ms)':<9} | {'p50 (ms)':<9} | "
        f"{'p95 (ms)':<9} | {'p99 (ms)':<9}"
    )
    row_b = f"{'BM25':<15} | {b_mean:<9.3f} | {b_p50:<9.3f} | {b_p95:<9.3f} | {b_p99:<9.3f}"
    row_d = (
        f"{'Dense (FAISS)':<15} | {d_mean:<9.3f} | {d_p50:<9.3f} | {d_p95:<9.3f} | {d_p99:<9.3f}"
    )
    row_h = f"{'Hybrid (RRF)':<15} | {h_mean:<9.3f} | {h_p50:<9.3f} | {h_p95:<9.3f} | {h_p99:<9.3f}"

    lines = [
        "=" * 67,
        "RAGLens Retrieval Latency Benchmark",
        "=" * 67,
        f"Corpus documents : {len(docs)}",
        f"Total chunks     : {len(chunks)}",
        f"Queries tested   : {len(test_queries)}",
        f"Repetitions      : {repetitions}",
        f"Total calls/mode : {len(test_queries) * repetitions}",
        "-" * 67,
        header,
        "-" * 67,
        row_b,
        row_d,
        row_h,
        "=" * 67,
    ]
    report = "\n".join(lines) + "\n"
    return report


if __name__ == "__main__":
    report_text = run_benchmark()
    print(report_text)
    out_path = Path("docs/proof/retrieval_bench.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Benchmark results written to {out_path}")
