# Retrieval-Augmented Generation: Concepts and Architecture

Retrieval-Augmented Generation (RAG) is an architectural pattern that combines parametric knowledge stored within large language model weights with non-parametric external knowledge retrieved dynamically from index stores.

## The Motivation for RAG

While modern LLMs exhibit impressive reasoning and generative fluency, they suffer from well-documented operational weaknesses:

1. **Hallucination**: Generating plausible-sounding yet factually incorrect or unsupported claims.
2. **Knowledge Cutoffs**: Inability to reflect events, discoveries, or data updates occurring after model pre-training.
3. **Lack of Provenance**: Inability to inspect, verify, or audit the underlying source material for a generated claim.
4. **Data Privacy**: Storing proprietary institutional knowledge directly in public model weights is insecure and financially costly.

RAG resolves these limitations by separating knowledge storage from reasoning capabilities. At inference time, the system retrieves relevant documents from an indexed corpus and injects them directly into the LLM context window alongside the user prompt.

## Core Stages of a RAG Pipeline

A production RAG workflow comprises four primary stages:

1. **Ingestion and Chunking**: External source materials (PDFs, Markdown notes, database records) are cleaned, extracted, and segmented into semantic chunks. Metadata such as document IDs, page offsets, and section headings are retained.
2. **Indexing**: Chunks are transformed into searchable formats, including inverted indices for lexical search and vector databases (such as FAISS) for dense semantic retrieval.
3. **Retrieval and Reranking**: For an incoming query, a hybrid retriever extracts candidate chunks using reciprocal rank fusion. A cross-encoder reranker rescores candidates to maximize precision.
4. **Augmented Generation**: Retrieved chunks are formatted into numbered contexts (e.g. `[1]`, `[2]`). The LLM is instructed to answer the prompt strictly based on the provided evidence, citing sources explicitly or issuing an explicit refusal if evidence is missing.
