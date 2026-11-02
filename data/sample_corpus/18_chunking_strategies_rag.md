# Chunking Strategies and Document Segmentation for RAG

Chunking is the process of partitioning unstructured source documents into contiguous text segments suitable for embedding and retrieval.

## The Granularity Dilemma

Choosing chunk size involves an inherent tradeoff:
- **Small Chunks (e.g. 128 to 256 characters)**: Produce tightly focused embeddings that match narrow keyword queries accurately. However, small chunks frequently lack surrounding context, resulting in fragmented snippets that fail to provide complete answers during LLM synthesis.
- **Large Chunks (e.g. 1024 to 2048 characters)**: Retain complete narrative reasoning and topical context. However, embeddings of broad passages tend to average together multiple disparate concepts, diluting specific semantic signals and lowering retrieval sensitivity.

## Common Chunking Paradigms

### 1. Fixed-Size Chunking with Overlap

The simplest approach divides text into blocks of $N$ characters or tokens with an overlap of $M$ characters ($M < N$). The overlap ensures that sentences or entities spanning the boundary between two adjacent chunks are not severed, preserving continuity across windows.

### 2. Recursive Character Splitting

Rather than splitting arbitrarily at fixed character boundaries, recursive splitters employ a hierarchy of semantic delimiters:
- Double newline (`\n\n`) to preserve paragraph boundaries.
- Single newline (`\n`) to preserve lists and lines.
- Sentence terminal punctuation (`. `, `? `, `! `) to avoid splitting mid-sentence.
- Whitespace (` `) to prevent bisecting individual words.

The algorithm attempts to split at the highest-level delimiter first, recursing to more granular separators only when an individual block exceeds the maximum chunk size limit.

### 3. Contextual and Document Structure-Aware Splitting

Advanced chunkers parse document syntax (e.g. Markdown headers `#`, `##` or HTML/PDF sections) to ensure that logical sections remain intact as self-contained semantic units.
