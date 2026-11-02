# Cross-Encoder Reranking in Retrieval Architectures

In multi-stage information retrieval systems, reranking serves as a high-precision refinement stage that reorders candidates extracted by fast first-stage retrievers.

## Bi-Encoder vs. Cross-Encoder Paradigm

Information retrieval systems balance computational latency against contextual representation fidelity:

- **Bi-Encoders**: Embed query $q$ and document $d$ completely independently into separate vectors $E(q)$ and $E(d)$. Because documents are pre-embedded offline and stored in vector indexes, query-time candidate retrieval requires only approximate nearest neighbor search taking sub-5 milliseconds. However, because query and document tokens never interact during feature extraction, subtle relational nuances can be lost.
- **Cross-Encoders**: Feed the concatenated query and document string `[CLS] Query [SEP] Document [SEP]` directly into a unified transformer backbone. All query tokens attend to all document tokens across every self-attention layer. This complete token-level cross-attention yields substantially higher relevance accuracy, but requires running the full transformer for each candidate pair at query time.

## Two-Stage Retrieval Architecture

To achieve the best speed-accuracy balance, production architectures deploy a two-stage cascade:

1. **Stage 1 (Retrieval)**: Fast bi-encoders and BM25 retrieve the top 50 to 100 candidate passages from millions of documents within single-digit milliseconds.
2. **Stage 2 (Reranking)**: A cross-encoder model (such as `cross-encoder/ms-marco-MiniLM-L-6-v2`) scores only the top candidate passages, outputting calibrated scalar relevance probabilities between 0 and 1.

The top 3 to 10 reranked documents are then passed to the downstream language model generator.

## Operational Considerations and Lazy Loading

Because cross-encoders introduce non-trivial computational overhead (often 20 to 80 ms per batch), architectures should treat rerankers as optional, lazily loaded components. Systems that do not have GPU acceleration or that require sub-10ms response times can bypass the reranking stage with minimal architectural disruption.
