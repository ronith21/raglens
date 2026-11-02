# Hybrid Retrieval and Reciprocal Rank Fusion

Hybrid retrieval combines the orthogonal strengths of lexical search (BM25) and dense semantic vector search (FAISS) to achieve superior retrieval accuracy.

## Why Combine Lexical and Dense Search?

Dense vector embeddings excel at semantic paraphrasing, conceptual synonymy, and high-level thematic matching. However, dense retrieval frequently fails when users search for exact serial codes, acronyms, rare technical identifiers, or specialized entity names—domains where exact string matching in BM25 excels.

Conversely, BM25 fails completely when a user describes a concept using vocabulary that shares zero lexical overlap with the indexed document. A hybrid system harnesses both strategies.

## The Challenge of Score Normalization

A major hurdle in hybrid retrieval is that BM25 scores (unbounded positive floats) and cosine similarity scores (typically between -1.0 and 1.0) occupy entirely different distributions and scale ranges. Simple linear combination $\alpha S_{\text{dense}} + (1-\alpha) S_{\text{bm25}}$ is sensitive to out-of-distribution values and requires extensive tuning.

## Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion (Cormack et al., 2009) solves score normalization by operating solely on rank positions rather than arbitrary raw score magnitudes.

Given a set of rankings $R$ (e.g. $R = \{\text{dense}, \text{bm25}\}$), the RRF score of any document $d$ is defined as:

$$\text{RRF\_Score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $\text{rank}_r(d)$ is the 1-indexed position of document $d$ in the ranked results of retriever $r$, and $k$ is a constant smoothing hyperparameter (typically set to 60).

### Key Advantages of RRF

1. **Parameter Robustness**: The constant $k=60$ mitigates the disproportionate influence of high-ranking outliers while rewarding documents that appear near the top of both lists.
2. **Zero Calibration**: RRF requires no score scaling, normalization, or training data.
3. **Graceful Handling of Disjoint Results**: If a document appears in only one retriever's output, it still receives a score based on its single position without distorting comparisons.
