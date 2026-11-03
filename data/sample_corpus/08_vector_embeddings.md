# Dense Vector Embeddings and Semantic Representation

Dense vector embeddings map discrete entities—such as words, sentences, code snippets, or documents—into continuous, high-dimensional vector spaces where geometric proximity reflects semantic similarity.

## From Sparse to Dense Representations

Traditional information retrieval historically relied on sparse term incidence vectors, such as One-Hot encodings or Bag-of-Words matrices weighted by Term Frequency-Inverse Document Frequency (TF-IDF). While computationally sparse and interpretable, these representations treat synonyms as completely orthogonal: the terms "physician" and "doctor" have an inner product of zero in standard lexical space.

Dense embeddings project text into continuous spaces (typically ranging from 384 to 1536 dimensions). In these continuous geometries, distributed semantic features allow cosine similarity to capture conceptual equivalence regardless of lexical overlap.

## Bi-Encoder Architecture for Sentence Embeddings

Sentence Transformers (such as `all-MiniLM-L6-v2`) commonly adopt a bi-encoder architecture:

1. A shared transformer backbone independently ingests input strings.
2. Token representations from the final transformer layer are pooled (most commonly via mean pooling across non-padding token vectors) into a fixed-length sentence embedding $u \in \mathbb{R}^d$.
3. The embedding is normalized to unit length: $\hat{u} = \frac{u}{\|u\|_2}$, enabling cosine similarity to be computed efficiently via a single dot product: $\text{sim}(u, v) = \hat{u} \cdot \hat{v}$.

## Contrastive Loss and Training

Modern sentence embeddings are trained using contrastive objectives, such as Multiple Negatives Ranking Loss (MNRL) or InfoNCE:

$$\mathcal{L} = - \log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(q_i, d_j) / \tau)}$$

where $(q_i, d_i^+)$ represents a positive query-document pair, while other in-batch passages serve as negative samples. This pushes true semantic pairs close together while repelling unrelated contexts across the embedding manifold.
