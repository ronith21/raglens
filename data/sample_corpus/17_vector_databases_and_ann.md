# Vector Databases and Approximate Nearest Neighbor Search

Dense vector search requires matching a high-dimensional query vector against millions of candidate document vectors in sub-second timeframes.

## Exact vs. Approximate Nearest Neighbors (ANN)

Exact k-nearest neighbor search calculates the inner product or Euclidean distance between the query vector $q \in \mathbb{R}^d$ and every vector in dataset $X \in \mathbb{R}^{N \times d}$. This exhaustive brute-force scan requires $O(N \cdot d)$ floating-point operations. While mathematically exact, this linear complexity becomes unviable for production web applications where $N > 10^6$.

Approximate Nearest Neighbor (ANN) algorithms sacrifice strict 100% recall guarantees in exchange for logarithmic $O(\log N)$ or constant $O(1)$ query latencies, routinely returning 95-99% of the true nearest neighbors in single-digit milliseconds.

## Prominent ANN Indexing Paradigms

### Inverted File Index with Vector Quantization (IVF-PQ)

- **IVF (Inverted File)**: Partitions the vector space into $K$ Voronoi cells using k-means clustering. At query time, only the centroids closest to the query are inspected, pruning the majority of irrelevant database clusters.
- **PQ (Product Quantization)**: Compresses high-dimensional vectors by breaking them into $m$ orthogonal sub-vectors and quantizing each sub-vector into codebook centroids. This reduces memory footprint by up to 95% and enables fast distance computations using lookup tables.

### Hierarchical Navigable Small World (HNSW)

HNSW is a graph-based indexing algorithm inspired by skip lists. It constructs a multi-layer geometric graph where upper layers contain sparse long-range connections for fast greedy navigation across large vector distances, while lower layers contain dense local connections for fine-grained convergence. HNSW offers superior recall-speed tradeoffs for in-memory indexing, making it the industry standard in vector engines like FAISS, Weaviate, and Milvus.
