# Information Retrieval Evaluation Metrics

Evaluating the quality of retrieval systems requires quantitative metrics that capture both the presence and the ranking position of relevant documents.

## Hit Rate and Recall@k

Given a set of queries $Q$, let $R_q$ denote the set of ground-truth relevant documents for query $q$, and let $D_{q, k}$ represent the top-$k$ retrieved documents.

- **Hit@k**: Evaluates the binary proportion of queries for which at least one relevant document appears within the top $k$ positions:
  $$\text{Hit@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(|R_q \cap D_{q, k}| > 0)$$
- **Recall@k**: Measures the proportion of all relevant documents captured in the top $k$ results:
  $$\text{Recall@k} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|R_q \cap D_{q, k}|}{|R_q|}$$

## Mean Reciprocal Rank (MRR)

Mean Reciprocal Rank evaluates how high up the first relevant document is ranked. Let $\text{rank}_q$ denote the 1-based index of the first relevant document retrieved for query $q$:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

If no relevant document is retrieved within the evaluated window, the reciprocal rank is assigned a value of 0. MRR is especially suited for single-answer search tasks and question answering where user satisfaction depends primarily on encountering the first correct answer immediately.

## Normalized Discounted Cumulative Gain (nDCG@k)

Discounted Cumulative Gain (DCG) accounts for graded relevance scores $rel_i$ and penalizes relevant documents that appear further down the ranked list using logarithmic decay:

$$\text{DCG@k} = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

To normalize the metric between 0 and 1 across diverse queries, DCG@k is divided by the Ideal DCG (IDCG@k), which is the DCG achieved by an optimal ranking where documents are sorted in descending order of relevance:

$$\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$
