# Lexical Search, BM25, and Inverted Indices

Lexical search relies on exact token matching between user queries and indexed document passages. BM25 (Best Matching 25) remains the dominant probabilistic term-matching benchmark.

## Probabilistic Relevance and the Inverted Index

Lexical retrieval systems structure data using an inverted index—a dictionary mapping each vocabulary token to a postings list containing the document IDs and term frequencies where that word occurs. This structure allows queries with multiple words to be resolved in milliseconds through intersection or union operations on postings lists.

## The Okapi BM25 Scoring Function

Given a query $Q$ composed of keywords $q_1, q_2, \dots, q_n$, the BM25 score of a document $D$ is computed as:

$$\text{BM25}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

where:
- $f(q_i, D)$ is the term frequency of token $q_i$ in document $D$.
- $|D|$ represents the character or token length of document $D$, and $\text{avgdl}$ is the average document length across the entire corpus.
- $k_1$ is a positive tuning parameter (typically between 1.2 and 2.0) that controls term frequency saturation. Unlike raw TF, as $f(q_i, D)$ grows large, its marginal contribution asymptotically approaches $(k_1 + 1)$.
- $b$ is a tuning parameter (typically 0.75) that governs document length penalization. When $b=1$, the model completely scales term frequencies relative to document length; when $b=0$, length normalization is disabled.

## Inverse Document Frequency (IDF)

The IDF weight penalizes terms that appear ubiquitously across the corpus (such as "the", "system", "data") while prioritizing discriminative, rare words (such as "hyperparameter" or "eigenvalue"):

$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$

where $N$ is total documents and $n(q_i)$ is the number of documents containing term $q_i$.
