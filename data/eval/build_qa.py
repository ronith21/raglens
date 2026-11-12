"""Writes data/eval/qa.jsonl. Kept as code so the labelled set is reviewable in one place.

type: lexical    -> question reuses the document's own key terms
      paraphrase -> question deliberately avoids the document's key terms
      unanswerable -> answer is not in the corpus (off-topic or near-miss in-domain)
split: dev rows tune the refusal threshold; test rows are only used for the final numbers.
"""

import json
from pathlib import Path

ROWS: list[tuple[str, str, str, list[str]]] = [
    # (type, question, reference answer, relevant documents)
    (
        "lexical",
        "What is the update rule of gradient descent?",
        "theta_{t+1} = theta_t - eta times the gradient of the loss J(theta_t), where eta is the learning rate.",
        ["01_gradient_descent"],
    ),
    (
        "lexical",
        "How does mini-batch gradient descent differ from stochastic gradient descent?",
        "SGD uses a single random observation per step; mini-batch uses batches of typically 32 to 512 observations, balancing stability and efficiency.",
        ["01_gradient_descent"],
    ),
    (
        "lexical",
        "What does AdamW change compared to Adam?",
        "AdamW decouples L2 weight decay from the adaptive gradient moment updates, improving generalization.",
        ["01_gradient_descent"],
    ),
    (
        "lexical",
        "What is the difference between L1 and L2 regularization?",
        "L2 (ridge) penalizes squared weights and shrinks them toward zero without sparsity; L1 (lasso) drives non-essential coefficients exactly to zero, acting as feature selection.",
        ["02_overfitting_and_regularization"],
    ),
    (
        "lexical",
        "How does dropout behave during inference?",
        "Dropout is disabled at inference and activations are scaled by (1 - p) to preserve the expected magnitude.",
        ["02_overfitting_and_regularization"],
    ),
    (
        "lexical",
        "What are the three components of the bias-variance decomposition of expected mean squared error?",
        "Squared bias, variance and irreducible noise sigma squared.",
        ["03_bias_variance_tradeoff"],
    ),
    (
        "lexical",
        "How do bagging and boosting manipulate the bias-variance tradeoff?",
        "Bagging such as random forests reduces variance through averaging; boosting such as gradient boosted trees iteratively reduces bias.",
        ["03_bias_variance_tradeoff", "04_decision_trees_and_ensembles"],
    ),
    (
        "lexical",
        "What is the Gini impurity formula used in decision trees?",
        "Gini(S) = 1 minus the sum over classes of p_k squared.",
        ["04_decision_trees_and_ensembles"],
    ),
    (
        "lexical",
        "How do random forests decorrelate their trees?",
        "At each split only a random subset of about sqrt(p) features is considered, decorrelating the trees and reducing ensemble variance.",
        ["04_decision_trees_and_ensembles"],
    ),
    (
        "lexical",
        "What do XGBoost and LightGBM add to gradient boosting?",
        "Second-order Taylor expansions of the loss, histogram-based split finding, feature sub-sampling and explicit leaf regularization.",
        ["04_decision_trees_and_ensembles"],
    ),
    (
        "lexical",
        "What are the two bottlenecks of recurrent networks that transformers remove?",
        "Recurrence cannot be parallelized across the sequence, and gradients degrade over O(N) path lengths.",
        ["05_transformers_architecture"],
    ),
    (
        "lexical",
        "Which models are encoder-only, decoder-only and encoder-decoder transformers?",
        "Encoder-only: BERT, RoBERTa; decoder-only: GPT, LLaMA, Mistral; encoder-decoder: T5, BART.",
        ["05_transformers_architecture"],
    ),
    (
        "lexical",
        "Why are dot products divided by the square root of d_k in scaled dot-product attention?",
        "To avoid pushing the softmax into regions with vanishingly small gradients when the inner dimension is large.",
        ["06_attention_mechanisms"],
    ),
    (
        "lexical",
        "What are rotary position embeddings (RoPE)?",
        "RoPE incorporates relative token distances into query-key inner products through rotations, improving length extrapolation.",
        ["06_attention_mechanisms"],
    ),
    (
        "lexical",
        "How does Byte-Pair Encoding build its vocabulary?",
        "BPE starts from characters and repeatedly merges the most frequent adjacent symbol pair into a new token until the vocabulary size limit is reached.",
        ["07_tokenization_algorithms"],
    ),
    (
        "lexical",
        "How does WordPiece choose which pair to merge?",
        "It merges the pair that maximizes the likelihood of the training data under a unigram language model.",
        ["07_tokenization_algorithms"],
    ),
    (
        "lexical",
        "How are sentence embeddings pooled in a bi-encoder?",
        "Final-layer token representations are mean pooled into a fixed-length vector which is then normalized to unit length.",
        ["08_vector_embeddings"],
    ),
    (
        "lexical",
        "What is the Multiple Negatives Ranking Loss used for?",
        "It is a contrastive objective that trains sentence embeddings by pulling positive query-document pairs together and pushing in-batch negatives apart.",
        ["08_vector_embeddings"],
    ),
    (
        "lexical",
        "What are the four stages of a RAG pipeline?",
        "Ingestion and chunking, indexing, retrieval and reranking, and augmented generation.",
        ["09_retrieval_augmented_generation"],
    ),
    (
        "lexical",
        "How is Mean Reciprocal Rank computed?",
        "The average over queries of 1 divided by the rank of the first relevant document, with 0 when none is retrieved.",
        ["10_retrieval_evaluation_metrics"],
    ),
    (
        "lexical",
        "What is the difference between Hit@k and Recall@k?",
        "Hit@k is the share of queries with at least one relevant document in the top k; Recall@k is the share of all relevant documents captured in the top k.",
        ["10_retrieval_evaluation_metrics"],
    ),
    (
        "lexical",
        "What do the k1 and b parameters control in BM25?",
        "k1 controls term-frequency saturation and b controls document length normalization.",
        ["11_lexical_search_bm25"],
    ),
    (
        "lexical",
        "What is an inverted index?",
        "A dictionary mapping each vocabulary token to a postings list of the documents and term frequencies where it occurs.",
        ["11_lexical_search_bm25"],
    ),
    (
        "lexical",
        "What is the Reciprocal Rank Fusion formula?",
        "The RRF score of a document is the sum over retrievers of 1 / (k + rank), with k typically 60.",
        ["12_hybrid_retrieval_rrf"],
    ),
    (
        "lexical",
        "Why does hybrid retrieval avoid linear combination of BM25 and cosine scores?",
        "BM25 scores are unbounded while cosine scores lie in -1 to 1, so a linear mix is sensitive to scale and needs tuning; RRF uses only ranks.",
        ["12_hybrid_retrieval_rrf"],
    ),
    (
        "lexical",
        "What is the difference between a bi-encoder and a cross-encoder?",
        "Bi-encoders embed query and document independently so documents can be pre-indexed; cross-encoders read the concatenated pair with full cross-attention, more accurate but run per pair at query time.",
        ["13_cross_encoder_reranking", "08_vector_embeddings"],
    ),
    (
        "lexical",
        "Why is accuracy misleading for imbalanced datasets?",
        "With 99.5 percent negatives, always predicting negative gives 99.5 percent accuracy, so precision, recall, F1 and PR-AUC should be used.",
        ["14_class_imbalance_strategies"],
    ),
    (
        "lexical",
        "How does SMOTE create new samples?",
        "It interpolates new minority points along line segments between a minority instance and its nearest neighbours.",
        ["14_class_imbalance_strategies"],
    ),
    (
        "lexical",
        "How do Isolation Forests detect anomalies?",
        "They build random trees with random splits; anomalies are isolated with fewer splits, giving shorter average path lengths.",
        ["15_anomaly_detection"],
    ),
    (
        "lexical",
        "What is a contextual anomaly?",
        "A point that is abnormal only in a specific context, such as 30 degrees C being normal in July but anomalous in January.",
        ["15_anomaly_detection"],
    ),
    (
        "lexical",
        "What does DVC track in MLOps?",
        "DVC versions datasets, feature stores and model weights alongside code commits for reproducibility.",
        ["16_mlops_production_systems"],
    ),
    (
        "lexical",
        "How does HNSW search work?",
        "HNSW builds a multi-layer graph: sparse long-range links in upper layers for fast greedy navigation and dense local links in lower layers for fine convergence.",
        ["17_vector_databases_and_ann"],
    ),
    (
        "lexical",
        "What does product quantization do in IVF-PQ?",
        "It splits vectors into sub-vectors and quantizes each to codebook centroids, cutting memory by up to 95 percent.",
        ["17_vector_databases_and_ann"],
    ),
    (
        "lexical",
        "What separators does a recursive character splitter use?",
        "Double newline, single newline, sentence punctuation and whitespace, from coarsest to finest.",
        ["18_chunking_strategies_rag"],
    ),
    (
        "lexical",
        "What is the difference between intrinsic and extrinsic hallucination?",
        "Intrinsic hallucination contradicts the input context; extrinsic hallucination adds claims absent from the context.",
        ["19_hallucination_and_groundedness"],
    ),
    (
        "lexical",
        "What is the difference between post-training quantization and quantization-aware training?",
        "PTQ converts a trained model using a small calibration set without retraining; QAT simulates quantization during training so weights adapt.",
        ["20_quantization_and_inference"],
    ),
    (
        "lexical",
        "What does the KV-cache store during generation?",
        "Previous key and value tensors across decoding steps, avoiding recomputation.",
        ["20_quantization_and_inference"],
    ),
    # Paraphrases: the question avoids the document's key terms on purpose.
    (
        "paraphrase",
        "How do I stop a model from memorising its training examples instead of learning general patterns?",
        "Use regularization such as L1/L2 penalties, dropout, early stopping and data augmentation to reduce overfitting.",
        ["02_overfitting_and_regularization"],
    ),
    (
        "paraphrase",
        "Why might my training run blow up if each update step is too big?",
        "A learning rate that is too large makes updates overshoot minima, oscillate or diverge.",
        ["01_gradient_descent"],
    ),
    (
        "paraphrase",
        "How can text be split into pieces when a word was never seen before?",
        "Subword tokenization such as BPE breaks rare words into smaller known units, avoiding out-of-vocabulary tokens.",
        ["07_tokenization_algorithms"],
    ),
    (
        "paraphrase",
        "Why can a search for doctor miss a page that only says physician?",
        "Sparse lexical representations treat synonyms as orthogonal, so their inner product is zero; dense embeddings capture the similarity.",
        ["08_vector_embeddings"],
    ),
    (
        "paraphrase",
        "How can the system avoid answering when the documents do not contain the answer?",
        "Score gating and refusal: if retrieval confidence is below a threshold, return an explicit abstention instead of generating.",
        ["19_hallucination_and_groundedness", "09_retrieval_augmented_generation"],
    ),
    (
        "paraphrase",
        "How can a huge collection of vectors be searched quickly if a few misses are acceptable?",
        "Approximate nearest neighbour indexes such as HNSW or IVF-PQ trade a little recall for logarithmic or constant query time.",
        ["17_vector_databases_and_ann"],
    ),
    (
        "paraphrase",
        "How can I shrink a big language model so it fits on a smaller GPU?",
        "Quantize weights to lower precision such as FP16, INT8 or INT4.",
        ["20_quantization_and_inference"],
    ),
    (
        "paraphrase",
        "What should I watch for after a model is live and the incoming data slowly changes?",
        "Monitor for concept drift and covariate shift with statistics such as the KS test or PSI and trigger alerts or retraining.",
        ["16_mlops_production_systems"],
    ),
    (
        "paraphrase",
        "How should I judge a classifier when positives are extremely rare?",
        "Use precision, recall, F1 and PR-AUC rather than accuracy, and consider resampling or class weights.",
        ["14_class_imbalance_strategies"],
    ),
    (
        "paraphrase",
        "How does each word decide which other words in the sentence to focus on?",
        "Self-attention computes query-key dot products, scales them, applies softmax and takes a weighted sum of values.",
        ["06_attention_mechanisms"],
    ),
    (
        "paraphrase",
        "How can a neural network flag unusual inputs by trying to rebuild them?",
        "An autoencoder trained on normal data reconstructs anomalies poorly, so high reconstruction error signals an anomaly.",
        ["15_anomaly_detection"],
    ),
    (
        "paraphrase",
        "How can a second, slower model reorder the first-stage search results more accurately?",
        "A cross-encoder reranker scores each query-passage pair jointly and reorders the top candidates.",
        ["13_cross_encoder_reranking"],
    ),
    # Unanswerable: the corpus does not contain the answer.
    ("unanswerable", "What is the capital of Australia?", "", []),
    ("unanswerable", "Who won the 2018 FIFA World Cup?", "", []),
    ("unanswerable", "What is a good recipe for sourdough bread?", "", []),
    ("unanswerable", "How many moons does Jupiter have?", "", []),
    ("unanswerable", "What is the current price of Bitcoin?", "", []),
    ("unanswerable", "How much GPU memory did training GPT-4 require?", "", []),
    (
        "unanswerable",
        "What learning rate did the original BERT paper use for fine-tuning on SQuAD?",
        "",
        [],
    ),
    ("unanswerable", "Which vector database has the cheapest managed cloud pricing?", "", []),
    ("unanswerable", "What is the licence of the LLaMA 3 model weights?", "", []),
    ("unanswerable", "How does the Kubernetes scheduler assign pods to nodes?", "", []),
]


def main() -> None:
    """Write the JSONL file with alternating dev/test splits within each question type."""
    counters: dict[str, int] = {}
    out = Path(__file__).with_name("qa.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for i, (qtype, question, answer, docs) in enumerate(ROWS, start=1):
            n = counters.get(qtype, 0)
            counters[qtype] = n + 1
            row = {
                "id": f"q{i:02d}",
                "type": qtype,
                "split": "dev" if n % 2 == 0 else "test",
                "question": question,
                "answer": answer,
                "relevant_doc_ids": docs,
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(ROWS)} rows to {out}")


if __name__ == "__main__":
    main()
