# Tokenization Algorithms in Modern NLP

Tokenization serves as the bridge between raw textual characters and discrete numerical identifiers ingested by machine learning models.

## Evolution of Tokenization

Early NLP pipelines relied on rule-based word-level tokenization or character-level representations. Word-level tokenization struggles with out-of-vocabulary (OOV) terms, morphological variations, and large vocabulary tables. Conversely, pure character-level tokenization expands sequence lengths excessively, severely stressing the quadratic complexity $O(N^2)$ of self-attention.

Subword tokenization addresses both issues by breaking frequent words into coherent whole units while decomposing rare or unfamiliar terminology into constituent character chunks.

## Dominant Subword Algorithms

### Byte-Pair Encoding (BPE)

Originally an algorithmic data compression scheme, Byte-Pair Encoding was adapted for NLP by Sennrich et al. (2016). BPE begins with a base vocabulary of individual characters. The algorithm scans the corpus to identify the most frequently occurring pair of adjacent symbols and iteratively merges them into a new single token. This process repeats until the vocabulary reaches a predetermined size limit. Modern implementations (such as GPT-2/3/4) operate directly on UTF-8 bytes rather than Unicode characters, completely eliminating out-of-vocabulary tokens.

### WordPiece

Used extensively in BERT, WordPiece resembles BPE in vocabulary construction but alters the merge criterion. Rather than choosing the most frequent symbol pair, WordPiece merges the pair that maximizes the likelihood of the training data under a unigram language model, favoring pairs whose co-occurrence significantly exceeds what would be expected by random chance.

### SentencePiece and Unigram Language Model

SentencePiece treats raw text directly as a continuous stream of characters, treating whitespaces as explicit symbols (often encoded as `_`). It trains tokenizers without language-dependent pre-segmentation, making it suitable for languages without whitespace delimiters (such as Japanese and Chinese). The Unigram tokenization algorithm starts with an over-complete candidate vocabulary and iteratively prunes tokens whose removal minimizes the increase in corpus loss.
