# Self-Attention Mechanisms and Multi-Head Computation

Self-attention is the computational core of transformer architectures, enabling every token in an input sequence to dynamically route information from every other token based on semantic relevance.

## Scaled Dot-Product Attention

Given an input matrix $X \in \mathbb{R}^{N \times d_{\text{model}}}$, linear projection matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ map the representations into Queries ($Q$), Keys ($K$), and Values ($V$):

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

The compatibility score between query tokens and key tokens is computed via matrix dot product. To avoid pushing softmax outputs into regions of vanishingly small gradients when inner dimensions are large, the dot products are divided by the scaling factor $\sqrt{d_k}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

The softmax operation converts raw affinity logits into a normalized probability distribution across the sequence. The resulting attention weights dictate how much each token aggregates information from corresponding value vectors.

## Multi-Head Attention

A single attention mechanism might average together conflicting syntactic and semantic relationships. Multi-Head Attention mitigates this by projecting $Q$, $K$, and $V$ into $h$ distinct lower-dimensional subspaces of dimension $d_k = d_{\text{model}} / h$:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$

where each individual head is parameterized as:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

This multi-head formulation allows the network to simultaneously attend to disparate linguistic features—such as coreference resolution, subject-verb agreement, and topical association—across distinct subspaces.

## Positional Encodings and RoPE

Because dot-product attention is inherently permutation-invariant, positional information must be injected explicitly. Early architectures utilized static sinusoidal positional embeddings. Modern architectures favor relative or rotary position embeddings (RoPE), which incorporate relative token distances directly into query-key inner products through rotational complex matrices, enhancing length extrapolation and context handling.
