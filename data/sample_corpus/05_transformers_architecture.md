# The Transformer Architecture in Natural Language Processing

The Transformer architecture, introduced by Vaswani et al. in 2017, completely transformed sequence modeling by replacing recurrent neural networks (RNNs) and convolutional layers with stacked multi-head self-attention mechanisms.

## Limitations of Recurrence

Traditional recurrent networks like LSTMs and GRUs process sequential data step-by-step, updating an internal hidden state $h_t = f(h_{t-1}, x_t)$. This sequential dependency creates two fundamental bottlenecks:

1. **Inability to parallelize**: Training computations cannot be parallelized across the temporal sequence dimension, imposing steep wall-clock training times on large texts.
2. **Gradient degradation**: Despite gating mechanisms, path lengths between distant sequence tokens remain $O(N)$, causing signal dissipation over long document contexts.

Transformers circumvent recurrence by establishing direct $O(1)$ pairwise attention connections between all tokens in a sequence, unlocking massive horizontal hardware acceleration on modern GPU/TPU clusters.

## The Encoder-Decoder Structure

The canonical transformer consists of two primary modules:

- **Encoder**: Receives input tokens, incorporates positional representations, and processes them through $N$ identical stacked layers. Each layer contains a multi-head self-attention block followed by a position-wise feed-forward network, accompanied by residual skip-connections and layer normalization.
- **Decoder**: Generates output tokens auto-regressively. In addition to multi-head self-attention (which is causally masked to prevent attending to future tokens), the decoder incorporates cross-attention sub-layers that condition predictions on the encoded input representations.

Modern language modeling has diverged into distinct architectural paradigms derived from this template:
- Encoder-only models (e.g. BERT, RoBERTa) optimized for dense sequence representation and classification.
- Decoder-only models (e.g. GPT series, LLaMA, Mistral) optimized for generative autoregressive next-token prediction.
- Encoder-decoder models (e.g. T5, BART) optimized for sequence-to-sequence translation and abstractive summarization.
