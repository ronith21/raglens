# Quantization and Efficient Inference for Deep Learning Models

Deploying modern deep learning models—including transformer-based bi-encoders and generative LLMs—demands substantial computational memory and matrix multiplication bandwidth. Model quantization reduces this resource footprint.

## Fundamentals of Numerical Precision

Neural network weights and activations are traditionally stored as 32-bit single-precision floating-point numbers (`FP32`). Quantization maps continuous floating-point values into lower-bit discrete representations:

- `FP16` and `BF16` (16-bit floats): Retain high dynamic range and are supported natively by modern tensor cores, halving memory requirements with negligible loss in accuracy.
- `INT8` (8-bit integer): Maps weights into the range $[-128, 127]$ using scale factors $S$ and zero-points $Z$:
  $$x_{\text{int8}} = \text{round}\left(\frac{x}{S}\right) + Z$$
- `INT4` (4-bit integer): Enables massive compression, allowing 70-billion-parameter LLMs to fit into 40 GB of VRAM.

## Post-Training Quantization (PTQ) vs. QAT

- **Post-Training Quantization (PTQ)**: Converts a previously trained model directly to lower precision using a small calibration dataset to establish activation scales. PTQ requires minimal compute and no gradient retraining.
- **Quantization-Aware Training (QAT)**: Simulates quantization errors during the forward pass of model training, allowing backpropagation gradients to adjust weights to compensate for precision loss.

## Memory and Cache Optimizations: FlashAttention and KV-Cache

In addition to weight quantization, inference systems optimize memory access patterns during generation:
- **KV-Cache**: Stores previous key and value tensor representations in memory across autoregressive decoding steps, avoiding redundant re-computation of past token representations.
- **FlashAttention**: Reorders the attention computation into tiled SRAM blocks, minimizing high-latency reads and writes to GPU High-Bandwidth Memory (HBM) and yielding 2x to 4x throughput improvements.
