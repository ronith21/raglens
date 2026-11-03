# Class Imbalance and Resampling Strategies

Supervised learning algorithms typically assume balanced class distributions. However, real-world domains—such as credit card fraud detection, medical anomaly diagnosis, and rare event prediction—frequently exhibit extreme class imbalance, where the positive class comprises less than 1% of total observations.

## The Pitfalls of Accuracy in Imbalanced Regimes

When a dataset contains 99.5% negative samples and 0.5% positive samples, a trivial model that constantly predicts the negative class achieves 99.5% classification accuracy. Consequently, accuracy is fundamentally misleading. Model evaluation must instead utilize:

- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (proportion of predicted positives that are true positives).
- **Recall / Sensitivity**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (proportion of actual positives successfully captured).
- **F1 Score**: Harmonic mean of precision and recall.
- **PR-AUC**: Area under the Precision-Recall curve, which is more informative than ROC-AUC when positive instances are extremely sparse.

## Algorithmic and Resampling Solutions

Strategies for handling imbalanced datasets operate at either the data level or the algorithm level:

### Data-Level Approaches

1. **Random Undersampling**: Discards randomly chosen observations from the majority class until parity is reached. While computationally fast, this risks discarding valuable decision-boundary information.
2. **Random Oversampling**: Duplicates instances of the minority class. This can induce overfitting to identical duplicated examples.
3. **SMOTE (Synthetic Minority Over-sampling Technique)**: Synthesizes novel minority instances by identifying nearest neighbors in feature space and interpolating new points along the connecting line segments.

### Cost-Sensitive and Algorithmic Approaches

1. **Class-Weighted Loss Functions**: Scales the loss contributed by minority class errors inversely proportional to class frequency:
   $$w_c = \frac{N}{K \cdot N_c}$$
2. **Focal Loss**: Modifies cross-entropy by introducing a modulating factor $(1 - p_t)^\gamma$ to downweight easy, well-classified examples and focus training gradients on difficult, ambiguous samples.
