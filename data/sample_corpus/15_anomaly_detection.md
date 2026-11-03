# Anomaly Detection Methodologies and Applications

Anomaly detection identifies observations or patterns that deviate significantly from the expected behavior established by the majority of a dataset.

## Taxonomy of Anomalies

Anomalies are broadly categorized into three types:

1. **Point Anomalies**: A single data point lies far outside the distribution of the rest of the data (e.g. an individual credit transaction of $50,000 when normal transactions average $40).
2. **Contextual Anomalies**: A data point appears abnormal only within a specific temporal or behavioral context (e.g. a temperature reading of 30°C is normal in July but anomalous in January).
3. **Collective Anomalies**: A sequence or cluster of data points is anomalous as a group, even though individual values may fall within normal thresholds (e.g. a sudden steady sequence of identical heartbeats).

## Statistical and Distance-Based Methods

Traditional anomaly detection includes statistical $Z$-score testing and distance-based heuristics:

- **k-Nearest Neighbors (k-NN)**: Assigns anomaly scores based on distance to the $k$-th nearest neighbor. Points situated in sparse regions exhibit significantly higher distances.
- **Local Outlier Factor (LOF)**: Compares the local density of an observation to the local densities of its neighboring points, effectively detecting anomalies across datasets with non-uniform cluster densities.

## Machine Learning Approaches

### Isolation Forests

Isolation Forests operate on the principle that anomalies are "few and different." The algorithm recursively constructs random decision trees by randomly selecting features and random split values. Because outliers are isolated in sparse regions, they require significantly fewer splits to isolate into leaf nodes, resulting in shorter average path lengths from the tree root.

### Autoencoders and Reconstruction Error

Deep learning anomaly detection frequently leverages autoencoders. Trained solely on normal data, the network learns an information bottleneck that compresses and reconstructs standard patterns. When an abnormal input is presented at inference, the model fails to reconstruct unfamiliar features, resulting in elevated reconstruction error (e.g. Mean Squared Error between input and output) that serves as an anomaly signal.
