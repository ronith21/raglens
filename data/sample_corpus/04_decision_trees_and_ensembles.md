# Decision Trees and Ensemble Methods

Decision trees and ensemble architectures are among the most robust, interpretable, and computationally efficient algorithms available for tabular data analysis and structured classification tasks.

## Anatomy of a Decision Tree

A decision tree recursively partitions the feature space into orthogonal hyper-rectangles by applying greedy, axis-aligned splits. At each non-terminal node, the algorithm evaluates candidate features and thresholds to maximize the purity of descendant child nodes.

For classification tasks, node impurity is typically evaluated using either Information Gain (Entropy reduction) or the Gini Impurity metric:

$$\text{Gini}(S) = 1 - \sum_{k=1}^K p_k^2$$

For continuous regression targets, split criteria prioritize the reduction of Mean Squared Error or Mean Absolute Error across child partitions.

While single decision trees provide straightforward interpretability, deep unpruned trees suffer from severe variance and high vulnerability to data perturbations.

## Bagging and Random Forests

Bootstrap Aggregation (Bagging) reduces the variance of unstable base estimators by training $B$ distinct predictors on bootstrap samples—datasets constructed by sampling $N$ instances from the original dataset with replacement. The final prediction aggregates base outputs via majority voting (classification) or mean averaging (regression).

Random Forests enhance bagging by introducing random feature subspace selection. At every candidate split within each tree, only a random subset of size $m \approx \sqrt{p}$ features is considered. This feature subsampling decorrelates individual trees, dramatically suppressing the variance of the overall ensemble according to the formula:

$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

where $\rho$ represents the pairwise correlation between individual trees.

## Boosting Algorithms

Unlike bagging, which trains models in parallel, boosting constructs an ensemble sequentially. Each new base learner is trained to predict or correct the residual errors produced by preceding iterations:

1. **AdaBoost**: Iteratively upweights misclassified samples, forcing subsequent weak learners to concentrate on challenging boundary regions.
2. **Gradient Boosting Machines (GBM)**: Formulates boosting as gradient descent in function space, fitting each sequential tree directly to the negative gradient of the loss function with respect to the current ensemble's predictions.
3. **XGBoost and LightGBM**: Introduce second-order Taylor expansions of the loss function, histogram-based split finding, feature sub-sampling, and explicit tree-leaf regularization penalties, achieving dominant benchmark performance across structured data competitions.
