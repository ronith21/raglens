# The Bias-Variance Tradeoff in Machine Learning

The bias-variance tradeoff provides a foundational theoretical framework for analyzing the predictive errors of supervised learning algorithms.

## Decomposition of Expected Mean Squared Error

Consider an underlying data-generating process governed by $y = f(x) + \epsilon$, where $f(x)$ is the true underlying function and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ represents irreducible noise inherent to the measurement process or environment.

Given a training dataset $\mathcal{D}$, a learning algorithm estimates an approximating hypothesis $\hat{f}(x; \mathcal{D})$. The expected mean squared prediction error at a fixed test point $x$, taken across all possible training realizations $\mathcal{D}$, decomposes analytically into three orthogonal components:

$$\mathbb{E}_{\mathcal{D}} \left[ (y - \hat{f}(x))^2 \right] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2$$

### Bias

Bias measures the difference between the expected prediction of the hypothesis, averaged over hypothetical repetitions of training, and the true underlying signal:

$$\text{Bias}(\hat{f}(x)) = \mathbb{E}_{\mathcal{D}}[\hat{f}(x)] - f(x)$$

High bias indicates systematic erroneous assumptions within the model class. For example, fitting a linear regression model to inherently non-linear parabolic data imposes high bias, resulting in consistent underfitting.

### Variance

Variance quantifies the sensitivity of the learned predictor to variations in the specific training sample $\mathcal{D}$:

$$\text{Var}(\hat{f}(x)) = \mathbb{E}_{\mathcal{D}} \left[ (\hat{f}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)])^2 \right]$$

High variance signifies that small fluctuations in the training observations induce dramatic shifts in model parameters. Unconstrained deep decision trees and high-degree polynomial regressions typically exhibit elevated variance, memorizing individual partitions at the expense of stability.

### Irreducible Noise

The irreducible error term $\sigma^2$ is bounded by the stochasticity of the environment, missing unobserved covariates, and measurement noise. No algorithmic refinement can lower prediction error below this fundamental threshold.

## Navigating the Tradeoff

As model complexity increases, bias monotonically declines while variance rises. The optimal generalization performance is attained at the intermediate complexity level that minimizes their sum. Modern ensemble methods deliberately manipulate this tradeoff: bagging methods (e.g. Random Forests) decrease variance through independent averaging, whereas boosting algorithms (e.g. Gradient Boosted Trees) iteratively reduce bias.
