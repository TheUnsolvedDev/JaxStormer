# Kernel Support Vector Machine (SVM)

Kernel Support Vector Machine (SVM) is an extension of the linear SVM that allows for the classification of non-linearly separable data by transforming the input features into higher-dimensional spaces using kernel functions. This transformation enables the algorithm to find a linear hyperplane in this higher-dimensional space, which corresponds to a non-linear decision boundary in the original feature space.

## Algorithm Overview

### Objective

Similar to linear SVM, the objective of Kernel SVM is to find the hyperplane that maximizes the margin between classes. However, by using kernel functions, Kernel SVM can handle non-linear relationships between the features.

### Mathematical Formulation

#### Hyperplane in Higher-Dimensional Space

The decision boundary in the higher-dimensional space can be defined as:

$$ w \cdot \phi(x) + b = 0 $$

where:
- $w$ is the weight vector in the higher-dimensional space.
- $\phi(x)$ is the mapping function that transforms the original feature vector $x$ into the higher-dimensional space.
- $b$ is the bias term.

#### Decision Rule

The decision rule for classifying a new data point $x$ is:

$$ f(x) = \text{sign}(w \cdot \phi(x) + b) $$

### Kernel Trick

The key idea behind Kernel SVM is the kernel trick, which allows the algorithm to compute the dot product $\phi(x_i) \cdot \phi(x_j)$ in the higher-dimensional space without explicitly computing $\phi(x)$. This is achieved through a kernel function $K(x_i, x_j)$:

$$ K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j) $$

Common kernel functions include:
- **Linear Kernel**: $K(x_i, x_j) = x_i \cdot x_j$
- **Polynomial Kernel**: $K(x_i, x_j) = (\gamma x_i \cdot x_j + r)^d$
- **Radial Basis Function (RBF) Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
- **Sigmoid Kernel**: $K(x_i, x_j) = \tanh(\gamma x_i \cdot x_j + r)$

### Optimization Problem

The optimization problem for Kernel SVM with a soft margin is formulated as:

$$ \min_{w,b,\xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i $$

subject to the constraints:

$$ y_i (w \cdot \phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i $$

Using the kernel trick, the dual form of the optimization problem is solved instead of the primal form:

$$ \max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(x_i, x_j) $$

subject to:

$$ 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0 $$

where $\alpha_i$ are the Lagrange multipliers.

## Assumptions

Kernel SVM relies on several key assumptions:
1. **Non-Linearly Separable Data**: Assumes that the data may not be linearly separable in the original feature space.
2. **Appropriate Kernel Choice**: The performance depends on the choice of the kernel function and its parameters.

## Advantages

1. **Handles Non-Linearity**: Can effectively classify non-linearly separable data.
2. **Flexibility**: The choice of kernel functions allows for flexibility in modeling complex relationships.
3. **Robustness**: Effective in high-dimensional spaces and less prone to overfitting with proper regularization.

## Disadvantages

1. **Computational Complexity**: More computationally intensive than linear SVM, especially for large datasets.
2. **Parameter Tuning**: Requires careful tuning of the kernel parameters and the regularization parameter $C$.
3. **Interpretability**: The resulting model is less interpretable compared to linear SVMs.

## Evaluation Metrics

Common metrics to evaluate the performance of a Kernel SVM classifier include:
- **Accuracy**: The proportion of correctly classified instances.
- **Confusion Matrix**: A table that describes the performance of the classifier.
- **Precision, Recall, F1 Score**: Metrics to evaluate the performance for each class.
- **ROC Curve and AUC**: Performance measurement for classification problems at various threshold settings.

## Conclusion

Kernel Support Vector Machine (SVM) is a powerful extension of the linear SVM that allows for the classification of non-linearly separable data through the use of kernel functions. Its flexibility and effectiveness in handling complex relationships make it a popular choice for various classification tasks. However, careful consideration of computational resources and parameter tuning is essential for optimal performance.