# Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' Theorem with the assumption of independence between the features. It is widely used for text classification, spam detection, and other applications where the independence assumption holds reasonably well.

## Bayes' Theorem

Bayes' Theorem describes the probability of an event based on prior knowledge of conditions related to the event. The theorem is stated as:

$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

where:
- $P(C|X)$ is the posterior probability of class $C$ given the feature vector $X$.
- $P(X|C)$ is the likelihood of the feature vector $X$ given class $C$.
- $P(C)$ is the prior probability of class $C$.
- $P(X)$ is the marginal probability of the feature vector $X$.

## Naive Bayes Assumption

The "naive" assumption is that all features are independent given the class label. This simplifies the calculation of the likelihood $P(X|C)$ as the product of individual probabilities:

$$
P(X|C) = P(x_1, x_2, \ldots, x_n|C) = \prod_{i=1}^{n} P(x_i|C)
$$

## Classification

To classify a new instance, the classifier computes the posterior probability for each class and assigns the class with the highest posterior probability:

$$
\hat{C} = \arg\max_{C} P(C|X) = \arg\max_{C} \left[ P(C) \cdot \prod_{i=1}^{n} P(x_i|C) \right]
$$

### Types of Naive Bayes Classifiers

1. **Gaussian Naive Bayes**: Assumes that the continuous features follow a Gaussian (normal) distribution.
2. **Multinomial Naive Bayes**: Used for discrete features such as word counts in text classification.
3. **Bernoulli Naive Bayes**: Used for binary/boolean features.

## Estimation of Probabilities

### Prior Probability ($P(C)$)

The prior probability is estimated from the training data as the relative frequency of each class:

$$
P(C) = \frac{\text{Number of instances in class } C}{\text{Total number of instances}}
$$

### Likelihood ($P(x_i|C)$)

- **Gaussian Naive Bayes**: For continuous features, the likelihood is assumed to follow a Gaussian distribution with parameters estimated from the training data.

$$
P(x_i|C) = \frac{1}{\sqrt{2 \pi \sigma_C^2}} \exp\left( -\frac{(x_i - \mu_C)^2}{2 \sigma_C^2} \right)
$$

where $\mu_C$ and $\sigma_C$ are the mean and standard deviation of feature $x_i$ for class $C$.

- **Multinomial Naive Bayes**: For discrete features, the likelihood is estimated as the relative frequency of the feature given the class.

$$
P(x_i|C) = \frac{\text{Count of } x_i \text{ in class } C}{\text{Total count of all features in class } C}
$$

- **Bernoulli Naive Bayes**: For binary features, the likelihood is estimated as the probability of the feature being 1 given the class.

$$
P(x_i|C) = \frac{\text{Number of instances with } x_i = 1 \text{ in class } C}{\text{Total number of instances in class } C}
$$

## Assumptions

Naive Bayes relies on several key assumptions:
1. **Feature Independence**: Features are independent given the class label.
2. **Feature Relevance**: All features contribute to the classification.
3. **No Zero Probability**: Smoothing techniques like Laplace smoothing are used to handle zero probabilities in the likelihood estimation.

## Evaluation Metrics

Common metrics to evaluate the performance of a Naive Bayes classifier include:
- **Accuracy**: The proportion of correctly classified instances.
- **Confusion Matrix**: A table that describes the performance of the classifier.
- **Precision, Recall, F1 Score**: Metrics to evaluate the performance for each class.
- **ROC Curve and AUC**: Performance measurement for classification problems at various threshold settings.

## Advantages

1. **Simple and Fast**: Easy to implement and computationally efficient.
2. **Works Well with High-Dimensional Data**: Performs well with large feature spaces, commonly used in text classification.
3. **Robust to Irrelevant Features**: Handles irrelevant features well due to the independence assumption.

## Disadvantages

1. **Independence Assumption**: The strong independence assumption may not hold in practice, affecting the classifier's performance.
2. **Zero Probability**: Requires smoothing techniques to handle features that do not appear in the training data for a given class.
3. **Non-Informative Features**: Assumes all features contribute equally to the outcome, which may not be the case.

## Conclusion

Naive Bayes is a powerful and efficient probabilistic classifier suitable for a wide range of applications. Despite its simplicity and strong assumptions, it often performs surprisingly well in practice, especially for text classification and other tasks with high-dimensional feature spaces.