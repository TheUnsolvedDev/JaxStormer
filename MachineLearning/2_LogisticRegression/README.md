# Logistic Regression

Logistic regression is a statistical method used to model the relationship between a dependent binary variable and one or more independent variables. Unlike linear regression, which predicts continuous outcomes, logistic regression predicts probabilities of the outcome that can be used to classify the observations into one of the two categories.

## Mathematical Formulation

### Simple Logistic Regression

In the case of simple logistic regression, where there is only one independent variable, the relationship is modeled using the logistic function:

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

where:
- $P(y=1|x)$ is the probability of the dependent variable $y$ being 1 given the independent variable $x$.
- $x$ is the independent variable.
- $\beta_0$ is the intercept.
- $\beta_1$ is the coefficient for the independent variable.

### Multiple Logistic Regression

When there are multiple independent variables, the relationship is extended to:

$$P(y=1|x_1, x_2, \ldots, x_p) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p)}}$$

where:
- $P(y=1|x_1, x_2, \ldots, x_p)$ is the probability of the dependent variable $y$ being 1 given the independent variables $x_1, x_2, \ldots, x_p$.
- $x_1, x_2, \ldots, x_p$ are the independent variables.
- $\beta_0$ is the intercept.
- $\beta_1, \beta_2, \ldots, $\beta_p$ are the coefficients for the independent variables.

## Objective

The objective of logistic regression is to find the best-fitting model to describe the relationship between the binary dependent variable and the independent variables. This is done by maximizing the likelihood function:

$$L(\beta_0, \beta_1, \ldots, \beta_p) = \prod_{i=1}^{n} P(y_i|x_i)^{y_i} [1 - P(y_i|x_i)]^{1 - y_i}$$

or equivalently, by minimizing the negative log-likelihood:

$$\text{Minimize} \quad - \sum_{i=1}^{n} \left[ y_i \log(P(y_i|x_i)) + (1 - y_i) \log(1 - P(y_i|x_i)) \right]$$

where:
- $y_i$ are the observed values (0 or 1).
- $P(y_i|x_i)$ are the predicted probabilities given by the logistic regression model.

## Estimation of Coefficients

The coefficients $\beta_0, \beta_1, \ldots, \beta_p$ are estimated using the Maximum Likelihood Estimation (MLE) method, which maximizes the likelihood function.

## Assumptions

Logistic regression relies on several key assumptions:
1. **Linearity in the Logit**: The logit (log-odds) of the outcome is a linear combination of the independent variables.
2. **Independence of Observations**: The observations are independent of each other.
3. **No Multicollinearity**: The independent variables are not highly correlated with each other.
4. **Large Sample Size**: Logistic regression requires a large sample size to provide reliable results.

## Evaluation Metrics

Common metrics to evaluate the performance of a logistic regression model include:
- **Accuracy**: The proportion of correctly classified observations.
- **Precision**: The proportion of positive predictions that are actually positive.
- **Recall (Sensitivity)**: The proportion of actual positives that are correctly identified.
- **F1 Score**: The harmonic mean of precision and recall.
- **ROC Curve and AUC**: The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate at various threshold settings, and the Area Under the Curve (AUC) measures the overall performance of the model.

## Conclusion

Logistic regression is a powerful and widely-used statistical method for binary classification problems. It provides a way to understand the relationship between the dependent binary variable and the independent variables, and it offers a probabilistic framework for making predictions.