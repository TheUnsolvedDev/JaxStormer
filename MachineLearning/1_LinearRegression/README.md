# Linear Regression

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable $y$ and one or more independent variables $x$. The goal of linear regression is to find the best-fitting linear relationship between these variables.

## Mathematical Formulation

### Simple Linear Regression

In the case of simple linear regression, where there is only one independent variable, the relationship is modeled as:

$$y = \beta_0 + \beta_1 x + \epsilon$$

where:
- $y$ is the dependent variable.
- $x$ is the independent variable.
- $\beta_0$ is the intercept (the value of $y$ when $x = 0$).
- $\beta_1$ is the slope of the line (the change in $y$ for a one-unit change in $x$).
- $\epsilon$ is the error term (the difference between the observed and predicted values).

### Multiple Linear Regression

When there are multiple independent variables, the relationship is extended to:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

where:
- $y$ is the dependent variable.
- $x_1, x_2, \ldots, x_p$ are the independent variables.
- $\beta_0$ is the intercept.
- $\beta_1, \beta_2, \ldots, \beta_p$ are the coefficients for the independent variables.
- $\epsilon$ is the error term.

## Objective

The objective of linear regression is to minimize the sum of the squared differences between the observed values $y_i$ and the predicted values $\hat{y}_i$:

$$\text{Minimize} \quad \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where:
- $y_i$ are the observed values.
- $\hat{y}_i$ are the predicted values given by the regression model.

## Estimation of Coefficients

The coefficients $\beta_0, \beta_1, \ldots, \beta_p$ are estimated using the Ordinary Least Squares (OLS) method, which minimizes the sum of the squared residuals. The estimated coefficients are given by:

$$\mathbf{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

where:
- $\mathbf{\beta}$ is the vector of estimated coefficients.
- $\mathbf{X}$ is the matrix of independent variables.
- $\mathbf{X}^\top$ is the transpose of $\mathbf{X}$.
- $\mathbf{y}$ is the vector of observed values.

## Assumptions

Linear regression relies on several key assumptions:
1. **Linearity**: The relationship between the independent and dependent variables is linear.
2. **Independence**: The residuals (errors) are independent.
3. **Homoscedasticity**: The residuals have constant variance at every level of $x$.
4. **Normality**: The residuals of the model are normally distributed.

## Evaluation Metrics

Common metrics to evaluate the performance of a linear regression model include:
- **R-squared ($R^2$)**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Squared Error (MSE)**: The average of the squared differences between observed and predicted values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which brings the error to the same unit as the dependent variable.
