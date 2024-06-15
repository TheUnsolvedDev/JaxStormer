# k-Nearest Neighbors (k-NN)

k-Nearest Neighbors (k-NN) is a simple, non-parametric, and lazy learning algorithm used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.

## Algorithm Overview

### Classification

For classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small).

### Regression

For regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.

## Mathematical Formulation

### Distance Metric

The key aspect of k-NN is the distance metric used to identify the nearest neighbors. Common distance metrics include:
- **Euclidean Distance**: $d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
- **Manhattan Distance**: $d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$
- **Minkowski Distance**: $d(x, y) = (\sum_{i=1}^{n} |x_i - y_i|^p)^{1/p}$

where $x$ and $y$ are two points in the feature space.

### Algorithm Steps

1. **Store the Training Data**: Store all the training examples.
2. **Compute Distance**: For a new data point, compute the distance between this point and all training examples.
3. **Identify Neighbors**: Identify the k nearest neighbors based on the computed distances.
4. **Predict Output**:
   - **Classification**: Take a majority vote among the k neighbors to assign the class label.
   - **Regression**: Calculate the average of the k neighbors' values to assign the predicted value.

## Choosing k

The choice of k is crucial:
- **Small k**: Can lead to noisy predictions due to overfitting.
- **Large k**: Can smooth out predictions but might include too many points from other classes or regions, leading to underfitting.

A common approach is to use cross-validation to determine the optimal k.

## Assumptions

k-NN makes minimal assumptions about the data:
1. **Locality**: The method assumes that similar points are close in the feature space.
2. **Smoothness**: The underlying assumption is that points that are close to each other are likely to have similar outcomes.

## Advantages

1. **Simple to Implement**: The algorithm is straightforward and easy to implement.
2. **No Training Phase**: k-NN is a lazy learner and does not involve a training phase.
3. **Flexible**: Can be used for both classification and regression tasks.

## Disadvantages

1. **Computationally Intensive**: The algorithm can be slow during prediction, especially with large datasets.
2. **Sensitive to Irrelevant Features**: The presence of irrelevant features can impact the performance of k-NN.
3. **Curse of Dimensionality**: The algorithm's performance can degrade with high-dimensional data due to the sparse nature of the feature space.

## Evaluation Metrics

For **classification**, common evaluation metrics include:
- **Accuracy**: The proportion of correctly classified instances.
- **Confusion Matrix**: A table that describes the performance of the classifier.
- **Precision, Recall, F1 Score**: Metrics to evaluate the performance for each class.
- **ROC Curve and AUC**: Performance measurement for classification problems at various threshold settings.

For **regression**, common evaluation metrics include:
- **Mean Squared Error (MSE)**: The average of the squared differences between observed and predicted values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE.
- **Mean Absolute Error (MAE)**: The average of the absolute differences between observed and predicted values.
- **R-squared (\(R^2\))**: The proportion of the variance in the dependent variable that is predictable from the independent variables.

## Conclusion

k-Nearest Neighbors is a versatile and intuitive algorithm suitable for a variety of classification and regression tasks. Despite its simplicity, it can provide competitive performance, particularly for small datasets with low-dimensional features. However, its computational cost and sensitivity to irrelevant features and high-dimensional data necessitate careful preprocessing and consideration in practical applications.