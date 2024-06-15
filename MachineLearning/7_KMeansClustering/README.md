# k-Means Clustering

k-Means clustering is an unsupervised learning algorithm used to partition a dataset into k distinct, non-overlapping subsets (clusters). Each cluster is defined by its centroid, which is the mean of the points in the cluster. The algorithm aims to minimize the variance within each cluster.

## Algorithm Overview

### Steps

1. **Initialization**:
   - Select k initial centroids randomly from the dataset.
   
2. **Assignment**:
   - Assign each data point to the nearest centroid based on a chosen distance metric (commonly Euclidean distance).

3. **Update**:
   - Recalculate the centroids as the mean of the data points assigned to each cluster.

4. **Repeat**:
   - Repeat the assignment and update steps until the centroids do not change significantly (convergence) or a maximum number of iterations is reached.

### Distance Metric

The most commonly used distance metric in k-Means is Euclidean distance:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

where $x$ and $y$ are two points in the feature space.

## Mathematical Formulation

### Objective Function

The goal of k-Means is to minimize the within-cluster sum of squares (WCSS), also known as inertia:

$$
\text{WCSS} = \sum_{j=1}^{k} \sum_{i=1}^{n_j} ||x_i^{(j)} - \mu_j||^2
$$

where:
- $k$ is the number of clusters.
- $n_j$ is the number of points in cluster $j$.
- $x_i^{(j)}$ is the $i$-th point in cluster $j$.
- $\mu_j$ is the centroid of cluster $j$.
- $|| \cdot ||^2$ denotes the squared Euclidean distance.

## Choosing k

The choice of k is crucial and can be determined using methods like:
- **Elbow Method**: Plotting the WCSS against the number of clusters and looking for an "elbow point" where the rate of decrease sharply slows.
- **Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters, ranging from -1 to 1.

## Assumptions

k-Means clustering relies on several key assumptions:
1. **Clusters are Spherical**: Assumes clusters are spherical and equally sized, which may not hold in practice.
2. **Variance Homogeneity**: Assumes that the variance of the distribution of each attribute is equal across all clusters.
3. **Large Data Sets**: Works better with large datasets where clusters are more distinct.

## Advantages

1. **Simple and Fast**: Easy to implement and computationally efficient, especially for large datasets.
2. **Scalable**: Can handle large datasets and is easily scalable.
3. **Works Well with Well-Separated Clusters**: Performs well when clusters are distinct and well-separated.

## Disadvantages

1. **Fixed Number of Clusters**: Requires specifying the number of clusters beforehand.
2. **Sensitive to Initialization**: The final clusters can depend on the initial selection of centroids, often mitigated by running the algorithm multiple times with different initializations.
3. **Not Suitable for Non-Spherical Clusters**: Performs poorly when clusters have non-spherical shapes or different sizes and densities.
4. **Sensitive to Outliers**: Outliers can significantly affect the cluster centroids and the resulting clusters.

## Evaluation Metrics

Common metrics to evaluate the performance of k-Means clustering include:
- **Within-Cluster Sum of Squares (WCSS)**: Measures the variance within each cluster.
- **Silhouette Score**: Measures how similar each point is to its own cluster compared to other clusters.
- **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with the cluster that is most similar to it.
- **Calinski-Harabasz Index**: Ratio of the sum of between-cluster dispersion and within-cluster dispersion.

## Conclusion

k-Means clustering is a powerful and widely-used clustering algorithm suitable for a variety of applications. Despite its simplicity and certain limitations, it provides an effective way to partition data into meaningful clusters, especially when the assumptions of spherical and equally sized clusters are reasonably met. Its ease of implementation and computational efficiency make it a popular choice for exploratory data analysis and preprocessing tasks.