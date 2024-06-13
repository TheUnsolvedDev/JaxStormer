# Simple Neural Network in JAX

This project implements a simple feedforward neural network using JAX, a high-performance numerical computing library. The implementation includes data preparation, model definition, training, and evaluation.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Neural Network Definition](#neural-network-definition)
- [Training and Evaluation](#training-and-evaluation)
- [Mathematical Notations](#mathematical-notations)
- [Running the Code](#running-the-code)

## Installation

To run the code, ensure you have JAX and the necessary dependencies installed. You can install them using:

```bash
pip install jax jaxlib tqdm numpy
```

## Dataset Preparation

The `Dataset` class is responsible for loading and preprocessing the dataset. It scales the dataset features to a range of [0, 1] and converts labels to one-hot encoded vectors. This step is crucial for neural network training.

## Neural Network Definition

The `NeuralNetwork` class defines the architecture, initialization, and training process of the neural network. Key parameters include:
- `hidden_layers`: Specifies the number and size of hidden layers.
- `alpha`: Learning rate for gradient descent.
- `batch_size`: Size of the mini-batches for training.
- `n_epochs`: Number of training epochs.

### Parameter Initialization

Parameters (weights and biases) are initialized using a random normal distribution. This is critical to start the training process with diverse initial values.

### Neural Network Function

The neural network function computes the output by performing matrix multiplications and applying activation functions across layers. The final layer uses a softmax function to produce probabilities for each class.

## Training and Evaluation

### Loss Function

The cross-entropy loss function is used to measure the performance of the neural network. It calculates the difference between the predicted probabilities and the actual labels.

### Gradient Descent

The network parameters are updated using gradient descent, which involves computing the gradient of the loss function with respect to the parameters and adjusting the parameters in the opposite direction of the gradient.

### Accuracy Calculation

The accuracy of the model is calculated by comparing the predicted class labels with the actual labels and determining the proportion of correct predictions.

## Mathematical Notations

1. **Feedforward Network**
   - Input layer: \( X \)
   - Hidden layers: \( H_i \) where \( i \) is the layer index
   - Output layer: \( O \)
   - Weights: \( W_i \)
   - Biases: \( b_i \)
   - Activation function: \( \sigma \)

2. **Forward Pass**
   - For each hidden layer \( i \):
     \[
     H_i = \sigma(X W_i + b_i)
     \]
   - Output layer:
     \[
     O = \text{softmax}(H_{n-1} W_{n} + b_{n})
     \]

3. **Loss Function (Cross-Entropy)**
   \[
   L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
   \]
   where \( N \) is the number of samples, \( C \) is the number of classes, \( y \) is the true label, and \( \hat{y} \) is the predicted probability.

4. **Gradient Descent Update**
   \[
   W \leftarrow W - \alpha \nabla_W L
   \]
   \[
   b \leftarrow b - \alpha \nabla_b L
   \]
   where \( \alpha \) is the learning rate.

## Running the Code

1. **Prepare the Dataset**: Load and preprocess the dataset using the `Dataset` class.
2. **Initialize the Network**: Create an instance of the `NeuralNetwork` class with desired parameters.
3. **Train the Network**: Call the `train` method with the training and test data to start the training process.
4. **Evaluate the Network**: Monitor the loss and accuracy during training to evaluate the model performance.

By following these steps, you can train a simple neural network using JAX and apply it to various machine learning tasks.