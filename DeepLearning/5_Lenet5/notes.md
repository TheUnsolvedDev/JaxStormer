# LeNet-5 Implementation in JAX

This project implements the LeNet-5 architecture using JAX, a high-performance numerical computing library. LeNet-5 is a pioneering convolutional neural network (CNN) architecture designed for handwritten digit recognition. The implementation includes data preparation, model definition, training, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [LeNet-5 Architecture](#lenet-5-architecture)
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

## LeNet-5 Architecture

The LeNet-5 architecture is composed of the following layers:

1. **Input Layer**: Takes the input image (typically 32x32 pixels for the original LeNet-5).
2. **Convolutional Layer (C1)**: Applies 6 filters of size 5x5.
3. **Average Pooling Layer (S2)**: Applies a 2x2 filter with stride 2.
4. **Convolutional Layer (C3)**: Applies 16 filters of size 5x5.
5. **Average Pooling Layer (S4)**: Applies a 2x2 filter with stride 2.
6. **Fully Connected Layer (C5)**: Fully connected layer with 120 units.
7. **Fully Connected Layer (F6)**: Fully connected layer with 84 units.
8. **Output Layer**: Fully connected layer with 10 units for classification.

### Parameter Initialization

Parameters (weights and biases) are initialized using a random normal distribution. This is critical to start the training process with diverse initial values.

### LeNet-5 Function

The LeNet-5 function computes the output by performing convolution operations followed by activation functions and pooling layers. The final fully connected layers use a softmax function to produce probabilities for each class.

## Training and Evaluation

### Loss Function

The cross-entropy loss function is used to measure the performance of the neural network. It calculates the difference between the predicted probabilities and the actual labels.

### Optimization with Adam

The Adam optimizer, an adaptive learning rate optimization algorithm, is used to update network parameters. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

### Accuracy Calculation

The accuracy of the model is calculated by comparing the predicted class labels with the actual labels and determining the proportion of correct predictions.

## Mathematical Notations

```
                                       LeNet5 Summary                                       
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ path    ┃ module ┃ inputs              ┃ outputs             ┃ params                    ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         │ LeNet5 │ float32[10,28,28,1] │ float32[10,10]      │                           │
├─────────┼────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
│ Conv_0  │ Conv   │ float32[10,28,28,1] │ float32[10,14,14,6] │ bias: float32[6]          │
│         │        │                     │                     │ kernel: float32[5,5,1,6]  │
│         │        │                     │                     │                           │
│         │        │                     │                     │ 156 (624 B)               │
├─────────┼────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
│ Conv_1  │ Conv   │ float32[10,7,7,6]   │ float32[10,4,4,16]  │ bias: float32[16]         │
│         │        │                     │                     │ kernel: float32[5,5,6,16] │
│         │        │                     │                     │                           │
│         │        │                     │                     │ 2,416 (9.7 KB)            │
├─────────┼────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
│ Dense_0 │ Dense  │ float32[10,64]      │ float32[10,120]     │ bias: float32[120]        │
│         │        │                     │                     │ kernel: float32[64,120]   │
│         │        │                     │                     │                           │
│         │        │                     │                     │ 7,800 (31.2 KB)           │
├─────────┼────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
│ Dense_1 │ Dense  │ float32[10,120]     │ float32[10,84]      │ bias: float32[84]         │
│         │        │                     │                     │ kernel: float32[120,84]   │
│         │        │                     │                     │                           │
│         │        │                     │                     │ 10,164 (40.7 KB)          │
├─────────┼────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
│ Dense_2 │ Dense  │ float32[10,84]      │ float32[10,10]      │ bias: float32[10]         │
│         │        │                     │                     │ kernel: float32[84,10]    │
│         │        │                     │                     │                           │
│         │        │                     │                     │ 850 (3.4 KB)              │
├─────────┼────────┼─────────────────────┼─────────────────────┼───────────────────────────┤
│         │        │                     │               Total │ 21,386 (85.5 KB)          │
└─────────┴────────┴─────────────────────┴─────────────────────┴───────────────────────────┘
                                                                                            
                             Total Parameters: 21,386 (85.5 KB)                             
```

1. **LeNet-5 Architecture**

   - Input layer: $X$
   - Convolutional layers: $C_i$ where $i$ is the layer index
   - Pooling layers: $S_i$ where $i$ is the layer index
   - Fully connected layers: $F_i$ where $i$ is the layer index
   - Output layer: $O$
   - Weights: $W_i$
   - Biases: $b_i$
   - Activation function: $\sigma$

2. **Convolution Operation**

   - For each convolutional layer $i$:
     $$C_i = \sigma(\text{conv}(C_{i-1}, W_i) + b_i)$$

3. **Pooling Operation**

   - For each pooling layer $i$:
     $$S_i = \text{pool}(C_i)$$

4. **Fully Connected Layers**

   - Flatten the output of the last pooling layer and pass through fully connected layers:
     $$F_i = \sigma(P_{n-1} W_i + b_i)$$

5. **Output Layer**

   - The final layer uses softmax activation:
     $$O = \text{softmax}(F_{m-1} W_m + b_m)$$

6. **Loss Function (Cross-Entropy)**

   $$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$$

   where $N$ is the number of samples, $C$ is the number of classes, $y$ is the true label, and $\hat{y}$ is the predicted probability.

7. **Adam Optimizer Update**

   The Adam optimizer adjusts the learning rate for each parameter individually based on the first moment (mean) and the second moment (uncentered variance) of the gradients.

   - Compute gradients: $g_t$
   - Update biased first moment estimate: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   - Update biased second moment estimate: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   - Compute bias-corrected first moment estimate: $\hat{m}_t = m_t / (1 - \beta_1^t)$
   - Compute bias-corrected second moment estimate: $\hat{v}_t = v_t / (1 - \beta_2^t)$
   - Update parameters: $\theta_t = \theta_{t-1} - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

## Running the Code

1. **Prepare the Dataset**: Load and preprocess the dataset using the `Dataset` class.
2. **Initialize the Network**: Create an instance of the `ConvolutionalNeuralNetwork` class with desired parameters.
3. **Train the Network**: Call the `train` method with the training and test data to start the training process.
4. **Evaluate the Network**: Monitor the loss and accuracy during training to evaluate the model performance.

By following these steps, you can train a convolutional neural network using JAX and apply it to various machine learning tasks.