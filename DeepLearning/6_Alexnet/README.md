# AlexNet Implementation in JAX

This project implements the AlexNet architecture using JAX, a high-performance numerical computing library. AlexNet is a deep convolutional neural network that was designed to classify images into a large number of categories. The implementation includes data preparation, model definition, training, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [AlexNet Architecture](#alexnet-architecture)
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

## AlexNet Architecture

The AlexNet architecture is composed of the following layers:

1. **Input Layer**: Takes the input image (typically 224x224 pixels for the original AlexNet).
2. **Convolutional Layer (C1)**: Applies 96 filters of size 11x11 with stride 4 and ReLU activation.
3. **Max Pooling Layer (P1)**: Applies a 3x3 filter with stride 2.
4. **Convolutional Layer (C2)**: Applies 256 filters of size 5x5 with stride 1 and ReLU activation.
5. **Max Pooling Layer (P2)**: Applies a 3x3 filter with stride 2.
6. **Convolutional Layer (C3)**: Applies 384 filters of size 3x3 with stride 1 and ReLU activation.
7. **Convolutional Layer (C4)**: Applies 384 filters of size 3x3 with stride 1 and ReLU activation.
8. **Convolutional Layer (C5)**: Applies 256 filters of size 3x3 with stride 1 and ReLU activation.
9. **Max Pooling Layer (P3)**: Applies a 3x3 filter with stride 2.
10. **Fully Connected Layer (F6)**: Fully connected layer with 4096 units and ReLU activation.
11. **Fully Connected Layer (F7)**: Fully connected layer with 4096 units and ReLU activation.
12. **Output Layer**: Fully connected layer with 1000 units (for 1000 classes) and softmax activation.

### Parameter Initialization

Parameters (weights and biases) are initialized using a random normal distribution. This is critical to start the training process with diverse initial values.

### AlexNet Function

The AlexNet function computes the output by performing convolution operations followed by activation functions and pooling layers. The final fully connected layers use a softmax function to produce probabilities for each class.

## Training and Evaluation

### Loss Function

The cross-entropy loss function is used to measure the performance of the neural network. It calculates the difference between the predicted probabilities and the actual labels.

### Optimization with Adam

The Adam optimizer, an adaptive learning rate optimization algorithm, is used to update network parameters. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

### Accuracy Calculation

The accuracy of the model is calculated by comparing the predicted class labels with the actual labels and determining the proportion of correct predictions.

## Mathematical Notations
```
                                                           AlexNet Summary                                                           
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ path        ┃ module    ┃ inputs                      ┃ outputs               ┃ params                       ┃ batch_stats        ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│             │ AlexNet   │ - float32[10,96,96,1]       │ float32[10,10]        │                              │                    │
│             │           │ - train: False              │                       │                              │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Conv_0      │ Conv      │ float32[10,96,96,1]         │ float32[10,22,22,96]  │ bias: float32[96]            │                    │
│             │           │                             │                       │ kernel: float32[11,11,1,96]  │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 11,712 (46.8 KB)             │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ BatchNorm_0 │ BatchNorm │ - float32[10,22,22,96]      │ float32[10,22,22,96]  │ bias: float32[96]            │ mean: float32[96]  │
│             │           │ - use_running_average: True │                       │ scale: float32[96]           │ var: float32[96]   │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 192 (768 B)                  │ 192 (768 B)        │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Conv_1      │ Conv      │ float32[10,10,10,96]        │ float32[10,10,10,256] │ bias: float32[256]           │                    │
│             │           │                             │                       │ kernel: float32[5,5,96,256]  │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 614,656 (2.5 MB)             │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ BatchNorm_1 │ BatchNorm │ - float32[10,10,10,256]     │ float32[10,10,10,256] │ bias: float32[256]           │ mean: float32[256] │
│             │           │ - use_running_average: True │                       │ scale: float32[256]          │ var: float32[256]  │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 512 (2.0 KB)                 │ 512 (2.0 KB)       │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Conv_2      │ Conv      │ float32[10,4,4,256]         │ float32[10,4,4,384]   │ bias: float32[384]           │                    │
│             │           │                             │                       │ kernel: float32[3,3,256,384] │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 885,120 (3.5 MB)             │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ BatchNorm_2 │ BatchNorm │ - float32[10,4,4,384]       │ float32[10,4,4,384]   │ bias: float32[384]           │ mean: float32[384] │
│             │           │ - use_running_average: True │                       │ scale: float32[384]          │ var: float32[384]  │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 768 (3.1 KB)                 │ 768 (3.1 KB)       │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Conv_3      │ Conv      │ float32[10,4,4,384]         │ float32[10,4,4,384]   │ bias: float32[384]           │                    │
│             │           │                             │                       │ kernel: float32[3,3,384,384] │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 1,327,488 (5.3 MB)           │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ BatchNorm_3 │ BatchNorm │ - float32[10,4,4,384]       │ float32[10,4,4,384]   │ bias: float32[384]           │ mean: float32[384] │
│             │           │ - use_running_average: True │                       │ scale: float32[384]          │ var: float32[384]  │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 768 (3.1 KB)                 │ 768 (3.1 KB)       │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Conv_4      │ Conv      │ float32[10,4,4,384]         │ float32[10,4,4,256]   │ bias: float32[256]           │                    │
│             │           │                             │                       │ kernel: float32[3,3,384,256] │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 884,992 (3.5 MB)             │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ BatchNorm_4 │ BatchNorm │ - float32[10,4,4,256]       │ float32[10,4,4,256]   │ bias: float32[256]           │ mean: float32[256] │
│             │           │ - use_running_average: True │                       │ scale: float32[256]          │ var: float32[256]  │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 512 (2.0 KB)                 │ 512 (2.0 KB)       │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Dense_0     │ Dense     │ float32[10,256]             │ float32[10,4096]      │ bias: float32[4096]          │                    │
│             │           │                             │                       │ kernel: float32[256,4096]    │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 1,052,672 (4.2 MB)           │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Dropout_0   │ Dropout   │ float32[10,4096]            │ float32[10,4096]      │                              │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Dense_1     │ Dense     │ float32[10,4096]            │ float32[10,4096]      │ bias: float32[4096]          │                    │
│             │           │                             │                       │ kernel: float32[4096,4096]   │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 16,781,312 (67.1 MB)         │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Dropout_1   │ Dropout   │ float32[10,4096]            │ float32[10,4096]      │                              │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│ Dense_2     │ Dense     │ float32[10,4096]            │ float32[10,10]        │ bias: float32[10]            │                    │
│             │           │                             │                       │ kernel: float32[4096,10]     │                    │
│             │           │                             │                       │                              │                    │
│             │           │                             │                       │ 40,970 (163.9 KB)            │                    │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼──────────────────────────────┼────────────────────┤
│             │           │                             │                 Total │ 21,601,674 (86.4 MB)         │ 2,752 (11.0 KB)    │
└─────────────┴───────────┴─────────────────────────────┴───────────────────────┴──────────────────────────────┴────────────────────┘
                                                                                                                                     
                                               Total Parameters: 21,604,426 (86.4 MB)        
```

1. **AlexNet Architecture**

   - Input layer: $X$
   - Convolutional layers: $C_i$ where $i$ is the layer index
   - Pooling layers: $P_i$ where $i$ is the layer index
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
     $$P_i = \text{pool}(C_i)$$

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
2. **Initialize the Network**: Create an instance of the `AlexNet` class with desired parameters.
3. **Train the Network**: Call the `train` method with the training and test data to start the training process.
4. **Evaluate the Network**: Monitor the loss and accuracy during training to evaluate the model performance.

By following these steps, you can train an AlexNet model using JAX and apply it to various machine learning tasks.