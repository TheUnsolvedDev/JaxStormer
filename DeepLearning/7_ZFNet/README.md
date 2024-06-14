# ZFNet Implementation in JAX

This project implements the ZFNet (Zeiler and Fergus Net) architecture using JAX, a high-performance numerical computing library. ZFNet is a convolutional neural network designed to improve upon the AlexNet architecture by using larger receptive fields in the first convolutional layer and adjusting the stride and filter sizes. The implementation includes data preparation, model definition, training, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [ZFNet Architecture](#zfnet-architecture)
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

## Mathematical Notations
```
                                                             ZFNet Summary                                                             
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ path        ┃ module    ┃ inputs                      ┃ outputs               ┃ params                        ┃ batch_stats         ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│             │ ZFNet     │ - float32[10,48,48,1]       │ float32[10,10]        │                               │                     │
│             │           │ - train: False              │                       │                               │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Conv_0      │ Conv      │ float32[10,48,48,1]         │ float32[10,22,22,96]  │ bias: float32[96]             │                     │
│             │           │                             │                       │ kernel: float32[7,7,1,96]     │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 4,800 (19.2 KB)               │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ BatchNorm_0 │ BatchNorm │ - float32[10,22,22,96]      │ float32[10,22,22,96]  │ bias: float32[96]             │ mean: float32[96]   │
│             │           │ - use_running_average: True │                       │ scale: float32[96]            │ var: float32[96]    │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 192 (768 B)                   │ 192 (768 B)         │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Conv_1      │ Conv      │ float32[10,10,10,96]        │ float32[10,10,10,256] │ bias: float32[256]            │                     │
│             │           │                             │                       │ kernel: float32[5,5,96,256]   │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 614,656 (2.5 MB)              │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ BatchNorm_1 │ BatchNorm │ - float32[10,10,10,256]     │ float32[10,10,10,256] │ bias: float32[256]            │ mean: float32[256]  │
│             │           │ - use_running_average: True │                       │ scale: float32[256]           │ var: float32[256]   │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 512 (2.0 KB)                  │ 512 (2.0 KB)        │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Conv_2      │ Conv      │ float32[10,4,4,256]         │ float32[10,4,4,512]   │ bias: float32[512]            │                     │
│             │           │                             │                       │ kernel: float32[3,3,256,512]  │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 1,180,160 (4.7 MB)            │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ BatchNorm_2 │ BatchNorm │ - float32[10,4,4,512]       │ float32[10,4,4,512]   │ bias: float32[512]            │ mean: float32[512]  │
│             │           │ - use_running_average: True │                       │ scale: float32[512]           │ var: float32[512]   │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 1,024 (4.1 KB)                │ 1,024 (4.1 KB)      │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Conv_3      │ Conv      │ float32[10,4,4,512]         │ float32[10,4,4,1024]  │ bias: float32[1024]           │                     │
│             │           │                             │                       │ kernel: float32[3,3,512,1024] │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 4,719,616 (18.9 MB)           │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ BatchNorm_3 │ BatchNorm │ - float32[10,4,4,1024]      │ float32[10,4,4,1024]  │ bias: float32[1024]           │ mean: float32[1024] │
│             │           │ - use_running_average: True │                       │ scale: float32[1024]          │ var: float32[1024]  │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 2,048 (8.2 KB)                │ 2,048 (8.2 KB)      │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Conv_4      │ Conv      │ float32[10,4,4,1024]        │ float32[10,4,4,512]   │ bias: float32[512]            │                     │
│             │           │                             │                       │ kernel: float32[3,3,1024,512] │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 4,719,104 (18.9 MB)           │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ BatchNorm_4 │ BatchNorm │ - float32[10,4,4,512]       │ float32[10,4,4,512]   │ bias: float32[512]            │ mean: float32[512]  │
│             │           │ - use_running_average: True │                       │ scale: float32[512]           │ var: float32[512]   │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 1,024 (4.1 KB)                │ 1,024 (4.1 KB)      │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Dense_0     │ Dense     │ float32[10,512]             │ float32[10,1024]      │ bias: float32[1024]           │                     │
│             │           │                             │                       │ kernel: float32[512,1024]     │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 525,312 (2.1 MB)              │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Dropout_0   │ Dropout   │ float32[10,1024]            │ float32[10,1024]      │                               │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Dense_1     │ Dense     │ float32[10,1024]            │ float32[10,1024]      │ bias: float32[1024]           │                     │
│             │           │                             │                       │ kernel: float32[1024,1024]    │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 1,049,600 (4.2 MB)            │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Dropout_1   │ Dropout   │ float32[10,1024]            │ float32[10,1024]      │                               │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│ Dense_2     │ Dense     │ float32[10,1024]            │ float32[10,10]        │ bias: float32[10]             │                     │
│             │           │                             │                       │ kernel: float32[1024,10]      │                     │
│             │           │                             │                       │                               │                     │
│             │           │                             │                       │ 10,250 (41.0 KB)              │                     │
├─────────────┼───────────┼─────────────────────────────┼───────────────────────┼───────────────────────────────┼─────────────────────┤
│             │           │                             │                 Total │ 12,828,298 (51.3 MB)          │ 4,800 (19.2 KB)     │
└─────────────┴───────────┴─────────────────────────────┴───────────────────────┴───────────────────────────────┴─────────────────────┘
                                                                                                                                       
                                                Total Parameters: 12,833,098 (51.3 MB)    
```

## ZFNet Architecture

ZFNet, or Zeiler and Fergus Net, is a convolutional neural network architecture designed to improve upon the performance of AlexNet. It was introduced by Matthew D. Zeiler and Rob Fergus in their paper "Visualizing and Understanding Convolutional Networks" in 2013. ZFNet brought several innovations that enhanced the understanding and performance of convolutional networks, primarily focusing on adjustments to filter size, stride, and the overall architecture.

### Key Features of ZFNet:

1. **Improved Filter Sizes and Strides**: 
   - ZFNet uses larger receptive fields in the initial convolutional layers compared to AlexNet. This allows the network to capture more detailed features from the input images at the early stages.
   - The first convolutional layer uses 96 filters of size 7x7 with a stride of 2. This is a significant change from AlexNet's 11x11 filters with a stride of 4, which provides a more detailed analysis of the input data.

2. **Adjustments to the Convolutional Layers**:
   - ZFNet has five convolutional layers, where the first two layers have larger filters and strides, and the subsequent layers use smaller 3x3 filters, allowing for a more complex hierarchical feature extraction.
   - The architecture reduces the stride in the initial layers, leading to higher resolution feature maps and better feature representation.

3. **Deconvolutional Layers for Visualization**:
   - One of the notable contributions of ZFNet is the introduction of deconvolutional layers for visualizing intermediate activations and understanding what each layer of the network learns. This helped in gaining insights into how convolutional networks process information.

### ZFNet Architecture Details:

1. **Input Layer**: The network takes input images of size 224x224 pixels.
2. **Convolutional Layer 1 (C1)**: 96 filters of size 7x7, stride 2, followed by ReLU activation.
3. **Max Pooling Layer 1 (P1)**: 3x3 filter with stride 2.
4. **Convolutional Layer 2 (C2)**: 256 filters of size 5x5, stride 2, followed by ReLU activation.
5. **Max Pooling Layer 2 (P2)**: 3x3 filter with stride 2.
6. **Convolutional Layer 3 (C3)**: 384 filters of size 3x3, stride 1, followed by ReLU activation.
7. **Convolutional Layer 4 (C4)**: 384 filters of size 3x3, stride 1, followed by ReLU activation.
8. **Convolutional Layer 5 (C5)**: 256 filters of size 3x3, stride 1, followed by ReLU activation.
9. **Max Pooling Layer 3 (P3)**: 3x3 filter with stride 2.
10. **Fully Connected Layer 1 (F6)**: 4096 units with ReLU activation.
11. **Fully Connected Layer 2 (F7)**: 4096 units with ReLU activation.
12. **Output Layer**: 1000 units with softmax activation for classification into 1000 categories.

### Contributions to Convolutional Networks:

- **Visualization Techniques**: By introducing deconvolutional layers, ZFNet provided a method to visualize the learned features and understand how different layers contribute to the final decision, significantly advancing the interpretability of CNNs.
- **Performance Improvements**: The architectural changes, including the use of larger filters in initial layers and better stride management, resulted in improved performance over AlexNet on the ImageNet dataset.

ZFNet's innovations and refinements over AlexNet helped in pushing the boundaries of convolutional neural networks, contributing to the development of more advanced and interpretable deep learning models.

### Parameter Initialization

Parameters (weights and biases) are initialized using a random normal distribution. This is critical to start the training process with diverse initial values.

### ZFNet Function

The ZFNet function computes the output by performing convolution operations followed by activation functions and pooling layers. The final fully connected layers use a softmax function to produce probabilities for each class.

## Training and Evaluation

### Loss Function

The cross-entropy loss function is used to measure the performance of the neural network. It calculates the difference between the predicted probabilities and the actual labels.

### Optimization with Adam

The Adam optimizer, an adaptive learning rate optimization algorithm, is used to update network parameters. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

### Accuracy Calculation

The accuracy of the model is calculated by comparing the predicted class labels with the actual labels and determining the proportion of correct predictions.

## Mathematical Notations

1. **ZFNet Architecture**

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
2. **Initialize the Network**: Create an instance of the `ZFNet` class with desired parameters.
3. **Train the Network**: Call the `train` method with the training and test data to start the training process.
4. **Evaluate the Network**: Monitor the loss and accuracy during training to evaluate the model performance.

By following these steps, you can train a ZFNet model using JAX and apply it to various machine learning tasks.