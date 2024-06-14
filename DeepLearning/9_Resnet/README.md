# ResNet Implementation in JAX

This project implements the ResNet architecture using JAX, a high-performance numerical computing library. ResNet, short for Residual Network, was introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in the paper "Deep Residual Learning for Image Recognition." ResNet solves the problem of training deep neural networks by introducing residual connections, which allow gradients to flow directly through the network, making it easier to train much deeper networks.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [ResNet Architecture](#resnet-architecture)
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

```
                                                                        ResNet Summary                                                                        
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ path                            ┃ module            ┃ inputs                       ┃ outputs              ┃ batch_stats       ┃ params                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                                 │ ResNet            │ float32[10,28,28,1]          │ float32[10,10]       │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ Conv_0                          │ Conv              │ float32[10,28,28,1]          │ float32[10,28,28,16] │                   │ kernel: float32[3,3,1,16]  │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 144 (576 B)                │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_0             │ PreActResNetBlock │ - float32[10,28,28,16]       │ float32[10,28,28,16] │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_0/BatchNorm_0 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_0/Conv_0      │ Conv              │ float32[10,28,28,16]         │ float32[10,28,28,16] │                   │ kernel: float32[3,3,16,16] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,304 (9.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_0/BatchNorm_1 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_0/Conv_1      │ Conv              │ float32[10,28,28,16]         │ float32[10,28,28,16] │                   │ kernel: float32[3,3,16,16] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,304 (9.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_1             │ PreActResNetBlock │ - float32[10,28,28,16]       │ float32[10,28,28,16] │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_1/BatchNorm_0 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_1/Conv_0      │ Conv              │ float32[10,28,28,16]         │ float32[10,28,28,16] │                   │ kernel: float32[3,3,16,16] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,304 (9.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_1/BatchNorm_1 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_1/Conv_1      │ Conv              │ float32[10,28,28,16]         │ float32[10,28,28,16] │                   │ kernel: float32[3,3,16,16] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,304 (9.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_2             │ PreActResNetBlock │ - float32[10,28,28,16]       │ float32[10,28,28,16] │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_2/BatchNorm_0 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_2/Conv_0      │ Conv              │ float32[10,28,28,16]         │ float32[10,28,28,16] │                   │ kernel: float32[3,3,16,16] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,304 (9.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_2/BatchNorm_1 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_2/Conv_1      │ Conv              │ float32[10,28,28,16]         │ float32[10,28,28,16] │                   │ kernel: float32[3,3,16,16] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,304 (9.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3             │ PreActResNetBlock │ - float32[10,28,28,16]       │ float32[10,14,14,32] │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3/BatchNorm_0 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3/Conv_0      │ Conv              │ float32[10,28,28,16]         │ float32[10,14,14,32] │                   │ kernel: float32[3,3,16,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 4,608 (18.4 KB)            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3/BatchNorm_1 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3/Conv_1      │ Conv              │ float32[10,14,14,32]         │ float32[10,14,14,32] │                   │ kernel: float32[3,3,32,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 9,216 (36.9 KB)            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3/BatchNorm_2 │ BatchNorm         │ - float32[10,28,28,16]       │ float32[10,28,28,16] │ mean: float32[16] │ bias: float32[16]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[16]  │ scale: float32[16]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 32 (128 B)        │ 32 (128 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_3/Conv_2      │ Conv              │ float32[10,28,28,16]         │ float32[10,14,14,32] │                   │ kernel: float32[1,1,16,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 512 (2.0 KB)               │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_4             │ PreActResNetBlock │ - float32[10,14,14,32]       │ float32[10,14,14,32] │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_4/BatchNorm_0 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_4/Conv_0      │ Conv              │ float32[10,14,14,32]         │ float32[10,14,14,32] │                   │ kernel: float32[3,3,32,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 9,216 (36.9 KB)            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_4/BatchNorm_1 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_4/Conv_1      │ Conv              │ float32[10,14,14,32]         │ float32[10,14,14,32] │                   │ kernel: float32[3,3,32,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 9,216 (36.9 KB)            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_5             │ PreActResNetBlock │ - float32[10,14,14,32]       │ float32[10,14,14,32] │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_5/BatchNorm_0 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_5/Conv_0      │ Conv              │ float32[10,14,14,32]         │ float32[10,14,14,32] │                   │ kernel: float32[3,3,32,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 9,216 (36.9 KB)            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_5/BatchNorm_1 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_5/Conv_1      │ Conv              │ float32[10,14,14,32]         │ float32[10,14,14,32] │                   │ kernel: float32[3,3,32,32] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 9,216 (36.9 KB)            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6             │ PreActResNetBlock │ - float32[10,14,14,32]       │ float32[10,7,7,64]   │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6/BatchNorm_0 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6/Conv_0      │ Conv              │ float32[10,14,14,32]         │ float32[10,7,7,64]   │                   │ kernel: float32[3,3,32,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 18,432 (73.7 KB)           │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6/BatchNorm_1 │ BatchNorm         │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │ mean: float32[64] │ bias: float32[64]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[64]  │ scale: float32[64]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 128 (512 B)       │ 128 (512 B)                │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6/Conv_1      │ Conv              │ float32[10,7,7,64]           │ float32[10,7,7,64]   │                   │ kernel: float32[3,3,64,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 36,864 (147.5 KB)          │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6/BatchNorm_2 │ BatchNorm         │ - float32[10,14,14,32]       │ float32[10,14,14,32] │ mean: float32[32] │ bias: float32[32]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[32]  │ scale: float32[32]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 64 (256 B)        │ 64 (256 B)                 │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_6/Conv_2      │ Conv              │ float32[10,14,14,32]         │ float32[10,7,7,64]   │                   │ kernel: float32[1,1,32,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 2,048 (8.2 KB)             │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_7             │ PreActResNetBlock │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_7/BatchNorm_0 │ BatchNorm         │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │ mean: float32[64] │ bias: float32[64]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[64]  │ scale: float32[64]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 128 (512 B)       │ 128 (512 B)                │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_7/Conv_0      │ Conv              │ float32[10,7,7,64]           │ float32[10,7,7,64]   │                   │ kernel: float32[3,3,64,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 36,864 (147.5 KB)          │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_7/BatchNorm_1 │ BatchNorm         │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │ mean: float32[64] │ bias: float32[64]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[64]  │ scale: float32[64]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 128 (512 B)       │ 128 (512 B)                │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_7/Conv_1      │ Conv              │ float32[10,7,7,64]           │ float32[10,7,7,64]   │                   │ kernel: float32[3,3,64,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 36,864 (147.5 KB)          │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_8             │ PreActResNetBlock │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │                   │                            │
│                                 │                   │ - train: True                │                      │                   │                            │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_8/BatchNorm_0 │ BatchNorm         │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │ mean: float32[64] │ bias: float32[64]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[64]  │ scale: float32[64]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 128 (512 B)       │ 128 (512 B)                │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_8/Conv_0      │ Conv              │ float32[10,7,7,64]           │ float32[10,7,7,64]   │                   │ kernel: float32[3,3,64,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 36,864 (147.5 KB)          │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_8/BatchNorm_1 │ BatchNorm         │ - float32[10,7,7,64]         │ float32[10,7,7,64]   │ mean: float32[64] │ bias: float32[64]          │
│                                 │                   │ - use_running_average: False │                      │ var: float32[64]  │ scale: float32[64]         │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │ 128 (512 B)       │ 128 (512 B)                │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ PreActResNetBlock_8/Conv_1      │ Conv              │ float32[10,7,7,64]           │ float32[10,7,7,64]   │                   │ kernel: float32[3,3,64,64] │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 36,864 (147.5 KB)          │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│ Dense_0                         │ Dense             │ float32[10,64]               │ float32[10,10]       │                   │ bias: float32[10]          │
│                                 │                   │                              │                      │                   │ kernel: float32[64,10]     │
│                                 │                   │                              │                      │                   │                            │
│                                 │                   │                              │                      │                   │ 650 (2.6 KB)               │
├─────────────────────────────────┼───────────────────┼──────────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────┤
│                                 │                   │                              │                Total │ 1,344 (5.4 KB)    │ 271,962 (1.1 MB)           │
└─────────────────────────────────┴───────────────────┴──────────────────────────────┴──────────────────────┴───────────────────┴────────────────────────────┘
                                                                                                                                                              
                                                              Total Parameters: 273,306 (1.1 MB)                                                              

```

## ResNet Architecture

ResNet, introduced by Kaiming He et al., is characterized by the use of residual connections, which help in training very deep networks by allowing the network to learn residual functions instead of directly learning the desired underlying mappings.

### Key Features of ResNet:

1. **Residual Connections**:
   - Each residual block contains a shortcut connection that bypasses one or more layers, enabling the construction of very deep networks.

2. **Identity Mapping**:
   - The shortcut connections perform identity mapping, and their outputs are added to the outputs of the stacked layers.

3. **Bottleneck Design**:
   - For deeper networks, a bottleneck design is used to reduce the number of parameters while maintaining computational efficiency.

### ResNet Architecture Details:

1. **Input Layer**: The network takes input images of size 224x224 pixels.
2. **Initial Convolution and Pooling**:
   - Convolutional layer with 64 filters of size 7x7, stride 2, followed by max pooling.
3. **Residual Blocks**:
   - Each residual block consists of multiple convolutional layers with a shortcut connection that adds the input to the block to the output.
   - Bottleneck blocks are used for deeper architectures (e.g., ResNet-50, ResNet-101).
4. **Final Layers**:
   - Global average pooling followed by a fully connected layer with softmax activation for classification.

### Contributions to Convolutional Networks:

- **Residual Learning**: ResNet's residual connections help mitigate the vanishing gradient problem, enabling the training of very deep networks.
- **Efficiency**: The bottleneck design in ResNet allows for deeper architectures with fewer parameters.

## Training and Evaluation

### Loss Function

The cross-entropy loss function is used to measure the performance of the neural network. It calculates the difference between the predicted probabilities and the actual labels.

### Optimization with Adam

The Adam optimizer, an adaptive learning rate optimization algorithm, is used to update network parameters. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

### Accuracy Calculation

The accuracy of the model is calculated by comparing the predicted class labels with the actual labels and determining the proportion of correct predictions.

## Mathematical Notations

1. **ResNet Architecture**

   - Input layer: $X$
   - Convolutional layers: $C_i$ where $i$ is the layer index
   - Pooling layers: $P_i$ where $i$ is the layer index
   - Residual blocks: $R_j$ where $j$ is the block index
   - Fully connected layers: $F_k$ where $k$ is the layer index
   - Output layer: $O$
   - Weights: $W_i$
   - Biases: $b_i$
   - Activation function: $\sigma$

2. **Residual Block**

   - For each residual block $j$:
     $$R_j = \sigma(\text{conv}(R_{j-1}, W_{j,1}) + b_{j,1})$$
     $$R_j = \sigma(\text{conv}(R_j, W_{j,2}) + b_{j,2})$$
     - With shortcut connection:
     $$R_j = R_j + R_{j-1}$$

3. **Convolution Operation**

   - For each convolutional layer $i$:
     $$C_i = \sigma(\text{conv}(C_{i-1}, W_i) + b_i)$$

4. **Pooling Operation**

   - For each pooling layer $i$:
     $$P_i = \text{pool}(C_i)$$

5. **Fully Connected Layers**

   - Flatten the output of the last pooling layer and pass through fully connected layers:
     $$F_k = \sigma(P_{m-1} W_k + b_k)$$

6. **Output Layer**

   - The final layer uses softmax activation:
     $$O = \text{softmax}(F_{n-1} W_n + b_n)$$

7. **Loss Function (Cross-Entropy)**

   $$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$$

   where $N$ is the number of samples, $C$ is the number of classes, $y$ is the true label, and $\hat{y}$ is the predicted probability.

8. **Adam Optimizer Update**

   The Adam optimizer adjusts the learning rate for each parameter individually based on the first moment (mean) and the second moment (uncentered variance) of the gradients.

   - Compute gradients: $g_t$
   - Update biased first moment estimate: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   - Update biased second moment estimate: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   - Compute bias-corrected first moment estimate: $\hat{m}_t = m_t / (1 - \beta_1^t)$
   - Compute bias-corrected second moment estimate: $\hat{v}_t = v_t / (1 - \beta_2^t)$
   - Update parameters: $\theta_t = \theta_{t-1} - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

## Running the Code

1. **Prepare the Dataset**: Load and preprocess the dataset using the `Dataset` class.
2. **Initialize the Network**: Create an instance of the `ResNet` class with desired parameters.
3. **Train the Network**: Call the `train` method with the training and test data to start the training process.
4. **Evaluate the Network**: Monitor the loss and accuracy during training to evaluate the model performance.

By following these steps, you can train a ResNet model using JAX and apply it to various machine learning tasks.