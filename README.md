# Data-Driven Clothing for Interactive Applications

## ML Mini Project

This project implements a simplified machine learning approach inspired by the paper **"Data-Driven Clothing for Interactive Applications"** by Kangrui Xue and Jane Wu.

The objective is to learn the relationship between human pose parameters and clothing deformation using machine learning.

## Problem Statement

Traditional physics-based clothing simulation can be computationally expensive. This project explores a data-driven approach in which a machine learning model learns to predict clothing deformation from human pose parameters.

## Dataset

A synthetic, simulation-inspired dataset was generated for this implementation.

- Number of samples: 2,000
- Number of joints: 14
- Quaternion values per joint: 4
- Pose input features: 56
- Clothing vertices: 2,170
- Clothing deformation values: 6,510

The original numerical dataset from the reference paper was not available for this implementation. Therefore, a synthetic dataset was generated following the input/output dimensional structure described in the paper.

## Methodology

The project follows these steps:

1. Generate the pose and clothing deformation dataset.
2. Split the dataset into training, validation, and test sets.
3. Apply PCA to reduce the 6,510-dimensional clothing deformation output.
4. Train Linear Regression as a baseline model.
5. Train a fully connected neural network.
6. Evaluate the models using Mean Squared Error (MSE).
7. Convert the neural network predictions back to the original 6,510-dimensional space.

### PCA

The clothing deformation output is reduced from 6,510 dimensions to 128 PCA components.

The 128 components retain approximately 99.9829% of the variance in this implementation.

### Neural Network Architecture

```text
Input: 56
   ↓
Linear: 128
   ↓
ReLU
   ↓
Linear: 256
   ↓
ReLU
   ↓
Linear: 128
