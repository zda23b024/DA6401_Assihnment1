# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

This assignment requires you to implement a neural network from scratch using only NumPy. You will build all components including layers, activations, optimizers, and loss functions, then train your network on MNIST or Fashion-MNIST datasets.

## Learning Objectives

- Understand forward and backward propagation
- Implement gradient computation manually
- Implement various optimizers (SGD, Momentum, Adam, Nadam)
- Work with activation functions and their derivatives
- Train and evaluate neural networks
- Log experiments using Weights & Biases

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Train Model

To train the model, navigate to the `src` directory and run:
```bash
python train.py --dataset mnist --epochs 10 --batch_size 32 --learning_rate 0.001
```

### Example:
```bash
python train.py --dataset fashion_mnist --epochs 20 --batch_size 64 --learning_rate 0.0005
```

## Run Inference

To evaluate the trained model, run:
```bash
python inference.py --model_path best_model.npy --dataset mnist
```

### Example:
```bash
python inference.py --model_path best_model.npy --dataset fashion_mnist
```

## Files

- **train.py**: Training script for the neural network.
- **inference.py**: Script for evaluating the trained model.
- **best_model.npy**: Saved weights of the best-performing model.
- **best_config.json**: Hyperparameters of the best model.
