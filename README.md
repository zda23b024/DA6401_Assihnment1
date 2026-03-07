Here is a **clean, simple, and neat README** based on your text. I **kept it short**, **added how to run**, **added the automation with W&B**, **removed unnecessary parts**, and **added your report link and author**.

You can paste this directly as your **README.md**.

---

# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

This project implements a **Multi-Layer Perceptron (MLP)** neural network from scratch using **NumPy**.
The goal is to understand how neural networks work internally by manually implementing forward propagation, backpropagation, and weight updates without using deep learning libraries such as TensorFlow or PyTorch.

The model is trained to classify images from the **MNIST** or **Fashion-MNIST** datasets.

Experiments and results are tracked using **Weights & Biases (W&B)**.

---

# Learning Objectives

The main objectives of this assignment are:

* Understand forward propagation in neural networks
* Implement backpropagation and gradient computation manually
* Implement different optimizers such as SGD, Momentum, Adam, and Nadam
* Work with activation functions and their derivatives
* Train and evaluate neural networks on image classification tasks
* Track experiments using Weights & Biases

---

# Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Required packages include:

* numpy
* matplotlib
* pandas
* scikit-learn
* wandb

---

# Training the Model

To train the neural network, run the training script from the **src** directory.

```bash
python train.py --dataset mnist --epochs 10 --batch_size 32 --learning_rate 0.001
```

Example using Fashion-MNIST:

```bash
python train.py --dataset fashion_mnist --epochs 20 --batch_size 64 --learning_rate 0.0005
```

The training script will:

* Load and preprocess the dataset
* Train the neural network
* Log metrics such as accuracy and loss to Weights & Biases
* Save the trained model weights

The best model weights are saved as:

```
best_model.npy
```

---

# Running Inference

To evaluate the trained model, run:

```bash
python inference.py --model_path best_model.npy --dataset mnist
```

Example:

```bash
python inference.py --model_path best_model.npy --dataset fashion_mnist
```

This script loads the trained weights and evaluates the model on the test dataset.

---

# Automation with Weights & Biases

Hyperparameter experiments are automated using **Weights & Biases sweeps**.

Sweeps allow multiple training runs with different hyperparameter combinations to be executed automatically. This helps identify the best performing model configuration.

The sweep experiments explore parameters such as:

* Learning rate
* Batch size
* Number of hidden layers
* Number of neurons
* Activation functions
* Optimizers

Each run logs training metrics and hyperparameters to the W&B dashboard for comparison and analysis.

---

# Weights & Biases Report

All experiment results and analysis are available in the following report:

https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1/reports/DA6401_Assignment_1-Complete-Analysis-Report--VmlldzoxNjEzNDY1NQ?accessToken=3fe328d5tmjohfgt3bdwginnnz9ucmjzeit1swyy7ooixd49sh5mipfytripf1ei

---

