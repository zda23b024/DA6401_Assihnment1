---

# DA6401 Assignment 1

Neural Network Implementation from Scratch

## Overview

This project implements a fully connected neural network from scratch using **NumPy** without using deep learning frameworks such as TensorFlow or PyTorch. The model is trained on the **MNIST dataset** to perform handwritten digit classification.

The implementation includes forward propagation, backpropagation, weight updates, and evaluation. Experiments are tracked and analyzed using **Weights & Biases (W&B)**.

---

# Neural Network Implementation

The neural network consists of:

* Input layer with 784 features (28×28 flattened image)
* One or more hidden layers
* Output layer with 10 neurons representing the digit classes

Activation functions supported:

* ReLU
* Sigmoid
* Tanh
* Softmax (output layer)

Weight initialization methods:

* Xavier initialization
* Random initialization

The network is trained using mini-batch gradient descent.

---

# Automation Using Weights & Biases

The experiments in this project are automated using **Weights & Biases (W&B) sweeps**.
Sweeps allow multiple training runs with different hyperparameter configurations to be executed automatically.

Instead of manually testing different configurations, the sweep system automatically runs experiments and records the results.

The automated sweep explores combinations of:

* Learning rate
* Batch size
* Number of hidden layers
* Number of neurons per layer
* Activation functions
* Weight initialization methods

Each run logs metrics such as:

* Validation accuracy
* Test accuracy
* Training progress across epochs

These results are visualized in the W&B dashboard, which helps identify the best performing hyperparameter configuration.

---

# Running the Training Script

From the project root directory:

```
python src/train.py
```

This will train the neural network on the MNIST dataset and save the model weights.

---

# Running Inference

```
python src/inference.py
```

This script loads the saved model and evaluates its performance on the test dataset.

---



# Experiment Tracking

All experiments are automatically logged to Weights & Biases.
The dashboard provides visualizations of training progress and hyperparameter performance.

Report link:

[https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1/reports](https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1/reports)

---

# Author

Zakariya Yahya
Student ID: Ge26z813

---

If you want, I can also give you a **slightly better academic version (still simple)** that **looks more professional for GitHub submission and grading**.
