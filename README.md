Below is a **cleaned version of your README** with **all emojis removed**, **unnecessary tables removed**, and **simplified formatting** while keeping the important information needed for the assignment.

---

# DA6401 Assignment 1: Neural Network Implementation from Scratch

## Overview

This assignment involves building a custom neural network from scratch (without TensorFlow or Keras) and conducting hyperparameter optimization experiments on the MNIST and Fashion-MNIST datasets.

Project link:
[https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1](https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1)

---

# Model Architecture and Hyperparameters

## Activations Implemented

* ReLU – commonly used for deep networks
* Sigmoid – classical activation but can suffer from vanishing gradients
* Tanh – centered around zero
* Softmax – used in the output layer for multi-class classification

## Optimizers Implemented

* SGD
* Momentum
* NAG
* RMSProp
* Adam
* Nadam

Adam with learning rate 0.001 is typically the most stable for this assignment.

## Weight Initialization Methods

* Xavier (Glorot) initialization
* Random initialization

---

# Dataset Information

## MNIST Dataset

* 60,000 training samples
* 10,000 test samples
* Image size: 28 × 28 grayscale
* Input dimension after flattening: 784
* Number of classes: 10
* Pixel values normalized to range [0,1]

Typical accuracy: 93–99 percent.

## Fashion-MNIST Dataset

* 60,000 training samples
* 10,000 test samples
* Image size: 28 × 28 grayscale
* Input dimension after flattening: 784
* Number of classes: 10 clothing categories

Fashion-MNIST is more difficult than MNIST and usually gives lower accuracy.

---

# Default Training Parameters

```
Input Size: 784
Output Size: 10
Hidden Layers: 2–3
Neurons per Layer: 64–128
Activation: ReLU (hidden layers), Softmax (output)
Weight Initialization: Xavier
Optimizer: Adam
Learning Rate: 0.001
Epochs: 20
Batch Size: 32
Validation Split: 20%
```

---

# Hyperparameter Sweep Configuration

The sweep explores multiple combinations of hyperparameters.

Hidden layers: 1, 2, 3
Neurons per layer: 32, 64, 128, 256
Activation functions: relu, tanh, sigmoid
Optimizers: sgd, adam, rmsprop
Learning rates: 0.0001, 0.001, 0.01
Batch sizes: 16, 32, 64

More than 100 runs are executed using Weights and Biases sweep.

---

# Project Structure

```
da6401_assignment_1/

README.md
requirements.txt

src/
    train.py
    inference.py
    config.json
    best_model.npy

    ann/
        neural_network.py
        neural_layer.py
        activations.py
        objective_functions.py
        optimizers.py

    utils/
        data_loader.py

    Report_files/
        sweep_config.yaml
        Data_Exploration.py
        wandb_sweep_train.py
        wandb_2_3_optimizer_showdown.py
        Vanishes_Gradind_Analysis.py
        wandb_2_5_dead_neurons.py
        wandb_2_6_loss_comparison.py
        wandb_2_7_global_performance.py
        wandb_2_8_error_analysis.py
        wandb_2_9_weight_symmetry.py
        transfer_challenge_fashion_mnist.py
```

---

# Quick Start

## Install Dependencies

```
cd da6401_assignment_1
pip install -r requirements.txt
```

Required packages:

* numpy
* matplotlib
* pandas
* wandb
* scikit-learn

---

# Train the Model

Basic training:

```
cd da6401_assignment_1
python src/train.py
```

Example output:

```
Loading MNIST dataset
Training with 60000 samples

Epoch 1/20
Epoch 2/20
...
Epoch 20/20

Model saved to src/best_model.npy
```

Training time is typically 5–10 minutes.

---

# Run Inference

```
cd da6401_assignment_1
python src/inference.py
```

The script loads the trained model and evaluates it on test samples.

---

# Run Analysis Scripts

Each assignment question corresponds to one script.

Data exploration:

```
cd src/Report_files
python Data_Exploration.py
```

Optimizer comparison:

```
python wandb_2_3_optimizer_showdown.py
```

Vanishing gradient analysis:

```
python Vanishes_Gradind_Analysis.py
```

Dead neuron analysis:

```
python wandb_2_5_dead_neurons.py
```

Loss comparison:

```
python wandb_2_6_loss_comparison.py
```

Global performance evaluation:

```
python wandb_2_7_global_performance.py
```

Error analysis:

```
python wandb_2_8_error_analysis.py
```

Weight initialization comparison:

```
python wandb_2_9_weight_symmetry.py
```

Transfer challenge using Fashion-MNIST:

```
python transfer_challenge_fashion_mnist.py
```

---

# Hyperparameter Sweep

Step 1: Initialize sweep

```
cd da6401_assignment_1
wandb sweep src/Report_files/sweep_config.yaml
```

This command returns a sweep ID.

Step 2: Run sweep agents

```
wandb agent zda23m016-iit-madras-zanzibar/da6401-assignment1/sweeps/<SWEEP_ID>
```

You can run multiple agents in parallel using different terminals.

Expected time for 100 runs: about 1–4 hours depending on hardware.

---

# W&B Dashboard

Project link:

[https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1](https://wandb.ai/zda23m016-iit-madras-zanzibar/da6401-assignment1)

The dashboard includes:

* Runs
* Charts
* Sweeps
* Reports
* Artifacts

Metrics logged include training accuracy, validation accuracy, test accuracy, loss values, gradient norms, learning rate, batch size, activation function, optimizer, and weight initialization.

---

# Expected Results

For MNIST:

Best configurations typically achieve 97–99 percent test accuracy using Adam optimizer and ReLU activation.

For Fashion-MNIST:

Best configurations typically achieve 88–91 percent test accuracy.

---

# Key Findings

Adam optimizer consistently performs well across different configurations.

ReLU activation works better than sigmoid or tanh in deeper networks because it reduces vanishing gradient problems.

Networks with 2–3 hidden layers and around 128 neurons per layer usually provide a good balance between performance and complexity.

Batch size around 32 tends to give stable training.

Fashion-MNIST is more complex than MNIST, so hyperparameters optimized for MNIST may not work equally well for Fashion-MNIST.

---

# Troubleshooting

Module import error:

Always run scripts from the project root directory.

```
cd da6401_assignment_1
python src/train.py
```

Missing model file:

Train the model first before running inference.

```
python src/train.py
python src/inference.py
```

W&B login required:

```
wandb login
```

Enter your API key from the W&B website.

---

# Assignment Sections

The assignment contains the following parts:

Data exploration
Hyperparameter sweep
Optimizer comparison
Vanishing gradient analysis
Dead neuron analysis
Loss comparison
Global performance analysis
Error analysis
Weight initialization analysis
Transfer learning experiment with Fashion-MNIST

---

# Requirements

```
numpy
matplotlib
scikit-learn
pandas
wandb
```

Install with:

```
pip install -r requirements.txt
```

---

# Author

Student ID: zda23m016
Institute: IIT Madras
Course: DA6401 – Advanced Deep Learning

---

If you want, I can also **rewrite this README to be more “assignment-friendly” (what professors expect)** so it looks **more like a proper ML project report rather than a software README**. That usually helps in grading.
