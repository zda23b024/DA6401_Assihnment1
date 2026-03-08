"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from ann import activations


class NeuralLayer:
    def __init__(self, input_size, output_size, activation='relu', weight_init='xavier'):
        """
        Initialize weights, biases, and activation function
        """
        self.input_size = input_size
        self.output_size = output_size

        # -----------------------
        # Weight Initialization
        # -----------------------
        if weight_init == 'xavier':
           self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
           self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # -----------------------
        # Activation Selection
        # -----------------------
        self.activation_name = activation

        if activation == 'relu':
            self.activation = activations.relu
            self.activation_derivative = activations.relu_derivative

        elif activation == 'sigmoid':
            self.activation = activations.sigmoid
            self.activation_derivative = activations.sigmoid_derivative

        elif activation == 'tanh':
            self.activation = activations.tanh
            self.activation_derivative = activations.tanh_derivative

        elif activation == 'softmax':
            self.activation = activations.softmax
            self.activation_derivative = None  # handled in cross-entropy loss

        else:
            raise ValueError("Unsupported activation function")

        # -----------------------
        # Caches for Backprop
        # -----------------------
        self.input = None
        self.Z = None
        self.A = None

    # ------------------------
    # Forward Pass
    # ------------------------
    def forward(self, X):
        """
        Forward pass
        X: input matrix (batch_size x input_size)
        """
        self.input = X
        self.Z = np.dot(X, self.W) + self.b

        if self.activation_name == 'softmax':
            self.A = self.activation(self.Z)  # softmax only at output
        else:
            self.A = self.activation(self.Z)

        return self.A

    # ------------------------
    # Backward Pass
    # ------------------------
    def backward(self, dA):
        """
        Backward pass
        dA: gradient of loss w.r.t activation output
        """
        m = self.input.shape[0]

        if self.activation_name == 'softmax':
            # derivative already handled by cross-entropy
            dZ = dA
        else:
            dZ = dA * self.activation_derivative(self.Z)

        # Gradients
        self.grad_W = np.dot(self.input.T, dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m

        # Gradient for previous layer
        dX = np.dot(dZ, self.W.T)

        return dX
