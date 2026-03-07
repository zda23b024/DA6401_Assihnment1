"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU activation function
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU
    """
    return np.where(x > 0, 1, 0)


def tanh(x):
    """
    Tanh activation function
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of tanh
    """
    return 1 - np.tanh(x) ** 2


def softmax(x):
    """
    Softmax activation for output layer
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)