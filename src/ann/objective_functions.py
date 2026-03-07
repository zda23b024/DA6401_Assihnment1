"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


def cross_entropy_loss(y_true, y_pred):
    """
    Computes cross-entropy loss.
    y_true: one-hot labels (batch_size x num_classes)
    y_pred: predicted logits (batch_size x num_classes)
    """
    # clip to avoid log(0)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    return loss


def cross_entropy_derivative(y_true, y_pred):
    """
    Derivative of cross-entropy w.r.t logits
    Assumes softmax/sigmoid output is already applied
    """
    # gradient shape same as y_pred
    grad = (y_pred - y_true) / y_true.shape[0]
    return grad


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error
    """
    loss = np.mean((y_true - y_pred) ** 2)
    return loss


def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE
    """
    grad = 2 * (y_pred - y_true) / y_true.shape[0]
    return grad