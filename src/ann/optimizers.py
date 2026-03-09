"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""

import numpy as np
from src.ann.optimizers import * 

class Optimizer:
    """
    Base optimizer class. Each optimizer will inherit from this.
    """

    def step(self, layers, grad_W, grad_b):
        """
        Update weights in layers based on gradients.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class SGD(Optimizer):
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, layers, grad_W, grad_b):
        for layer, gw, gb in zip(layers, grad_W, grad_b):
            layer.W -= self.lr * gw
            layer.b -= self.lr * gb


class Momentum(Optimizer):
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v_W = []
        self.v_b = []

    def step(self, layers, grad_W, grad_b):
        if not self.v_W:
            # initialize velocities
            self.v_W = [np.zeros_like(gw) for gw in grad_W]
            self.v_b = [np.zeros_like(gb) for gb in grad_b]

        for i, (layer, gw, gb) in enumerate(zip(layers, grad_W, grad_b)):
            self.v_W[i] = self.momentum * self.v_W[i] - self.lr * gw
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * gb
            layer.W += self.v_W[i]
            layer.b += self.v_b[i]


class NAG(Optimizer):
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v_W = []
        self.v_b = []

    def step(self, layers, grad_W, grad_b):
        if not self.v_W:
            self.v_W = [np.zeros_like(gw) for gw in grad_W]
            self.v_b = [np.zeros_like(gb) for gb in grad_b]

        for i, (layer, gw, gb) in enumerate(zip(layers, grad_W, grad_b)):
            # lookahead step
            lookahead_W = layer.W + self.momentum * self.v_W[i]
            lookahead_b = layer.b + self.momentum * self.v_b[i]

            # update velocities
            self.v_W[i] = self.momentum * self.v_W[i] - self.lr * gw
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * gb

            # update parameters
            layer.W += self.v_W[i]
            layer.b += self.v_b[i]


class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s_W = []
        self.s_b = []

    def step(self, layers, grad_W, grad_b):
        if not self.s_W:
            self.s_W = [np.zeros_like(gw) for gw in grad_W]
            self.s_b = [np.zeros_like(gb) for gb in grad_b]

        for i, (layer, gw, gb) in enumerate(zip(layers, grad_W, grad_b)):
            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * gw ** 2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * gb ** 2
            layer.W -= self.lr * gw / (np.sqrt(self.s_W[i]) + self.eps)
            layer.b -= self.lr * gb / (np.sqrt(self.s_b[i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []
        self.t = 0

    def step(self, layers, grad_W, grad_b):
        if not self.m_W:
            self.m_W = [np.zeros_like(gw) for gw in grad_W]
            self.m_b = [np.zeros_like(gb) for gb in grad_b]
            self.v_W = [np.zeros_like(gw) for gw in grad_W]
            self.v_b = [np.zeros_like(gb) for gb in grad_b]

        self.t += 1
        for i, (layer, gw, gb) in enumerate(zip(layers, grad_W, grad_b)):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * gw
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (gw ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb ** 2)

            m_W_corr = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_corr = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_corr = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_W_corr / (np.sqrt(v_W_corr) + self.eps)
            layer.b -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.eps)


class Nadam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []
        self.t = 0

    def step(self, layers, grad_W, grad_b):
        if not self.m_W:
            self.m_W = [np.zeros_like(gw) for gw in grad_W]
            self.m_b = [np.zeros_like(gb) for gb in grad_b]
            self.v_W = [np.zeros_like(gw) for gw in grad_W]
            self.v_b = [np.zeros_like(gb) for gb in grad_b]

        self.t += 1
        for i, (layer, gw, gb) in enumerate(zip(layers, grad_W, grad_b)):
            # momentum estimates
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * gw
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (gw ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb ** 2)

            m_W_corr = (self.beta1 * self.m_W[i] + (1 - self.beta1) * gw) / (1 - self.beta1 ** self.t)
            m_b_corr = (self.beta1 * self.m_b[i] + (1 - self.beta1) * gb) / (1 - self.beta1 ** self.t)
            v_W_corr = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_W_corr / (np.sqrt(v_W_corr) + self.eps)
            layer.b -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.eps)
