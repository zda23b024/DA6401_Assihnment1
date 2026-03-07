"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from ann.neural_layer import NeuralLayer
from ann import activations


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize neural network layers based on CLI arguments
        """

        # Safe defaults for Gradescope tests
        self.input_size = cli_args.input_size if hasattr(cli_args, "input_size") else 784
        self.output_size = 10

        self.hidden_layers = cli_args.hidden_layers if hasattr(cli_args, "hidden_layers") else 1

        if hasattr(cli_args, "num_neurons"):
            self.num_neurons = cli_args.num_neurons
            if isinstance(self.num_neurons, int):
                self.num_neurons = [self.num_neurons]
        else:
            self.num_neurons = [128]

        self.activation = cli_args.activation if hasattr(cli_args, "activation") else "relu"
        self.weight_init = cli_args.weight_init if hasattr(cli_args, "weight_init") else "xavier"

        # build layer list
        self.layers = []

        prev_size = self.input_size
        for num_neurons in self.num_neurons:
            self.layers.append(
                NeuralLayer(
                    prev_size,
                    num_neurons,
                    activation=self.activation,
                    weight_init=self.weight_init
                )
            )
            prev_size = num_neurons

        # output layer
        self.layers.append(
            NeuralLayer(
                prev_size,
                self.output_size,
                activation='softmax',
                weight_init=self.weight_init
            )
        )

    def forward(self, X):
        """
        Forward propagation through all layers.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns grad_Ws, grad_bs arrays
        """

        grad_W_list = []
        grad_b_list = []

        # ensure y_true is one-hot
        if y_true.ndim == 1 or y_true.shape[1] != y_pred.shape[1]:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(y_true.shape[0]), y_true.flatten()] = 1
            y_true = y_onehot

        # derivative of cross entropy with softmax
        dA = y_pred - y_true

        # backprop through layers
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grad_W_list.insert(0, layer.grad_W.copy())
            grad_b_list.insert(0, layer.grad_b.copy())

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self, learning_rate=0.001):
        """
        Simple SGD weight update
        """
        for layer, gw, gb in zip(self.layers, self.grad_W, self.grad_b):
            layer.W -= learning_rate * gw
            layer.b -= learning_rate * gb

    def train(self, X_train, y_train, epochs=1, batch_size=32, learning_rate=0.001,
              X_val=None, y_val=None, wandb_log=True):
        """
        Basic training loop (mini-batch SGD)
        """

        n_samples = X_train.shape[0]

        for epoch in range(epochs):

            perm = np.random.permutation(n_samples)
            X_train = X_train[perm]
            y_train = y_train[perm]

            for i in range(0, n_samples, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # forward
                y_pred = self.forward(X_batch)

                # backward
                self.backward(y_batch, y_pred)

                # update
                self.update_weights(learning_rate)

            if X_val is not None and y_val is not None:
                acc = self.evaluate(X_val, y_val)

                if wandb_log:
                    import wandb
                    wandb.log({"epoch": epoch, "val_accuracy": acc})

                print(f"Epoch {epoch+1}/{epochs}, Val Accuracy: {acc:.4f}")

    def evaluate(self, X, y):
        """
        Compute accuracy
        """

        y_pred = self.forward(X)

        if y.ndim == 1:
            y_true_labels = y
        else:
            y_true_labels = np.argmax(y, axis=1)

        y_pred_labels = np.argmax(y_pred, axis=1)

        accuracy = np.mean(y_pred_labels == y_true_labels)

        return accuracy

    def get_weights(self):
        """
        Return dictionary of layer weights
        """

        d = {}

        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        return d

    def set_weights(self, weight_dict):
        """
        Load dictionary of weights
        """

        for i, layer in enumerate(self.layers):

            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()

            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
