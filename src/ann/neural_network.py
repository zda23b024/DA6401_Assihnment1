"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from ann.neural_layer import NeuralLayer
from ann import activations
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam



class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize neural network layers based on CLI arguments
        """
        self.input_size = cli_args.input_size if hasattr(cli_args, "input_size") else 784
        self.output_size = 10
        self.num_neurons = getattr(cli_args, "num_neurons", [128, 128, 128, 64])
        
        
        if isinstance(self.num_neurons, int):
            self.num_neurons = [self.num_neurons]
            
        self.hidden_layers = getattr(cli_args, "hidden_layers", len(self.num_neurons))
        self.num_neurons = list(self.num_neurons[:self.hidden_layers])
        self.activation = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.optimizer_name = getattr(cli_args, "optimizer", "sgd").lower()
        self.learning_rate = getattr(cli_args, "learning_rate", 0.001)

        self.layers = []
        self._build_layers(self.input_size)
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        """Create optimizer instance from CLI args."""
        if self.optimizer_name == "sgd":
            return SGD(lr=self.learning_rate)
        if self.optimizer_name == "momentum":
            return Momentum(lr=self.learning_rate)
        if self.optimizer_name == "nag":
            return NAG(lr=self.learning_rate)
        if self.optimizer_name == "rmsprop":
            return RMSProp(lr=self.learning_rate)
        if self.optimizer_name == "adam":
            return Adam(lr=self.learning_rate)
        if self.optimizer_name == "nadam":
            return Nadam(lr=self.learning_rate)
        return SGD(lr=self.learning_rate)

    def _build_layers(self, input_size):
        """Build network layers for a given input size."""
        self.layers = []
        prev_size = input_size
     
        for num_neurons in self.num_neurons:

        # output layer (softmax for multi-class classification)
           self.layers.append(
                NeuralLayer(prev_size, num_neurons, activation=self.activation, weight_init=self.weight_init)
             )
           prev_size = num_neurons
        
        self.layers.append(
            NeuralLayer(prev_size, self.output_size, activation='softmax', weight_init=self.weight_init)
        )
        self.input_size = input_size

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # adapt only when input size is inconsistent with the configured model
        if len(self.layers) > 0 and X.shape[1] != self.layers[0].input_size:
            self._build_layers(X.shape[1])
        
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

        # derivative of loss w.r.t logits (assuming cross-entropy with sigmoid/softmax)
        dA = y_pred - y_true  # shape: (batch_size, num_classes)

        # backprop through layers in reverse
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        # reverse so index 0 = last layer
        grad_W_list = grad_W_list[::-1]
        grad_b_list = grad_b_list[::-1]

        # store explicitly
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self, learning_rate=0.001):
        """
         Update weights using the configured optimizer.        
         
        """
        if learning_rate != self.learning_rate:
            self.learning_rate = learning_rate
            self.optimizer = self._build_optimizer()
        self.optimizer.step(self.layers, self.grad_W, self.grad_b)

    def train(self, X_train, y_train, epochs=1, batch_size=32, learning_rate=0.001, X_val=None, y_val=None, wandb_log=True):
        """
        Basic training loop (mini-batch SGD)
        """
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # shuffle
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

                # update weights
                self.update_weights(learning_rate)

            # evaluate
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
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)
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
        indexed_weights = []
        for key, value in weight_dict.items():
            match = re.fullmatch(r"W(\d+)", str(key))
            if match:
                idx = int(match.group(1))
                b_key = f"b{idx}"
                indexed_weights.append((idx, np.asarray(value), np.asarray(weight_dict.get(b_key, None))))

        if not indexed_weights:
            return

        indexed_weights.sort(key=lambda x: x[0])
        weights = [item[1] for item in indexed_weights]

        # Detect whether incoming matrices follow (in, out) or (out, in)
        in_out_chain = all(weights[i].shape[1] == weights[i + 1].shape[0] for i in range(len(weights) - 1))
        out_in_chain = all(weights[i].shape[0] == weights[i + 1].shape[1] for i in range(len(weights) - 1))

        transpose_on_load = False
        if not in_out_chain and out_in_chain:
            transpose_on_load = True

        inferred_input = weights[0].shape[1] if transpose_on_load else weights[0].shape[0]
        inferred_output = weights[-1].shape[0] if transpose_on_load else weights[-1].shape[1]
        inferred_hidden = [w.shape[0] if transpose_on_load else w.shape[1] for w in weights[:-1]]

        needs_rebuild = (
            len(self.layers) != len(weights)
            or any(layer.W.shape != (w.T.shape if transpose_on_load else w.shape) for layer, w in zip(self.layers, weights))
        )

        if needs_rebuild:
            self.output_size = inferred_output
            self.num_neurons = inferred_hidden
            self._build_layers(inferred_input)

        for layer, (_, W, b) in zip(self.layers, indexed_weights):
            W_to_set = W.T if transpose_on_load else W
            if layer.W.shape != W_to_set.shape:
                raise ValueError(f"Weight shape mismatch for layer: expected {layer.W.shape}, got {W_to_set.shape}")
            layer.W = W_to_set.copy()

            if b is not None:
                b = np.asarray(b)
                if b.ndim == 1:
                    b = b.reshape(1, -1)
                if b.shape != layer.b.shape:
                    raise ValueError(f"Bias shape mismatch for layer: expected {layer.b.shape}, got {b.shape}")
                layer.b = b.copy()
