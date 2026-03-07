"""
2.4 Vanishing Gradient Analysis
Compare Sigmoid vs ReLU - log gradient norms for first hidden layer
"""
import wandb
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def train_with_activation(activation_name):
    wandb.init(project="da6401-assignment1", name=f"2.4_vanishing_gradients_{activation_name}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")
    
    class Args:
        pass
    
    args = Args()
    args.input_size = 784
    args.output_size = 10
    args.hidden_layers = 3
    args.num_neurons = [128, 128, 128]
    args.activation = activation_name
    args.weight_init = 'xavier'
    
    model = NeuralNetwork(args)
    
    n_train = X_train.shape[0]
    
    for epoch in range(10):
        indices = np.random.permutation(n_train)
        
        for i in range(0, n_train, 32):
            batch_indices = indices[i:i+32]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            y_pred = model.forward(X_batch)
            model.backward(y_batch, y_pred)
            
            # Log gradient norms from first hidden layer
            if len(model.grad_W) > 0:
                grad_norm_layer1 = np.linalg.norm(model.grad_W[0])
                wandb.log({
                    "epoch": epoch,
                    "batch": epoch * (n_train // 32) + (i // 32),
                    "grad_norm_layer1": grad_norm_layer1
                })
            
            model.update_weights(learning_rate=0.001)
        
        # Validation
        y_pred_val = model.forward(X_val)
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(y_pred_val_labels == y_val_labels)
        
        wandb.log({"epoch_val_accuracy": val_acc})
    
    wandb.finish()


# Compare Sigmoid vs ReLU
train_with_activation("sigmoid")
train_with_activation("relu")