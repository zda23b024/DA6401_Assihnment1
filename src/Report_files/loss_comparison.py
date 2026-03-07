"""
2.6 Loss Function Comparison
Compare MSE vs Cross-Entropy with same architecture
"""
import wandb
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy_loss(y_true, y_pred):
    eps = 1e-7
    return -np.mean(np.sum(y_true * np.log(np.clip(y_pred, eps, 1-eps)), axis=1))


def train_with_loss(loss_name, loss_fn):
    wandb.init(project="da6401-assignment1", name=f"2.6_loss_{loss_name}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")
    
    class Args:
        pass
    
    args = Args()
    args.input_size = 784
    args.output_size = 10
    args.hidden_layers = 2
    args.num_neurons = [128, 128]
    args.activation = 'relu'
    args.weight_init = 'xavier'
    
    model = NeuralNetwork(args)
    
    n_train = X_train.shape[0]
    learning_rate = 0.01
    
    for epoch in range(20):
        indices = np.random.permutation(n_train)
        epoch_loss = []
        
        for i in range(0, n_train, 128):
            batch_indices = indices[i:i+128]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            y_pred = model.forward(X_batch)
            batch_loss = loss_fn(y_batch, y_pred)
            epoch_loss.append(batch_loss)
            
            model.backward(y_batch, y_pred)
            model.update_weights(learning_rate=learning_rate)
        
        # Validation
        y_pred_val = model.forward(X_val)
        val_loss = loss_fn(y_val, y_pred_val)
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(y_pred_val_labels == y_val_labels)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": np.mean(epoch_loss),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "loss_type": loss_name
        })
    
    wandb.finish()


# Compare MSE and Cross-Entropy
train_with_loss("MSE", mse_loss)
train_with_loss("CrossEntropy", cross_entropy_loss)