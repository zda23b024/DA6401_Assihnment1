"""
Hyperparameter Sweep Script for MNIST Neural Network
Logs to W&B for analysis using Parallel Coordinates plot
Performs 100+ runs with different hyperparameter combinations
"""

import wandb
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def train_with_sweep():
    """Train model with hyperparameters from W&B sweep configuration"""
    
    # Initialize W&B run
    wandb.init()
    
    config = wandb.config
    
    # Use defaults if no sweep config provided
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    num_neurons = config.get("num_neurons", 128)
    hidden_layers = config.get("hidden_layers", 2)
    activation = config.get("activation", "relu")
    weight_init = config.get("weight_init", "xavier")
    epochs = config.get("epochs", 20)
    
    # Ensure num_neurons is a list
    if isinstance(num_neurons, int):
        num_neurons = [num_neurons] * hidden_layers
    
    print(f"\n{'='*70}")
    print(f"Starting run with config:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Neurons: {num_neurons}")
    print(f"  Hidden Layers: {hidden_layers}")
    print(f"  Activation: {activation}")
    print(f"  Weight Init: {weight_init}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*70}\n")
    
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")
    
    # Create args-like object for NeuralNetwork
    class Args:
        pass
    
    args = Args()
    args.input_size = X_train.shape[1]
    args.output_size = 10
    args.hidden_layers = hidden_layers
    args.num_neurons = num_neurons
    args.activation = activation
    args.weight_init = weight_init
    
    # Initialize model
    model = NeuralNetwork(args)
    
    # Training loop with validation
    best_val_accuracy = 0
    patience = 5
    patience_counter = 0
    
    n_train = X_train.shape[0]
    
    for epoch in range(epochs):
        # Mini-batch training
        indices = np.random.permutation(n_train)
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Backward pass
            model.backward(y_batch, y_pred)
            
            # Update weights
            model.update_weights(learning_rate=learning_rate)
        
        # Validation after each epoch
        y_pred_val = model.forward(X_val)
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(y_pred_val_labels == y_val_labels)
        
        # Test accuracy
        y_pred_test = model.forward(X_test)
        y_pred_test_labels = np.argmax(y_pred_test, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(y_pred_test_labels == y_test_labels)
        
        wandb.log({
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        })
        
        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val_accuracy: {best_val_accuracy:.4f}")
            break
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    
    # Final evaluation
    y_pred_test = model.forward(X_test)
    y_pred_test_labels = np.argmax(y_pred_test, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    final_test_accuracy = np.mean(y_pred_test_labels == y_test_labels)
    
    print(f"\nFinal Test Accuracy: {final_test_accuracy:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}\n")
    
    # Log final metrics for sweep analysis
    wandb.log({
        "final_val_accuracy": best_val_accuracy,
        "final_test_accuracy": final_test_accuracy
    })
    
    wandb.finish()


if __name__ == "__main__":
    train_with_sweep()
