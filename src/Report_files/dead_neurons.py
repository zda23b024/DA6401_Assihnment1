"""
2.5 Dead Neuron Investigation
Using ReLU with high learning rate - monitor for dead neurons
Compare with Tanh activation
"""
import wandb
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def train_and_monitor_dead_neurons(activation, learning_rate, run_name):
    wandb.init(project="da6401-assignment1", name=run_name)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")
    
    # Sample for monitoring activations
    X_sample = X_val[:256]
    
    class Args:
        pass
    
    args = Args()
    args.input_size = 784
    args.output_size = 10
    args.hidden_layers = 2
    args.num_neurons = [128, 128]
    args.activation = activation
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
            model.update_weights(learning_rate=learning_rate)
        
        # Monitor activations on sample
        activations_hidden = model.layers[0].forward(X_sample)  # First hidden layer
        dead_neurons = np.sum(np.all(activations_hidden == 0, axis=0))
        
        # Validation
        y_pred_val = model.forward(X_val)
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(y_pred_val_labels == y_val_labels)
        
        wandb.log({
            "epoch": epoch,
            "dead_neurons_layer1": dead_neurons,
            "val_accuracy": val_acc,
            "activation": activation,
            "learning_rate": learning_rate
        })
    
    wandb.finish()


# Test 1: ReLU with high learning rate (0.1) - likely to have dead neurons
train_and_monitor_dead_neurons("relu", 0.1, "2.5_dead_neurons_relu_high_lr")

# Test 2: ReLU with normal learning rate (0.001) - fewer dead neurons
train_and_monitor_dead_neurons("relu", 0.001, "2.5_dead_neurons_relu_normal_lr")

# Test 3: Tanh with high learning rate - should not have dead neurons
train_and_monitor_dead_neurons("tanh", 0.1, "2.5_no_dead_neurons_tanh_high_lr")