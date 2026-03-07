"""
2.7 Global Performance Analysis
Training vs Test Accuracy across all hyperparameter combinations
"""
import wandb
import numpy as np
import sys
import os
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def train_and_log(hidden_layers, neurons, activation, run_id):
    run_name = f"2.7_hl{hidden_layers}_n{neurons}_{activation}_run{run_id}"
    wandb.init(project="da6401-assignment1", name=run_name, reinit=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")
    
    class Args:
        pass
    
    args = Args()
    args.input_size = 784
    args.output_size = 10
    args.hidden_layers = hidden_layers
    args.num_neurons = [neurons] * hidden_layers
    args.activation = activation
    args.weight_init = 'xavier'
    
    model = NeuralNetwork(args)
    
    n_train = X_train.shape[0]
    learning_rate = 0.001
    
    for epoch in range(15):
        indices = np.random.permutation(n_train)
        
        for i in range(0, n_train, 128):
            batch_indices = indices[i:i+128]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            y_pred = model.forward(X_batch)
            model.backward(y_batch, y_pred)
            model.update_weights(learning_rate=learning_rate)
        
        # Training accuracy
        y_pred_train = model.forward(X_train)
        y_pred_train_labels = np.argmax(y_pred_train, axis=1)
        y_train_labels = np.argmax(y_train, axis=1)
        train_acc = np.mean(y_pred_train_labels == y_train_labels)
        
        # Test accuracy
        y_pred_test = model.forward(X_test)
        y_pred_test_labels = np.argmax(y_pred_test, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        test_acc = np.mean(y_pred_test_labels == y_test_labels)
        
        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": train_acc - test_acc,
            "hidden_layers": hidden_layers,
            "neurons": neurons,
            "activation": activation
        })
    
    wandb.finish()


# Run multiple configurations
hidden_layers_list = [1, 2, 3]
neurons_list = [64, 128]
activations_list = ['relu', 'tanh', 'sigmoid']

run_id = 0
for hl, n, act in product(hidden_layers_list, neurons_list, activations_list):
    train_and_log(hl, n, act, run_id)
    run_id += 1