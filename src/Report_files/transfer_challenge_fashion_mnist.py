"""
Transfer Challenge: MNIST to Fashion-MNIST (5 Marks)
=====================================================

Task: Based on MNIST experiments, choose 3 hyperparameter configurations
and apply them to Fashion-MNIST. Analyze if the best MNIST config transfers well.

Key Questions:
1. Which 3 configs worked best on MNIST?
2. Do they also work well on Fashion-MNIST?
3. Why does dataset complexity affect hyperparameter effectiveness?
"""

import wandb
import numpy as np
import sys
import os
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam
from utils.data_loader import load_dataset


def train_config_on_fashion_mnist(config_id, hidden_layers, num_neurons, 
                                   activation, optimizer_name, epochs=20):
    """
    Train a single configuration on Fashion-MNIST
    """
    run_name = f"transfer_fashion_mnist_config{config_id}_{optimizer_name}"
    wandb.init(project="da6401-assignment1", name=run_name, reinit=True)
    
    # Load Fashion-MNIST (use "fashion_mnist" instead of "mnist")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("fashion_mnist")
    
    # Create model
    class Args:
        pass
    
    args = Args()
    args.input_size = 784
    args.output_size = 10
    args.hidden_layers = hidden_layers
    args.num_neurons = [num_neurons] * hidden_layers if isinstance(num_neurons, int) else num_neurons
    args.activation = activation
    args.weight_init = 'xavier'
    
    model = NeuralNetwork(args)
    
    # Create optimizer
    optimizer_map = {
        'sgd': SGD(lr=0.001),
        'momentum': Momentum(lr=0.001, momentum=0.9),
        'nag': NAG(lr=0.001, momentum=0.9),
        'rmsprop': RMSProp(lr=0.001),
        'adam': Adam(lr=0.001),
        'nadam': Nadam(lr=0.001)
    }
    optimizer = optimizer_map.get(optimizer_name, SGD(lr=0.001))
    
    # Training loop
    n_train = X_train.shape[0]
    batch_size = 32
    best_val_acc = 0
    best_test_acc = 0
    
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
            
            # Update weights with optimizer
            optimizer.step(model.layers, model.grad_W, model.grad_b)
        
        # Validation accuracy
        y_pred_val = model.forward(X_val)
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(y_pred_val_labels == y_val_labels)
        
        # Test accuracy
        y_pred_test = model.forward(X_test)
        y_pred_test_labels = np.argmax(y_pred_test, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        test_acc = np.mean(y_pred_test_labels == y_test_labels)
        
        best_val_acc = max(best_val_acc, val_acc)
        best_test_acc = max(best_test_acc, test_acc)
        
        wandb.log({
            "epoch": epoch,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "dataset": "Fashion-MNIST"
        })
    
    # Log final metrics
    wandb.log({
        "config_id": config_id,
        "hidden_layers": hidden_layers,
        "num_neurons": num_neurons,
        "activation": activation,
        "optimizer": optimizer_name,
        "final_val_accuracy": best_val_acc,
        "final_test_accuracy": best_test_acc,
        "dataset": "Fashion-MNIST"
    })
    
    wandb.finish()
    
    return {
        'config_id': config_id,
        'val_acc': best_val_acc,
        'test_acc': best_test_acc,
        'hidden_layers': hidden_layers,
        'num_neurons': num_neurons,
        'activation': activation,
        'optimizer': optimizer_name
    }


def transfer_challenge():
    """
    Run transfer challenge with 3 recommended configurations
    
    Based on typical MNIST learnings:
    - Config 1: Deep + ReLU + Adam (usually works well)
    - Config 2: Medium + Tanh + Momentum (stable alternative)
    - Config 3: Shallow + ReLU + RMSProp (fast training)
    """
    
    print("\n" + "="*70)
    print("FASHION-MNIST TRANSFER CHALLENGE (5 Marks)")
    print("="*70)
    print("\nTesting 3 configurations based on MNIST learnings...\n")
    
    # Define the 3 configurations
    # MODIFY THESE based on your MNIST experiment findings
    configs = [
        {
            'id': 1,
            'name': 'Deep + ReLU + Adam',
            'hidden_layers': 3,
            'num_neurons': 128,
            'activation': 'relu',
            'optimizer': 'adam',
            'description': 'Deeper network with ReLU and Adam - typically best for complex patterns'
        },
        {
            'id': 2,
            'name': 'Medium + Tanh + Momentum',
            'hidden_layers': 2,
            'num_neurons': 128,
            'activation': 'tanh',
            'optimizer': 'momentum',
            'description': 'Balanced architecture with Tanh and Momentum - stable regularization'
        },
        {
            'id': 3,
            'name': 'Shallow + ReLU + RMSProp',
            'hidden_layers': 2,
            'num_neurons': 64,
            'activation': 'relu',
            'optimizer': 'rmsprop',
            'description': 'Lighter network with adaptive learning - good for varied features'
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'─'*70}")
        print(f"Config {config['id']}: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Architecture: {config['hidden_layers']} layers × {config['num_neurons']} neurons")
        print(f"Activation: {config['activation']}, Optimizer: {config['optimizer']}")
        print(f"Training on Fashion-MNIST...")
        
        result = train_config_on_fashion_mnist(
            config_id=config['id'],
            hidden_layers=config['hidden_layers'],
            num_neurons=config['num_neurons'],
            activation=config['activation'],
            optimizer_name=config['optimizer'],
            epochs=20
        )
        
        results.append({**result, 'description': config['description']})
        
        print(f"✓ Val Acc: {result['val_acc']:.4f} | Test Acc: {result['test_acc']:.4f}")
    
    # Summary analysis
    print(f"\n{'='*70}")
    print("TRANSFER ANALYSIS SUMMARY")
    print(f"{'='*70}\n")
    
    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    print("Results ranked by Test Accuracy:\n")
    for i, r in enumerate(results_sorted, 1):
        rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{rank_symbol} Config {r['config_id']}: {r['test_acc']:.4f} test accuracy")
        print(f"   {r['num_neurons']} neurons × {r['hidden_layers']} layers, {r['activation']}, {r['optimizer']}\n")
    
    best_config = results_sorted[0]
    worst_config = results_sorted[-1]
    
    print("\nKEY FINDINGS:\n")
    print(f"Best: Config {best_config['config_id']} ({best_config['test_acc']:.4f})")
    print(f"Worst: Config {worst_config['config_id']} ({worst_config['test_acc']:.4f})")
    print(f"Performance Range: {(best_config['test_acc'] - worst_config['test_acc']):.4f}\n")
    
    # Analysis questions
    print("ANALYSIS QUESTIONS:\n")
    print("1. Did the MNIST best config also work best on Fashion-MNIST?")
    print("   → Compare your MNIST findings with these Fashion-MNIST results\n")
    
    print("2. Why might Fashion-MNIST need different hyperparameters than MNIST?")
    print("   → Fashion items are more varied in appearance than digits")
    print("   → More fine-grained features (textures, patterns)")
    print("   → Greater intra-class variation (shirts, shirts, shirts...)\n")
    
    print("3. What complexity factors affect hyperparameter effectiveness?")
    print("   → Dataset visual complexity")
    print("   → Feature diversity within classes")
    print("   → Within-class vs between-class variance")
    print("   → Non-linear decision boundaries\n")
    
    # Log comparison to W&B
    wandb.init(project="da6401-assignment1", name="transfer_challenge_summary", reinit=True)
    
    for r in results_sorted:
        wandb.log({
            "config_id": r['config_id'],
            "test_accuracy": r['test_acc'],
            "val_accuracy": r['val_acc'],
            "architecture": f"{r['hidden_layers']}x{r['num_neurons']}_{r['activation']}",
            "optimizer": r['optimizer'],
            "dataset": "Fashion-MNIST"
        })
    
    wandb.finish()
    
    print("\n" + "="*70)
    print("Transfer challenge complete! Check W&B dashboard for detailed logs.")
    print("="*70 + "\n")


if __name__ == "__main__":
    transfer_challenge()
