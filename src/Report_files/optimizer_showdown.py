"""
2.3 Optimizer Showdown
Compare 6 optimizers: SGD, Momentum, NAG, RMSProp, Adam, Nadam
Same architecture: 3 hidden layers, 128 neurons, ReLU activation
"""
import wandb
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam
from utils.data_loader import load_dataset


def train_with_optimizer(optimizer_class, optimizer_name):
    wandb.init(project="da6401-assignment1", name=f"2.3_{optimizer_name}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset("mnist")
    
    # Fixed architecture: 3 layers × 128 neurons, ReLU
    class Args:
        pass
    
    args = Args()
    args.input_size = 784
    args.output_size = 10
    args.hidden_layers = 3
    args.num_neurons = [128, 128, 128]
    args.activation = 'relu'
    args.weight_init = 'xavier'
    
    model = NeuralNetwork(args)
    optimizer = optimizer_class(lr=0.001)
    
    n_train = X_train.shape[0]
    learning_rate = 0.001
    
    # Train for 5 epochs to compare convergence  
    for epoch in range(5):
        indices = np.random.permutation(n_train)
        
        for i in range(0, n_train, 32):
            batch_indices = indices[i:i+32]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            y_pred = model.forward(X_batch)
            model.backward(y_batch, y_pred)
            
            # Use custom optimizer
            optimizer.step(model.layers, model.grad_W, model.grad_b)
        
        # Validation
        y_pred_val = model.forward(X_val)
        y_pred_val_labels = np.argmax(y_pred_val, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        val_acc = np.mean(y_pred_val_labels == y_val_labels)
        
        # Calculate loss (cross-entropy)
        eps = 1e-7
        loss = -np.mean(np.sum(y_val * np.log(np.clip(y_pred_val, eps, 1-eps)), axis=1))
        
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "val_accuracy": val_acc
        })
    
    wandb.finish()


# Compare all 6 optimizers
optimizers = [
    (SGD, "SGD"),
    (Momentum, "Momentum"),
    (NAG, "NAG"),
    (RMSProp, "RMSProp"),
    (Adam, "Adam"),
    (Nadam, "Nadam")
]

for opt_class, opt_name in optimizers:
    train_with_optimizer(opt_class, opt_name)