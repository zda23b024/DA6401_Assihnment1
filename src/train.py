"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json
import os
import wandb

from keras.datasets import mnist, fashion_mnist
from ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('--dataset',
                        type=str,
                        default='mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Mini batch size')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Learning rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer')

    parser.add_argument('--hidden_layers',
                        type=int,
                        default=4,
                        help='Number of hidden layers')

    # <-- accept a space-separated list of neurons per layer
    parser.add_argument('--num_neurons',
                        type=int,
                        nargs='+',
                        required=True,
                        help='Number of neurons in each hidden layer (space-separated list)')

    parser.add_argument('--activation',
                        type=str,
                        default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function')

    parser.add_argument('--loss',
                        type=str,
                        default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss function')

    parser.add_argument('--weight_init',
                        type=str,
                        default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization')

    parser.add_argument('--wandb_project',
                        type=str,
                        default='da6401-assignment1',
                        help='W&B project name')

    parser.add_argument('--model_save_path',
                        type=str,
                        default='best_model.npy',
                        help='Relative path to save model')

    return parser.parse_args()


def load_dataset(dataset_name):
    """
    Load dataset and preprocess it.
    """

    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # one hot encoding
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]

    return X_train, y_train_onehot, X_test, y_test_onehot


def main():
    """
    Main training function.
    """

    args = parse_arguments()

    # initialize wandb
    wandb.init(project=args.wandb_project, config=vars(args))

    print("Loading dataset...")

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    print("Dataset loaded")
    print("Training samples:", X_train.shape[0])

    # Create model using NeuralNetwork class
    model = NeuralNetwork(args)

    print("Model initialized")

    best_accuracy = 0

    for epoch in range(args.epochs):

        print(f"Epoch {epoch+1}/{args.epochs}")

        # Mini-batch training
        n_samples = X_train.shape[0]
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, args.batch_size):
            batch_indices = indices[i:i+args.batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            y_pred = model.forward(X_batch)

            # Backward pass
            model.backward(y_batch, y_pred)

            # Update weights
            model.update_weights(learning_rate=args.learning_rate)

        # Evaluate on test set after each epoch
        y_pred_test = model.forward(X_test)
        y_pred_labels = np.argmax(y_pred_test, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_pred_labels == y_test_labels)

        wandb.log({
            "epoch": epoch,
            "accuracy": accuracy
        })

        print("Accuracy:", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(args.model_save_path) or '.', exist_ok=True)
            
            # Save weights
            weights = model.get_weights()
            np.save(args.model_save_path, np.array(weights, dtype=object))

            with open("config.json", "w") as f:
                json.dump(vars(args), f, indent=4)

    print("Training complete!")
    print("Best accuracy:", best_accuracy)


if __name__ == '__main__':
    main()