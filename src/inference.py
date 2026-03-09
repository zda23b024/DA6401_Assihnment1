"""Inference Script
Evaluate trained models on test sets
"""

import argparse

import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from ann import activations
from utils.data_loader import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--model_path", type=str, default="best_model.npy", help="Path to the saved model (.npy)")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the config file (.json)")


    return parser.parse_args()


def load_model(model_path, config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create args object from config
    class Args:
        pass

    args = Args()
    for key, value in config.items():
        setattr(args, key, value)

    # Initialize model
    model = NeuralNetwork(args)

    # Load weights from numpy file
    weights_data = np.load(model_path, allow_pickle=True)
    
    # Handle both dictionary format and tuple array format
    if isinstance(weights_data.item(), dict):
        # Dictionary format (from get_weights())
        weight_dict = weights_data.item()
    else:
        # Tuple array format (legacy)
        weight_dict = {}
        for i, (W, b) in enumerate(weights_data):
            weight_dict[f"W{i}"] = W
            weight_dict[f"b{i}"] = b
    
    model.set_weights(weight_dict)

    return model


def evaluate_model(model, X_test, y_test):
    logits = model.forward(X_test)
    probs = activations.softmax(logits)
    preds = np.argmax(probs, axis=1)
    
    # Convert y_test from one-hot encoding to class indices
    y_test_labels = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test_labels, preds)
    precision = precision_score(y_test_labels, preds, average='macro')
    recall = recall_score(y_test_labels, preds, average='macro')
    f1 = f1_score(y_test_labels, preds, average='macro')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics


def main():
    args = parse_arguments()

    # Load test data (assuming mnist for now, but should be from config)
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    dataset = config.get('dataset', 'mnist')

    _, _, _, _, X_test, y_test = load_dataset(dataset)

    model = load_model(args.model_path, args.config_path)

    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")

    return results


if __name__ == "__main__":
    main()
