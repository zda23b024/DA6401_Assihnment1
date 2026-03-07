"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def one_hot_encode(y, num_classes=10):
    """
    Convert labels to one-hot encoding
    """
    return np.eye(num_classes)[y]


def load_dataset(name='mnist', test_size=0.2, random_state=42):
    """
    Load and preprocess dataset
    Arguments:
        name: 'mnist' or 'fashion_mnist'
        test_size: fraction of training set to use as validation
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    if name.lower() == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name.lower() == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # normalize images to [0,1] and flatten
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)  # flatten 28x28 -> 784
    X_test = X_test.reshape(X_test.shape[0], -1)

    # one-hot encode labels
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # split train into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state, shuffle=True
    )

    return X_train, y_train, X_val, y_val, X_test, y_test