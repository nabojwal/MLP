"""
Data Loading and intensity normalization steps
Handles MNIST and Fashion-MNIST datasets for assignment-1
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name, validation_split=0.1, random_seed=42):
    """
    Args:
        dataset_name: "mnist" or "fashion_mnist"
        validation_split: fraction of training data for validation
        random_seed: An int for reproducibility

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """

    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset not matching to: MNIST or fashion-MNIST")

   
    #Image Normalization
    
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0


    #Flattening the images

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

   
    # One-hot encodeing of labels

    num_classes = 10

    y_train_onehot = np.zeros((y_train.shape[0], num_classes))
    y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

    y_test_onehot = np.zeros((y_test.shape[0], num_classes))
    y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1


    #Train-Validation spliting

    X_train, X_val, y_train_onehot, y_val = train_test_split(
        X_train,
        y_train_onehot,
        test_size=validation_split,
        random_state=random_seed,
        shuffle=True,
        stratify=y_train
    )

    return X_train, y_train_onehot, X_val, y_val, X_test, y_test_onehot