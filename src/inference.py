import argparse
import numpy as np
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description="Model Inference")

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--config_path", type=str, required=True)

    return parser.parse_args()


def main():
    """
    Load trained model from disk.
    """
    args = parse_arguments()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    from types import SimpleNamespace
    config = SimpleNamespace(**config)

    _, _, _, _, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(config)

    weights = np.load(args.model_path, allow_pickle=True)

    for layer, saved_layer in zip(model.layers, weights):
        layer.W = saved_layer["W"]
        layer.b = saved_layer["b"]

    logits = model.forward(X_test)

    predictions = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    f1 = f1_score(true_labels, predictions, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()

