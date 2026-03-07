
import argparse
import numpy as np
import json
import os
import wandb

from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a modular MLP for DA6401 Assignment-1"
    )

    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, required=True)

    parser.add_argument("-b", "--batch_size", type=int, required=True)

    parser.add_argument("-lr", "--learning_rate", type=float, required=True)

    parser.add_argument("-o", "--optimizer", type=str, required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])

    parser.add_argument("-nhl", "--num_layers", type=int, required=True)

    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", required=True)

    parser.add_argument("-a", "--activation", type=str, required=True,
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("-l", "--loss", type=str, required=True,
                        choices=["cross_entropy", "mse"])

    parser.add_argument("-wi", "--weight_init", type=str, required=True,
                        choices=["random", "zeros", "xavier"])

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("--model_save_path", type=str,
                        default="models/model.npy")

    return parser.parse_args()


def main():

    args = parse_arguments()

    wandb.init(
        project="da6401_assignment_1",
        config=vars(args),
        mode="online"  # change to "offline" if internet slow
    )

    
    config = wandb.config

   
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config.dataset)

   
    model = NeuralNetwork(config)

    print("Starting training...")

    
    history = model.train(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size
    )

  
    for epoch in range(len(history["train_loss"])):

        log_dict = {
            "epoch": epoch,
            "train_loss": history["train_loss"][epoch],
            "train_accuracy": history["train_accuracy"][epoch],
        }

       
        if "val_loss" in history:
            log_dict["val_loss"] = history["val_loss"][epoch]

        if "val_accuracy" in history:
            log_dict["val_accuracy"] = history["val_accuracy"][epoch]

        wandb.log(log_dict)

  
    val_loss, val_acc = model.evaluate(X_val, y_val)

    wandb.log({
        "final_val_loss": val_loss,
        "final_val_accuracy": val_acc
    })

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    weights = []
    for layer in model.layers:
        weights.append({
            "W": layer.W,
            "b": layer.b
        })

    np.save(config.model_save_path, weights, allow_pickle=True)


    with open("models/config.json", "w") as f:
        json.dump(dict(config), f, indent=4)

    print("Training complete!")
    print(f"Model saved at {config.model_save_path}")

    wandb.finish()


if __name__ == "__main__":
    main()