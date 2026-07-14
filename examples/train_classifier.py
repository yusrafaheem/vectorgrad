"""
Train a small MLP with vectorgrad on a synthetic multi-class classification
problem, and plot the loss/accuracy curves.

Run:
    python examples/train_classifier.py

Swapping in real MNIST instead of synthetic blobs is a one-line change:
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    y = y.astype(int)
Everything downstream (model, optimizer, training loop) is unchanged.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

from vectorgrad import Adam, Linear, ReLU, Sequential, Tensor
from vectorgrad.data import iterate_minibatches, make_blobs, standardize, train_test_split


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == y).mean())


def main():
    np.random.seed(42)

    X, y = make_blobs(n_samples=3000, n_features=8, centers=6, cluster_std=3.2, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=1)
    X_train, X_test = standardize(X_train, X_test)

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    model = Sequential(
        [
            Linear(n_features, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, n_classes),
        ]
    )
    print(model)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"trainable parameters: {n_params}")

    optimizer = Adam(model.parameters(), lr=0.01)

    epochs = 40
    batch_size = 64
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            x_t = Tensor(xb)
            logits = model(x_t)
            loss = logits.softmax_cross_entropy(yb)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.data))

        train_logits = model(Tensor(X_train)).data
        test_logits = model(Tensor(X_test)).data
        history["train_loss"].append(np.mean(epoch_losses))
        history["train_acc"].append(accuracy(train_logits, y_train))
        history["test_acc"].append(accuracy(test_logits, y_test))

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | loss {history['train_loss'][-1]:.4f} "
                f"| train acc {history['train_acc'][-1]:.3f} "
                f"| test acc {history['test_acc'][-1]:.3f}"
            )

    out_path = os.path.join(os.path.dirname(__file__), "training_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("training loss")
    axes[0].set_xlabel("epoch")

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["test_acc"], label="test")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved plot to {out_path}")
    print(f"final test accuracy: {history['test_acc'][-1]:.3f}")


if __name__ == "__main__":
    main()
