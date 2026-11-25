
"""Train a dynamic landmark classifier."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

try:
    from .config import (
        DEFAULT_CACHE_PATH,
        DEFAULT_MODEL_PATH,
        FEATURE_SIZE,
    )
    from .model import LandmarkClassifier
except ImportError:
    from config import (
        DEFAULT_CACHE_PATH,
        DEFAULT_MODEL_PATH,
        FEATURE_SIZE,
    )
    from model import LandmarkClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for label in np.unique(y):

        indices = np.where(y == label)[0].copy()

        if len(indices) < 2:
            raise ValueError(
                f"Class index {label} has fewer than 2 samples."
            )

        rng.shuffle(indices)

        val_count = max(
            1,
            int(round(len(indices) * val_ratio)),
        )

        val_count = min(
            val_count,
            len(indices) - 1,
        )

        val_indices.extend(
            indices[:val_count].tolist()
        )

        train_indices.extend(
            indices[val_count:].tolist()
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def evaluate(model, loader, criterion, device, num_classes):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    with torch.inference_mode():

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = criterion(logits, yb)

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * len(yb)

            total_correct += (
                predictions == yb
            ).sum().item()

            total_samples += len(yb)

            true_values = yb.cpu().numpy()
            predicted_values = predictions.cpu().numpy()

            for true_label, predicted_label in zip(
                true_values,
                predicted_values,
            ):
                confusion[true_label, predicted_label] += 1

    average_loss = total_loss / max(total_samples, 1)

    accuracy = total_correct / max(total_samples, 1)

    return average_loss, accuracy, confusion


def main(
    data_path: Path,
    model_path: Path,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 15,
    val_ratio: float = 0.2,
    seed: int = 42,
):

    set_seed(seed)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Run build_dataset.py first."
        )

    saved = np.load(
        data_path,
        allow_pickle=False,
    )

    X = saved["X"].astype(np.float32)
    y = saved["y"].astype(np.int64)

    class_names = saved["class_names"].astype(str).tolist()

    saved.close()

    if X.ndim != 2 or X.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"Expected X shape (N, {FEATURE_SIZE}), got {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError("X and y have different sample counts.")

    if len(class_names) < 2:
        raise ValueError("At least two classes are required.")

    if len(set(class_names)) != len(class_names):
        raise ValueError("Duplicate class names found.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    num_classes = len(class_names)

    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError(
            "Labels are outside the valid class index range."
        )

    class_counts = np.bincount(
        y,
        minlength=num_classes,
    )

    missing_classes = [
        class_names[index]
        for index, count in enumerate(class_counts)
        if count == 0
    ]

    insufficient_classes = [
        class_names[index]
        for index, count in enumerate(class_counts)
        if count < 2
    ]

    if missing_classes:
        raise ValueError(
            f"Classes with no samples: {missing_classes}"
        )

    if insufficient_classes:
        raise ValueError(
            "Each class needs at least 2 samples. "
            f"Insufficient: {insufficient_classes}"
        )

    print("=" * 70)
    print("SIGNFLOW - LANDMARK MODEL TRAINING")
    print("=" * 70)

    print("Samples:", len(X))
    print("Features:", X.shape[1])
    print("Classes:", num_classes)
    print("Class names:", class_names)

    print("\nClass distribution:")

    for index, name in enumerate(class_names):
        print(
            f"{name:<15} : {class_counts[index]:>7} samples"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    train_indices, val_indices = stratified_indices(
        y,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = LandmarkClassifier(
        input_size=X.shape[1],
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * len(yb)

            total_correct += (
                predictions == yb
            ).sum().item()

            total_samples += len(yb)

        train_loss = total_loss / max(total_samples, 1)

        train_accuracy = (
            total_correct / max(total_samples, 1)
        )

        val_loss, val_accuracy, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"lr={current_lr:.6f}"
        )

        # Save the checkpoint with the lowest validation loss.
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": X.shape[1],
                "num_classes": num_classes,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "class_counts": class_counts.tolist(),
            }

            temporary_path = model_path.with_name(
                model_path.name + ".tmp"
            )

            torch.save(
                checkpoint,
                temporary_path,
            )

            temporary_path.replace(model_path)

            print("  -> Best checkpoint saved.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                "Validation loss did not improve."
            )
            break

    # Reload the best checkpoint for final evaluation.
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_loss, final_accuracy, confusion = evaluate(
        model,
        val_loader,
        criterion,
        device,
        num_classes,
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION REPORT")
    print("=" * 70)

    print("Best epoch:", best_epoch)
    print(f"Validation loss: {final_loss:.4f}")
    print(f"Validation accuracy: {final_accuracy:.4%}")

    print("\nPer-class recall:")

    for index, name in enumerate(class_names):

        support = int(confusion[index].sum())

        correct = int(confusion[index, index])

        recall = (
            correct / support
            if support > 0 else 0.0
        )

        print(
            f"{name:<15} "
            f"{recall:.2%} "
            f"({correct}/{support})"
        )

    print("\nModel saved:", model_path.resolve())
    print("Device:", device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_CACHE_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(
        data_path=args.data,
        model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
