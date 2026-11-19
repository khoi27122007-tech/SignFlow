"""Train the landmark classifier. Class count is read from the dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Subset

try:
    from .config import DEFAULT_CACHE_PATH, DEFAULT_MODEL_PATH, FEATURE_SIZE
    from .model import LandmarkClassifier
except ImportError:
    from config import DEFAULT_CACHE_PATH, DEFAULT_MODEL_PATH, FEATURE_SIZE
    from model import LandmarkClassifier


def stratified_indices(y: np.ndarray, val_ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) >= 2 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def main(data_path: Path, model_path: Path, epochs: int, batch_size: int, learning_rate: float):
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {data_path}. Run build_dataset.py first.")
    saved = np.load(data_path, allow_pickle=False)
    X = saved['X'].astype(np.float32)
    y = saved['y'].astype(np.int64)
    class_names = saved['class_names'].astype(str).tolist()
    if X.ndim != 2 or X.shape[1] != FEATURE_SIZE:
        raise ValueError(f"Expected X shape (N, {FEATURE_SIZE}), got {X.shape}")
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two classes.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_idx, val_idx = stratified_indices(y, 0.2)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False) if val_idx else None

    model = LandmarkClassifier(X.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            count += len(yb)
        train_acc = correct / max(count, 1)

        val_acc = train_acc
        if val_loader:
            model.eval()
            val_correct, val_count = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_correct += (model(xb).argmax(1) == yb).sum().item()
                    val_count += len(yb)
            val_acc = val_correct / max(val_count, 1)
        print(f"Epoch {epoch:03d}/{epochs} | loss={total_loss/max(count,1):.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names,
                        'input_size': X.shape[1], 'num_classes': len(class_names)}, model_path)

    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Device: {device}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args.data, args.output, args.epochs, args.batch_size, args.lr)
