from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sequence_model import LandmarkGRU

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    data_dir = ROOT / "data/sequences"
    paths = sorted(data_dir.glob("*.npy"))
    if len(paths) < 10:
        raise RuntimeError("Chưa đủ sequence. Hãy thu thập nhiều chuỗi J/Z bằng data_collector.py.")
    labels = sorted({p.stem.rsplit("_", 1)[0] for p in paths})
    mapping = {name: i for i, name in enumerate(labels)}
    xs, ys = [], []
    for path in paths:
        label = path.stem.rsplit("_", 1)[0]
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 63:
            continue
        # Normalize sequence length to 32 frames by evenly sampling.
        ids = np.linspace(0, len(arr)-1, 32).round().astype(int)
        xs.append(arr[ids]); ys.append(mapping[label])
    if len(xs) < 10:
        raise RuntimeError("Dữ liệu sequence không hợp lệ hoặc không đủ.")
    x, y = np.asarray(xs, np.float32), np.asarray(ys, np.int64)
    rng = np.random.default_rng(42); ids = rng.permutation(len(y))
    n_val = max(1, int(0.2 * len(ids)))
    tr, va = ids[n_val:], ids[:n_val]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LandmarkGRU(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train = DataLoader(TensorDataset(torch.tensor(x[tr]), torch.tensor(y[tr])),
                       batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TensorDataset(torch.tensor(x[va]), torch.tensor(y[va])),
                     batch_size=args.batch_size)
    best = 0.0; out = ROOT / "models"; out.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
        model.eval(); good = total = 0
        with torch.no_grad():
            for xb, yb in val:
                pred = model(xb.to(device)).argmax(1).cpu()
                good += (pred == yb).sum().item(); total += len(yb)
        acc = good / max(total, 1)
        print(f"Epoch {epoch:03d} | val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model_state_dict": model.state_dict(), "classes": labels,
                        "val_accuracy": best}, out / "sequence_model.pth")
            (out / "sequence_classes.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print("Best validation accuracy:", best)


if __name__ == "__main__":
    main()
