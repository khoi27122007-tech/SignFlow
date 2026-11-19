from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from landmark_model import LandmarkMLP

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data/landmarks/landmarks.csv"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    if len(rows) < 20:
        raise RuntimeError("Cần ít nhất 20 mẫu landmarks; nên thu thập nhiều hơn và cân bằng lớp.")
    classes = sorted({r["label"] for r in rows})
    mapping = {name: i for i, name in enumerate(classes)}
    x = np.asarray([[float(r[f"f{i}"]) for i in range(63)] for r in rows], np.float32)
    y = np.asarray([mapping[r["label"]] for r in rows], np.int64)

    # Stratified split without requiring scikit-learn.
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for cls in range(len(classes)):
        ids = np.where(y == cls)[0]
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * 0.2)) if len(ids) > 2 else 0
        val_idx.extend(ids[:n_val]); train_idx.extend(ids[n_val:])
    if not val_idx:
        raise RuntimeError("Mỗi lớp cần tối thiểu 3 mẫu để tạo validation split.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LandmarkMLP(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    tx, ty = torch.tensor(x[train_idx]), torch.tensor(y[train_idx])
    vx, vy = torch.tensor(x[val_idx]), torch.tensor(y[val_idx])
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=args.batch_size)

    best = 0.0
    out = ROOT / "models"; out.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total = correct = loss_sum = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item(); total += len(yb)
        model.eval(); vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                vc += (logits.argmax(1).cpu() == yb).sum().item(); vt += len(yb)
        val_acc = vc / max(vt, 1)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={loss_sum/max(total,1):.4f} | train_acc={correct/max(total,1):.4f} | val_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model_state_dict": model.state_dict(), "classes": classes,
                        "val_accuracy": best}, out / "landmark_model.pth")
            (out / "landmark_classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
