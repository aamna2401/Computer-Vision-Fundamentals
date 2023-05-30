"""Train the NumPy CNN on 8x8 digits, verify gradients, compare with PyTorch, and test focal loss on imbalanced data."""
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_numpy import Adam, Conv2D, Dense, Flatten, MaxPool2, Net, ReLU, focal_loss, gradient_check, softmax

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def make_net(seed=0):
    r = np.random.default_rng(seed)
    return Net([Conv2D(1, 8, 3, r), ReLU(), MaxPool2(), Flatten(), Dense(8 * 3 * 3, 10, r)])


def train(Xtr, ytr, gamma=0.0, epochs=25, bs=32, lr=5e-3, seed=0):
    net, opt, rng = make_net(seed), Adam(lr), np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        tot = 0
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            loss, dz = focal_loss(net.forward(Xtr[b]), ytr[b], gamma)
            net.backward(dz)
            opt.step(net.parameters())
            tot += loss * len(b)
        acc = (net.forward(Xte).argmax(1) == yte).mean()
        hist.append((tot / len(idx), acc))
    return net, hist


metrics = {}
# 1. gradient check (tiny batch, float64)
chk_net = make_net(1)
metrics["gradient_check_max_relative_error"] = {f"gamma={g}": float(f"{gradient_check(chk_net, Xtr[:6], ytr[:6], gamma=g):.2e}") for g in (0.0, 2.0)}

# 2. train NumPy CNN
t = time.perf_counter()
net, hist = train(Xtr, ytr)
np_time = time.perf_counter() - t
np_acc = float((net.forward(Xte).argmax(1) == yte).mean())

# 3. same architecture in PyTorch
tnet = nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(72, 10))
opt = torch.optim.Adam(tnet.parameters(), 5e-3)
Xt, yt, Xv, yv = map(torch.tensor, (Xtr.astype(np.float32), ytr, Xte.astype(np.float32), yte))
t = time.perf_counter(); th = []
for ep in range(25):
    perm = torch.randperm(len(Xt))
    for s in range(0, len(perm), 32):
        b = perm[s:s + 32]
        opt.zero_grad(); nn.functional.cross_entropy(tnet(Xt[b]), yt[b]).backward(); opt.step()
    th.append((tnet(Xv).argmax(1) == yv).float().mean().item())
torch_time = time.perf_counter() - t
metrics["digits"] = {"train": len(Xtr), "test": len(Xte), "numpy_cnn_test_acc": round(np_acc, 4), "pytorch_same_arch_test_acc": round(th[-1], 4),
                     "numpy_train_seconds": round(np_time, 1), "pytorch_train_seconds": round(torch_time, 1)}

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot([h[0] for h in hist], "o-"); ax[0].set_title("NumPy CNN training loss"); ax[0].set_xlabel("epoch")
ax[1].plot([h[1] for h in hist], label="NumPy (ours)"); ax[1].plot(th, "--", label="PyTorch"); ax[1].set_title("test accuracy"); ax[1].legend(); ax[1].set_xlabel("epoch")
cm = confusion_matrix(yte, net.forward(Xte).argmax(1))
ax[2].imshow(cm, cmap="Blues"); ax[2].set_title("confusion matrix (NumPy CNN)")
for i in range(10):
    for j in range(10):
        ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig(OUT / "training_and_confusion.png", dpi=110); plt.close()

# learned filters
W = net.layers[0].W[:, 0]
fig, ax = plt.subplots(1, 8, figsize=(12, 1.8))
for a, w in zip(ax, W):
    a.imshow(w, cmap="RdBu", vmin=-abs(w).max(), vmax=abs(w).max()); a.axis("off")
plt.suptitle("Learned 3x3 filters"); plt.tight_layout(); plt.savefig(OUT / "learned_filters.png", dpi=110); plt.close()

# 4. imbalance: keep only 8% of classes 3 and 8 in training; compare CE and focal loss on those minority classes
rng = np.random.default_rng(0)
keep = np.ones(len(ytr), bool)
for c in (3, 8):
    ids = np.where(ytr == c)[0]
    keep[rng.choice(ids, int(0.92 * len(ids)), replace=False)] = False
Xi, yi = Xtr[keep], ytr[keep]
rows = {}
for gamma in (0.0, 2.0):
    recs = []
    for seed in range(5):
        n, _ = train(Xi, yi, gamma=gamma, seed=seed)
        pred = n.forward(Xte).argmax(1)
        recs.append([recall_score(yte, pred, labels=[3, 8], average=None).mean(), (pred == yte).mean()])
    r = np.mean(recs, 0)
    rows[f"gamma={gamma}"] = {"minority_recall(3,8)": round(float(r[0]), 3), "overall_acc": round(float(r[1]), 3)}
metrics["imbalance_experiment_5_seeds"] = {"train_samples": int(keep.sum()), **rows}
plt.figure(figsize=(5.5, 3.6))
lab = list(rows); plt.bar(np.arange(2) - 0.18, [rows[l]["minority_recall(3,8)"] for l in lab], 0.36, label="minority recall (digits 3, 8)")
plt.bar(np.arange(2) + 0.18, [rows[l]["overall_acc"] for l in lab], 0.36, label="overall accuracy")
plt.xticks(range(2), ["cross-entropy (gamma=0)", "focal loss (gamma=2)"]); plt.ylim(0, 1.05); plt.legend(loc="lower right"); plt.title("Class imbalance: CE vs focal loss")
plt.tight_layout(); plt.savefig(OUT / "focal_vs_ce_imbalance.png", dpi=110); plt.close()

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
