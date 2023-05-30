"""Spatio-temporal GCN vs an MLP on synthetic skeleton clips (stand / walk / fall), with noise and time-shift robustness."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gcn import EDGES, STGCN, normalised_adjacency, synth_skeletons

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

A_sk = normalised_adjacency(torch.tensor(EDGES).T, 14)
Xs, ys = synth_skeletons(120, rng=np.random.default_rng(0))
idx = np.random.default_rng(1).permutation(len(ys)); cut = int(0.7 * len(ys))
Xtr, ytr, Xte, yte = map(torch.tensor, (Xs[idx[:cut]], ys[idx[:cut]], Xs[idx[cut:]], ys[idx[cut:]]))
mu, sd = Xtr.flatten(1).mean(0), Xtr.flatten(1).std(0) + 1e-3  # MLP gets standardised features: a fair baseline


class StandardisedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3 * 48 * 14, 128), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(128, 3))

    def forward(self, x):
        return self.net((x.flatten(1) - mu) / sd)


def fit(model, epochs=40, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    hist = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 32):
            b = perm[s:s + 32]; opt.zero_grad(); F.cross_entropy(model(Xtr[b]), ytr[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hist.append((model(Xte).argmax(1) == yte).float().mean().item())
    return hist


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return round((model(X).argmax(1) == y).float().mean().item(), 3)


st = STGCN(A_sk, 3); h_st = fit(st)
mlp = StandardisedMLP(); h_mlp = fit(mlp)

noise = {}
for s in [0.03, 0.08, 0.15]:
    Xn, yn = map(torch.tensor, synth_skeletons(60, rng=np.random.default_rng(5), noise=s))
    noise[str(s)] = {"stgcn": acc(st, Xn, yn), "mlp": acc(mlp, Xn, yn)}

# time-shift robustness: roll each test clip by a random number of frames (a clip that starts mid-action)
shift = {}
g = np.random.default_rng(7)
for max_shift in [0, 4, 8, 16]:
    sh = torch.stack([torch.roll(x, int(g.integers(-max_shift, max_shift + 1)), dims=1) for x in Xte]) if max_shift else Xte
    shift[str(max_shift)] = {"stgcn": acc(st, sh, yte), "mlp": acc(mlp, sh, yte)}

per_class = {}
with torch.no_grad():
    for name, m in [("stgcn", st), ("mlp", mlp)]:
        m.eval(); p = m(Xte).argmax(1)
        per_class[name] = {c: round((p[yte == i] == i).float().mean().item(), 3) for i, c in enumerate(["stand", "walk", "fall"])}

metrics = {"clips": len(ys), "train": cut, "test": len(ys) - cut, "classes": ["stand", "walk", "fall"],
           "stgcn_test_acc": round(h_st[-1], 3), "mlp_test_acc": round(h_mlp[-1], 3),
           "stgcn_params": sum(p.numel() for p in st.parameters()), "mlp_params": sum(p.numel() for p in mlp.parameters()),
           "per_class_recall": per_class, "joint_noise_test_acc": noise, "time_shift_frames_test_acc": shift,
           "note": "synthetic data: demonstrates the architecture and its invariances, not real-world performance"}

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(h_st, label="ST-GCN"); ax[0].plot(h_mlp, "--", label="MLP (standardised, flattened clip)")
ax[0].set_title("test accuracy during training"); ax[0].set_xlabel("epoch"); ax[0].legend()
xs_ = [float(k) for k in noise]
ax[1].plot(xs_, [noise[k]["stgcn"] for k in noise], "o-", label="ST-GCN"); ax[1].plot(xs_, [noise[k]["mlp"] for k in noise], "s--", label="MLP")
ax[1].set_xlabel("test-time joint noise (std)"); ax[1].set_ylabel("accuracy"); ax[1].set_title("keypoint-noise robustness"); ax[1].legend()
xs_ = [int(k) for k in shift]
ax[2].plot(xs_, [shift[k]["stgcn"] for k in shift], "o-", label="ST-GCN"); ax[2].plot(xs_, [shift[k]["mlp"] for k in shift], "s--", label="MLP")
ax[2].set_xlabel("max random time shift (frames, of 48)"); ax[2].set_ylabel("accuracy"); ax[2].set_title("robustness to when the action starts"); ax[2].legend()
plt.tight_layout(); plt.savefig(OUT / "skeleton_stgcn.png", dpi=110); plt.close()

clip = Xs[idx[cut:]][np.where(yte.numpy() == 2)[0][0]]
fig, ax = plt.subplots(1, 4, figsize=(11, 3.2))
for a, t in zip(ax, [0, 18, 30, 47]):
    pts = clip[:2, t].T
    for u, v in EDGES:
        a.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], color="#3b6ea5", lw=2)
    a.scatter(pts[:, 0], pts[:, 1], s=14, c="k"); a.set_aspect("equal"); a.set_xlim(-2.0, 2.0); a.set_ylim(-.3, 2); a.set_title(f"frame {t}"); a.axis("off")
plt.suptitle("A synthetic 'fall' clip on the 14-joint skeleton graph"); plt.tight_layout(); plt.savefig(OUT / "skeleton_fall_clip.png", dpi=110); plt.close()

(OUT / "skeleton_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
