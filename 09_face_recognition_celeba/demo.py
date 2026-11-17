"""Face classification (scratch CNN vs pretrained ResNet-18 features) and Siamese verification on unseen identities.

usage: python demo.py --data PATH   where PATH/<identity>/*.jpg holds CelebA-aligned face crops.
Face images are not redistributed here (CelebA licence): results contain curves and metrics only.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faces import SmallCNN, augment, contrastive_loss, load_identities, roc_eer, to_tensor

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).parent.parent / "data" / "faces"))
ap.add_argument("--epochs", type=int, default=30)
args = ap.parse_args()
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
rng = np.random.default_rng(0)

ids = load_identities(args.data)
names = sorted(ids, key=lambda n: -len(ids[n]))
print(len(names), "identities,", sum(len(v) for v in ids.values()), "images")
metrics = {"identities_total": len(names), "images_total": int(sum(len(v) for v in ids.values()))}

# ---------- A. closed-set classification on the 30 identities with the most images ----------
cls_names = names[:30]
Xtr, ytr, Xte, yte = [], [], [], []
for ci, n in enumerate(cls_names):
    a = ids[n][rng.permutation(len(ids[n]))]
    k = int(0.7 * len(a))
    Xtr += list(a[:k]); ytr += [ci] * k; Xte += list(a[k:]); yte += [ci] * (len(a) - k)
Xtr, Xte = to_tensor(np.stack(Xtr)), to_tensor(np.stack(Xte)); ytr, yte = torch.tensor(ytr), torch.tensor(yte)
metrics["classification"] = {"classes": 30, "train": len(Xtr), "test": len(Xte), "chance": round(1 / 30, 3)}

net = SmallCNN(30); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=args.epochs * ((len(Xtr) + 31) // 32))
curve = []
for ep in range(args.epochs):
    net.train(); perm = torch.randperm(len(Xtr)); tot = 0
    for s in range(0, len(perm), 32):
        b = perm[s:s + 32]
        loss = F.cross_entropy(net(augment(Xtr[b])), ytr[b], label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
    net.eval()
    with torch.no_grad():
        curve.append((tot / len(perm), (net(Xte).argmax(1) == yte).float().mean().item()))
scratch_acc = curve[-1][1]

# baselines: raw-pixel kNN and pretrained ResNet-18 embedding + nearest centroid
flat = lambda x: F.normalize((x - x.mean((1, 2, 3), keepdim=True)).flatten(1))
sim = flat(Xte) @ flat(Xtr).T
pix_acc = (ytr[sim.argmax(1)] == yte).float().mean().item()

resnet_note = "not available"
res_acc = None
try:
    import torchvision
    r18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT); r18.fc = nn.Identity(); r18.eval()
    mean, std = torch.tensor([.485, .456, .406])[None, :, None, None], torch.tensor([.229, .224, .225])[None, :, None, None]
    def emb(x):
        with torch.no_grad():
            return F.normalize(r18(F.interpolate((x - mean) / std, size=112, mode="bilinear")))
    etr, ete = emb(Xtr), emb(Xte)
    cent = F.normalize(torch.stack([etr[ytr == c].mean(0) for c in range(30)]))
    res_acc = ((ete @ cent.T).argmax(1) == yte).float().mean().item(); resnet_note = "ImageNet ResNet-18, nearest class centroid"
except Exception as e:  # offline: skip the pretrained baseline
    resnet_note = f"skipped ({type(e).__name__})"
metrics["classification"].update({"scratch_cnn_test_acc": round(scratch_acc, 3), "raw_pixel_nn_acc": round(pix_acc, 3),
                                  "pretrained_resnet18_centroid_acc": None if res_acc is None else round(res_acc, 3), "pretrained_note": resnet_note})

# ---------- B. Siamese verification on identities never seen in training ----------
perm_ids = [n for n in names]; rng.shuffle(perm_ids)
train_ids, test_ids = perm_ids[:int(.7 * len(perm_ids))], perm_ids[int(.7 * len(perm_ids)):]
metrics["verification"] = {"train_identities": len(train_ids), "unseen_test_identities": len(test_ids)}


def sample_pairs(pool, n, rng):
    A, B, S = [], [], []
    for _ in range(n):
        if rng.random() < 0.5:
            c = pool[rng.integers(len(pool))]; i, j = rng.choice(len(ids[c]), 2, replace=False); A.append(ids[c][i]); B.append(ids[c][j]); S.append(1)
        else:
            c1, c2 = rng.choice(len(pool), 2, replace=False); A.append(ids[pool[c1]][rng.integers(len(ids[pool[c1]]))])
            B.append(ids[pool[c2]][rng.integers(len(ids[pool[c2]]))]); S.append(0)
    return to_tensor(np.stack(A)), to_tensor(np.stack(B)), torch.tensor(S).float()

sia = SmallCNN(None); opt = torch.optim.AdamW(sia.parameters(), 2e-3, weight_decay=1e-3)
losses = []
for ep in range(args.epochs):
    sia.train(); tot = 0
    for _ in range(40):
        a, b, s = sample_pairs(train_ids, 64, rng)
        loss = contrastive_loss(sia(augment(a)), sia(augment(b)), s)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    losses.append(tot / 40)
ta, tb, ts = sample_pairs(test_ids, 3000, np.random.default_rng(123))
sia.eval()
with torch.no_grad():
    sims = {"siamese (ours)": F.cosine_similarity(sia(ta), sia(tb)).numpy(),
            "raw pixels": F.cosine_similarity(flat(ta), flat(tb)).numpy()}
    if res_acc is not None:
        sims["pretrained ResNet-18"] = (emb(ta) * emb(tb)).sum(1).numpy()
labels = ts.numpy().astype(int)
ver = {}
for k, s in sims.items():
    auc, eer, acc = roc_eer(s, labels)
    ver[k] = {"AUC": round(auc, 3), "EER": round(eer, 3), "best_threshold_acc": round(acc, 3)}
metrics["verification"].update({"test_pairs": 3000, "results": ver})

# ---------- figures ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot([c[0] for c in curve]); ax[0].set_title("classification training loss"); ax[0].set_xlabel("epoch")
ax[1].plot([c[1] for c in curve], label="scratch CNN"); ax[1].axhline(pix_acc, color="gray", ls="--", label="raw-pixel 1-NN")
if res_acc is not None: ax[1].axhline(res_acc, color="green", ls="-.", label="pretrained ResNet-18")
ax[1].axhline(1 / 30, color="red", ls=":", label="chance"); ax[1].set_title("30-identity test accuracy"); ax[1].legend(); ax[1].set_xlabel("epoch")
ax[2].plot(losses); ax[2].set_title("Siamese contrastive loss"); ax[2].set_xlabel("epoch")
plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, s in sims.items():
    o = np.argsort(-s); l = labels[o]
    ax[0].plot(np.cumsum(1 - l) / (1 - l).sum(), np.cumsum(l) / l.sum(), label=f"{k} (AUC {ver[k]['AUC']})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("false accept rate"); ax[0].set_ylabel("true accept rate"); ax[0].legend(); ax[0].set_title("Verification ROC, unseen identities")
for k, s in sims.items():
    ax[1].hist(s[labels == 1], 30, alpha=.45, label=f"{k}: same", density=True) if k == "siamese (ours)" else None
    ax[1].hist(s[labels == 0], 30, alpha=.45, label=f"{k}: different", density=True) if k == "siamese (ours)" else None
ax[1].set_title("Siamese similarity distributions"); ax[1].legend(); ax[1].set_xlabel("cosine similarity")
plt.tight_layout(); plt.savefig(OUT / "verification_roc.png", dpi=110); plt.close()

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
