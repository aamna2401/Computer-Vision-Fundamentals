"""Face classification and Siamese face verification on CelebA identities (PyTorch)."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_identities(root, size=64, min_images=8):
    """root/<identity>/*.jpg -> dict identity -> uint8 array (n, size, size, 3). Centre-crops the aligned face first."""
    out = {}
    for d in sorted(Path(root).iterdir()):
        files = sorted(d.glob("*.jpg"))
        if d.is_dir() and len(files) >= min_images:
            imgs = []
            for f in files:
                im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                h, w = im.shape[:2]
                im = im[int(.18 * h):int(.88 * h), int(.1 * w):int(.9 * w)]  # drop hair/background, keep the face
                imgs.append(cv2.resize(im, (size, size)))
            out[d.name] = np.stack(imgs)
    return out


def to_tensor(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0


def augment(x):
    """Random horizontal flip + brightness/contrast jitter + small translation on a batch."""
    flip = torch.rand(len(x)) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = x * (1 + 0.3 * (torch.rand(len(x), 1, 1, 1) - 0.5)) + 0.2 * (torch.rand(len(x), 1, 1, 1) - 0.5)
    dx, dy = np.random.randint(-3, 4, 2)
    return torch.roll(x.clamp(0, 1), shifts=(int(dy), int(dx)), dims=(2, 3))


class SmallCNN(nn.Module):
    """4 conv blocks -> global average pool -> 128-d embedding -> optional classifier."""

    def __init__(self, n_classes=None, emb=128):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 192))
        self.embed = nn.Linear(192, emb)
        self.head = nn.Linear(emb, n_classes) if n_classes else None

    def embedding(self, x):
        return self.embed(self.features(x).mean((2, 3)))

    def forward(self, x):
        e = self.embedding(x)
        return self.head(F.relu(e)) if self.head is not None else e


def contrastive_loss(a, b, same, margin=1.0):
    d = F.pairwise_distance(F.normalize(a), F.normalize(b))
    return (same * d ** 2 + (1 - same) * F.relu(margin - d) ** 2).mean()


def roc_eer(scores, labels):
    """Return (AUC, EER, best-threshold accuracy) for similarity scores (higher = same person)."""
    order = np.argsort(-scores)
    l = labels[order]
    tps, fps = np.cumsum(l), np.cumsum(1 - l)
    tpr, fpr = tps / l.sum(), fps / (1 - l).sum()
    auc = float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr, fpr))
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    acc = float(np.max((tps + ((1 - l).sum() - fps)) / len(l)))
    return auc, float((fnr[i] + fpr[i]) / 2), acc
