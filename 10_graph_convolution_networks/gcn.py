"""Graph convolution from scratch, and a spatio-temporal GCN block for skeleton sequences (the ST-GCN idea)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalised_adjacency(edge_index, n):
    """A_hat = D^-1/2 (A + I) D^-1/2 as a dense matrix."""
    A = torch.zeros(n, n)
    A[edge_index[0], edge_index[1]] = 1
    A = ((A + A.T) > 0).float() + torch.eye(n)
    A = (A > 0).float()
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


class GCN(nn.Module):
    """H' = act(A_hat H W), repeated; `depth` graph-convolution layers."""

    def __init__(self, in_dim, hid, out_dim, depth=2, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hid] * (depth - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, A_hat):
        for i, lin in enumerate(self.layers):
            x = F.dropout(x, self.dropout, self.training)
            x = A_hat @ lin(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.dropout = nn.Linear(in_dim, hid), nn.Linear(hid, out_dim), dropout

    def forward(self, x, A_hat=None):
        return self.l2(F.dropout(F.relu(self.l1(F.dropout(x, self.dropout, self.training))), self.dropout, self.training))


class STGCNBlock(nn.Module):
    """Spatial graph conv over joints, then a temporal conv over frames. Input (N, C, T, V)."""

    def __init__(self, in_c, out_c, A_hat, t_kernel=9, stride=1):
        super().__init__()
        self.register_buffer("A", A_hat)
        self.spatial = nn.Conv2d(in_c, out_c, 1)
        self.temporal = nn.Sequential(nn.BatchNorm2d(out_c), nn.ReLU(),
                                      nn.Conv2d(out_c, out_c, (t_kernel, 1), (stride, 1), (t_kernel // 2, 0)), nn.BatchNorm2d(out_c))
        self.res = nn.Conv2d(in_c, out_c, 1, (stride, 1)) if (in_c != out_c or stride != 1) else nn.Identity()

    def forward(self, x):
        y = torch.einsum("nctv,vw->nctw", self.spatial(x), self.A)
        return F.relu(self.temporal(y) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, A_hat, n_classes, in_c=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_c * A_hat.shape[0])
        self.blocks = nn.Sequential(STGCNBlock(in_c, 32, A_hat), STGCNBlock(32, 64, A_hat, stride=2), STGCNBlock(64, 64, A_hat, stride=2))
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):  # (N, C, T, V)
        n, c, t, v = x.shape
        x = self.bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(n, c, v, t).permute(0, 1, 3, 2)
        return self.fc(self.blocks(x).mean((2, 3)))


# 14-joint skeleton (AlphaPose-style): head, neck, shoulders, elbows, wrists, hips, knees, ankles
EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7), (1, 8), (1, 9), (8, 10), (10, 12), (9, 11), (11, 13)]
REST = np.array([[0, 1.7], [0, 1.5], [-.2, 1.45], [.2, 1.45], [-.3, 1.2], [.3, 1.2], [-.35, .95], [.35, .95],
                 [-.1, 1.0], [.1, 1.0], [-.12, .55], [.12, .55], [-.12, .1], [.12, .1]])


def synth_skeletons(n_per_class, T=48, rng=None, noise=0.03):
    """Synthetic activity clips on the 14-joint skeleton: 0 stand/sway, 1 walk (leg+arm swing), 2 fall (tilt + drop)."""
    rng = rng or np.random.default_rng(0)
    X, y = [], []
    for c in range(3):
        for _ in range(n_per_class):
            t = np.linspace(0, 1, T)[:, None, None]
            seq = np.repeat(REST[None], T, 0).copy()
            ph = rng.uniform(0, 2 * np.pi)
            if c == 0:
                seq[:, :, 0] += 0.03 * np.sin(2 * np.pi * 0.5 * t[:, :, 0] + ph)
            elif c == 1:
                f = rng.uniform(1.5, 2.5)
                swing = 0.18 * np.sin(2 * np.pi * f * t[:, 0, 0] + ph)
                seq[:, [10, 12], 1] += 0.0
                seq[:, 10, 0] += swing; seq[:, 12, 0] += swing * 1.4; seq[:, 11, 0] -= swing; seq[:, 13, 0] -= swing * 1.4
                seq[:, 6, 0] -= swing * 0.8; seq[:, 7, 0] += swing * 0.8
            else:
                start = rng.uniform(0.2, 0.5)
                prog = np.clip((t[:, 0, 0] - start) / 0.3, 0, 1)
                theta = prog * rng.uniform(1.2, 1.5)
                pivot = REST[[12, 13]].mean(0)
                R = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], 1).reshape(T, 2, 2)
                seq = np.einsum("tij,tvj->tvi", R, seq - pivot) + pivot
            seq = seq + rng.normal(0, noise, seq.shape)
            conf = np.clip(rng.normal(0.9, 0.05, seq.shape[:2] + (1,)), 0, 1)
            X.append(np.concatenate([seq, conf], 2).transpose(2, 0, 1))  # C, T, V
            y.append(c)
    return np.array(X, np.float32), np.array(y)
