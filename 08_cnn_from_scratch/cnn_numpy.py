"""A small CNN in pure NumPy with full forward AND backward passes (conv, ReLU, max-pool, dense, softmax + focal loss).

Layout is NCHW. Convolution uses sliding windows (im2col-style) and einsum so it stays vectorised.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    def __init__(self, in_c, out_c, k, rng):
        self.k = k
        self.W = rng.normal(0, np.sqrt(2 / (in_c * k * k)), (out_c, in_c, k, k))  # He init
        self.b = np.zeros(out_c)

    def forward(self, x):
        self.x = x
        self.cols = sliding_window_view(x, (self.k, self.k), axis=(2, 3))  # N,C,H',W',k,k
        return np.einsum("nchwij,fcij->nfhw", self.cols, self.W) + self.b[None, :, None, None]

    def backward(self, d):
        self.dW = np.einsum("nchwij,nfhw->fcij", self.cols, d)
        self.db = d.sum((0, 2, 3))
        dx = np.zeros_like(self.x)
        Ho, Wo = d.shape[2:]
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + Ho, j:j + Wo] += np.einsum("nfhw,fc->nchw", d, self.W[:, :, i, j])
        return dx

    params = property(lambda s: [(s.W, s.dW), (s.b, s.db)])


class ReLU:
    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, d):
        return d * self.m

    params = property(lambda s: [])


class MaxPool2:
    def forward(self, x):
        n, c, h, w = x.shape
        self.shape = x.shape
        r = x[:, :, :h // 2 * 2, :w // 2 * 2].reshape(n, c, h // 2, 2, w // 2, 2)
        out = r.max(axis=(3, 5))
        self.mask = (r == out[:, :, :, None, :, None])
        self.mask = self.mask / self.mask.sum(axis=(3, 5), keepdims=True)  # split gradient across ties
        return out

    def backward(self, d):
        n, c, h, w = self.shape
        dr = self.mask * d[:, :, :, None, :, None]
        dx = np.zeros(self.shape)
        dx[:, :, :h // 2 * 2, :w // 2 * 2] = dr.reshape(n, c, h // 2 * 2, w // 2 * 2)
        return dx

    params = property(lambda s: [])


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, d):
        return d.reshape(self.shape)

    params = property(lambda s: [])


class Dense:
    def __init__(self, i, o, rng):
        self.W = rng.normal(0, np.sqrt(2 / i), (i, o))
        self.b = np.zeros(o)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW, self.db = self.x.T @ d, d.sum(0)
        return d @ self.W.T

    params = property(lambda s: [(s.W, s.dW), (s.b, s.db)])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def focal_loss(logits, y, gamma=0.0):
    """Mean focal loss FL = -(1 - p_t)^gamma log p_t and its gradient w.r.t. the logits (gamma=0 is cross-entropy)."""
    p = softmax(logits)
    n = len(y)
    pt = np.clip(p[np.arange(n), y], 1e-12, 1)
    loss = -((1 - pt) ** gamma) * np.log(pt)
    dL_dpt = gamma * (1 - pt) ** max(gamma - 1, 0) * np.log(pt) * (gamma > 0) - (1 - pt) ** gamma / pt
    onehot = np.eye(p.shape[1])[y]
    dz = (dL_dpt * pt)[:, None] * (onehot - p)
    return loss.mean(), dz / n


class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, d):
        for l in reversed(self.layers):
            d = l.backward(d)

    def parameters(self):
        return [p for l in self.layers for p in l.params]


class Adam:
    def __init__(self, lr=1e-2, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.t, self.m, self.v = lr, b1, b2, 0, {}, {}

    def step(self, params):
        self.t += 1
        for i, (p, g) in enumerate(params):
            m = self.m[i] = self.b1 * self.m.get(i, 0) + (1 - self.b1) * g
            v = self.v[i] = self.b2 * self.v.get(i, 0) + (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + 1e-8)


def gradient_check(net, x, y, gamma=0.0, eps=1e-5, n_checks=6, seed=0):
    """Max relative error between analytic and numerical gradients over random parameter entries (1e-6 floor so exactly-zero gradients of dead ReLUs do not divide by ~0)."""
    rng = np.random.default_rng(seed)
    for l in net.layers:  # zero biases put flat background patches exactly on the ReLU kink; nudge them off it
        if hasattr(l, "b"):
            l.b += rng.normal(0, 0.1, l.b.shape)
    loss, dz = focal_loss(net.forward(x), y, gamma)
    net.backward(dz)
    worst = 0.0
    for p, g in net.parameters():
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; lp, _ = focal_loss(net.forward(x), y, gamma)
            p[idx] = old - eps; lm, _ = focal_loss(net.forward(x), y, gamma)
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / max(abs(num) + abs(g[idx]), 1e-6))
    return worst
