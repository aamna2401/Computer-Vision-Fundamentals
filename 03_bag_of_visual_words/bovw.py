"""Bag of visual words scene recognition: dense SIFT -> k-means vocabulary -> histograms -> classifier."""
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def dense_sift(gray, step=8, size=16):
    sift = cv2.SIFT_create()
    h, w = gray.shape
    kps = [cv2.KeyPoint(float(x), float(y), size) for y in range(size, h - size, step) for x in range(size, w - size, step)]
    _, desc = sift.compute(gray, kps)
    return desc


def load_split(root, per_class_train=30, size=200, seed=0):
    """Read class folders, resize to a fixed size, split each class into train / test."""
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te, classes = [], [], [], [], sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    for ci, c in enumerate(classes):
        files = sorted((Path(root) / c).glob("*.jpg"))
        rng.shuffle(files)
        for i, f in enumerate(files):
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (size, size))
            (X_tr if i < per_class_train else X_te).append(g)
            (y_tr if i < per_class_train else y_te).append(ci)
    return X_tr, np.array(y_tr), X_te, np.array(y_te), classes


def build_vocabulary(train_descs, k, seed=0, max_desc=60000):
    allv = np.vstack(train_descs).astype(np.float32)
    if len(allv) > max_desc:
        allv = allv[np.random.default_rng(seed).choice(len(allv), max_desc, replace=False)]
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048).fit(allv)


def histogram(desc, kmeans):
    words = kmeans.predict(desc.astype(np.float32))
    h = np.bincount(words, minlength=kmeans.n_clusters).astype(float)
    return h / (np.linalg.norm(h) + 1e-9)


def tiny_image(gray, s=16):
    v = cv2.resize(gray, (s, s)).astype(float).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-9)
