"""k-nearest-neighbour classifier written from scratch (used on bag-of-visual-words histograms)."""
import numpy as np
from scipy.stats import mode


def euclidean(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def predict(x_train, y, x_input, k):
    """Predict a label for every row of x_input by majority vote of its k nearest training rows."""
    y = np.asarray(y)
    op_labels = []
    for item in x_input:
        point_dist = np.array([euclidean(x_train[j, :], item) for j in range(len(x_train))])
        nearest = np.argsort(point_dist)[:k]
        labels = y[nearest]
        lab = mode(labels, keepdims=False).mode
        op_labels.append(lab)
    return op_labels
