"""Camera calibration (DLT) and two-view geometry (normalised 8-point) from first principles."""
import numpy as np
from scipy.linalg import rq


def homogenize(x):
    return np.hstack([x, np.ones((len(x), 1))])


def dlt_projection(pts2d, pts3d):
    """Direct linear transform: 3x4 projection matrix P with [u v 1] ~ P [X Y Z 1]; solved by SVD."""
    X = homogenize(pts3d)
    rows = []
    for (u, v), Xi in zip(pts2d, X):
        rows.append(np.concatenate([Xi, np.zeros(4), -u * Xi]))
        rows.append(np.concatenate([np.zeros(4), Xi, -v * Xi]))
    _, _, vt = np.linalg.svd(np.array(rows))
    return vt[-1].reshape(3, 4)


def project(P, pts3d):
    p = (P @ homogenize(pts3d).T).T
    return p[:, :2] / p[:, 2:3]


def decompose(P):
    """P = K [R | t] with K upper-triangular (RQ decomposition)."""
    K, R = rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    t = np.linalg.solve(K, P[:, 3])
    if np.linalg.det(R) < 0:
        R, t = -R, -t
    return K / K[2, 2], R, t


def normalise(pts):
    """Translate to the centroid and scale so the mean distance is sqrt(2) (Hartley normalisation)."""
    c = pts.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (T @ homogenize(pts).T).T, T


def eight_point(p1, p2, normalised=True):
    """Fundamental matrix F with p2^T F p1 = 0; rank-2 enforced."""
    if normalised:
        (a, T1), (b, T2) = normalise(p1), normalise(p2)
    else:
        a, b, T1, T2 = homogenize(p1), homogenize(p2), np.eye(3), np.eye(3)
    A = np.stack([b[:, 0] * a[:, 0], b[:, 0] * a[:, 1], b[:, 0], b[:, 1] * a[:, 0], b[:, 1] * a[:, 1], b[:, 1],
                  a[:, 0], a[:, 1], np.ones(len(a))], axis=1)
    F = np.linalg.svd(A)[2][-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0]) @ Vt
    F = T2.T @ F @ T1
    return F / F[2, 2]


def sampson_error(F, p1, p2):
    x1, x2 = homogenize(p1), homogenize(p2)
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / den)


def epipolar_lines(F, pts, which=1):
    """Lines in the other image for points in image `which` (1: lines in image 2)."""
    x = homogenize(pts)
    return x @ F.T if which == 1 else x @ F
