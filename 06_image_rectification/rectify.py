"""Homography estimation (normalised DLT) and plane rectification."""
import cv2
import numpy as np


def _norm(pts):
    c = pts.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T, T


def homography(src, dst):
    """H such that dst ~ H src, from >= 4 point pairs."""
    (a, Ta), (b, Tb) = _norm(np.asarray(src, float)), _norm(np.asarray(dst, float))
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ H @ Ta
    return H / H[2, 2]


def apply_h(H, pts):
    p = (H @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    return p[:, :2] / p[:, 2:3]


def rectify(img, quad, tile_px=140, out_size=None):
    """Map the image quadrilateral `quad` (TL, TR, BR, BL) onto an axis-aligned square and warp the whole image."""
    square = np.array([[0, 0], [tile_px, 0], [tile_px, tile_px], [0, tile_px]], float)
    H = homography(quad, square)
    h, w = img.shape[:2]
    corners = apply_h(H, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    mn, mx = corners.min(0), corners.max(0)
    scale = 1.0
    if out_size:
        scale = out_size / max(mx - mn)
    T = np.array([[scale, 0, -mn[0] * scale], [0, scale, -mn[1] * scale], [0, 0, 1]])
    size = tuple(np.ceil((mx - mn) * scale).astype(int))
    return cv2.warpPerspective(img, T @ H, size), T @ H
