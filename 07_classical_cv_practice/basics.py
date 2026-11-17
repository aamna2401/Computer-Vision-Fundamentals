"""Small, self-contained building blocks: image pyramids, Laplacian-of-Gaussian zero crossings, RANSAC line fitting."""
import cv2
import numpy as np
from scipy import ndimage


def gaussian_pyramid(img, levels=4):
    out = [img.astype(np.float32)]
    for _ in range(levels - 1):
        out.append(cv2.pyrDown(out[-1]))
    return out


def laplacian_pyramid(img, levels=4):
    g = gaussian_pyramid(img, levels)
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(levels - 1)]
    return lap + [g[-1]]


def reconstruct(lap):
    img = lap[-1]
    for band in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=band.shape[1::-1]) + band
    return img


def log_zero_crossings(gray, sigma=2.0, min_contrast=2.0):
    """Marr-Hildreth edges: zero crossings of the Laplacian of Gaussian with enough local slope."""
    log = ndimage.gaussian_laplace(gray.astype(float), sigma)
    mx = ndimage.maximum_filter(log, size=3)
    mn = ndimage.minimum_filter(log, size=3)
    crossing = (mn < 0) & (mx > 0) & ((mx - mn) > min_contrast)
    return crossing, log


def ransac_line(points, iters=200, tol=1.5, seed=0):
    """Fit y = a x + b robustly: sample 2 points, count inliers within `tol`, keep the best, refit on inliers."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(points), 2, replace=False)
        (x1, y1), (x2, y2) = points[i], points[j]
        if abs(x2 - x1) < 1e-9:
            continue
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        inl = np.abs(points[:, 1] - (a * points[:, 0] + b)) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    a, b = np.polyfit(points[best, 0], points[best, 1], 1)
    return a, b, best
