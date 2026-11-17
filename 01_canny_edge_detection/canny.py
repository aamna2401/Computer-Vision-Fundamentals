"""Canny edge detector from first principles (NumPy / SciPy only).

Pipeline: Gaussian-derivative filters -> gradient magnitude and orientation ->
non-maximum suppression (4 quantised directions) -> double threshold + hysteresis.
"""
import numpy as np
from scipy import ndimage


def gaussian_derivative_kernels(sigma, truncate=3.0):
    """Return (Gx, Gy): first derivative of a 2-D Gaussian, so smoothing and differentiation are one step."""
    half = int(np.ceil(truncate * sigma))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return -x / sigma ** 2 * g, -y / sigma ** 2 * g


def gradients(gray, sigma=1.0):
    gx, gy = gaussian_derivative_kernels(sigma)
    fx = ndimage.convolve(gray.astype(float), gx, mode="nearest")
    fy = ndimage.convolve(gray.astype(float), gy, mode="nearest")
    magnitude = np.hypot(fx, fy)
    angle = (np.degrees(np.arctan2(fy, fx)) + 180.0) % 180.0  # edge normal direction in [0, 180)
    return fx, fy, magnitude, angle


def non_max_suppression(magnitude, angle):
    """Keep a pixel only if it is a local maximum along the gradient direction."""
    h, w = magnitude.shape
    m = np.pad(magnitude, 1)
    out = np.zeros_like(magnitude)
    # neighbours for the four quantised directions: 0, 45, 90, 135 degrees
    q = np.zeros_like(angle, dtype=int)
    q[(angle >= 22.5) & (angle < 67.5)] = 1
    q[(angle >= 67.5) & (angle < 112.5)] = 2
    q[(angle >= 112.5) & (angle < 157.5)] = 3
    offsets = {0: ((0, -1), (0, 1)), 1: ((-1, -1), (1, 1)), 2: ((-1, 0), (1, 0)), 3: ((-1, 1), (1, -1))}
    for k, ((dy1, dx1), (dy2, dx2)) in offsets.items():
        n1 = m[1 + dy1:1 + dy1 + h, 1 + dx1:1 + dx1 + w]
        n2 = m[1 + dy2:1 + dy2 + h, 1 + dx2:1 + dx2 + w]
        keep = (q == k) & (magnitude >= n1) & (magnitude >= n2)
        out[keep] = magnitude[keep]
    return out


def hysteresis(nms, low, high):
    """Strong pixels (>= high) seed edges; weak pixels (>= low) survive only if connected to a strong one."""
    strong = nms >= high
    weak = nms >= low
    labels, n = ndimage.label(weak, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def canny(gray, sigma=1.0, low_ratio=0.1, high_ratio=0.25):
    """Full detector. Thresholds are fractions of the maximum gradient magnitude."""
    fx, fy, mag, ang = gradients(gray, sigma)
    nms = non_max_suppression(mag, ang)
    peak = nms.max() + 1e-12
    edges = hysteresis(nms, low_ratio * peak, high_ratio * peak)
    return {"fx": fx, "fy": fy, "magnitude": mag, "angle": ang, "nms": nms, "edges": edges}
