"""Harris corner detector from first principles.

Structure tensor M = G_sigma * [[Ix^2, IxIy], [IxIy, Iy^2]]; response R = det(M) - k * trace(M)^2.
Corners are local maxima of R above a fraction of the strongest response.
"""
import numpy as np
from scipy import ndimage


def harris_response(gray, sigma=1.5, k=0.05):
    g = ndimage.gaussian_filter(gray.astype(float), 1.0)
    ix = ndimage.sobel(g, axis=1) / 8.0
    iy = ndimage.sobel(g, axis=0) / 8.0
    sxx = ndimage.gaussian_filter(ix * ix, sigma)
    syy = ndimage.gaussian_filter(iy * iy, sigma)
    sxy = ndimage.gaussian_filter(ix * iy, sigma)
    det = sxx * syy - sxy ** 2
    trace = sxx + syy
    return det - k * trace ** 2


def detect_corners(gray, sigma=1.5, k=0.05, rel_threshold=0.01, nms_size=7, border=8):
    r = harris_response(gray, sigma, k)
    local_max = r == ndimage.maximum_filter(r, size=nms_size)
    mask = local_max & (r > rel_threshold * r.max())
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    order = np.argsort(-r[ys, xs])
    return np.stack([xs[order], ys[order]], axis=1), r
