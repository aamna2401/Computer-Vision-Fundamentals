"""Run the from-scratch Canny on sample images, compare with OpenCV, save figures to results/."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from canny import canny

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "images"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def agreement(a, b, tol=1):
    """Precision / recall / F1 of edge map a against reference b, allowing a `tol` pixel tolerance."""
    db = ndimage.binary_dilation(b, iterations=tol)
    da = ndimage.binary_dilation(a, iterations=tol)
    p = (a & db).sum() / max(a.sum(), 1)
    r = (b & da).sum() / max(b.sum(), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


metrics = {}
for name in ["circle.jpg", "wallLahoreFortBBC.jpg", "Img1.jpg"]:
    gray = cv2.imread(str(DATA / name), cv2.IMREAD_GRAYSCALE)
    if max(gray.shape) > 600:
        gray = cv2.resize(gray, None, fx=600 / max(gray.shape), fy=600 / max(gray.shape))
    res = canny(gray, sigma=1.4)
    ref = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.4), 50, 150) > 0
    p, r, f = agreement(res["edges"], ref)
    metrics[name] = {"edge_pixels_ours": int(res["edges"].sum()), "edge_pixels_opencv": int(ref.sum()),
                     "precision_vs_opencv": round(p, 3), "recall_vs_opencv": round(r, 3), "f1_vs_opencv": round(f, 3)}

    fig, ax = plt.subplots(1, 6, figsize=(21, 3.8))
    for a, im, t in zip(ax, [gray, res["magnitude"], res["angle"], res["nms"], res["edges"], ref],
                        ["input", "gradient magnitude", "gradient orientation", "after NMS", "hysteresis (ours)", "cv2.Canny"]):
        a.imshow(im, cmap="twilight" if t == "gradient orientation" else "gray")
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"canny_{Path(name).stem}.png", dpi=110)
    plt.close()

# effect of sigma on the wall image
gray = cv2.imread(str(DATA / "wallLahoreFortBBC.jpg"), cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, None, fx=600 / max(gray.shape), fy=600 / max(gray.shape))
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
ax[0].imshow(gray, cmap="gray"); ax[0].set_title("input")
for a, s in zip(ax[1:], [0.8, 1.6, 3.0]):
    e = canny(gray, sigma=s)["edges"]
    a.imshow(e, cmap="gray"); a.set_title(f"sigma = {s}  ({int(e.sum())} edge px)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "canny_sigma_sweep.png", dpi=110); plt.close()

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
