"""Harris demo: corners on sample images, threshold sweep, and repeatability under rotation / brightness change."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harris import detect_corners, harris_response

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "images"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def load(name, size=400):
    im = cv2.imread(str(DATA / name))
    s = size / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s)


def draw(im, pts, color=(0, 0, 255)):
    out = im.copy()
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 3, color, -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# 1. detections on three images, ours vs OpenCV
fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
counts = {}
for j, name in enumerate(["Img1.jpg", "Img3.png", "brick_wall.png"]):
    im = load(name)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    pts, _ = detect_corners(gray, rel_threshold=0.02)
    cv = cv2.cornerHarris(np.float32(gray), 3, 3, 0.05)
    ys, xs = np.nonzero(cv > 0.02 * cv.max())
    ax[0, j].imshow(draw(im, pts)); ax[0, j].set_title(f"{name}: ours ({len(pts)} corners)")
    ax[1, j].imshow(draw(im, np.stack([xs, ys], 1), (0, 160, 0))); ax[1, j].set_title(f"cv2.cornerHarris ({len(xs)} raw px, no NMS)")
    counts[name] = {"ours": int(len(pts)), "opencv_raw_pixels": int(len(xs))}
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "harris_detections.png", dpi=110); plt.close()

# 2. threshold sweep on a textured facade (the clean checkerboard always gives its 49 interior corners)
im = load("brick_wall.png", 400); gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
sweep = {}
for a, t in zip(ax, [0.001, 0.01, 0.05, 0.2]):
    pts, _ = detect_corners(gray, rel_threshold=t)
    a.imshow(draw(im, pts)); a.set_title(f"threshold {t} -> {len(pts)} corners"); a.axis("off")
    sweep[str(t)] = int(len(pts))
plt.tight_layout(); plt.savefig(OUT / "harris_threshold_sweep.png", dpi=110); plt.close()

# 3. repeatability: detect, transform image, detect again, map back, count matches within 3 px
im = load("brick_wall.png", 400); gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
base, _ = detect_corners(gray, rel_threshold=0.02)
base = base[:300]
rep = {}
for ang in [0, 15, 30, 45, 90]:
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h))
    pts, _ = detect_corners(rot, rel_threshold=0.02)
    mapped = (M[:, :2] @ base.T + M[:, 2:3]).T
    inside = (mapped[:, 0] > 10) & (mapped[:, 0] < w - 10) & (mapped[:, 1] > 10) & (mapped[:, 1] < h - 10)
    mapped = mapped[inside]
    d = np.linalg.norm(mapped[:, None] - pts[None], axis=2).min(1) if len(pts) else np.full(len(mapped), 99)
    rep[f"rotation_{ang}deg"] = round(float((d < 3).mean()), 3)
for gain in [0.5, 0.75, 1.5]:  # x1.5 clips highlights, which destroys some corners
    adj = np.clip(gray * gain, 0, 255).astype(np.uint8)
    pts, _ = detect_corners(adj, rel_threshold=0.02)
    d = np.linalg.norm(base[:, None] - pts[None], axis=2).min(1)
    rep[f"brightness_x{gain}"] = round(float((d < 3).mean()), 3)
plt.figure(figsize=(7, 3.6))
plt.bar(list(rep.keys()), list(rep.values()), color="#3b6ea5")
plt.xticks(rotation=35, ha="right"); plt.ylim(0, 1.05); plt.ylabel("repeatability")
plt.title("Harris corner repeatability (3 px tolerance)")
plt.tight_layout(); plt.savefig(OUT / "harris_repeatability.png", dpi=110); plt.close()

metrics = {"corner_counts": counts, "threshold_sweep_facade": sweep, "repeatability": rep}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
