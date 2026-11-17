"""Rectify a perspective-distorted tiled floor and measure how homography error grows with click noise."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rectify import apply_h, homography, rectify

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
rng = np.random.default_rng(0)

# 1. real image: four grout intersections around one tile (clicked by hand), assumed to be a square
img = cv2.imread(str(DATA / "rectification" / "img6.jpg"))
quad = np.array([[102, 44], [194, 73], [142, 131], [38, 94]], float)
rect, H = rectify(img, quad, tile_px=140, out_size=700)
marked = img.copy()
cv2.polylines(marked, [quad.astype(np.int32)], True, (0, 255, 255), 2)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(rgb(marked)); ax[0].set_title("input: perspective view, one tile marked")
ax[1].imshow(rgb(rect)); ax[1].set_title("rectified: tiles become squares")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "tile_rectification.png", dpi=110); plt.close()

# 2. quantitative test on synthetic ground truth: warp a checkerboard, re-estimate H from noisy clicks
board = cv2.imread(str(DATA / "images" / "Img1.jpg"))
board = cv2.resize(board, (480, 480))
true_src = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], float)
true_dst = np.array([[120, 60], [430, 20], [470, 420], [40, 380]], float)
Ht = homography(true_src, true_dst)
grid = np.array([[x, y] for x in np.linspace(40, 440, 9) for y in np.linspace(40, 440, 9)], float)
gt = apply_h(Ht, grid)
res = {}
for noise in [0, 0.5, 1, 2, 4, 8]:
    errs = []
    for _ in range(200):
        Hn = homography(true_src, true_dst + rng.normal(0, noise, (4, 2)))
        errs.append(np.linalg.norm(apply_h(Hn, grid) - gt, axis=1).mean())
    res[str(noise)] = round(float(np.mean(errs)), 3)
plt.figure(figsize=(6.2, 3.8))
plt.plot([float(k) for k in res], list(res.values()), "o-", color="#3b6ea5")
plt.xlabel("click noise (px std on the 4 corners)"); plt.ylabel("mean grid error after warp (px)")
plt.title("Homography sensitivity to point-picking error"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(OUT / "homography_noise_sensitivity.png", dpi=110); plt.close()

# 3. synthetic round trip picture
warped = cv2.warpPerspective(board, Ht, (520, 460))
back, _ = rectify(warped, apply_h(Ht, np.array([[0, 0], [480, 0], [480, 480], [0, 480]], float)), tile_px=480)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for a, im, t in zip(ax, [board, warped, back], ["ground truth plane", "synthetic perspective view", "recovered by homography"]):
    a.imshow(rgb(im)); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "synthetic_roundtrip.png", dpi=110); plt.close()

metrics = {"synthetic_mean_grid_error_px_vs_click_noise": res,
           "tile_homography": np.round(H, 4).tolist()}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics["synthetic_mean_grid_error_px_vs_click_noise"]))
