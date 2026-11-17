"""Classical CV toolbox demo: pyramids, LoG edges, RANSAC, SIFT + RANSAC homography, Viola-Jones, background subtraction, optical flow."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basics import laplacian_pyramid, log_zero_crossings, ransac_line, reconstruct

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
rng = np.random.default_rng(0)
metrics = {}

wall = cv2.imread(str(DATA / "images" / "wallLahoreFortBBC.jpg"))
wall = cv2.resize(wall, (640, int(640 * wall.shape[0] / wall.shape[1])))
gray = cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY)

# 1. Laplacian pyramid: band-pass decomposition and lossless reconstruction
lap = laplacian_pyramid(gray, 4)
rec = reconstruct(lap)
metrics["laplacian_pyramid_reconstruction_max_abs_error"] = float(np.abs(rec - gray).max())
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for a, band, t in zip(ax, lap, ["band 0 (finest)", "band 1", "band 2", "low-pass residual"]):
    a.imshow(band, cmap="gray"); a.set_title(f"{t} {band.shape[1]}x{band.shape[0]}"); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "laplacian_pyramid.png", dpi=100); plt.close()

# 2. LoG zero crossings at three scales
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
ax[0].imshow(gray, cmap="gray"); ax[0].set_title("input")
for a, s in zip(ax[1:], [1.5, 3.0, 5.0]):
    z, _ = log_zero_crossings(gray, s, min_contrast=1.0 / s)
    a.imshow(z, cmap="gray"); a.set_title(f"LoG zero crossings, sigma={s}")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "log_edges.png", dpi=100); plt.close()

# 3. RANSAC vs least squares as outliers grow
xs = np.linspace(0, 100, 80)
true_a, true_b = 0.8, 10
ls_err, rs_err, ratios = [], [], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for r in ratios:
    e1, e2 = [], []
    for seed in range(30):
        g = np.random.default_rng(seed)
        pts = np.stack([xs, true_a * xs + true_b + g.normal(0, 1, len(xs))], 1)
        n_out = int(r * len(xs)); idx = g.choice(len(xs), n_out, replace=False)
        pts[idx, 1] = g.uniform(0, 100, n_out)
        a1, b1 = np.polyfit(pts[:, 0], pts[:, 1], 1)
        a2, b2, _ = ransac_line(pts, iters=300, tol=3, seed=seed)
        e1.append(abs(a1 - true_a)); e2.append(abs(a2 - true_a))
    ls_err.append(np.mean(e1)); rs_err.append(np.mean(e2))
metrics["ransac_vs_least_squares_slope_error"] = {f"{int(r*100)}% outliers": {"least_squares": round(l, 3), "ransac": round(q, 3)}
                                                   for r, l, q in zip(ratios, ls_err, rs_err)}
g = np.random.default_rng(1)
pts = np.stack([xs, true_a * xs + true_b + g.normal(0, 1, len(xs))], 1)
idx = g.choice(len(xs), 32, replace=False); pts[idx, 1] = g.uniform(0, 100, 32)
a2, b2, inl = ransac_line(pts, 300, 3); a1, b1 = np.polyfit(pts[:, 0], pts[:, 1], 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(*pts[~inl].T, c="red", s=12, label="outliers"); ax[0].scatter(*pts[inl].T, c="tab:blue", s=12, label="inliers")
ax[0].plot(xs, a1 * xs + b1, "k--", label="least squares"); ax[0].plot(xs, a2 * xs + b2, "g-", label="RANSAC")
ax[0].legend(); ax[0].set_title("40% outliers")
ax[1].plot([int(r * 100) for r in ratios], ls_err, "o--", c="k", label="least squares")
ax[1].plot([int(r * 100) for r in ratios], rs_err, "s-", c="g", label="RANSAC")
ax[1].set_xlabel("% outliers"); ax[1].set_ylabel("slope error"); ax[1].legend(); ax[1].set_title("Robustness to outliers")
plt.tight_layout(); plt.savefig(OUT / "ransac_vs_least_squares.png", dpi=100); plt.close()

# 4. SIFT + RANSAC homography between an image and a rotated, scaled, perspective-warped copy
h, w = gray.shape
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst = np.float32([[60, 40], [w - 30, 10], [w - 80, h - 20], [20, h - 60]])
Hgt = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(wall, Hgt, (w, h))
warped = cv2.GaussianBlur(warped, (5, 5), 0)
sift = cv2.SIFT_create()
k1, d1 = sift.detectAndCompute(gray, None); k2, d2 = sift.detectAndCompute(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), None)
m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
good = [a for a, b in m if a.distance < 0.75 * b.distance]
p1 = np.float32([k1[x.queryIdx].pt for x in good]); p2 = np.float32([k2[x.trainIdx].pt for x in good])
Hest, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
grid = np.float32([[x, y] for x in range(0, w, 80) for y in range(0, h, 80)]).reshape(-1, 1, 2)
err = np.linalg.norm(cv2.perspectiveTransform(grid, Hest) - cv2.perspectiveTransform(grid, Hgt), axis=2).mean()
metrics["sift_ransac"] = {"keypoints": [len(k1), len(k2)], "ratio_test_matches": len(good), "ransac_inliers": int(mask.sum()),
                          "inlier_ratio": round(float(mask.mean()), 3), "mean_homography_error_px": round(float(err), 3)}
vis = cv2.drawMatches(wall, k1, warped, k2, [g for g, mk in zip(good, mask.ravel()) if mk][:80], None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(14, 4.2)); plt.imshow(rgb(vis)); plt.axis("off")
plt.title(f"SIFT + RANSAC: {int(mask.sum())}/{len(good)} inliers, homography error {err:.2f}px")
plt.tight_layout(); plt.savefig(OUT / "sift_ransac_matches.png", dpi=100); plt.close()

# 5. Viola-Jones face detection
face_img = cv2.imread(str(DATA / "images" / "index.jpeg"))
casc = cv2.CascadeClassifier(str(DATA / "images" / "haarcascade_frontalface_alt.xml"))
faces = casc.detectMultiScale(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
for x, y, fw, fh in faces:
    cv2.rectangle(face_img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
metrics["viola_jones_faces_found"] = int(len(faces))

# 6. video: MOG2 background subtraction + Farneback optical flow
cap = cv2.VideoCapture(str(DATA / "video" / "shibuya_clip.mp4"))
mog = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=32, detectShadows=False)
frames, fg_frac, prev = [], [], None
while True:
    ok, f = cap.read()
    if not ok:
        break
    mask_fg = mog.apply(f)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg_frac.append(float((mask_fg > 0).mean()))
    frames.append((f, mask_fg))
sel = len(frames) - 1
f, fgm = frames[sel]
g0 = cv2.cvtColor(frames[sel - 3][0], cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros_like(f); hsv[..., 0] = ang * 90 / np.pi; hsv[..., 1] = 255; hsv[..., 2] = np.clip(mag * 25, 0, 255)
flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
fig, ax = plt.subplots(1, 4, figsize=(18, 3.4))
for a, im, t, cm in zip(ax, [f, fgm, flow_vis, f], ["frame", "MOG2 foreground", "Farneback flow (hue = direction)", "flow arrows"], [None, "gray", None, None]):
    a.imshow(rgb(im) if im.ndim == 3 else im, cmap=cm); a.set_title(t); a.axis("off")
ys, xs_ = np.mgrid[8:f.shape[0]:16, 8:f.shape[1]:16]
ax[3].quiver(xs_, ys, flow[ys, xs_, 0], flow[ys, xs_, 1], color="yellow", angles="xy", scale_units="xy", scale=0.6, width=0.003)
plt.tight_layout(); plt.savefig(OUT / "video_motion.png", dpi=100); plt.close()
metrics["video"] = {"frames": len(frames), "foreground_fraction_mean_after_warmup": round(float(np.mean(fg_frac[30:])), 4)}

# face figure
plt.figure(figsize=(3.4, 3.4)); plt.imshow(rgb(face_img)); plt.axis("off"); plt.title(f"Viola-Jones: {len(faces)} face(s)")
plt.tight_layout(); plt.savefig(OUT / "viola_jones.png", dpi=100); plt.close()

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
