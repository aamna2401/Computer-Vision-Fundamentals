"""Calibration from 20 known 3-D/2-D correspondences, then fundamental matrix, epipolar lines and 3-D structure on a temple pair."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from geometry import decompose, dlt_projection, eight_point, epipolar_lines, project, sampson_error

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "calibration"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
metrics = {}

# ---- calibration (DLT) ----
p2d, p3d = np.loadtxt(DATA / "pts2d.txt"), np.loadtxt(DATA / "pts3d.txt")
P = dlt_projection(p2d, p3d)
err = np.linalg.norm(project(P, p3d) - p2d, axis=1)
K, R, t = decompose(P)
metrics["calibration"] = {"points": len(p2d), "reprojection_rmse_image_units": round(float(np.sqrt((err ** 2).mean())), 5),
                          "K": np.round(K, 3).tolist()}
# how does error behave when image points are noisy?
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
rmse_true = []
for s in noise_levels:
    errs = []
    for _ in range(50):
        Pn = dlt_projection(p2d + rng.normal(0, s, p2d.shape), p3d)
        errs.append(np.sqrt((np.linalg.norm(project(Pn, p3d) - p2d, axis=1) ** 2).mean()))
    rmse_true.append(float(np.mean(errs)))
metrics["calibration"]["rmse_vs_noise"] = dict(zip(map(str, noise_levels), np.round(rmse_true, 5).tolist()))

# ---- two-view geometry on the temple pair ----
d = np.load(DATA / "data.npz")
im1, im2 = cv2.imread(str(DATA / "im1.png")), cv2.imread(str(DATA / "im2.png"))
p1, p2, K1, K2 = d["pts1"].astype(float), d["pts2"].astype(float), d["K1"], d["K2"]
F_raw = eight_point(p1, p2, normalised=False)
F = eight_point(p1, p2, normalised=True)
F_cv, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT)
metrics["fundamental"] = {
    "points": len(p1),
    "sampson_mean_px_unnormalised": round(float(sampson_error(F_raw, p1, p2).mean()), 3),
    "sampson_mean_px_normalised": round(float(sampson_error(F, p1, p2).mean()), 3),
    "sampson_mean_px_opencv_8pt": round(float(sampson_error(F_cv / F_cv[2, 2], p1, p2).mean()), 3),
    "rank": int(np.linalg.matrix_rank(F, tol=1e-8)),
}

# epipolar lines for 8 random points
idx = rng.choice(len(p1), 8, replace=False)
lines = epipolar_lines(F, p1[idx], 1)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
colors = plt.cm.tab10(np.arange(8))
h, w = im2.shape[:2]
ax[0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)); ax[1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
for c, pt, l in zip(colors, p1[idx], lines):
    ax[0].scatter(*pt, color=c, s=40, edgecolors="w")
    xs = np.array([0, w]); ys = -(l[0] * xs + l[2]) / l[1]
    ax[1].plot(xs, ys, color=c, lw=1.5)
ax[1].scatter(*p2[idx].T, c=colors, s=40, edgecolors="w"); ax[1].set_xlim(0, w); ax[1].set_ylim(h, 0)
ax[0].set_title("image 1: chosen points"); ax[1].set_title("image 2: epipolar lines from F (points should lie on their line)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "epipolar_lines.png", dpi=110); plt.close()

# ---- pose from essential matrix, triangulation ----
E = K2.T @ F @ K1
_, Rr, tr, mask = cv2.recoverPose(E, p1, p2, K1)
P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))]); P2 = K2 @ np.hstack([Rr, tr])
X = cv2.triangulatePoints(P1, P2, p1.T, p2.T); X = (X[:3] / X[3]).T
back = project(P2, X); reproj = np.linalg.norm(back - p2, axis=1)
metrics["structure"] = {"triangulated_points": int(len(X)), "mean_reprojection_px_view2": round(float(reproj.mean()), 3),
                        "baseline_direction": np.round((tr / np.linalg.norm(tr)).ravel(), 3).tolist()}
fig = plt.figure(figsize=(12, 5))
a = fig.add_subplot(121, projection="3d"); a.scatter(X[:, 0], X[:, 2], -X[:, 1], c=X[:, 2], s=8, cmap="viridis")
a.set_title("sparse 3-D structure (triangulated)"); a.set_xlabel("x"); a.set_ylabel("z"); a.set_zlabel("-y")
b = fig.add_subplot(122); b.bar(["raw 8-pt", "normalised 8-pt", "OpenCV 8-pt"],
                                [metrics["fundamental"][k] for k in ["sampson_mean_px_unnormalised", "sampson_mean_px_normalised", "sampson_mean_px_opencv_8pt"]],
                                color=["#c0504d", "#3b6ea5", "#7f7f7f"])
b.set_ylabel("mean Sampson error (px)"); b.set_title("Why normalise the points?")
plt.tight_layout(); plt.savefig(OUT / "structure_and_normalisation.png", dpi=110); plt.close()

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
