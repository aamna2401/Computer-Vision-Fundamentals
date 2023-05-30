"""Run the lane pipeline on sample dashcam frames; save a stage-by-stage panel, a grid of results and metrics."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lane import apply_roi, colour_mask, detect_edges, process

HERE = Path(__file__).parent
FRAMES = sorted((HERE.parent / "data" / "lane_frames").glob("*.jpg"))
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# stage-by-stage on the first frame
im = cv2.imread(str(FRAMES[0]))
res = process(im)
vis_roi = im.copy(); cv2.polylines(vis_roi, res["roi"], True, (0, 255, 255), 4)
stages = [("input + ROI", rgb(vis_roi)), ("colour mask (yellow/white)", colour_mask(im)),
          ("Canny edges AND colour", detect_edges(im)), ("edges in ROI", res["edges"]), ("result", rgb(res["annotated"]))]
fig, ax = plt.subplots(1, 5, figsize=(22, 3.6))
for a, (t, s) in zip(ax, stages):
    a.imshow(s, cmap=None if s.ndim == 3 else "gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "pipeline_stages.png", dpi=100); plt.close()

# grid over all sample frames
metrics = {}
fig, ax = plt.subplots(2, 3, figsize=(16, 6.5))
for a, f in zip(ax.ravel(), FRAMES):
    im = cv2.imread(str(f)); r = process(im)
    a.imshow(rgb(r["annotated"])); a.set_title(f.name); a.axis("off")
    metrics[f.name] = {"both_lanes_found": bool(r["left"] and r["right"]), "segments": int(len(r["segments"])),
                       "offset": None if r["offset"] is None else round(r["offset"], 3)}
plt.tight_layout(); plt.savefig(OUT / "lane_results_grid.png", dpi=100); plt.close()

# speed
im = cv2.imread(str(FRAMES[0])); im = cv2.resize(im, (960, 540))
import time
t = time.perf_counter()
for _ in range(30): process(im)
ms = (time.perf_counter() - t) / 30 * 1000
summary = {"frames": len(FRAMES), "both_lanes_found": sum(v["both_lanes_found"] for v in metrics.values()),
           "ms_per_frame_960x540": round(ms, 1), "per_frame": metrics}
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
