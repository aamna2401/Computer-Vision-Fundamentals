"""Scene recognition on a 15-class subset: tiny-image baseline vs bag of SIFT words (kNN and linear SVM)."""
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bovw import build_vocabulary, dense_sift, histogram, load_split, tiny_image
from knn import predict

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

X_tr, y_tr, X_te, y_te, classes = load_split(HERE.parent / "data" / "scenes" / "train")
print(len(X_tr), "train", len(X_te), "test", len(classes), "classes")

res = {}
tr_tiny = np.array([tiny_image(g) for g in X_tr]); te_tiny = np.array([tiny_image(g) for g in X_te])
res["tiny_image_knn"] = float(np.mean(np.array(predict(tr_tiny, y_tr, te_tiny, 5)) == y_te))

tr_desc = [dense_sift(g) for g in X_tr]; te_desc = [dense_sift(g) for g in X_te]
best = None
for k in [20, 50, 100, 200, 400]:
    km = build_vocabulary(tr_desc, k)
    tr_h = np.array([histogram(d, km) for d in tr_desc]); te_h = np.array([histogram(d, km) for d in te_desc])
    acc_knn = float(np.mean(np.array(predict(tr_h, y_tr, te_h, 5)) == y_te))
    svm = LinearSVC(C=1.0, max_iter=5000).fit(tr_h, y_tr)
    pred = svm.predict(te_h)
    acc_svm = float(np.mean(pred == y_te))
    res[f"bovw_k{k}"] = {"knn": round(acc_knn, 3), "linear_svm": round(acc_svm, 3)}
    print(k, res[f"bovw_k{k}"])
    if best is None or acc_svm > best[0]:
        best = (acc_svm, k, pred, km, te_h)
res["tiny_image_knn"] = round(res["tiny_image_knn"], 3)
res["best_vocab_size"] = best[1]
res["chance"] = round(1 / len(classes), 3)

ks = [k for k in [20, 50, 100, 200, 400]]
plt.figure(figsize=(7, 4))
plt.plot(ks, [res[f"bovw_k{k}"]["knn"] for k in ks], "o-", label="BoVW + kNN (k=5)")
plt.plot(ks, [res[f"bovw_k{k}"]["linear_svm"] for k in ks], "s-", label="BoVW + linear SVM")
plt.axhline(res["tiny_image_knn"], color="gray", ls="--", label="tiny image + kNN")
plt.axhline(res["chance"], color="red", ls=":", label="chance")
plt.xscale("log"); plt.xlabel("vocabulary size"); plt.ylabel("test accuracy"); plt.legend(); plt.grid(alpha=.3)
plt.title("Scene recognition accuracy vs vocabulary size"); plt.tight_layout()
plt.savefig(OUT / "accuracy_vs_vocab.png", dpi=110); plt.close()

cm = confusion_matrix(y_te, best[2], normalize="true")
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=75, ha="right"); ax.set_yticklabels(classes)
ax.set_title(f"Confusion matrix, BoVW k={best[1]} + linear SVM (acc {best[0]:.2f})")
plt.colorbar(im); plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=110); plt.close()

per_class = {c: round(float(cm[i, i]), 2) for i, c in enumerate(classes)}
res["per_class_recall_best"] = per_class
(OUT / "metrics.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
