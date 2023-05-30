"""GCN on Cora: scratch vs PyG vs MLP baseline, and a depth sweep."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gcn import GCN, MLP, normalised_adjacency

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
metrics = {}

# ---------------- A. node classification on Cora ----------------
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

data = Planetoid(str(HERE.parent / "data" / "cora"), "Cora")[0]
A_hat = normalised_adjacency(data.edge_index, data.num_nodes)
X = F.normalize(data.x, p=1)


def fit(model, forward, epochs=200, lr=0.01, wd=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
    best_val, test_at_best = 0, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = forward()
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = forward().argmax(1)
        val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val >= best_val:
            best_val, test_at_best = val, (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return test_at_best


class PyG(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.c1, self.c2 = GCNConv(1433, 16), GCNConv(16, 7)

    def forward(self):
        x = F.dropout(X, 0.5, self.training); x = F.relu(self.c1(x, data.edge_index))
        return self.c2(F.dropout(x, 0.5, self.training), data.edge_index)

res = {"mlp": [], "gcn_scratch": [], "gcn_pyg": []}
for seed in range(5):
    torch.manual_seed(seed)
    m = MLP(1433, 16, 7); res["mlp"].append(fit(m, lambda: m(X)))
    g = GCN(1433, 16, 7, 2); res["gcn_scratch"].append(fit(g, lambda: g(X, A_hat)))
    p = PyG(); res["gcn_pyg"].append(fit(p, p))
metrics["cora"] = {"nodes": data.num_nodes, "edges": int(data.edge_index.shape[1] // 2), "train_labels": int(data.train_mask.sum()),
                   **{k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} for k, v in res.items()}}

depth = {}
for d in [1, 2, 3, 4, 6, 8]:
    accs = []
    for seed in range(3):
        torch.manual_seed(seed); g = GCN(1433, 16, 7, d); accs.append(fit(g, lambda: g(X, A_hat)))
    depth[d] = float(np.mean(accs))
metrics["cora"]["depth_sweep_test_acc"] = {str(k): round(v, 3) for k, v in depth.items()}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
names = ["MLP (no graph)", "GCN scratch", "GCN PyG"]
ax[0].bar(names, [np.mean(res[k]) for k in res], yerr=[np.std(res[k]) for k in res], color=["#999", "#3b6ea5", "#6aa84f"], capsize=4)
ax[0].set_ylim(0.4, 0.9); ax[0].set_ylabel("test accuracy"); ax[0].set_title("Cora node classification (140 labels)")
ax[1].plot(list(depth), list(depth.values()), "o-"); ax[1].set_xlabel("number of graph-conv layers"); ax[1].set_ylabel("test accuracy"); ax[1].set_title("Over-smoothing: deeper is not better")
plt.tight_layout(); plt.savefig(OUT / "cora_results.png", dpi=110); plt.close()


(OUT / "cora_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
