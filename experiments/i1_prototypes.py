"""INTERPRETATION 1: what a 4 KB object prototype encodes (exploratory, offline).  For the library's prototypes
(results/robot/q4_object_protos.npz: the policy's own SigLIP features at a demo's first grasp), the pairwise cosine similarity is
explained by three factors: the same object TYPE, the same ROOM (LIBERO scene), and the distance between the two grasp patches in
the image (PLACE).  Nested least squares give each factor's share of the explained variance; average-linkage clustering on
1 - cosine gives the prototypes' concept hierarchy (figures/robot/i1_prototype_tree.png).

    sbatch --export=ALL,SCRIPT="experiments/i1_prototypes.py --record" slurm/jobs/cpu.sbatch
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")


def grasp_tokens(tids):
    """The demo's first-grasp patch (row, col) in the 16 x 16 token grid, as q4_template.py builds the prototype."""
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import where as W
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90 = D.Demos90()
    out = {}
    for t in tids:
        f = os.path.join(R, "f0b_firstframes", f"libero_90_{t}.npz")
        eps = d90.episodes_of_task(t)
        if not eps or not os.path.exists(f):
            continue
        ep = d90.load(eps[0])
        closed = np.where(ep["actions"][:, 6] > 0)[0]
        if len(closed) == 0:
            continue
        z = np.load(f, allow_pickle=True)
        rc = W.project(ep["state"][int(closed[0])][:3], z["K"], z["E"][0])
        out[t] = (int(np.clip(rc[0] // 16, 0, 15)), int(np.clip(rc[1] // 16, 0, 15)))
    return out


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2), b


def main():
    from lifelong_ood.policy import factors as F
    z = np.load(os.path.join(R, "q4_object_protos.npz"), allow_pickle=True)
    tids = [int(t) for t in z["tids"]]
    types = {int(t): str(ty) for t, ty in zip(z["tids"], z["types"])}
    scene = {t.index: t.scene for t in F.load() if t.suite == "libero_90"}
    tok = grasp_tokens(tids)
    keep = [t for t in tids if t in tok]
    V = np.stack([z[str(t)] for t in keep])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    C = V @ V.T
    iu = np.triu_indices(len(keep), 1)
    ti, tj = np.array(keep)[iu[0]], np.array(keep)[iu[1]]
    cos = C[iu]
    st = np.array([types[a] == types[b] for a, b in zip(ti, tj)], float)
    ss = np.array([scene[a] == scene[b] for a, b in zip(ti, tj)], float)
    dist = np.array([abs(tok[a][0] - tok[b][0]) + abs(tok[a][1] - tok[b][1]) for a, b in zip(ti, tj)], float)
    full, coef = r2([st, ss, dist], cos)
    shares = {}
    for name, cols in (("type", [ss, dist]), ("room", [st, dist]), ("place", [st, ss])):
        shares[name] = float(full - r2(cols, cos)[0])               # the variance this factor explains beyond the other two
    groups = {g: float(cos[m].mean()) for g, m in (("same type, same room", (st == 1) & (ss == 1)),
                                                     ("same type, other room", (st == 1) & (ss == 0)),
                                                     ("other type, same room", (st == 0) & (ss == 1)),
                                                     ("other type, other room", (st == 0) & (ss == 0))) if m.any()}
    near = (st == 1) & (ss == 1)
    place_within = float(np.corrcoef(dist[near], cos[near])[0, 1]) if near.sum() > 3 else None
    res = dict(n=len(keep), r2_full=float(full), coef=dict(zip(("intercept", "same_type", "same_room", "patch_distance"), map(float, coef))),
               unique_share=shares, group_mean_cos=groups, place_corr_same_type_room=place_within,
               labels={str(t): f"{types[t].replace('akita_', '').replace('_', ' ')} | {scene[t].lower()} | {tok[t]}" for t in keep})
    print(json.dumps({k: v for k, v in res.items() if k != "labels"}, indent=1), flush=True)
    figure(C, keep, res["labels"])
    json.dump(res, open(os.path.join(R, "i1_prototypes.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("i1_prototypes", bench="libero_90", tag="read", seed=0, metrics={k: v for k, v in res.items() if k != "labels"},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i1_prototypes.py", command=" ".join(sys.argv)))


def figure(C, keep, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    D_ = np.clip(1 - C, 0, None)
    np.fill_diagonal(D_, 0)
    Z = linkage(squareform(D_, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(8, 12))
    dendrogram(Z, labels=[labels[str(t)] for t in keep], orientation="left", ax=ax, color_threshold=0.0, above_threshold_color="#5d686f",
               leaf_font_size=7)
    ax.set_xlabel("1 - cosine (average linkage)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "robot", "i1_prototype_tree.png"), dpi=120)


if __name__ == "__main__":
    main()
