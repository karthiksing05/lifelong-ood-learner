"""What a composer borrows: per demo frame, the weight each vocabulary prototype carries (averaged over the chunk's steps
and dimensions), named with the atlas's motion words; and which action dimensions each prototype lends.

    python experiments/r6_borrow_viz.py --tasks 39,35,36,42,67,19
Writes figures/robot/borrow_<task>.png and prints each task's borrowing as a sequence of words.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import composer as C                          # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402

DIMS = ["fwd", "left", "up", "roll", "pitch", "yaw", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="39,35,36,42,67,19")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("atlas", os.path.join(ROOT, "experiments", "r2c_atlas.py"))
    atlas = importlib.util.module_from_spec(spec); spec.loader.exec_module(atlas)
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    for tid in [int(x) for x in a.tasks.split(",")]:
        cp = C.ComposerProgram.load(f"results/robot/r6_comp_{tid}{a.tag}.npz")
        tree = Tree.load(cp.library)
        comp = cp.model()
        Mk, Vk = C.prototypes(tree, cp.ids)
        ep = d90.load(d90.episodes_of_task(tid)[0])
        T = len(ep["actions"])
        Z = torch.as_tensor(C.features(ep["state"], np.arange(T) / max(T - 1, 1)))
        with torch.no_grad():
            _, _, w = comp(Z, Mk, Vk)
        w = w.numpy()                                          # [T, M, H, D]
        wt = w.mean(axis=(2, 3))                               # [T, M]
        names = [f"{atlas.describe(D.from_rows(tree.nodes[i].mode[None], norm, 10)[0])['name']} (t={tree.nodes[i].u})" for i in cp.ids]
        top = np.argsort(-wt.sum(0))[:8]
        dim_share = w.mean(axis=(0, 2))                        # [M, D]: how much each prototype lends to each dimension
        # the borrowing as a sequence of words: the dominant prototype per frame, run-length encoded
        dom = wt.argmax(1)
        seq = [(0, names[dom[0]])] + [(f, names[dom[f]]) for f in range(1, T) if dom[f] != dom[f - 1]]
        out[tid] = [(int(f), n) for f, n in seq]
        print(f"[borrow] task {tid} '{texts[tid]}': " + " -> ".join(f"{n} @{f}" for f, n in seq[:14]), flush=True)
        fig, axs = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw=dict(width_ratios=[3, 1.2]))
        axs[0].stackplot(np.arange(T), *[wt[:, k] for k in top], labels=[names[k] for k in top], alpha=0.85)
        for g in D.gripper_events(ep["actions"]):
            axs[0].axvline(g, color="k", lw=1, ls="--")
        axs[0].set_xlim(0, T - 1); axs[0].set_ylim(0, 1)
        axs[0].set_xlabel("demo frame (dashed: gripper events)"); axs[0].set_ylabel("share of the composed chunk")
        axs[0].set_title(f"'{texts[tid]}': which library prototypes the composer borrows, frame by frame")
        axs[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        axs[1].imshow(dim_share[top] / np.maximum(dim_share[top].sum(0, keepdims=True), 1e-9), aspect="auto", cmap="Greens")
        axs[1].set_xticks(range(7)); axs[1].set_xticklabels(DIMS, fontsize=7)
        axs[1].set_yticks(range(len(top))); axs[1].set_yticklabels([names[k][:26] for k in top], fontsize=6)
        axs[1].set_title("which dimensions each prototype lends", fontsize=9)
        fig.tight_layout(); fig.savefig(f"figures/robot/borrow_{tid}.png", dpi=110); plt.close(fig)
    json.dump(out, open("results/robot/borrow_sequences.json", "w"), indent=1)


if __name__ == "__main__":
    main()
