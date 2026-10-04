"""Prototype gallery: what a held-out task borrows, seen through the training tasks that share each prototype.

For a held-out LIBERO-90 task's dimension-wise program, the P prototypes lending the most (frame, dimension) pairs;
per prototype one row: the held-out demo's frame where it lends most, then the medoid member frame (the member chunk
nearest the prototype's mode) of each of the training tasks with the most members, with the prototype's motion word
(r2c_atlas.describe of its medoid chunk).  Runs with the openpi venv's python (PyAV for the LIBERO-90 videos):

    /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/r8_gallery.py --task 39
"""
import argparse
import importlib.util
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                     # noqa: E402
from lifelong_ood.tree.nodes import Tree                                # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=39)
    ap.add_argument("--P", type=int, default=4, help="prototypes (rows)")
    ap.add_argument("--S", type=int, default=4, help="training tasks per prototype (columns)")
    ap.add_argument("--tree", default="results/robot/r2b_libero_40_tree.npz")
    ap.add_argument("--npz", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--shared", type=float, default=0.0, help="instead of a task's lenders: the prototypes at this noise level "
                    "(e.g. 0.4) that the most training tasks share, one row each, its member frames in different tasks")
    a = ap.parse_args()
    os.chdir(ROOT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    atlas = load_module("atlas", os.path.join(ROOT, "experiments", "r2c_atlas.py"))
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    tree, z = Tree.load(a.tree), np.load(a.npz)
    norm = D.Norm.load()
    demos, d90 = D.Demos(), D.Demos90()
    texts, _ = r3p.libero90_texts()
    # node -> (level, member indices)
    members = {}
    for l in ((2, 3, 4) if not a.shared else (tree.u_grid.index(a.shared),)):
        owner = z[f"owner_policy_{tree.u_grid[l]}"]
        for k, nid in zip(sorted(set(owner.tolist())), tree.levels[l]):
            members[int(nid)] = (l, np.where(owner == k)[0])
    if a.shared:
        ntk = {n: len(set(z["task"][m].tolist())) for n, (_, m) in members.items()}
        top = sorted(members, key=lambda n: (-ntk[n], -len(members[n][1])))[:a.P]
        share, lent, img90 = {int(n): 0.0 for n in top}, None, None
    else:
        prog = DemoProgram.load(f"results/robot/r7_dim_{a.task}_s1.npz")
        ids, ch = prog.dim_ids, prog.dim_choice
        lent = np.array([[int(ids[f, ch[f, d]]) for d in range(ch.shape[1])] for f in range(len(ids))])     # [T, 7] node per dim
        nodes, counts = np.unique(lent, return_counts=True)
        top = nodes[np.argsort(-counts)][:a.P]
        share = {int(n): float(c) / lent.size for n, c in zip(nodes, counts)}
        ep90 = d90.load(d90.episodes_of_task(a.task)[0])
        img90, _ = d90.frames(ep90["episode_index"])
    SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    fig, axs = plt.subplots(len(top), a.S + 1, figsize=(1.55 * (a.S + 1) + 2.4, 1.72 * len(top) + 0.6), facecolor=SURFACE, squeeze=False)
    ep_cache = {}
    for r, nid in enumerate(top):
        l, mem = members[int(nid)]
        mode = tree.nodes[int(nid)].mode
        med = mem[np.argmin(((z["mu"][mem] - mode[None]) ** 2).sum(1))]
        desc = atlas.describe(D.from_rows(z["mu"][med][None], norm, 10)[0])
        ax = axs[r, 0]
        if lent is not None:
            f = int(np.argmax((lent == nid).sum(1)))                    # the held-out frame where it lends most
            dims = [d for d in range(7) if lent[f, d] == nid]
            ax.imshow(img90[f])
            ax.set_title(f"held out, frame {f}\nlends dims {''.join('xyzabcg'[d] for d in dims)}", fontsize=6.3, color=INK2, pad=2)
        else:
            raw = D.from_rows(z["mu"][med][None], norm, 10)[0]            # the medoid (the member nearest the mode), as named
            path = np.cumsum(raw[:, :3], 0) * atlas.POS * 100            # cm
            path = np.vstack([np.zeros((1, 3)), path])
            ax.plot(path[:, 0], path[:, 2], color="#2a78d6", linewidth=2, solid_capstyle="round")
            ax.scatter(path[-1:, 0], path[-1:, 2], s=14, color="#2a78d6")
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
            ax.axhline(0, color="#e1e0d9", linewidth=0.6, zorder=0); ax.axvline(0, color="#e1e0d9", linewidth=0.6, zorder=0)
            ax.plot([-3.0, -2.0], [-3.0, -3.0], color=INK2, linewidth=1.2)
            ax.text(-2.5, -2.85, "1 cm", fontsize=5.5, color=INK2, ha="center", va="bottom")
            ax.set_title("the medoid's hand path, side view\n(forward vs up; dot = end)", fontsize=6.3, color=INK2, pad=2)
        tasks, n = np.unique(z["task"][mem], return_counts=True)
        order = tasks[np.argsort(-n)][:a.S]
        for c, tk in enumerate(order):
            mt = mem[z["task"][mem] == tk]
            i = mt[np.argmin(((z["mu"][mt] - mode[None]) ** 2).sum(1))]
            e = int(z["episode"][i])
            if e not in ep_cache:
                ep_cache[e] = demos.load(e, images=True)
            png = ep_cache[e]["image"][int(z["frame"][i])]
            im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            ax = axs[r, c + 1]
            ax.imshow(im)
            tx = demos.tasks[int(tk)]
            ax.set_title((tx if len(tx) <= 34 else tx[:32] + "...") + f"\n{int(n[np.argsort(-n)][c])} member frames", fontsize=5.6, color=INK2, pad=2)
        for c in range(a.S + 1):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            for sp in axs[r, c].spines.values():
                sp.set_visible(False)
        for c in range(len(order) + 1, a.S + 1):
            axs[r, c].axis("off")
        axs[r, 0].set_ylabel(f"prototype {int(nid)}\n\"{desc['name']}\"\nt = {tree.u_grid[l]}, {len(mem)} frames, {len(tasks)} tasks\n"
                             + (f"{share[int(nid)]:.0%} of the task's pairs\n" if lent is not None else "")
                             + f"moves {desc['dist'] * 100:.1f} cm, turns {np.degrees(desc['rot']):.0f} deg",
                             fontsize=6.6, color=INK, rotation=0, ha="right", va="center", labelpad=6)
    title = (f"What \"{texts[a.task]}\" borrows: its {len(top)} most-lent prototypes, each seen in the training tasks that share it"
             if lent is not None else f"The {len(top)} prototypes at t = {a.shared} that the most training tasks share, each seen in four of them")
    fig.suptitle(title, fontsize=8.5, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(wspace=0.05, hspace=0.42)
    out = f"figures/robot/r8_gallery_{a.task}" if lent is not None else f"figures/robot/r8_gallery_shared_t{a.shared}"
    fig.savefig(out + ".png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    print(f"[gallery] {out}.png: prototypes {[int(x) for x in top]}" + (f" lend {sum(share[int(x)] for x in top):.0%} of the pairs" if lent is not None else ""))


if __name__ == "__main__":
    main()
