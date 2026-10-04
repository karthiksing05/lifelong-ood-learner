"""How a held-out task is spelled from the prior's own motions (segment borrowing, policy/segcode.py).

Top: the demo's action trajectory per dimension (gray) and the rebuilt one (blue), the sequence the program executes.
Bottom: one cell per 10-step segment and action dimension, coloured by the suite of the training task whose leaf lends it
(the policy's chunk mean at one training frame), with that task's index.  Writes figures/robot/segments_<task>.{png,pdf}.

    python experiments/fig_segments.py --task 39
"""
import argparse
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy import segcode                                 # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                     # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SUITE_COLOR = {"libero_spatial": "#2a78d6", "libero_object": "#eb6834", "libero_goal": "#1baf7a", "libero_10": "#4a3aa7"}
SUITE_NAME = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "LIBERO-10"}
DIMS = ["forward", "left", "up", "roll", "pitch", "yaw", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=39)
    a = ap.parse_args()
    os.chdir(ROOT)
    z = np.load("results/robot/r2b_libero_40.npz")
    leaves = z["mu"].astype(np.float32).reshape(-1, 10, 7)
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    ep = d90.load(d90.episodes_of_task(a.task)[0])
    T = len(ep["actions"])
    rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
    seq = segcode.sequence(rows)
    prog = DemoProgram.load(f"results/robot/r10_seg_{a.task}_s1.npz")
    hat = segcode.decode(leaves, prog.seg_ids, prog.seg_offs, len(seq))
    src_task = z["task"][prog.seg_ids]                                   # [nseg, 7]
    suite_of = {t: s for s, ts in D.SUITES.items() if s != "libero_40" for t in ts}
    train_text = D.Demos().tasks
    nseg = len(prog.seg_ids)
    fig = plt.figure(figsize=(11, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(8, 1, height_ratios=[1] * 7 + [2.4], hspace=0.12)
    x = np.arange(len(seq))
    for d in range(7):
        ax = fig.add_subplot(gs[d])
        ax.set_facecolor(SURFACE)
        for s in range(nseg):
            ax.axvspan(s * 10 - 0.5, min(s * 10 + 10, len(seq)) - 0.5, color=SUITE_COLOR[suite_of[int(src_task[s, d])]], alpha=0.10, linewidth=0)
        ax.plot(x, seq[:, d], color=MUTED, linewidth=1.6, label="demo" if d == 0 else None)
        ax.plot(x, hat[:, d], color="#2a78d6", linewidth=1.0, label="borrowed" if d == 0 else None)
        ax.set_ylabel(DIMS[d], rotation=0, ha="right", va="center", fontsize=7, color=INK)
        ax.set_xlim(-0.5, len(seq) - 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if d == 0:
            leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7)
            [t.set_color(INK) for t in leg.get_texts()]
    ax = fig.add_subplot(gs[7])
    ax.set_facecolor(SURFACE)
    for s in range(nseg):
        for d in range(7):
            tk = int(src_task[s, d])
            ax.add_patch(plt.Rectangle((s - 0.46, d - 0.44), 0.92, 0.88, color=SUITE_COLOR[suite_of[tk]], linewidth=0))
            ax.text(s, d, str(tk), ha="center", va="center", fontsize=5.2, color="white")
    ax.set_xlim(-0.6, nseg - 0.4)
    ax.set_ylim(6.6, -0.6)
    ax.set_yticks(range(7))
    ax.set_yticklabels(DIMS, fontsize=6.5, color=INK)
    ax.set_xticks(range(nseg))
    ax.set_xticklabels([str(s * 10) for s in range(nseg)], fontsize=5.5, color=MUTED)
    ax.set_xlabel("demo frame at the segment's start; cell = the training task whose motion lends that segment of that dimension",
                  fontsize=7.5, color=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SUITE_COLOR.values()]
    leg = ax.legend(handles, [f"{SUITE_NAME[s]} (tasks {D.SUITES[s][0]}-{D.SUITES[s][-1]})" for s in SUITE_COLOR], loc="upper left",
                    bbox_to_anchor=(0, -0.28), ncol=4, frameon=False, fontsize=7)
    [t.set_color(INK) for t in leg.get_texts()]
    vals, cnt = np.unique(src_task, return_counts=True)
    top = vals[np.argsort(-cnt)][:3]
    rms = float(np.sqrt(((segcode.windows(hat, T) - rows) ** 2).mean()))
    fig.suptitle(f"\"{texts[a.task]}\" (held out) spelled from pieces of the prior's own motions: {nseg} segments x 7 dimensions, "
                 f"{DemoProgram.load(f'results/robot/r10_seg_{a.task}_s5.npz').bytes()} B with alignment keys every 5th frame, {rms:.3f} from the demo's chunks.\nMost-used sources: "
                 + "; ".join(f"{train_text[int(t)]} ({int(c)})" for t, c in zip(top, np.sort(cnt)[::-1][:3])),
                 fontsize=8, color=INK, x=0.02, ha="left", y=0.99)
    out = f"figures/robot/segments_{a.task}"
    fig.savefig(out + ".png", dpi=190, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    print(f"[fig] {out}.png")


if __name__ == "__main__":
    main()
