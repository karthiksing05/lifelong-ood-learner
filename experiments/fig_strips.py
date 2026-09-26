"""Rollout filmstrips from the client's saved episode videos (the first two final episodes of each task and arm).

Rows = arms on the SAME init state, columns = evenly spaced frames of each episode (an episode ends at success or at the
step limit).  Runs with the openpi venv's python (PyAV + matplotlib; no torch):

    /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/fig_strips.py --task 39 --trial 30
"""
import argparse
import glob
import os

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GOOD, CRIT = "#0ca30c", "#d03b3b"
ARMS = [("zero-shot", "m3_final_{t}_zeroshot"), ("few-shot LoRA (18.7 MB)", "final_{t}_lora"), ("few-shot LoRA (18.7 MB)", "m3_final_{t}_lora"),
        ("demo replay", "m3_final_{t}_raw_replay"), ("borrowing, every 5th frame", "m3_final_{t}_dimcode_s5"),
        ("borrowing + 258 KB fold", "m3_final_{t}_dimfold_s1_last2")]


def frames(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--trial", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    rows, seen = [], set()
    for label, d in ARMS:
        hits = glob.glob(os.path.join(R, d.format(t=a.task), f"task{a.task}_trial{a.trial}_*.mp4"))
        if not hits or label in seen:
            continue
        seen.add(label)
        fr = frames(hits[0])
        ok = hits[0].endswith("_success.mp4")
        rows.append((label, ok, fr))
    if not rows:
        raise SystemExit(f"no videos for task {a.task} trial {a.trial}")
    n, m = len(rows), a.cols
    fig, axs = plt.subplots(n, m, figsize=(1.35 * m + 1.9, 1.38 * n + 0.5), facecolor=SURFACE, squeeze=False)
    for i, (label, ok, fr) in enumerate(rows):
        idx = np.linspace(0, len(fr) - 1, m).round().astype(int)
        for j, k in enumerate(idx):
            ax = axs[i, j]
            ax.imshow(fr[k])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == n - 1:
                ax.set_xlabel(f"{k / max(len(fr) - 1, 1):.0%} of the episode", fontsize=6.3, color=MUTED)
        axs[i, 0].set_ylabel(f"{label}\n{'success' if ok else 'failure'}, {len(fr)} steps", fontsize=7, color=INK, rotation=0,
                             ha="right", va="center", labelpad=8)
        # outcome marker: a status dot with its word (never color alone)
        axs[i, m - 1].text(1.04, 0.5, "✓" if ok else "✗", transform=axs[i, m - 1].transAxes, fontsize=11,
                           color=GOOD if ok else CRIT, va="center", ha="left")
    t = a.title or f"LIBERO-90 task {a.task}, init state {a.trial}: the same start under each method"
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(wspace=0.04, hspace=0.08, top=1 - 0.28 / h)
    fig.suptitle(t, fontsize=8.5, color=INK, x=0.02, ha="left", y=1 - 0.06 / h)
    out = os.path.join(ROOT, "figures", "robot", f"strip_{a.task}_t{a.trial}")
    fig.savefig(out + ".png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out + ".pdf", facecolor=SURFACE, bbox_inches="tight")
    print(f"[strips] {out}.png ({n} arms)")


if __name__ == "__main__":
    main()
