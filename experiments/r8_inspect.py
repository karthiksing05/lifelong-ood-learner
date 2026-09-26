"""Prototype inspection: where the motions a held-out task borrows come from.

Every (frame, action dimension) of a held-out LIBERO-90 task's dimension-wise program (results/robot/r7_dim_<t>_s1.npz)
is borrowed from ONE library prototype; a prototype's provenance is the training tasks of its member frames (the r2b
library's owner arrays at the prototype's level).  Per held-out task this gives the share of its borrowed pairs that
comes from each of the 40 training tasks.  Reported: the top sources, the effective number of sources (exp entropy),
the level mix, and the WORD OVERLAP between the sources' instructions and the task's own, against a permutation null
that pairs each program with ANOTHER held-out task's instruction (does a task borrow from related tasks more than an
unrelated task does?).

    python experiments/r8_inspect.py [--record]
Writes results/robot/r8_provenance.json and figures/robot/r8_provenance.{png,pdf}.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.tree.nodes import Tree                                # noqa: E402

# instruction words that carry no task identity (almost every LIBERO instruction is a pick-and-place)
STOP = {"the", "a", "an", "of", "on", "in", "to", "and", "it", "its", "put", "pick", "up", "place", "them", "at", "into", "inside",
        "from", "is", "both"}
SUITE_OF = {t: s for s, ts in D.SUITES.items() if s != "libero_40" for t in ts}


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def overlap(held, src):
    h = words(held)
    return len(h & words(src)) / max(len(h), 1)


def provenance(tree, z, levels=(2, 3, 4)):
    """node id -> (level, share of its member frames per training task [40], number of member frames)."""
    task = z["task"]
    out = {}
    for l in levels:
        u = tree.u_grid[l]
        owner = z[f"owner_policy_{u}"]
        ks = sorted(set(owner.tolist()))
        ids = tree.levels[l]
        assert len(ks) == len(ids), (l, len(ks), len(ids))
        for k, nid in zip(ks, ids):
            m = task[owner == k]
            out[int(nid)] = (l, np.bincount(m, minlength=40) / len(m), len(m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="results/robot/r2b_libero_40_tree.npz")
    ap.add_argument("--npz", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    tree, z = Tree.load(a.tree), np.load(a.npz)
    prov = provenance(tree, z)
    train_text = D.Demos().tasks
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    held = sorted(int(re.findall(r"r7_dim_(\d+)_s1", p)[0]) for p in glob.glob("results/robot/r7_dim_*_s1.npz"))
    from lifelong_ood.policy.compose import DemoProgram
    mass, lvl, res, lenders = {}, {}, {}, {}
    for t in held:
        p = DemoProgram.load(f"results/robot/r7_dim_{t}_s1.npz")
        ids, ch = p.dim_ids, p.dim_choice                     # [T, K], [T, 7]
        m, lv, members, ntask = np.zeros(40), np.zeros(len(tree.u_grid)), [], []
        for f in range(len(ids)):
            for d in range(ch.shape[1]):
                l, share, nm = prov[int(ids[f, ch[f, d]])]
                m += share
                lv[l] += 1
                members.append(nm)
                ntask.append(int((share > 0).sum()))
        mass[t], lvl[t] = m / m.sum(), lv / lv.sum()
        members, ntask = np.array(members), np.array(ntask)
        lenders[t] = dict(median_members=float(np.median(members)), le3_members=float(np.mean(members <= 3)),
                          single_task=float(np.mean(ntask == 1)), ge3_tasks=float(np.mean(ntask >= 3)),
                          distinct=int(len(set((int(ids[f, ch[f, d]]) for f in range(len(ids)) for d in range(ch.shape[1]))))))
    ov = np.array([[overlap(texts[t], train_text[s]) for s in range(40)] for t in held])       # [held, 40]
    M = np.stack([mass[t] for t in held])
    obs = float(np.mean((M * ov).sum(1)))
    rng = np.random.default_rng(0)
    null = np.array([np.mean((M[rng.permutation(len(held))] * ov).sum(1)) for _ in range(a.perms)])
    pval = float((1 + (null >= obs).sum()) / (1 + a.perms))
    base = np.bincount(z["task"], minlength=40) / len(z["task"])                              # library frame shares
    for i, t in enumerate(held):
        order = np.argsort(-M[i])
        ent = -(M[i][M[i] > 0] * np.log(M[i][M[i] > 0])).sum()
        res[t] = dict(text=texts[t], top=[dict(task=int(s), text=train_text[int(s)], suite=SUITE_OF.get(int(s), "?"), share=float(M[i][s]),
                                             overlap=float(ov[i][s])) for s in order[:5]],
                      effective_sources=float(np.exp(ent)), overlap_score=float((M[i] * ov[i]).sum()),
                      overlap_base=float((base * ov[i]).sum()), level_mix={str(tree.u_grid[l]): float(lvl[t][l]) for l in (2, 3, 4)},
                      lenders=lenders[t])
        print(f"[r8] {t:3d} '{texts[t][:44]}': {np.exp(ent):.1f} effective sources; top: "
              + "; ".join(f"{train_text[int(s)][:34]} {M[i][s]:.0%}" for s in order[:3])
              + f" | word overlap {res[t]['overlap_score']:.2f} (library base {res[t]['overlap_base']:.2f})", flush=True)
    print(f"[r8] mean word overlap of the borrowed sources {obs:.3f} vs {null.mean():.3f} +- {null.std():.3f} when programs are "
          f"paired with another held-out task's instruction (permutation p = {pval:.4f}, {a.perms} permutations)")
    agg = {k: float(np.mean([lenders[t][k] for t in held])) for k in ("median_members", "le3_members", "single_task", "ge3_tasks")}
    agg.update({f"level_{tree.u_grid[l]}": float(np.mean([lvl[t][l] for t in held])) for l in (2, 3, 4)})
    print(f"[r8] lenders over the {len(held)} programs: {agg['level_0.2']:.2f} / {agg['level_0.3']:.2f} / {agg['level_0.4']:.2f} of pairs from "
          f"t = 0.2 / 0.3 / 0.4; {agg['le3_members']:.2f} from prototypes with <= 3 member frames, {agg['single_task']:.2f} from single-task "
          f"prototypes, {agg['ge3_tasks']:.2f} from prototypes >= 3 tasks share")
    out = dict(tasks={str(t): res[t] for t in held}, lenders=agg, overlap_mean=obs, null_mean=float(null.mean()), null_std=float(null.std()),
               perm_p=pval, n_perms=a.perms, held=held, matrix=M.tolist(), overlap=ov.tolist(), base=base.tolist(),
               train_text={str(k): v for k, v in train_text.items()})
    json.dump(out, open("results/robot/r8_provenance.json", "w"), indent=1)
    figure(out, texts)
    if a.record:
        m = {k: v for k, v in out.items() if k not in ("matrix", "overlap", "train_text", "base")}
        ledger.record("r8_inspect", bench="libero_90", tag=f"provenance_{len(held)}tasks", seed=0, metrics=m,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/r8_inspect.py", command="python experiments/r8_inspect.py --record",
                      artifacts=["results/robot/r8_provenance.json", "figures/robot/r8_provenance.png"])
        print("[r8] ledger row written")


def figure(out, texts):
    """LIFT = a training task's share of the held-out task's borrowed pairs / its share of the library's frames (LIBERO-10's
    long demos own the most frames, so raw shares mostly show demo length).  Diverging blue (borrowed more than its frames)
    - gray (as expected) - red (less); ring = the largest lift; rows carry the effective number of sources."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    div = LinearSegmentedColormap.from_list("div", ["#b3261e", "#e34948", "#f3b5b0", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
    held, M, base = out["held"], np.array(out["matrix"]), np.array(out["base"])
    cols = [t for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for t in D.SUITES[s]]
    lift = np.log2(np.maximum(M, 1e-6) / base[None])
    fig, ax = plt.subplots(figsize=(10.5, 0.36 * len(held) + 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.clip(lift[:, cols], -2, 2), cmap=div, vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
    for i in range(len(held)):
        j = int(np.argmax(lift[i, cols]))
        ax.scatter([j], [i], s=46, facecolors="none", edgecolors=INK, linewidths=0.9)
    ax.set_yticks(range(len(held)))
    lab = []
    for t in held:
        x = texts[t] if len(texts[t]) < 48 else texts[t][:46] + "..."
        lab.append(f"{x}  ({out['tasks'][str(t)]['effective_sources']:.0f} sources)")
    ax.set_yticklabels(lab, fontsize=7, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols], fontsize=5.5, color=MUTED)
    for k, name in enumerate(("Spatial", "Object", "Goal", "LIBERO-10")):
        ax.text(10 * k + 4.5, len(held) - 0.5 + 0.75, f"{name} (training tasks {D.SUITES[('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')[k]][0]}-{D.SUITES[('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')[k]][-1]})", ha="center", va="top", fontsize=7.2, color=INK2)
        if k:
            ax.axvline(10 * k - 0.5, color=SURFACE, linewidth=2.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["1/4", "1/2", "1", "2", "4"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, colors=INK2, length=2)
    cb.set_label("borrowed share / share of the library's frames", fontsize=7, color=INK2)
    ax.set_title(f"Where each held-out task's borrowed motions come from (ring = largest lift).  Instruction-word overlap with "
                 f"the sources {out['overlap_mean']:.3f} vs {out['null_mean']:.3f} for shuffled pairings (p = {out['perm_p']:.3f})",
                 fontsize=8.2, color=INK, loc="left", pad=8)
    fig.savefig("figures/robot/r8_provenance.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig("figures/robot/r8_provenance.pdf", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("[r8] figures/robot/r8_provenance.png/.pdf")


if __name__ == "__main__":
    main()
