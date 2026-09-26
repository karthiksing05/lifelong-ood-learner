"""The prototype ATLAS: what pi0.5's agent prototypes are, in physical terms.

    python experiments/r2c_atlas.py [--tree results/robot/r2b_libero_40_tree.npz --npz results/robot/r2b_libero_40.npz]

Every library prototype (a mode of the policy's action marginal at noise level t; r2b_marginal) is a 10-step chunk of
7-D delta commands in the robot base frame (x forward, y left, z up; signs verified against the demos' hand positions,
corr 0.92-0.98).  A unit command moves the hand ~1.1 cm and turns it ~0.1 rad per step (measured; the OSC_POSE targets
are 5 cm / 0.5 rad but the impedance controller lags them); the gripper command is -1 open / +1 closed.  A chunk
moving less than 2 cm is "hold".  Each prototype
is described by its hand displacement over the chunk, its rotation, and its gripper behaviour, and named from those:
e.g. "down + open", "lift + closed", "closing", "twist wrist + closed", "hold".  Per level: the vocabulary, how many
tasks and frames use each word, examples as end-effector paths, and the coarse-to-fine tree of the top levels.
Writes docs/PROTOTYPES.md, figures/robot/atlas_*.png and a ledger row r2c_atlas (the numbers the doc quotes).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402

# Measured on the demos (state change per unit command per step): the impedance controller lags its 0.05 m / 0.5 rad
# targets, and the hand actually moves ~0.011 m (x 0.0103, y 0.0124, z 0.0114; corr 0.92-0.98) and turns ~0.10 rad
# (0.119, 0.086, 0.106; corr 0.78-0.98) per unit per step.
POS, ROT = 0.011, 0.10
AX = [("forward", "back"), ("left", "right"), ("up", "down")]


def describe(raw):
    """raw [10, 7] actions -> dict of physical descriptors and a name."""
    d = raw[:, :3].sum(0) * POS                      # displacement over the chunk (m)
    r = raw[:, 3:6].sum(0) * ROT                     # rotation (rad, axis-angle increments summed)
    g = raw[:, 6]
    g0, g1, gm = float(g[:3].mean()), float(g[-3:].mean()), float(g.mean())
    if g0 < -0.3 and g1 > 0.3:
        grip = "closing"
    elif g0 > 0.3 and g1 < -0.3:
        grip = "opening"
    else:
        grip = "closed" if gm > 0 else "open"
    dist, rot = float(np.linalg.norm(d)), float(np.linalg.norm(r))
    if grip in ("closing", "opening") and dist < 0.02:
        motion = ""                                  # the gripper event is the motion
    elif rot > 0.15 and rot / 0.15 > dist / 0.02:
        motion = "twist wrist"
    elif dist < 0.02:
        motion = "hold"
    else:
        order = np.argsort(-np.abs(d))
        words = []
        for i in order[:2]:
            if abs(d[i]) >= 0.4 * abs(d[order[0]]) and abs(d[i]) > 0.01:
                words.append(AX[i][0] if d[i] > 0 else AX[i][1])
        motion = "-".join(words) if words else "hold"
        if words == ["up"] or words[:1] == ["up"]:
            motion = motion.replace("up", "lift", 1)
    name = f"{motion} + {grip}" if motion else grip
    return dict(dx=float(d[0]), dy=float(d[1]), dz=float(d[2]), dist=dist, rot=rot, grip=grip, g0=g0, g1=g1, name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="results/robot/r2b_libero_40_tree.npz")
    ap.add_argument("--npz", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--levels", default="2,3,4,5,6")
    a = ap.parse_args()
    tree = Tree.load(a.tree)
    z = np.load(a.npz)
    demos = D.Demos()
    norm = D.Norm.load()
    task = z["task"]
    levels = [int(x) for x in a.levels.split(",")]
    out, per_level = {}, {}
    for l in levels:
        u = tree.u_grid[l]
        owner = z[f"owner_policy_{u}"]
        ks = sorted(set(owner.tolist()))
        ids = tree.levels[l]
        assert len(ks) == len(ids), (l, len(ks), len(ids))
        node_of = dict(zip(ks, ids))
        rows = []
        mu_all = z["mu"]
        member_names = [describe(D.from_rows(mu_all[i][None], norm, 10)[0])["name"] for i in range(len(mu_all))]
        member_grip = D.from_rows(mu_all, norm, 10)[:, :, 6].mean(1)
        for k in ks:
            nid = node_of[k]
            members = np.where(owner == k)[0]
            # a coarse mode is the centre of a blurred mixture and shrinks toward "hold": name the prototype by its MEDOID
            # (the member chunk nearest the mode), and report how many members carry the same word
            mode = tree.nodes[nid].mode
            med = members[np.argmin(((mu_all[members] - mode[None]) ** 2).sum(1))]
            raw = D.from_rows(mu_all[med][None], norm, 10)[0]
            desc = describe(raw)
            mode_desc = describe(D.from_rows(mode[None], norm, 10)[0])
            purity = float(np.mean([member_names[i] == desc["name"] for i in members]))
            closed = member_grip[members] > 0
            grip_purity = float(max(closed.mean(), 1 - closed.mean()))
            desc.update(node=int(nid), frames=int(len(members)), tasks=sorted(set(task[members].tolist())), raw=raw.tolist(),
                        mode_name=mode_desc["name"], purity=purity, grip_purity=grip_purity)
            rows.append(desc)
        per_level[l] = rows
        vocab = collections.defaultdict(lambda: dict(prototypes=0, frames=0, tasks=set()))
        for r in rows:
            v = vocab[r["name"]]
            v["prototypes"] += 1; v["frames"] += r["frames"]; v["tasks"] |= set(r["tasks"])
        nfr = sum(r["frames"] for r in rows)
        voc = sorted(({"name": k, "prototypes": v["prototypes"], "frame_share": v["frames"] / nfr, "tasks": len(v["tasks"])}
                      for k, v in vocab.items()), key=lambda x: -x["frame_share"])
        out[str(u)] = dict(n_prototypes=len(rows), n_words=len(voc), vocabulary=voc,
                           word_purity=float(np.average([r["purity"] for r in rows], weights=[r["frames"] for r in rows])),
                           gripper_purity=float(np.average([r["grip_purity"] for r in rows], weights=[r["frames"] for r in rows])),
                           median_dist_cm=float(np.median([r["dist"] for r in rows]) * 100),
                           frac_gripper_event=float(np.mean([r["grip"] in ("closing", "opening") for r in rows])))
        print(f"[atlas] t={u}: {len(rows)} prototypes, {len(voc)} words, members sharing their prototype's word "
              f"{out[str(u)]['word_purity']:.2f}, sharing its gripper state {out[str(u)]['gripper_purity']:.2f}; top: "
              + "; ".join(f"{v['name']} ({v['frame_share']:.0%} of frames, {v['tasks']} tasks)" for v in voc[:8]), flush=True)
    # coarse-to-fine: the top levels and their children
    tree_lines = []
    top = levels[-1]
    def kids(nid, l):
        return [m for m in tree.levels[l] if tree.nodes[m].parent == nid]
    from lifelong_ood.tree.build import link_parents
    link_parents(tree)
    name_of = {r["node"]: r for l in levels for r in per_level[l]}
    for n6 in tree.levels[top]:
        r = name_of[n6]
        tree_lines.append(f"- t={tree.u_grid[top]}: **{r['name']}** ({r['frames']} frames, {len(r['tasks'])} tasks)")
        for n5 in kids(n6, top - 1):
            r5 = name_of.get(n5)
            if r5 is None:
                continue
            tree_lines.append(f"  - t={tree.u_grid[top - 1]}: **{r5['name']}** ({r5['frames']} frames, {len(r5['tasks'])} tasks)")
            ch = sorted((name_of[m] for m in kids(n5, top - 2) if m in name_of), key=lambda x: -x["frames"])
            words = collections.Counter()
            for c in ch:
                words[c["name"]] += c["frames"]
            if ch:
                tree_lines.append(f"    - t={tree.u_grid[top - 2]}: {len(ch)} children; by frames: "
                                  + ", ".join(f"{w} ({n})" for w, n in words.most_common(8)))
    print("\n".join(tree_lines), flush=True)
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures/robot", exist_ok=True)
        l_mid = levels[0]                                          # the finest level: coherent motion primitives
        rows = sorted(per_level[l_mid], key=lambda r: -r["frames"])[:24]
        fig, axs = plt.subplots(4, 6, figsize=(18, 11), subplot_kw=dict(projection="3d"))
        for ax, r in zip(axs.ravel(), rows):
            raw = np.asarray(r["raw"])
            p = np.vstack([np.zeros(3), np.cumsum(raw[:, :3] * POS, 0)]) * 100
            col = ["tab:red" if g > 0 else "tab:green" for g in raw[:, 6]]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color="0.5", lw=1)
            ax.scatter(p[1:, 0], p[1:, 1], p[1:, 2], c=col, s=12)
            ax.scatter([0], [0], [0], c="k", s=10, marker="x")
            ax.set_title(f"{r['name']}\n{r['frames']} frames, {len(r['tasks'])} tasks", fontsize=8)
            ax.tick_params(labelsize=5)
            ax.set_xlabel("fwd cm", fontsize=6); ax.set_ylabel("left cm", fontsize=6); ax.set_zlabel("up cm", fontsize=6)
        fig.suptitle(f"The 24 most used agent prototypes at t={tree.u_grid[l_mid]} (hand path over 10 steps; green = gripper open, red = closed)")
        fig.tight_layout(); fig.savefig("figures/robot/atlas_prototypes.png", dpi=110); plt.close(fig)
        # vocabulary share per level
        fig, ax = plt.subplots(figsize=(10, 4.5))
        allw = collections.Counter()
        for l in levels:
            for v in out[str(tree.u_grid[l])]["vocabulary"]:
                allw[v["name"]] += v["frame_share"]
        words = [w for w, _ in allw.most_common(14)]
        bottom = np.zeros(len(levels))
        cmap = plt.get_cmap("tab20")
        for i, w in enumerate(words + ["other"]):
            vals = []
            for l in levels:
                voc = {v["name"]: v["frame_share"] for v in out[str(tree.u_grid[l])]["vocabulary"]}
                vals.append(voc.get(w, 0.0) if w != "other" else 1 - sum(voc.get(x, 0.0) for x in words))
            ax.bar([str(tree.u_grid[l]) for l in levels], vals, bottom=bottom, color=cmap(i % 20), label=w)
            bottom += np.asarray(vals)
        ax.set_xlabel("noise level t (coarser to the right)"); ax.set_ylabel("share of frames")
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title("What the prototypes are, by level: share of demo frames per prototype word")
        fig.tight_layout(); fig.savefig("figures/robot/atlas_vocabulary.png", dpi=110); plt.close(fig)
        # task x word incidence at the middle level
        rows = per_level[l_mid]
        wl = [v["name"] for v in out[str(tree.u_grid[l_mid])]["vocabulary"][:20]]
        tl = sorted(set(task.tolist()))
        Mx = np.zeros((len(wl), len(tl)))
        for r in rows:
            if r["name"] in wl:
                members = z[f"owner_policy_{tree.u_grid[l_mid]}"] == [k for k, n in node_of.items() if n == r["node"]][0] if False else None
        owner = z[f"owner_policy_{tree.u_grid[l_mid]}"]
        ks = sorted(set(owner.tolist())); node_of = dict(zip(ks, tree.levels[l_mid]))
        name_by_k = {k: name_of[node_of[k]]["name"] for k in ks}
        for i, k in enumerate(owner):
            w = name_by_k[k]
            if w in wl:
                Mx[wl.index(w), tl.index(task[i])] += 1
        Mx = Mx / np.maximum(Mx.sum(0, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(Mx, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(wl))); ax.set_yticklabels(wl, fontsize=7)
        ax.set_xticks(range(len(tl))); ax.set_xticklabels([demos.tasks[t][:28] for t in tl], rotation=90, fontsize=6)
        ax.set_title(f"Each trained task's frames by prototype word at t={tree.u_grid[l_mid]} (column-normalised)")
        fig.tight_layout(); fig.savefig("figures/robot/atlas_incidence.png", dpi=110); plt.close(fig)
    except Exception as ex:   # noqa: BLE001
        print(f"[atlas] figures skipped: {ex}", flush=True)
    m = {"levels": out, "tree": tree_lines}
    ledger.record("r2c_atlas", bench="libero_40", tag="", seed=0, metrics=m,
                  bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                  config=vars(a), script="experiments/r2c_atlas.py", command=" ".join(sys.argv))
    json.dump({str(l): per_level[l] for l in levels}, open("results/robot/atlas.json", "w"))


if __name__ == "__main__":
    main()
