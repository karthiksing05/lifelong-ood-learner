"""P1 of the prototype-PoE plan: does the frozen policy hold several OPTIONS at one observation, and do they point at objects?

On first frames with oracle segmentation and object positions (experiments/f0b_firstframes_seg.py; practice states 0-4), groups:
validity = LIBERO-Goal (the policy reads its words), class_a = LIBERO-90 tasks where only the object differs from a performed task.
Generators of options at the observation:
  (a) null  the policy's own score under the empty prompt, ascended (Wang Eq. 6, Adam) from 64 random starts at t in
            {0.4, 0.55, 0.7}; the clean modes are clustered (rms < 0.15): how many clusters hold >= 10% of the starts, and how
            many distinct objects do they point at?  (control: the same under the task's own instruction, expected unimodal)
  (d) grid  the action expert's attention raised to mass 0.7 on one of 4 x 4 blocks of agent-view tokens (no oracle): how many
            distinct objects do the 16 options point at, and does a block covering an object point at it?
(Generators (b) prompts and (c) oracle object focus were read by experiments/b2_steer_offline.py: class a contrast 0.31 / 0.62.)
Pointing as in experiments/f1_factor_probe.py: the first chunk's horizontal displacement, argmax cosine over hand->object vectors.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q1_options.py --record" slurm/jobs/robot_gpu.sbatch
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import attn_bias as AB        # noqa: E402
from lifelong_ood.policy import factors as F           # noqa: E402
from lifelong_ood.policy import where as W             # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FRAMES = os.path.join(R, "f0b_firstframes")
CLASS_A = (18, 26, 48, 49, 52, 53, 58, 59, 60, 62, 65, 66, 71, 38, 70)
LEVELS = (0.4, 0.55, 0.7)
STATES = 5


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def pointed(d, P):
    return int(np.argmax(unit(P) @ unit(d)))


def clusters(X, tol=0.15):
    """Greedy rms clustering of clean chunks [n, 70] -> list of (centre, count)."""
    out = []
    for x in X:
        for c in out:
            if np.sqrt(((c[0] - x) ** 2).mean()) < tol:
                c[1] += 1
                break
        else:
            out.append([x.copy(), 1])
    return sorted(out, key=lambda c: -c[1])


def main():
    import torch
    from lifelong_ood.backbones import score_ops as SO
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    keys = [f"libero_goal_{i}" for i in range(10)] + [f"libero_90_{i}" for i in CLASS_A]
    per = {}
    for key in keys:
        f = os.path.join(FRAMES, key + ".npz")
        if not os.path.exists(f):
            continue
        suite, tid = key.rsplit("_", 1)
        t = by[(suite, int(tid))]
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        objs = [n for n in names if n in t.objects]
        recs = []
        for k in range(min(STATES, len(z["states"]))):
            eef = np.asarray(z["state8"][k][:3], np.float64)
            Pv = np.stack([np.asarray(z["pos"][k][names.index(n)], np.float64)[:2] - eef[:2] for n in objs])
            seg = z["seg"][k]
            omap = {n: W.oracle_map(seg, W.geoms_of(z, name=n)) for n in objs}
            base = {"observation/image": z["img"][k], "observation/wrist_image": z["wimg"][k],
                    "observation/state": np.asarray(z["state8"][k], np.float32)}
            rec = {"state": int(z["states"][k])}
            for gen, prompt in (("null", ""), ("own", t.language)):
                pre = bk.prefix(bk.observation(dict(base, prompt=prompt)))
                bk.bind(pre)
                lv = {}
                for u in LEVELS:
                    _, x0 = SO.mode_ascent(bk, None, u, iters=100, lr=0.05, seed=k, init="noise", n_starts=64, batch=64)
                    cl = clusters(x0)
                    big = [c for c in cl if c[1] >= 0.1 * len(x0)]
                    acts = [bk.unnormalize(c[0][None])[0] for c in big]
                    tgt = sorted({objs[pointed(0.05 * a[:, :3].sum(0)[:2], Pv)] for a in acts})
                    lv[str(u)] = dict(n_clusters=len(cl), n_major=len(big), major_mass=[int(c[1]) for c in big], objects=tgt)
                rec[gen] = lv
            # (d) grid-cell options under the task's own instruction
            pre = bk.prefix(bk.observation(dict(base, prompt=t.language)))
            g = np.random.default_rng(k)
            noise = g.standard_normal(bk.dim_full).astype(np.float32)[None]
            cells = []
            for bi in range(4):
                for bj in range(4):
                    m = np.zeros((16, 16), np.float32)
                    m[4 * bi:4 * bi + 4, 4 * bj:4 * bj + 4] = 1.0
                    rows = bk.sample(1, pre, noise=noise, step_gate=AB.bias_gate(bk, {"agent": m}, pre, mode="mass", rho=0.7))
                    a = bk.unnormalize(rows)[0]
                    p = pointed(0.05 * a[:, :3].sum(0)[:2], Pv)
                    cov = {n: float(omap[n][4 * bi:4 * bi + 4, 4 * bj:4 * bj + 4].mean()) for n in objs}
                    owner = max(cov, key=cov.get) if max(cov.values()) > 0.02 else None
                    cells.append(dict(cell=(bi, bj), points=objs[p], owner=owner))
            own = [c for c in cells if c["owner"] is not None]
            rec["grid"] = dict(objects=sorted({c["points"] for c in cells}), n_objects=len({c["points"] for c in cells}),
                               owner_hit=float(np.mean([c["points"] == c["owner"] for c in own])) if own else None,
                               owner_chance=float(np.mean([1.0 / len(objs) for _ in own])) if own else None)
            recs.append(rec)
        per[key] = dict(group="validity" if suite == "libero_goal" else "class_a", objects=objs, states=recs)
        last = recs[-1]
        print(f"[q1] {key:14s} null majors {[last['null'][str(u)]['n_major'] for u in LEVELS]} own majors "
              f"{[last['own'][str(u)]['n_major'] for u in LEVELS]} | grid objects {last['grid']['n_objects']}/{len(objs)} "
              f"owner hit {last['grid']['owner_hit']}", flush=True)
    groups = {}
    for gname in ("validity", "class_a"):
        vs = [s for v in per.values() if v["group"] == gname for s in v["states"]]
        if not vs:
            continue
        groups[gname] = dict(
            frames=len(vs),
            null_multi_share={str(u): float(np.mean([s["null"][str(u)]["n_major"] >= 2 for s in vs])) for u in LEVELS},
            own_multi_share={str(u): float(np.mean([s["own"][str(u)]["n_major"] >= 2 for s in vs])) for u in LEVELS},
            null_objects_mean={str(u): float(np.mean([len(s["null"][str(u)]["objects"]) for s in vs])) for u in LEVELS},
            grid_objects_mean=float(np.mean([s["grid"]["n_objects"] for s in vs])),
            grid_owner_hit=float(np.nanmean([s["grid"]["owner_hit"] for s in vs if s["grid"]["owner_hit"] is not None])),
            grid_owner_chance=float(np.nanmean([s["grid"]["owner_chance"] for s in vs if s["grid"]["owner_chance"] is not None])))
        print(f"[q1] GROUP {gname}: {groups[gname]}", flush=True)
    res = dict(levels=list(LEVELS), groups=groups, per_task=per)
    json.dump(res, open(os.path.join(R, "q1_options.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("q1_options", bench="libero", tag="read", seed=0, metrics=dict(groups=groups, levels=list(LEVELS)),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/q1_options.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "q1_options.json")]))


if __name__ == "__main__":
    main()
