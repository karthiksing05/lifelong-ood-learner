"""P0b of the overhaul (pre-registered: ledger m3_prereg tag zc_probe): does a prompt that names an object move pi0.5's first
chunk toward that object, in a new scene?  Offline: no rollout, only the policy's first chunk at dumped first observations.

For each first observation (experiments/f0_firstframes.py; practice states 0-9) and each candidate object X of the scene
(movable types, the task's own targets excluded): the task's instruction with its object phrase replaced by X's
(lifelong_ood/policy/factors.substitute) and the lowest-index training instruction that moves X, all sampled from the same
noise (paired).  The first chunk's commanded hand displacement d = 0.05 m x sum of its 10 actions' xyz is compared, in the
table plane, with the direction from the hand to every candidate:
  PA    pointed-object accuracy (chance 1/m);  PA_w for the factor arithmetic d_ref + w (d(X) - d_ref), d_ref = the mean over
        the candidates' prompts, w in {1, 2, 4, 8};
  CC    contrast cosine cos(d(X) - d(X'), p_X - p_X');
  LS    language sensitivity: spread across prompts / spread across two noise seeds.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/f1_factor_probe.py" slurm/jobs/robot_gpu.sbatch   [--record]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import factors as F                                   # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FRAMES = os.path.join(R, "f0_firstframes")
SEEDS = (0, 1)
GAINS = (1.0, 2.0, 4.0, 8.0)


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def pointed(d, P):
    """d [2] displacement, P [m, 2] hand->object vectors -> index of the candidate the displacement points at."""
    return int(np.argmax(unit(P) @ unit(d)))


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    train = [t for t in tasks if t.suite != "libero_90"]
    phrases = F.object_phrases(tasks)
    cov = json.load(open(os.path.join(R, "f0_coverage.json")))
    zs = {int(k): v["zeroshot"] for k, v in cov["tasks"].items()}
    held = {int(k): v["group"] for k, v in cov["held_out"].items()}
    per_task = {}
    only = sys.argv[sys.argv.index("--only") + 1].split("+") if "--only" in sys.argv else None   # e.g. libero_goal_1+libero_90_48
    for f in sorted(glob.glob(os.path.join(FRAMES, "*.npz"))):
        if only and os.path.basename(f)[:-4] not in only:
            continue
        suite, tid = os.path.basename(f)[:-4].rsplit("_", 1)
        if suite not in ("libero_90", "libero_goal", "libero_object", "libero_10"):
            continue
        t = by[(suite, int(tid))]
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        types = {n: t.objects.get(n) for n in names if n in t.objects}
        cands = [c for c in F.candidates(t) if c in set(types.values())]
        if len(set(t.moved)) != 1 or len(cands) < 2:
            continue
        prompts = []
        ph_t = F.scene_phrases(t, tasks)
        for c in cands:
            p = F.substitute(t, c, ph_t)
            if p is None:
                break
            prompts.append(p)
        if len(prompts) != len(cands):
            continue
        matched = [next((x.language for x in train if c in x.moved), None) for c in cands]
        own = cands.index(t.moved[0]) if t.moved[0] in cands else None
        recs = []
        for k in range(len(z["states"])):
            eef = np.asarray(z["state8"][k][:3], np.float64)
            pos = {}
            for n, ty in types.items():
                pos.setdefault(ty, np.asarray(z["pos"][k][names.index(n)], np.float64))
            Pv = np.stack([pos[c][:2] - eef[:2] for c in cands])
            ds = {}
            for s in SEEDS:
                base = {"observation/image": z["img"][k], "observation/wrist_image": z["wimg"][k],
                        "observation/state": np.asarray(z["state8"][k], np.float32)}
                allp = prompts + [m for m in matched if m]
                obs = bk.observations([dict(base, prompt=p) for p in allp])
                pre = bk.prefix(obs)
                g = np.random.default_rng(1000 * k + s)
                noise = np.tile(g.standard_normal(bk.dim_full).astype(np.float32), (len(allp), 1))
                rows = bk.sample(len(allp), pre, seed=s, noise=noise)
                acts = np.stack([bk.unnormalize(r[None])[0] for r in rows])          # [P, H, 7]
                ds[s] = 0.05 * acts[:, :, :3].sum(1)                               # [P, 3] commanded displacement (m)
            d = ds[SEEDS[0]][: len(cands)]
            dm = [None if m is None else ds[SEEDS[0]][len(cands) + [x for x in matched if x].index(m)] for m in matched]
            rec = dict(state=int(z["states"][k]),
                       pa=[int(pointed(d[i][:2], Pv) == i) for i in range(len(cands))],
                       pa_matched=[None if dm[i] is None else int(pointed(dm[i][:2], Pv) == i) for i in range(len(cands))],
                       ls=float(np.std(d[:, :2], axis=0).mean() / max(np.abs(ds[SEEDS[0]][: len(cands), :2] - ds[SEEDS[1]][: len(cands), :2]).mean(), 1e-9)),
                       cc=[float(unit(d[i][:2] - d[j][:2]) @ unit(Pv[i] - Pv[j])) for i in range(len(cands)) for j in range(len(cands)) if i < j])
            dref = d.mean(0)
            for w in GAINS:
                rec[f"pa_w{w:g}"] = [int(pointed((dref + w * (d[i] - dref))[:2], Pv) == i) for i in range(len(cands))]
            recs.append(rec)
        grp = ("validity" if suite == "libero_goal" else "object" if suite == "libero_object" else "train10" if suite == "libero_10"
               else ("held_" + held[int(tid)]) if int(tid) in held else ("solved" if (zs.get(int(tid)) or 0) >= 0.6 else "failed"))
        mean = lambda key: float(np.mean([x for r in recs for x in r[key] if x is not None])) if recs else float("nan")  # noqa: E731
        per_task[f"{suite}:{tid}"] = dict(group=grp, language=t.language, candidates=cands, own=own, m=len(cands),
                                          chance=1.0 / len(cands), PA=mean("pa"), PA_matched=mean("pa_matched"),
                                          PA_own=float(np.mean([r["pa"][own] for r in recs])) if own is not None else None,
                                          CC=mean("cc"), LS=float(np.mean([r["ls"] for r in recs])),
                                          **{f"PA_w{w:g}": mean(f"pa_w{w:g}") for w in GAINS})
        v = per_task[f"{suite}:{tid}"]
        print(f"[f1] {suite}:{tid} {grp:9s} m={v['m']} PA {v['PA']:.2f} (chance {v['chance']:.2f}) w8 {v['PA_w8']:.2f} "
              f"matched {v['PA_matched']:.2f} CC {v['CC']:+.2f} LS {v['LS']:.2f} | {t.language}", flush=True)
    groups = {}
    for g in sorted({v["group"] for v in per_task.values()}):
        vs = [v for v in per_task.values() if v["group"] == g]
        groups[g] = dict(tasks=len(vs), chance=float(np.mean([v["chance"] for v in vs])),
                         **{k: float(np.nanmean([v[k] for v in vs if v[k] is not None])) for k in
                            ("PA", "PA_matched", "CC", "LS") + tuple(f"PA_w{w:g}" for w in GAINS)})
        print(f"[f1] group {g:9s} n={groups[g]['tasks']:2d} PA {groups[g]['PA']:.2f} (chance {groups[g]['chance']:.2f}) "
              + " ".join(f"w{w:g} {groups[g][f'PA_w{w:g}']:.2f}" for w in GAINS)
              + f" matched {groups[g]['PA_matched']:.2f} CC {groups[g]['CC']:+.2f} LS {groups[g]['LS']:.2f}", flush=True)
    res = dict(parser_hash=F.parser_hash(), seeds=list(SEEDS), gains=list(GAINS), groups=groups, per_task=per_task)
    json.dump(res, open(os.path.join(R, "f1_factor_probe" + ("_smoke" if only else "") + ".json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("f1_factor_probe", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/f1_factor_probe.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "f1_factor_probe.json")]))


if __name__ == "__main__":
    main()
