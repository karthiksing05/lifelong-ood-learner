"""Q1b of the Bind plan: does biasing the action expert's attention toward an object's tokens turn the policy's FIRST chunk
toward that object?  Offline, oracle maps, practice states only.

For every task with one moved object and >= 2 visible candidates (first frames of experiments/f0b_firstframes_seg.py), at
every state and for every candidate c, under the task's OWN instruction:
  steer[cfg][c]  the first chunk with c's oracle map biased (configs: target mass rho in RHOS x layer bands, and fixed beta)
  swap[c]        the first chunk under the instruction rewritten to name c, no bias (the CALIBRATION: what reading the word
                 does where the policy reads it, e.g. LIBERO-Object)
  base           the first chunk under the own instruction, no bias
Pointing (as experiments/f1_factor_probe.py): the commanded horizontal displacement points at candidate i = argmax cosine
with the hand->object vectors.  PA = share of (state, c) whose chunk points at c; CC = cosine between the difference of two
candidates' chunks and the difference of their positions (does steering toward i rather than j move the hand from j to i?).
Override diagnostics per config: OS = chunk distance between consecutive states under the bias / without it (the chunk must
still follow the observation); LS = chunk distance between the own instruction and an unrelated training instruction under
the bias / without it (the chunk must still follow the prompt); SHIFT = does a copy of c's mask moved onto empty table pull
the chunk toward that table point (pointing among candidates + the shifted point).
Pre-registered kill (ledger m3_prereg tag b_bind_q1): if CC < 0.1 and PA gain over base < 0.1 for every config, BIND stops
before any rollout.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/b2_steer_offline.py [--only libero_90_48] [--record]" slurm/jobs/robot_gpu.sbatch
-> results/robot/b2_steer_offline.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import attn_bias as AB          # noqa: E402
from lifelong_ood.policy import factors as F             # noqa: E402
from lifelong_ood.policy import where as W               # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FRAMES = os.path.join(R, "f0b_firstframes")
CLASS_A = (18, 26, 48, 49, 52, 53, 58, 59, 60, 62, 65, 66, 71, 38, 70)
MIN_COVER = 0.1
SEEDS = (0, 1)
CONFIGS = {f"mass{r:g}_{band}": dict(mode="mass", rho=r, layers=L) for r in (0.3, 0.5, 0.7)
           for band, L in (("all", None), ("mid", tuple(range(6, 12))))}
CONFIGS.update({f"fixed{b:g}_all": dict(mode="fixed", beta=b, layers=None) for b in (4.0, 8.0)})
UNRELATED = "open the middle drawer of the cabinet"          # a LIBERO-Goal training instruction (prompt-sensitivity check)


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def pointed(d, P):
    return int(np.argmax(unit(P) @ unit(d)))


def group(suite, tid):
    return ("validity" if suite == "libero_goal" else "object" if suite == "libero_object"
            else "class_a" if suite == "libero_90" and tid in CLASS_A else "other_" + suite)


def shifted(mask16, avoid16):
    """The same token pattern moved to the empty position (no object token) farthest from where it was; None if none fits."""
    ij = np.argwhere(mask16 > 0)
    if len(ij) == 0:
        return None
    best, far = None, -1.0
    for di in range(-15, 16, 2):
        for dj in range(-15, 16, 2):
            q = ij + [di, dj]
            if q.min() < 0 or q.max() > 15 or q[:, 0].min() < 7:          # stay on the lower (table) half of the image
                continue
            if avoid16[q[:, 0], q[:, 1]].max() > 0:
                continue
            dist = float(np.hypot(di, dj))
            if dist > far:
                far, best = dist, q
    if best is None:
        return None
    m = np.zeros((16, 16), np.float32)
    m[best[:, 0], best[:, 1]] = 1.0
    return m


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    only = sys.argv[sys.argv.index("--only") + 1].split("+") if "--only" in sys.argv else None
    per_task = {}
    for f in sorted(glob.glob(os.path.join(FRAMES, "*.npz"))):
        key = os.path.basename(f)[:-4]
        if only and key not in only:
            continue
        suite, tid = key.rsplit("_", 1)
        tid = int(tid)
        if suite not in ("libero_goal", "libero_object", "libero_90"):
            continue
        t = by.get((suite, tid))
        if t is None or len(set(t.moved)) != 1:
            continue
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        types = {n: t.objects.get(n) for n in names if n in t.objects}
        cands = [c for c in F.candidates(t) if c in set(types.values())]
        ph = F.scene_phrases(t, tasks)
        cands = [c for c in cands if c in ph]                 # objects no instruction names have no phrase to read
        prompts = {c: F.substitute(t, c, ph) for c in cands}
        if len(cands) < 2 or any(p is None for p in prompts.values()):
            continue
        geoms = {c: np.concatenate([W.geoms_of(z, name=n) for n, ty in types.items() if ty == c]) for c in cands}
        recs, chunks = [], {}
        for k in range(len(z["states"])):
            eef = np.asarray(z["state8"][k][:3], np.float64)
            truth = {c: (W.oracle_map(z["seg"][k], geoms[c]) >= MIN_COVER).astype(np.float32) for c in cands}
            vis = [c for c in cands if truth[c].max() > 0]
            if len(vis) < 2:
                continue
            pos = {}
            for n, ty in types.items():
                pos.setdefault(ty, np.asarray(z["pos"][k][names.index(n)], np.float64))
            Pv = np.stack([pos[c][:2] - eef[:2] for c in vis])
            allobj = np.clip(sum(truth.values()), 0, 1)
            base_obs = {"observation/image": z["img"][k], "observation/wrist_image": z["wimg"][k],
                        "observation/state": np.asarray(z["state8"][k], np.float32)}
            own_pre = bk.prefix(bk.observation(dict(base_obs, prompt=t.language)))
            unrel_pre = bk.prefix(bk.observation(dict(base_obs, prompt=UNRELATED)))
            rec = dict(state=int(z["states"][k]), vis=vis)
            for s in SEEDS:
                g = np.random.default_rng(1000 * k + s)
                noise = g.standard_normal(bk.dim_full).astype(np.float32)[None]

                def disp(pre, gate=None):
                    rows = bk.sample(1, pre, seed=s, noise=noise, step_gate=gate)
                    a = bk.unnormalize(rows)[0]
                    return a, 0.05 * a[:, :3].sum(0)
                a0, d0 = disp(own_pre)
                au, _ = disp(unrel_pre)
                chunks.setdefault(("base", s), {})[k] = a0
                r = dict(base=pointed(d0[:2], Pv), swap={}, steer={}, os_={}, ls={}, shift={})
                for i, c in enumerate(vis):
                    sp = bk.prefix(bk.observation(dict(base_obs, prompt=prompts[c])))
                    _, ds = disp(sp)
                    r["swap"][c] = (pointed(ds[:2], Pv), ds[:2].tolist())
                for name, cfg in CONFIGS.items():
                    r["steer"][name] = {}
                    for i, c in enumerate(vis):
                        gate = AB.bias_gate(bk, {"agent": truth[c]}, own_pre, **cfg)
                        a1, d1 = disp(own_pre, gate)
                        r["steer"][name][c] = (pointed(d1[:2], Pv), d1[:2].tolist())
                        chunks.setdefault((name, c, s), {})[k] = a1
                        if i == 0:
                            ua, _ = disp(unrel_pre, AB.bias_gate(bk, {"agent": truth[c]}, unrel_pre, **cfg))
                            r["ls"][name] = float(np.linalg.norm(a1 - ua) / max(np.linalg.norm(a0 - au), 1e-9))
                            sh = shifted(truth[c], allobj)
                            if sh is not None:
                                rc = np.argwhere(sh > 0).mean(0) * 16 + 7.5
                                pt = W.unproject(rc, z["depth"][k][int(rc[0]), int(rc[1])], z["K"], z["E"][k])
                                P2 = np.concatenate([Pv, (pt[:2] - eef[:2])[None]])
                                _, d2 = disp(own_pre, AB.bias_gate(bk, {"agent": sh}, own_pre, **cfg))
                                r["shift"][name] = int(pointed(d2[:2], P2) == len(vis))
                rec[s] = r
            rec["Pv"] = Pv.tolist()
            recs.append(rec)
        if not recs:
            continue
        own = t.moved[0]

        def pa(get):
            xs = [int(get(rec[s], c) == i) for rec in recs for s in SEEDS for i, c in enumerate(rec["vis"])]
            return float(np.mean(xs)) if xs else float("nan")

        def cc(get):
            out = []
            for rec in recs:
                Pv = np.asarray(rec["Pv"])
                for s in SEEDS:
                    ds = [np.asarray(get(rec[s], c)) for c in rec["vis"]]
                    for i in range(len(ds)):
                        for j in range(i + 1, len(ds)):
                            out.append(float(unit(ds[i] - ds[j]) @ unit(Pv[i] - Pv[j])))
            return float(np.mean(out)) if out else float("nan")
        base_pa = float(np.mean([int(rec[s]["base"] == i) for rec in recs for s in SEEDS for i, c in enumerate(rec["vis"])]))
        row = dict(group=group(suite, tid), language=t.language, candidates=cands, own=own, n_states=len(recs),
                   chance=float(np.mean([1.0 / len(r["vis"]) for r in recs])), PA_base=base_pa,
                   PA_swap=pa(lambda r, c: r["swap"][c][0]), CC_swap=cc(lambda r, c: r["swap"][c][1]), configs={})
        for name in CONFIGS:
            os_r = []
            for c in cands:
                for s in SEEDS:
                    st, bs = chunks.get((name, c, s), {}), chunks[("base", s)]
                    ks = sorted(set(st) & set(bs))
                    for k1, k2 in zip(ks, ks[1:]):
                        os_r.append(np.linalg.norm(st[k1] - st[k2]) / max(np.linalg.norm(bs[k1] - bs[k2]), 1e-9))
            row["configs"][name] = dict(
                PA=pa(lambda r, c: r["steer"][name][c][0]), CC=cc(lambda r, c: r["steer"][name][c][1]),
                OS=float(np.median(os_r)) if os_r else None,
                LS=float(np.median([rec[s]["ls"][name] for rec in recs for s in SEEDS if name in rec[s]["ls"]])),
                SHIFT=float(np.mean([rec[s]["shift"][name] for rec in recs for s in SEEDS if name in rec[s]["shift"]]))
                if any(name in rec[s]["shift"] for rec in recs for s in SEEDS) else None)
        per_task[key] = row
        bestc = max(row["configs"], key=lambda n: row["configs"][n]["CC"])
        v = row["configs"][bestc]
        print(f"[b2] {key:18s} {row['group']:9s} chance {row['chance']:.2f} | base PA {base_pa:.2f} | swap PA {row['PA_swap']:.2f} "
              f"CC {row['CC_swap']:+.2f} | best {bestc}: PA {v['PA']:.2f} CC {v['CC']:+.2f} OS {v['OS']} LS {v['LS']:.2f} "
              f"SHIFT {v['SHIFT']} | {t.language}", flush=True)
    groups = {}
    for gname in sorted({v["group"] for v in per_task.values()}):
        vs = [v for v in per_task.values() if v["group"] == gname]
        m = lambda xs: float(np.nanmean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None  # noqa: E731
        groups[gname] = dict(tasks=len(vs), chance=m([v["chance"] for v in vs]), PA_base=m([v["PA_base"] for v in vs]),
                             PA_swap=m([v["PA_swap"] for v in vs]), CC_swap=m([v["CC_swap"] for v in vs]),
                             configs={n: {k: m([v["configs"][n][k] for v in vs]) for k in ("PA", "CC", "OS", "LS", "SHIFT")}
                                      for n in CONFIGS})
        print(f"[b2] GROUP {gname}: tasks {len(vs)} chance {groups[gname]['chance']:.2f} base {groups[gname]['PA_base']:.2f} "
              f"swap PA {groups[gname]['PA_swap']:.2f} CC {groups[gname]['CC_swap']:+.2f}", flush=True)
        for n, v in groups[gname]["configs"].items():
            print(f"[b2]     {n:14s} PA {v['PA']:.2f} CC {v['CC']:+.2f} OS {v['OS']} LS {v['LS']} SHIFT {v['SHIFT']}", flush=True)
    res = dict(configs={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in CONFIGS.items()},
               seeds=list(SEEDS), groups=groups, per_task=per_task)
    TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    out = os.path.join(R, f"b2_steer_offline{'_' + TAG if TAG else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("b2_steer", bench="libero", tag="offline" + ("_" + TAG if TAG else ""), seed=0, metrics=dict(groups=groups, configs=res["configs"]),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/b2_steer_offline.py", command=" ".join(sys.argv),
                            artifacts=[out]))


if __name__ == "__main__":
    main()
