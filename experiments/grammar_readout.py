"""GRAMMAR x BINDINGS, READ OUT (interpretability of the prototypes and the grammar; archive/2026-10-program/PLAN_TRANSFER.md C1).

Two stages.
  labels (GPU): every frame's 10-step chunk of each task's ONE demo (results/robot/p0_demo) labelled by r19's basin rule at the
          coarse levels of the 40-task prototype tree (t = 0.4, 0.55) -> results/robot/grammar_labels.npz
  read (CPU):   - each coarse prototype NAMED by what its frames do: gripper closed share, hand speed, and where its frames fall in
                  the v2 phases (approach / carry / rest; lifelong_ood/policy/phase.py cut);
                - each task's PROGRAM as a sentence: its phases with their anchors, the coarse-prototype string of each phase, and
                  the bindings in readable units (the grasp: hand position in the object's frame and the approach direction; the
                  place: the object's release position in the receptacle's frame);
                - the grasp bindings' agreement WITHIN an object type vs across types (why a grasp source is keyed by type);
                - bytes per task.
                -> results/robot/grammar_readout.json, docs/GRAMMAR.md

    ~/flash/miniconda3/envs/chunks/bin/python experiments/grammar_readout.py labels [--device cuda]
    ~/flash/miniconda3/envs/chunks/bin/python experiments/grammar_readout.py read [--record]
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import phase as PH                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
LEVELS = (0.4, 0.55)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def demo_tasks():
    return sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(R, "p0_demo", "*.npz"))
                  if os.path.basename(f)[:-4].isdigit())


def labels(device="cuda"):
    from lifelong_ood.policy import data as D
    from lifelong_ood.tree.nodes import Tree
    r19 = load_mod("r19", os.path.join(ROOT, "experiments", "r19_invariance.py"))
    r2b = load_mod("r2b", os.path.join(ROOT, "experiments", "r2b_marginal.py"))
    lib = np.load(os.path.join(R, "r2b_libero_40.npz"))
    tree = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
    norm = D.Norm.load()
    tasks = [t for t in demo_tasks() if os.path.exists(os.path.join(R, "p0_demo", f"{t}_match.npz"))
             and "actions" in np.load(os.path.join(R, "p0_demo", f"{t}_match.npz"), allow_pickle=True).files]   # 46, 47, 59: seeds only
    demos = {t: PH.at_rest(PH.load_demo(R, t)) for t in tasks}
    X = [r19.chunks(demos[t]["action"], norm, every=1) for t in tasks]
    cut = np.cumsum([0] + [len(x) for x in X])
    allX = np.concatenate(X)
    out = dict(tasks=np.array(tasks), cut=cut)
    for u in LEVELS:
        li = int(np.argmin([abs(v - u) for v in tree.u_grid]))
        ids = tree.levels[li]
        modes = np.stack([tree.nodes[i].mode for i in ids]).astype(np.float32)
        out[f"lab_{u}"] = np.asarray(r19.labels(allX, lib["mu"], lib["var"], modes, float(tree.u_grid[li]), r2b, device))
        out[f"ids_{u}"] = np.asarray(ids)
        out[f"modes_{u}"] = modes
        print(f"[grammar] t = {u}: {len(ids)} prototypes, {len(np.unique(out[f'lab_{u}']))} used by the demos", flush=True)
    np.savez_compressed(os.path.join(R, "grammar_labels.npz"), **out)


def approach_dir(Rm):
    """The hand's approach axis (its z) in the object's frame, in words."""
    z = Rm[:, 2]
    k = int(np.argmax(np.abs(z)))
    return {(2, -1): "from above", (2, 1): "from below"}.get((k, int(np.sign(z[k]))), f"from the side ({'xyz'[k]}{'+' if z[k] > 0 else '-'})")


def read():
    L = np.load(os.path.join(R, "grammar_labels.npz"))
    tasks = [int(t) for t in L["tasks"]]
    cut = L["cut"]
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    lang = {int(t): v["language"] for t, v in pool.items()}
    # 1. name the coarse prototypes by what their frames do
    frames = []                                               # (task, frame, gripper closed, speed, phase kind)
    for k, t in enumerate(tasks):
        z = PH.at_rest(PH.load_demo(R, t))
        a = np.asarray(z["action"])
        parts = PH.cut(z)
        kind = np.empty(len(a), object)
        for (_, _, _, s0, s1, role, ref, held) in parts:
            kind[s0:s1] = ("approach" if role == "obj" else "carry" if held is not None else "fixture" if role == "fix" else "rest")
        dp = np.diff(np.asarray(z["eef_pos"]), axis=0, prepend=np.asarray(z["eef_pos"])[:1]) * 100
        sp = np.linalg.norm(dp, axis=-1)
        cmd = a[:, 6] > 0
        ev = np.zeros(len(a), bool)
        for i in np.where(cmd[1:] != cmd[:-1])[0] + 1:
            ev[max(0, i - 2):i + 3] = True
        for i in range(cut[k + 1] - cut[k]):
            frames.append((t, i, bool(a[i, 6] > 0), float(sp[i]), kind[i], float(dp[i, 2]), bool(ev[i])))
    res = dict(tasks=tasks, prototypes={}, programs={}, grasp_by_type={})
    from lifelong_ood.policy import data as D
    norm = D.Norm.load()
    for u in LEVELS:
        lab = L[f"lab_{u}"]
        ids = list(L[f"ids_{u}"])
        chunks = D.from_rows(L[f"modes_{u}"], norm, 10)          # each prototype's mode as 10 raw actions (OSC deltas, gripper)
        names = {}
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            cl = float(np.mean([frames[i][2] for i in idx]))
            spd = float(np.median([frames[i][3] for i in idx]))
            kinds = {k_: float(np.mean([frames[i][4] == k_ for i in idx])) for k_ in ("approach", "carry", "rest", "fixture")}
            top = max(kinds, key=kinds.get)
            dz = float(np.median([frames[i][5] for i in idx]))
            evs = float(np.mean([frames[i][6] for i in idx]))
            closed = cl > 0.5
            verb = ("grip change" if evs > 0.3 else ("lower" if closed else "reach down") if dz < -0.25 else
                    ("lift" if closed else "rise") if dz > 0.25 else ("transport" if closed else "move"))
            ch = chunks[int(c)]
            names[int(c)] = dict(frames=int(len(idx)), tasks=int(len({frames[i][0] for i in idx})), gripper_closed=cl,
                                 speed_cm_per_step=spd, dz_cm_per_step=dz, event_share=evs, phase_share=kinds,
                                 name=f"{'closed' if closed else 'open'} {verb} ({top})",
                                 mode_path_cm=(np.cumsum(ch[:, :3], 0) * 5.0).round(2).tolist(),   # commanded: goal = p + 0.05 a (m)
                                 mode_grip=ch[:, 6].round(2).tolist())
        res["prototypes"][str(u)] = names
    # 2. programs as sentences; 3. grasp bindings by object type
    grasps = {}
    for k, t in enumerate(tasks):
        z = PH.at_rest(PH.load_demo(R, t))
        parts = PH.cut(z)
        nm = [str(x) for x in z["names"]]
        lab = {u: L[f"lab_{u}"][cut[k]:cut[k + 1]] for u in LEVELS}
        ph = []
        for (_, _, _, s0, s1, role, ref, held) in parts:
            proto = {str(u): PH_collapse([int(x) for x in lab[u][s0:s1]]) for u in LEVELS}
            rec = dict(frames=[int(s0), int(s1)], anchor=None if ref is None else nm[ref], prototypes=proto)
            if role == "obj":
                e = s1 - 1
                T_ = np.linalg.inv(PH.pose(z["obj_pos"][e, ref], PH.q2m(z["obj_quat"][e, ref]))) @ PH.pose(z["eef_pos"][e], PH.q2m(z["eef_quat"][e]))
                rec.update(kind="grasp", hand_in_object_cm=(T_[:3, 3] * 100).round(1).tolist(), approach=approach_dir(T_[:3, :3]))
                grasps.setdefault(PH.type_of(nm[ref]), []).append((t, T_))
            elif held is not None:
                rel = [i for i in range(s0 + 1, s1) if z["action"][i, 6] <= 0 < z["action"][i - 1, 6]]
                e = rel[0] if rel else s1 - 1
                Rr = PH.q2m(z["obj_quat"][e, ref])
                p = Rr.T @ (np.asarray(z["obj_pos"][e, held]) - np.asarray(z["obj_pos"][e, ref]))
                rec.update(kind="carry", held=nm[held], release_in_receptacle_cm=(p * 100).round(1).tolist())
            else:
                rec.update(kind="fixture" if role == "fix" else "rest")
            ph.append(rec)
        sent = " -> ".join({"grasp": lambda r: f"GRASP {r['anchor']} ({r['approach']})",
                            "carry": lambda r: f"CARRY {r['held']} TO {r['anchor']}",
                            "rest": lambda r: f"FINISH at {r['anchor']}",
                            "fixture": lambda r: f"OPERATE {r['anchor']}"}[r["kind"]](r) for r in ph)
        T = len(z["action"])
        res["programs"][str(t)] = dict(language=lang.get(t), sentence=sent, phases=ph, bytes=2 * 8 * T + 28 * len(parts),
                                       grammar=" ".join(r["kind"] for r in ph))
    for ty, lst in grasps.items():
        res["grasp_by_type"][ty] = [dict(task=t, hand_in_object_cm=(T_[:3, 3] * 100).round(1).tolist(), approach=approach_dir(T_[:3, :3]))
                                    for t, T_ in lst]
    within, across = [], []
    items = [(ty, t, T_) for ty, lst in grasps.items() for t, T_ in lst]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            d = float(np.linalg.norm(items[i][2][:3, 3] - items[j][2][:3, 3]) * 100)
            (within if items[i][0] == items[j][0] else across).append(d)
    res["grasp_offset_cm"] = dict(within_type_median=float(np.median(within)), across_type_median=float(np.median(across)),
                                  within_pairs=len(within), across_pairs=len(across))
    # 4. do the prototype strings group phases by the binding table's keys?  normalized edit distance between the t = 0.4 strings
    #    of two tasks' grasp phases (same object type vs not) and carry phases (same goal relation vs not; p5_pool goal_key)
    def edit(a, b):
        n, m = len(a), len(b)
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[m] / max(n, m, 1)
    key_of = {"grasp": lambda t, r: PH.type_of(r["anchor"]), "carry": lambda t, r: (pool.get(str(t)) or {}).get("goal_key")}
    res["string_by_key"] = {}
    for kind in ("grasp", "carry"):
        items = [(int(t), key_of[kind](int(t), r), r["prototypes"]["0.4"]) for t, p in res["programs"].items() for r in p["phases"]
                 if r["kind"] == kind]
        items = [x for x in items if x[1] is not None]
        n_ = len(items)
        Dm = np.array([[edit(items[i][2], items[j][2]) for j in range(n_)] for i in range(n_)])
        iu = np.triu_indices(n_, 1)
        keys = np.array([str(x[1]) for x in items])

        def gap(k):
            sm = (k[:, None] == k[None, :])[iu]
            return Dm[iu][~sm].mean() - Dm[iu][sm].mean(), sm
        g0, sm = gap(keys)
        rng = np.random.default_rng(0)
        perm = [gap(rng.permutation(keys))[0] for _ in range(2000)]      # keys shuffled across phases: the pairs' dependence kept
        res["string_by_key"][kind] = dict(same_key_median=float(np.median(Dm[iu][sm])), other_key_median=float(np.median(Dm[iu][~sm])),
                                          same_key_mean=float(Dm[iu][sm].mean()), other_key_mean=float(Dm[iu][~sm].mean()),
                                          same_pairs=int(sm.sum()), other_pairs=int((~sm).sum()), phases=n_,
                                          perm_p=float((1 + sum(p >= g0 for p in perm)) / (1 + len(perm))))
    # the same permutation read for the grasp GEOMETRY by object type (the binding, not the string)
    gi = [(ty, T_[:3, 3]) for ty, lst in grasps.items() for _, T_ in lst]
    Gd = np.array([[np.linalg.norm(a[1] - b[1]) * 100 for b in gi] for a in gi])
    iu = np.triu_indices(len(gi), 1)
    gk = np.array([x[0] for x in gi])

    def ggap(k):
        sm = (k[:, None] == k[None, :])[iu]
        return Gd[iu][~sm].mean() - Gd[iu][sm].mean()
    rng = np.random.default_rng(0)
    g0 = ggap(gk)
    res["grasp_geometry_perm_p"] = float((1 + sum(ggap(rng.permutation(gk)) >= g0 for _ in range(2000))) / 2001)
    print("prototype strings by binding key:", json.dumps(res["string_by_key"]))
    fam = {}
    for t, p in res["programs"].items():
        fam.setdefault(p["grammar"], []).append(int(t))
    res["families"] = fam
    json.dump(res, open(os.path.join(R, "grammar_readout.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        keep = {k: res[k] for k in ("tasks", "prototypes", "grasp_offset_cm", "grasp_geometry_perm_p", "string_by_key", "families")}
        keep["programs"] = {t: {k: p[k] for k in ("language", "sentence", "bytes", "grammar", "phases")} for t, p in res["programs"].items()}
        keep["grasp_by_type"] = res["grasp_by_type"]
        print(ledger.record("grammar", bench="libero_90", tag="readout", seed=0, metrics=keep,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/grammar_readout.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "grammar_readout.json"), os.path.join(R, "grammar_labels.npz")]))
    print(json.dumps(res["grasp_offset_cm"]))
    for u in LEVELS:
        top = sorted(res["prototypes"][str(u)].items(), key=lambda kv: -kv[1]["frames"])
        print(f"t = {u}: {len(top)} prototypes used; " + "; ".join(f"{c}: {v['name']} ({v['frames']} frames, {v['tasks']} tasks, "
              f"{v['speed_cm_per_step']:.2f} cm/step, dz {v['dz_cm_per_step']:+.2f})" for c, v in top[:16]))
    for g, ts in fam.items():
        print(f"{g}: {len(ts)} tasks")


def PH_collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "labels":
        labels(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda")
    else:
        read()
