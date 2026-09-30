"""Which CARRY should a composed task use? (EXPLORATORY dev read on the practice states 0-9; ledger p5_stream tag dev_carries)

A composed program takes its grasp from one task and its place from another; between them the object must be carried through a
scene neither source was recorded in.  Carries compared on the 54 practised compositions (10 episodes each, paired by init state):
  streamc   the place source's own route, relative to the receptacle (experiment dev_p5)
  streamh   the frozen prior carries; the guide resumes at the descent (dev_handoff)
  streamte  a lift-across-descend bridge to the place source's descent, waypoints and gripper events passed on arrival (dev_carry)
  streamhe  streamh with events passed on arrival (dev_carry)
  streamtle streamte whose bridge first lifts by the place source's own lift (added after streamte dragged low-carried objects)
The carry set: greedy forward selection -- start from the carry that passes practice (>= 8 of 10) most often, add the carry that adds
the most passing compositions to the set's union, stop when the best addition adds < 2.  (First stated as "a carry joins if it adds
>= 2 beyond the union of all the others"; changed before the full read because a near-duplicate variant, streamtle, was added after
it, and under that rule two near-duplicates exclude each other.  Both outputs are reported.)

    python experiments/carry_read.py [--record]
"""
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
EXP = {"streamc": "dev_p5", "streamh": "dev_handoff", "streamte": "dev_carry", "streamhe": "dev_carry", "streamtle": "dev_carry"}


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    HR = load_mod("hr", os.path.join(ROOT, "experiments", "handoff_read.py"))
    PO = load_mod("po", os.path.join(ROOT, "experiments", "phase_outcomes.py"))
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    combos = [tuple(x) for x in json.load(open(os.path.join(R, "dev_handoff_combos.json")))]
    tab = {}
    for t, g, p in combos:
        row = {}
        for k, ex in EXP.items():
            f = os.path.join(R, f"{ex}_{t}_{k}_{g}_{p}.json")
            if os.path.exists(f):
                row[k] = HR.episodes(f)
        if len(row) == len(EXP):
            tab[(t, g, p)] = row
    kinds = list(EXP)
    succ = {key: {k: sum(v[k].values()) for k in kinds} for key, v in tab.items()}
    passes = {k: {key for key, s in succ.items() if s[k] >= 8} for k in kinds}
    res = dict(compositions=len(tab), totals={k: int(sum(s[k] for s in succ.values())) for k in kinds}, n=10 * len(tab),
               passes={k: len(v) for k, v in passes.items()})
    union_all = set().union(*passes.values())
    res["passes_union_all"] = len(union_all)
    res["best_of_all"] = int(sum(max(s.values()) for s in succ.values()))
    added = {k: len(passes[k] - set().union(*[passes[j] for j in kinds if j != k])) for k in kinds}
    res["unique_passes"] = added
    res["carry_set_first_rule"] = [k for k in kinds if added[k] >= 2]
    carries = [max(kinds, key=lambda k: (len(passes[k]), -kinds.index(k)))]
    cover = set(passes[carries[0]])
    while True:
        gain = {k: len(passes[k] - cover) for k in kinds if k not in carries}
        if not gain or max(gain.values()) < 2:
            break
        k = max(gain, key=lambda k: (gain[k], -kinds.index(k)))
        carries.append(k)
        cover |= passes[k]
    res["carry_set"] = carries
    res["passes_union_set"] = len(set().union(*[passes[k] for k in carries])) if carries else 0
    # by layout mismatch and by the place source's object type
    bands = {}
    for key in tab:
        t, g, p = key
        m = HR.mismatch(t, g, p, pool)
        band = "unknown" if m is None else ("<10" if m < 10 else "10-30" if m < 30 else ">30")
        cross = pool[str(p)]["object_type"] != pool[str(t)]["object_type"]
        for grp in (f"mismatch {band}", "place source of another type" if cross else "place source of the same type"):
            b = bands.setdefault(grp, dict(compositions=0, **{k: 0 for k in kinds}))
            b["compositions"] += 1
            for k in kinds:
                b[k] += succ[key][k]
    res["groups"] = bands
    # phase outcomes where light recordings exist
    res["phase_outcomes"] = {}
    LIGHT = {"streamc": "dev_p5light"}          # the guided carry's light replay (seed 0 replays the same episodes, now recorded)
    res["replay_identical"] = {}
    for k in kinds:
        ex = LIGHT.get(k, EXP[k])
        files = [os.path.join(R, f"{ex}_{t}_{k}_{g}_{p}.json") for (t, g, p) in tab]
        if k in LIGHT:
            same = [HR.episodes(f) == tab[key][k] for f, key in zip(files, tab) if os.path.exists(f)]
            res["replay_identical"][k] = [int(sum(same)), len(same)]
        tot = {s: 0 for s in PO.STAGES + ("success",)}
        n = 0
        for f in files:
            if not os.path.isdir(f[:-5] + "_rec"):
                continue
            _, rows = PO.outcomes(f, pool)
            for r in rows:
                tot[PO.stage_of(r)] += 1
                n += 1
        if n:
            res["phase_outcomes"][k] = dict(episodes=n, failed_at=tot)
    res["per_composition"] = {f"{t}_{g}_{p}": succ[(t, g, p)] for (t, g, p) in tab}
    print(f"{len(tab)} compositions x 10 practice episodes: " + ", ".join(f"{k} {res['totals'][k]}" for k in kinds) +
          f"; best carry per composition {res['best_of_all']}")
    print("passing practice (>= 8/10): " + ", ".join(f"{k} {res['passes'][k]} (alone {added[k]})" for k in kinds) +
          f"; union {len(union_all)}")
    print(f"carry set (greedy): {carries} -> {res['passes_union_set']} passing; first rule: {res['carry_set_first_rule']}")
    for grp, b in sorted(bands.items()):
        print(f"  {grp}: {b['compositions']} compositions: " + ", ".join(f"{k} {b[k]}" for k in kinds) + f" of {10 * b['compositions']}")
    for k, v in res["phase_outcomes"].items():
        print(f"  phases {k} ({v['episodes']} episodes): " + ", ".join(f"{s} {v['failed_at'][s]}" for s in PO.STAGES + ("success",)))
    json.dump(res, open(os.path.join(R, "carry_read.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("p5_stream", bench="libero_90", tag="dev_carries", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/carry_read.py", command=" ".join(sys.argv) + "  # EXPLORATORY, states 0-9"))


if __name__ == "__main__":
    main()
