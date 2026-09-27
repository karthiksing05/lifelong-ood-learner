"""Experiment 2's reads (pre-registered: ledger m3_prereg tags branch, branch_amend1).  Prefix gate (G1), full-replay gate (G2),
rescue per arm (D1), each knob vs the recorded-noise continuation (primary) and vs fresh noise (D2), early vs late fresh
noise (D3).

    python experiments/r18_branch_stats.py [--record]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger, stats                                 # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TOL = 1e-3
LATE = ["replay", "noise1", "noise2", "noise3", "noise4", "wx0.5", "wx2", "leaf2", "release", "nofold", "foldall"]
KNOBS = ["wx0.5", "wx2", "leaf2", "release", "nofold", "foldall"]
CONCENTRATED = {42, 32, 1, 35, 40, 36, 52, 62, 41}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def branches():
    """-> list of dicts: task, trial, id, variant, at, success, steps, rec path."""
    out = []
    for f in sorted(glob.glob(os.path.join(R, "r18_branch_*_*.json"))):
        if f.endswith(".list.json") or "specs" in f or "shards" in f:
            continue
        r = json.load(open(f))
        for tid, t in r["tasks"].items():
            for e in t["episodes"]:
                if e.get("error") is not None or "branch" not in e:
                    continue
                rec = os.path.join(f[:-5] + "_rec", f"task{tid}_trial{e['trial']}_{e['branch']}.npz")
                out.append(dict(task=int(tid), trial=int(e["trial"]), id=e["branch"], variant=e["variant"], at=int(e["at"]),
                                success=int(bool(e["success"])), steps=int(e["steps"]), rec=rec if os.path.exists(rec) else None))
    return out


def main(a=None):
    if a is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--record", action="store_true")
        ap.add_argument("--B", type=int, default=10000)
        a = ap.parse_args()
    os.chdir(ROOT)
    r17 = load_module("r17", os.path.join(ROOT, "experiments", "r17_nft.py"))
    from lifelong_ood.policy.compose import DemoProgram
    spec = json.load(open(os.path.join(R, "r18_branch_specs.json")))
    inst = {(i["task"], i["trial"]): i for i in spec["instances"]}
    demo = {}
    br = branches()
    # G1 / G2: every branch's prefix against the practice recording
    for b in br:
        i = inst[(b["task"], b["trial"])]
        tag = spec["tasks"][str(b["task"])]["tag"]
        ref = np.load(os.path.join(R, f"m5_practice_{b['task']}_{tag}_rec", f"task{b['task']}_trial{b['trial']}.npz"))["state"]
        if b["rec"] is None:
            b["prefix_ok"], b["prefix_err"] = False, float("nan")
            continue
        z = np.load(b["rec"])
        st = z["state"]
        # ACTIVE: the branch's actions differ from the recorded ones from its branch point on (the recorded-noise continuation
        # reproduces the recording, so an inactive knob changed nothing, e.g. a guide knob inside a fold window)
        ra = np.load(os.path.join(R, f"m5_practice_{b['task']}_{tag}_rec", f"task{b['task']}_trial{b['trial']}.npz"))["action"]
        m = min(len(z["action"]), len(ra))
        b["active"] = int(len(z["action"]) != len(ra) or bool(np.any(z["action"][5 * b["at"]:m] != ra[5 * b["at"]:m])))
        n = min(len(st), len(ref), 5 * b["at"])
        err = float(np.abs(st[:n] - ref[:n]).max()) if n else 0.0
        b["prefix_err"], b["prefix_ok"] = err, err <= TOL
        if "_full_" in b["id"]:
            diff = np.where(np.abs(st[:min(len(st), len(ref))] - ref[:min(len(st), len(ref))]).max(1) > TOL)[0]
            b["replay_same_len"] = len(st) == len(ref)
            b["first_diverging_step"] = int(diff[0]) if len(diff) else None
        # secondary: does the pointer pass the recorded stall frame?
        if b["task"] not in demo:
            demo[b["task"]] = DemoProgram.load(os.path.join(R, f"r3_prog_{b['task']}_raw_lib40.npz"))
        _, trace = r17.stall_queries(demo[b["task"]], st, 1)
        b["passes_stall"] = int(trace.max() > i["stall_frame"])
    ok = [b for b in br if "_full_" not in b["id"]]
    g1 = dict(branches=len(ok), passing=int(sum(b["prefix_ok"] for b in ok)),
              exact=int(sum(b["prefix_err"] == 0.0 for b in ok)),
              err_quantiles=[float(np.nanquantile([b["prefix_err"] for b in ok], q)) for q in (0.5, 0.9, 0.99, 1.0)] if ok else [])
    reps = [b for b in br if "_full_" in b["id"]]
    g2 = dict(replays=len(reps), outcome_reproduced=int(sum(1 - b["success"] for b in reps)),
              states_reproduced=int(sum(b.get("first_diverging_step") is None and b.get("replay_same_len", False) for b in reps)),
              diverging=[dict(task=b["task"], trial=b["trial"], step=b.get("first_diverging_step"), success=b["success"]) for b in reps
                         if b.get("first_diverging_step") is not None or not b.get("replay_same_len", False)])
    print(f"[G1] prefix within {TOL}: {g1['passing']}/{g1['branches']} (exact {g1['exact']}); max-error quantiles 50/90/99/100% "
          f"{['%.1e' % x for x in g1['err_quantiles']]}")
    print(f"[G2] replays reproducing the failure {g2['outcome_reproduced']}/{g2['replays']}, the states {g2['states_reproduced']}/{g2['replays']}; "
          f"diverging {g2['diverging']}")
    # table[(task, trial)][arm] = success, over prefix-passing branches
    table = {}
    for b in ok:
        if not b["prefix_ok"]:
            continue
        arm = ("late_" if b["id"].split("_")[1] == "late" else "early_") + b["variant"]
        table.setdefault((b["task"], b["trial"]), {})[arm] = b
    # G3 (amendment 4): the late replay -- recorded noise, nothing changed -- must reproduce the recorded episode's actions
    lr = [b for b in ok if b["id"].split("_", 1)[1] == "late_replay"]
    g3 = dict(late_replays=len(lr), reproduced=int(sum(1 - b["active"] for b in lr)),
              not_reproduced=[dict(task=b["task"], trial=b["trial"], success=b["success"]) for b in lr if b["active"]])
    print(f"[G3] late replays reproducing the recorded actions {g3['reproduced']}/{g3['late_replays']}; not: {g3['not_reproduced']}")
    res = dict(G1=g1, G2=g2, G3=g3, D1={}, D2={}, D3={}, per_task={}, tol=TOL)
    arms = [f"late_{v}" for v in LATE] + ["early_noise1", "early_noise2"]
    for arm in arms:
        xs = [v[arm]["success"] for v in table.values() if arm in v]
        ps = [v[arm]["passes_stall"] for v in table.values() if arm in v]
        k, n = int(sum(xs)), len(xs)
        res["D1"][arm] = dict(k=k, n=n, rate=k / n if n else float("nan"), wilson=list(stats.wilson(k, n)),
                              passes_stall=int(sum(ps)), active=int(sum(v[arm]["active"] for v in table.values() if arm in v)))
    for name, group in (("any_late", [f"late_{v}" for v in LATE]), ("any_late_noise", [f"late_noise{s}" for s in (1, 2, 3, 4)]),
                        ("any_late_knob", [f"late_{v}" for v in KNOBS]), ("any_early", ["early_noise1", "early_noise2"]),
                        ("any", arms)):
        full = [v for v in table.values() if any(g in v for g in group)]
        k = int(sum(max(v[g]["success"] for g in group if g in v) for v in full))
        res["D1"][name] = dict(k=k, n=len(full), rate=k / len(full) if full else float("nan"), wilson=list(stats.wilson(k, len(full))))
    for arm in arms + ["any_late", "any_late_noise", "any_late_knob", "any_early", "any"]:
        d = res["D1"][arm]
        print(f"[D1] {arm:18s} {d['k']:3d}/{d['n']:3d} = {d['rate']:.3f} ({d['wilson'][0]:.2f}-{d['wilson'][1]:.2f})"
              + (f", pointer past the stall {d['passes_stall']}, actions changed in {d['active']}" if "passes_stall" in d else ""))
    # D2: each knob vs the recorded-noise continuation (primary) and vs fresh noise1 (secondary), paired by instance
    for ctrl in ("replay", "noise1"):
        for kn in KNOBS:
            pairs = [(v[f"late_{kn}"]["success"], v[f"late_{ctrl}"]["success"]) for v in table.values()
                     if f"late_{kn}" in v and f"late_{ctrl}" in v]
            b_ = sum(1 for x, y in pairs if x and not y); c_ = sum(1 for x, y in pairs if y and not x)
            res["D2"][f"{kn}-{ctrl}"] = d = dict(n=len(pairs), knob=int(sum(x for x, _ in pairs)), control=int(sum(y for _, y in pairs)),
                                                 discordant=[b_, c_], sign_p=stats.sign_test(b_, c_))
        # Holm over the knobs (amendment 3: the decision rule uses the Holm-adjusted p)
        ks = sorted(KNOBS, key=lambda kn: res["D2"][f"{kn}-{ctrl}"]["sign_p"])
        run = 0.0
        for r, kn in enumerate(ks):
            d = res["D2"][f"{kn}-{ctrl}"]
            run = max(run, min(1.0, (len(ks) - r) * d["sign_p"]))
            d["holm_p"] = run
        for kn in KNOBS:
            d = res["D2"][f"{kn}-{ctrl}"]
            print(f"[D2] {kn:8s} vs {ctrl:7s}: {d['knob']} vs {d['control']} of {d['n']}, discordant {d['discordant'][0]}/{d['discordant'][1]}, "
                  f"sign p {d['sign_p']:.4f} (Holm {d['holm_p']:.4f})")
    # D3: early vs late fresh noise, seeds 1 and 2, paired by (instance, seed)
    pairs = [(v[f"early_noise{s}"]["success"], v[f"late_noise{s}"]["success"]) for v in table.values() for s in (1, 2)
             if f"early_noise{s}" in v and f"late_noise{s}" in v]
    b_ = sum(1 for x, y in pairs if x and not y); c_ = sum(1 for x, y in pairs if y and not x)
    res["D3"] = dict(n=len(pairs), early=int(sum(x for x, _ in pairs)), late=int(sum(y for _, y in pairs)), discordant=[b_, c_],
                     sign_p=stats.sign_test(b_, c_))
    print(f"[D3] early vs late fresh noise: {res['D3']['early']} vs {res['D3']['late']} of {len(pairs)}, discordant {b_}/{c_}, "
          f"sign p {res['D3']['sign_p']:.4f}")
    # per task: instances, rescued by any arm, by noise, by a knob
    for t in sorted({t for t, _ in table}):
        vs = [v for (tt, _), v in table.items() if tt == t]
        res["per_task"][str(t)] = dict(instances=len(vs), rescued=int(sum(max(b["success"] for b in v.values()) for v in vs)),
                                       noise=int(sum(max([v[f"late_noise{s}"]["success"] for s in (1, 2, 3, 4) if f"late_noise{s}" in v] or [0]) for v in vs)),
                                       knob=int(sum(max([v[f"late_{k}"]["success"] for k in KNOBS if f"late_{k}" in v] or [0]) for v in vs)),
                                       concentrated=t in CONCENTRATED)
    # EXPLORATORY (not pre-registered): where the stalls are.  The pointer aligns the HAND to the demo, so a stall at the demo's
    # last frames means the hand finished the demo's path and the task still failed; and whether the gripper, where the demo
    # holds something at the stall frame, is closed tighter than the demo's (it holds less than the demo did)
    ex = dict(at_end=0, at_end_rescued=0, mid=0, mid_rescued=0, demo_holds=0, tighter=0, tighter_by_task={})
    for (t, k), v in table.items():
        i = inst[(t, k)]
        tag = spec["tasks"][str(t)]["tag"]
        st = np.load(os.path.join(R, f"m5_practice_{t}_{tag}_rec", f"task{t}_trial{k}.npz"))["state"]
        dst = np.load(os.path.join(R, f"r3_prog_{t}_raw_lib40.npz"), allow_pickle=True)["states"].astype(np.float32)
        gap = lambda s: float(s[6] - s[7])                                                   # noqa: E731
        end = i["stall_frame"] >= i["demo_frames"] - 3
        rescued = max(b["success"] for b in v.values())
        ex["at_end" if end else "mid"] += 1
        ex["at_end_rescued" if end else "mid_rescued"] += int(rescued)
        g_r, g_d, g_open = gap(st[min(5 * i["stall_query"], len(st) - 1)]), gap(dst[i["stall_frame"]]), max(gap(x) for x in dst)
        if g_d < 0.5 * g_open:
            ex["demo_holds"] += 1
            if g_r < g_d - 0.005:
                ex["tighter"] += 1
                ex["tighter_by_task"][str(t)] = ex["tighter_by_task"].get(str(t), 0) + 1
    res["exploratory"] = ex
    print(f"[exploratory] stall at the demo's last frames {ex['at_end']} (rescued {ex['at_end_rescued']}), elsewhere {ex['mid']} "
          f"(rescued {ex['mid_rescued']}); demo holds at the stall frame {ex['demo_holds']}, robot's grip > 5 mm tighter {ex['tighter']} "
          f"{ex['tighter_by_task']}")
    conc = sum(r["rescued"] for t, r in res["per_task"].items() if int(t) in CONCENTRATED)
    res["rescued_concentrated"] = conc
    print(f"[per task] {res['per_task']}")
    print(f"[R] instances of the concentrated tasks rescued by some branch: {conc} (part b runs if >= 10)")
    json.dump(dict(res, branches=[{k: v for k, v in b.items() if k != 'rec'} for b in br]),
              open(os.path.join(R, "r18_branch_stats.json"), "w"), indent=1)
    if a.record:
        ledger.record("r18_branch", bench="libero_90", tag="diagnosis", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(tol=TOL),
                      script="experiments/r18_branch_stats.py", command="python experiments/r18_branch_stats.py --record",
                      artifacts=[os.path.join(R, "r18_branch_stats.json")])
        print("[r18] ledger row written")


if __name__ == "__main__":
    main()
