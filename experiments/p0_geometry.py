"""P0b (docs/PLAN_TRANSFER.md; ledger m3_prereg tag p0): are the persistent practice failures BINDING (aim) failures?

Reads the bit-exact replays of experiments/p0_replay.py.  Per task: the manipulated object = the movable object of interest grasped
in the most successful episodes; the success reference = the mean eef position in that object's frame at the close event (first
switch of the gripper command to close) and, for a movable receptacle among the objects of interest, in the receptacle's frame at
the release event.  Each failure is classified (pre-registered order):
    aim      never closes while the object is never grasped; or closes > 2 cm from the reference; or is released > 2 cm from the
             release reference
    closure  closes within 2 cm and no grasp within 20 steps
    carry    grasped, then the grasp is lost (no contact for 5 consecutive steps) while the gripper still commands close
    other    everything else
Decision: C2 (binding) survives iff >= 50% of the primary set (the 123 persistent instances, movable-object tasks) are aim failures.

    python experiments/p0_geometry.py [--record]
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
TOL, GRASP_WINDOW = 0.02, 20


def rot(q):
    """xyzw quaternion -> 3x3 rotation."""
    x, y, z, w = q / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def local(z, t, j):
    """The eef position in object j's frame at step t."""
    return rot(z["obj_quat"][t, j]).T @ (z["eef_pos"][t] - z["obj_pos"][t, j])


def events(cmd):
    closing = cmd > 0
    close = [t for t in range(len(cmd)) if closing[t] and (t == 0 or not closing[t - 1])]
    opening = [t for t in range(1, len(cmd)) if not closing[t] and closing[t - 1]]
    return close, opening


def load(task):
    out = []
    for f in sorted(glob.glob(os.path.join(R, "p0_replay", f"*_{task}_trial*.npz"))):
        z = dict(np.load(f, allow_pickle=True))
        z["file"] = os.path.basename(f)
        z["experiment"] = z["file"].rsplit(f"_{task}_trial", 1)[0]
        out.append(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(os.path.join(R, "r18_branch_specs.json")))
    persistent = {(i["task"], i["trial"]) for i in spec["instances"]}
    tasks = sorted({int(os.path.basename(f).split("_trial")[0].rsplit("_", 1)[1]) for f in glob.glob(os.path.join(R, "p0_replay", "*.npz"))})
    res = dict(tol_m=TOL, tasks={}, gate=dict(replayed=0, exact=0), primary={}, all_failures={})
    rows = []
    for t in tasks:
        eps = load(t)
        res["gate"]["replayed"] += len(eps)
        eps = [z for z in eps if bool(z["exact"])]
        res["gate"]["exact"] += len(eps)
        names = [str(x) for x in eps[0]["names"]]
        movable = list(eps[0]["movable"])
        succ = [z for z in eps if bool(z["rec_success"])]
        counts = collections.Counter(names[j] for z in succ for j in range(len(names)) if movable[j] and z["grasp"][:, j].any())
        info = dict(names=names, movable=[bool(m) for m in movable], successes=len(succ), episodes=len(eps), grasp_counts=dict(counts))
        if not counts:
            info["kind"] = "fixture"
            res["tasks"][str(t)] = info
            for z in eps:
                if not bool(z["rec_success"]):
                    rows.append(dict(task=t, trial=int(z["trial"]), experiment=z["experiment"], cls="fixture",
                                     persistent=(t, int(z["trial"])) in persistent and z["experiment"] == "m5_practice"))
            continue
        obj = names.index(counts.most_common(1)[0][0])
        recs = [j for j in range(len(names)) if j != obj and movable[j]]
        rec = recs[0] if recs else None
        ref_close, ref_rel = [], []
        for z in succ:
            close, opening = events(z["action"][:, 6])
            if close:
                ref_close.append(local(z, close[0], obj))
            g = np.where(z["grasp"][:, obj])[0]
            if rec is not None and len(g):
                rel = [o for o in opening if o > g[0]]
                if rel:
                    ref_rel.append(local(z, rel[0], rec))
        rc = np.mean(ref_close, 0) if ref_close else None
        rr = np.mean(ref_rel, 0) if ref_rel else None
        info.update(kind="movable", object=names[obj], receptacle=names[rec] if rec is not None else None,
                    ref_close=None if rc is None else rc.tolist(), ref_release=None if rr is None else rr.tolist(),
                    success_close_spread_cm=float(100 * np.mean([np.linalg.norm(x - rc) for x in ref_close])) if ref_close else None)
        res["tasks"][str(t)] = info
        for z in eps:
            if bool(z["rec_success"]):
                continue
            cmd = z["action"][:, 6]
            close, opening = events(cmd)
            g = z["grasp"][:, obj]
            r = dict(task=t, trial=int(z["trial"]), experiment=z["experiment"],
                     persistent=(t, int(z["trial"])) in persistent and z["experiment"] == "m5_practice")
            off = None if (not close or rc is None) else float(100 * np.linalg.norm(local(z, close[0], obj) - rc))
            r["close_offset_cm"] = off
            grasped_after = bool(close) and bool(g[close[0]: close[0] + GRASP_WINDOW + 1].any())
            r["grasped"] = bool(g.any())
            if not close and not g.any():
                r["cls"] = "aim"; r["why"] = "never closes"
            elif off is not None and off > 100 * TOL:
                r["cls"] = "aim"; r["why"] = "close off the reference"
            elif not grasped_after and not g.any():
                r["cls"] = "closure"
            else:
                gi = np.where(g)[0]
                # a grasp is LOST when contact is gone for 5 consecutive steps while the gripper has commanded close throughout
                lost = [s for s in range(gi[0] + 1, len(g) - 4) if not g[s:s + 5].any() and np.all(cmd[gi[0]:s + 5] > 0)]
                rel = [o for o in opening if o > gi[0]]
                roff = None if (rec is None or rr is None or not rel) else float(100 * np.linalg.norm(local(z, rel[0], rec) - rr))
                r["release_offset_cm"] = roff
                if lost:
                    r["cls"] = "carry"
                elif roff is not None and roff > 100 * TOL:
                    r["cls"] = "aim"; r["why"] = "release off the reference"
                else:
                    r["cls"] = "other"
            r["aim_if_ungrasped"] = r["cls"] == "aim" and not r["grasped"]
            rows.append(r)
    for scope, sel in (("primary", [r for r in rows if r["persistent"]]), ("all_failures", rows)):
        mv = [r for r in sel if r["cls"] != "fixture"]
        c = collections.Counter(r["cls"] for r in mv)
        n = len(mv)
        res[scope] = dict(n=n, fixture=len(sel) - n, classes=dict(c), aim_share=c["aim"] / n if n else float("nan"),
                          aim_or_grasp_miss_share=(c["aim"] + c["closure"]) / n if n else float("nan"),
                          aim_ungrasped_share=sum(r["aim_if_ungrasped"] for r in mv) / n if n else float("nan"),
                          why=dict(collections.Counter(r.get("why") for r in mv if r["cls"] == "aim")),
                          by_task={str(t): dict(collections.Counter(r["cls"] for r in sel if r["task"] == t)) for t in sorted({r["task"] for r in sel})})
    res["decision_C2"] = "survives" if res["primary"]["aim_share"] >= 0.5 else "fails"
    print(f"[p0] replays exact {res['gate']['exact']}/{res['gate']['replayed']}")
    for t, v in res["tasks"].items():
        print(f"   task {t}: {v['kind']}" + (f", object {v['object']}, receptacle {v['receptacle']}, successes {v['successes']}, success close spread "
                                             f"{v['success_close_spread_cm']:.1f} cm" if v["kind"] == "movable" and v["success_close_spread_cm"] is not None else ""))
    for scope in ("primary", "all_failures"):
        d = res[scope]
        print(f"[{scope}] {d['n']} movable-object failures (+{d['fixture']} fixture): {d['classes']}; aim share {d['aim_share']:.2f} "
              f"(aim or grasp-miss {d['aim_or_grasp_miss_share']:.2f}; aim and never grasped {d['aim_ungrasped_share']:.2f}); why {d['why']}")
        print(f"   by task {d['by_task']}")
    print(f"[p0] C2 (binding) {res['decision_C2']}")
    json.dump(dict(res, rows=rows), open(os.path.join(R, "p0_geometry.json"), "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p0_audit", bench="libero_90", tag="geometry", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p0_geometry.py", command=" ".join(sys.argv), artifacts=[os.path.join(R, "p0_geometry.json")])
        print("[p0] ledger row written")


if __name__ == "__main__":
    main()
