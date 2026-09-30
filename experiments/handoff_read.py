"""Item 4 dev read (EXPLORATORY; states 0-9, the practice states): does letting the frozen prior CARRY the object between the grasp
binding and the place binding (PhaseBindHooks handoff=1; arm streamh) rescue compositions whose place source was recorded in another
object-receptacle layout (the diagnosed S2 failure)?

Paired per composition (task, grasp source g, place source p) and init state against the guided carry (streamc, experiment dev_p5).
Layout mismatch = the horizontal distance between where the target's object starts relative to its receptacle and where the place
source's object started its carry relative to its receptacle (both in the receptacle's frame; the demos' at-rest poses).

    python experiments/handoff_read.py [--experiment dev_handoff] [--json] [--record]
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                                          # noqa: E402
from lifelong_ood.policy import phase as PH                             # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def episodes(f):
    e = json.load(open(f))
    t = next(iter(e["tasks"].values()))["episodes"]
    return {int(x["trial"]): int(x["success"]) for x in t}


def mismatch(t, g, p, pool):
    """Horizontal distance (cm) between the target's object-in-receptacle start and the place source's carry start."""
    v = pool[str(t)]
    zt = PH.at_rest(PH.load_demo(R, t))
    nt = [str(x) for x in zt["names"]]
    if v["object"] not in nt or v["receptacle"] not in nt:
        return None
    i, j = nt.index(v["object"]), nt.index(v["receptacle"])
    rel_t = PH.q2m(zt["obj_quat"][0, j]).T @ (zt["obj_pos"][0, i] - zt["obj_pos"][0, j])
    zp = PH.at_rest(PH.load_demo(R, p))
    parts = PH.cut(zp)
    if len(parts) < 2 or parts[1][7] is None:
        return None
    s0, ref, held = parts[1][3], parts[1][6], parts[1][7]
    rel_p = PH.q2m(zp["obj_quat"][s0, ref]).T @ (zp["obj_pos"][s0, held] - zp["obj_pos"][s0, ref])
    return float(np.linalg.norm((rel_t - rel_p)[:2]) * 100)


def main():
    exp = sys.argv[sys.argv.index("--experiment") + 1] if "--experiment" in sys.argv else "dev_handoff"
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    table, rows = {}, []
    for f in sorted(glob.glob(os.path.join(R, f"{exp}_*_streamh_*_*.json"))):
        t, g, p = map(int, re.match(rf".*{exp}_(\d+)_streamh_(\d+)_(\d+)\.json", f).groups())
        c = os.path.join(R, f"dev_p5_{t}_streamc_{g}_{p}.json")
        if not os.path.exists(c):
            continue
        key = f"{t}_{g}_{p}"
        table[key] = {"streamh": episodes(f), "streamc": episodes(c)}
        m = mismatch(t, g, p, pool)
        rows.append(dict(key=key, task=t, g=g, p=p, streamh=sum(table[key]["streamh"].values()), streamc=sum(table[key]["streamc"].values()),
                         n=len(table[key]["streamh"]), mismatch_cm=m))
    keys = list(table)
    pairs = stats.paired(table, "streamh", "streamc", keys)
    b, c = stats.discordant(pairs)
    d = stats.bootstrap_diff(pairs)
    tot = stats.pooled(table, ["streamh", "streamc"], keys)
    res = dict(experiment=exp, compositions=len(keys), streamh=tot["streamh"], streamc=tot["streamc"], diff=d, discordant=[b, c],
               sign_p=stats.sign_test(b, c), rows=rows)
    for thr in (0.8,):
        res[f"pass_{thr}"] = dict(streamh=sum(r["streamh"] >= thr * r["n"] for r in rows), streamc=sum(r["streamc"] >= thr * r["n"] for r in rows))
    ms = [r for r in rows if r["mismatch_cm"] is not None]
    for lo, hi in ((0, 10), (10, 30), (30, 1e9)):
        sel = [r for r in ms if lo <= r["mismatch_cm"] < hi]
        res[f"mismatch_{lo}_{int(min(hi, 999))}"] = dict(compositions=len(sel), streamh=sum(r["streamh"] for r in sel),
                                                          streamc=sum(r["streamc"] for r in sel), n=sum(r["n"] for r in sel))
    for r in sorted(rows, key=lambda r: (r["task"], r["g"], r["p"])):
        print(f"{r['key']:>10s}  mismatch {r['mismatch_cm'] if r['mismatch_cm'] is None else round(r['mismatch_cm'], 1)!s:>6} cm  "
              f"streamc {r['streamc']:2d}  streamh {r['streamh']:2d} / {r['n']}")
    print(f"{len(keys)} compositions: streamh {tot['streamh'][0]}/{tot['streamh'][1]} vs streamc {tot['streamc'][0]}/{tot['streamc'][1]}; "
          f"diff {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], discordant {b}/{c}, sign p {res['sign_p']:.2g}")
    print("passing practice (>= 8/10): " + json.dumps(res["pass_0.8"]))
    for k in [k for k in res if k.startswith("mismatch_")]:
        v = res[k]
        print(f"{k}: {v['compositions']} compositions, streamc {v['streamc']}/{v['n']}, streamh {v['streamh']}/{v['n']}")
    po = os.path.join(R, "phase_outcomes_streamh_dev.json")                 # experiments/phase_outcomes.py --json (the light recordings)
    if os.path.exists(po):
        res["phase_outcomes_streamh"] = json.load(open(po))["total"]
    if "--json" in sys.argv:
        json.dump(res, open(os.path.join(R, f"{exp}_read.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("p5_stream", bench="libero_90", tag=exp, seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/handoff_read.py", command=" ".join(sys.argv) + "  # EXPLORATORY dev read, states 0-9"))


if __name__ == "__main__":
    main()
