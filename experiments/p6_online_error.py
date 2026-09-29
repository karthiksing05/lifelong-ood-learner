"""P6 descriptive: the learned pose reader's error on the SCORED episodes (init states 30-49 of m3_final), from the positions the
phasebind server logs at each episode's first query ("[phasebind] reader episode k: ...", lifelong_ood/policy/phase.py) against
the settled evaluation placements of experiments/p0_demo_init.py (results/robot/p0_demo/<t>.npz eval_obj_pos, read after the
client's 10 settle steps, as the first query is).  A server log results/robot/server_<job>_<i>.log belongs to line i of the pack
file its job ran (slurm/<job>.out names it).  Also: success by the episode's worst object error.

    python experiments/p6_online_error.py [--record]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
TAG, EXP = "phasebind_reader", "m3_final"


def logs():
    """{task: [server log paths]} of every pack that ran TAG in EXP."""
    out = {}
    for so in glob.glob(os.path.join(ROOT, "slurm", "*.out")):
        txt = open(so, errors="replace").read()
        m = re.search(r"specs (\S+\.tsv)", txt)
        if not m or TAG not in txt or not os.path.exists(m.group(1)):
            continue
        job = os.path.basename(so)[:-4]
        for i, line in enumerate(open(m.group(1)).read().splitlines()):
            kv = dict(x.split("=", 1) for x in line.split("\t"))
            if kv.get("TAG") == TAG and kv.get("EXPERIMENT") == EXP:
                p = os.path.join(R, f"server_{job}_{i}.log")
                if os.path.exists(p):
                    out.setdefault(int(kv["TASKS"]), []).append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    res = dict(per_task={})
    all_err, all_succ = [], []
    for t, paths in sorted(logs().items()):
        pred = {}
        for p in paths:
            for m in re.finditer(r"reader episode (\d+): (\[\[.*\]\])", open(p, errors="replace").read()):
                pred[int(m.group(1))] = np.asarray(json.loads(m.group(2)))
        d = np.load(os.path.join(R, "p0_demo", f"{t}.npz"), allow_pickle=True)
        names = [str(n) for n in d["names"]]
        r = json.load(open(os.path.join(R, f"{EXP}_{t}_{TAG}.json")))
        succ = {e["trial"]: bool(e["success"]) for e in r["tasks"][str(t)]["episodes"]}
        ks = sorted(k for k in pred if 30 <= k < 50)
        err = np.stack([np.linalg.norm(pred[k] - d["eval_obj_pos"][k - 30], axis=-1) * 100 for k in ks])      # cm [episode, object]
        worst = err.max(1)
        all_err.append(err.reshape(-1)); all_succ += [(float(w), succ[k]) for w, k in zip(worst, ks)]
        res["per_task"][str(t)] = dict(episodes=len(ks), err_cm_median={n: float(np.median(err[:, j])) for j, n in enumerate(names)},
                                       err_cm_max={n: float(err[:, j].max()) for j, n in enumerate(names)},
                                       successes=int(sum(succ[k] for k in ks)))
        print(f"[p6-online] task {t}: " + ", ".join(f"{n} {np.median(err[:, j]):.2f} cm (max {err[:, j].max():.2f})" for j, n in enumerate(names))
              + f"; {sum(succ[k] for k in ks)}/{len(ks)} successes", flush=True)
    e = np.concatenate(all_err)
    w = np.array([x for x, _ in all_succ]); s = np.array([y for _, y in all_succ])
    res["median_cm"] = float(np.median(e))
    res["p90_cm"] = float(np.percentile(e, 90))
    res["success_by_worst_error"] = {f"<{c} cm": dict(n=int((w < c).sum()), k=int(s[w < c].sum())) for c in (1, 2)}
    res["success_by_worst_error"][">=2 cm"] = dict(n=int((w >= 2).sum()), k=int(s[w >= 2].sum()))
    print(f"[p6-online] median {res['median_cm']:.2f} cm, 90th percentile {res['p90_cm']:.2f} cm; success by the episode's worst error: "
          f"{res['success_by_worst_error']}")
    json.dump(res, open(os.path.join(R, "p6_online_error.json"), "w"), indent=1)
    if a.record:
        from lifelong_ood import ledger
        ledger.record("p6_reader", bench="libero_90", tag="online", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/p6_online_error.py", command=" ".join(sys.argv), artifacts=[os.path.join(R, "p6_online_error.json")])


if __name__ == "__main__":
    main()
