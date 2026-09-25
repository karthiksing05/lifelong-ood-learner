"""Tables for the c-series, straight from the result files: mean +- SE over seeds.

    python experiments/c_report.py c1 results/c1_cobweb_update_colormnist_FINAL__seed*.json
    python experiments/c_report.py c2 results/c2_router_loop_colormnist_FINAL__seed*.json

C1: per arm, the LAST task's taught / untaught / old-class numbers, and the mean over tasks of the
untaught joint (how early the value became addressable).  C2: per population, the tiers the router
chose, what was stored, its accuracy at storage and at the end of the stream, and the cost.
No number here is typed by hand; the documents quote this output.
"""
import json
import sys

import numpy as np


def mse(x):
    x = np.asarray([v for v in x if v == v], float)
    if not len(x):
        return "   -   "
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{x.mean():.3f}±{se:.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def c1(files):
    runs = [json.load(open(f)) for f in files]
    arms = []
    for r in runs:
        for row in r["rows"]:
            if row["arm"] not in arms:
                arms.append(row["arm"])
    nf = runs[0].get("new_factor", "new")
    print(f"C1 {runs[0].get('bench', '?')} -- {len(runs)} seed(s); new value on '{nf}'; "
          f"last task unless noted\n")
    hdr = (f"{'arm':30s} {'taught':>12s} {'t-new':>12s} {'untaught':>12s} {'u-new':>12s} "
           f"{'untaught(mean)':>14s} {'old drop':>12s} {'drift %':>10s} {'changed %':>10s} "
           f"{'added B':>9s} {'steps':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for a in arms:
        last = [max((x for x in r["rows"] if x["arm"] == a), key=lambda x: x["task"])
                for r in runs if any(x["arm"] == a for x in r["rows"])]
        allt = [[x for x in r["rows"] if x["arm"] == a] for r in runs]
        um = [np.mean([x["untaught_joint"] for x in rs]) for rs in allt if rs]
        steps = [np.mean([x["steps_weight"] for x in rs]) for rs in allt if rs]
        tn = [x.get("taught_new", x.get("taught_bg")) for x in last]
        un = [x.get("untaught_new", x.get("untaught_bg")) for x in last]
        added = [4 * x["lora_params"] if not x.get("consolidation") else 0 for x in last]
        print(f"{a[:30]:30s} {mse([x['taught_joint'] for x in last]):>12s} {mse(tn):>12s} "
              f"{mse([x['untaught_joint'] for x in last]):>12s} {mse(un):>12s} {mse(um):>14s} "
              f"{mse([x['old_drop'] for x in last]):>12s} "
              f"{mse([100 * x['drift_mean'] for x in last]):>10s} "
              f"{mse([100 * x['changed_frac'] for x in last]):>10s} {int(np.mean(added)):>9d} "
              f"{np.mean(steps):>6.0f}")


def c2(files):
    runs = [json.load(open(f)) for f in files]
    print(f"C2 -- {len(runs)} seed(s)\n")
    for pop in ("in-dist", "new-comb", "new-value"):
        tiers, joint, jend, cost, ok = {}, [], [], [], []
        for r in runs:
            rs = [x for x in r["records"] if x["pop"] == pop]
            for x in rs:
                tiers[x["tier"]] = tiers.get(x["tier"], 0) + 1
            if rs:
                joint.append(np.mean([x["joint"] for x in rs]))
                jend.append(np.mean([x["joint_end"] for x in rs]))
                cost.append(np.sum([x["cost_bytes"] for x in rs]) / len(rs))
                ok.append(np.mean([x.get("stored_ok", True) for x in rs]))
        print(f"{pop:10s} tiers {tiers} | joint at storage {mse(joint)} end {mse(jend)} | "
              f"stored as the right tuple {mse(ok)} | bytes/concept {np.mean(cost):.0f}")
    print()
    for key in ("old_joint_start", "old_joint_end", "changed_frac", "bytes_total"):
        print(f"{key:18s} {mse([r['summary'][key] for r in runs])}")
    # tier of each new-value concept by its position among new-value arrivals: does it get cheaper?
    pos = {}
    for r in runs:
        k = 0
        for x in r["records"]:
            if x["pop"] == "new-value":
                pos.setdefault(k, []).append(x["tier"])
                k += 1
    print("\nnew-value concepts by arrival order (tiers over seeds):")
    for k in sorted(pos):
        c = {}
        for t in pos[k]:
            c[t] = c.get(t, 0) + 1
        print(f"  #{k + 1}: {c}")


if __name__ == "__main__":
    kind, files = sys.argv[1], sys.argv[2:]
    {"c1": c1, "c2": c2}[kind](files)
