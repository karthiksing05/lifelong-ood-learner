"""The family-transfer read (pre-registered: ledger m3_prereg tag family), init states 30-49.

    F1 (primary)   r20fam vs r20rnd over the 12 targets: exact two-sided sign test, pooled paired difference (two-level bootstrap)
    F2             the same within the 7-task pick-and-place family
    F3             each pooled arm vs the one-demo LoRA (m3_final tag lora)
    descriptive    vs the target's own practice learners (lora_r16L2, lora_r17bal)

    python experiments/r20_family_stats.py [--record]
"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger, stats                                 # noqa: E402

R = os.path.join(ROOT, "results", "robot")
ARMS = {"fam": "lora_r20fam", "rnd": "lora_r20rnd", "onedemo": "lora", "L2": "lora_r16L2", "nft": "lora_r17bal"}
PAIRS = [("fam", "rnd"), ("fam", "onedemo"), ("rnd", "onedemo"), ("fam", "L2"), ("fam", "nft")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--prefix", default="r20", help="r20: pooled training (tag family); r21: two-stage (tag family2)")
    a = ap.parse_args()
    ARMS["fam"], ARMS["rnd"] = f"lora_{a.prefix}fam", f"lora_{a.prefix}rnd"
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    fams = json.load(open(os.path.join(R, "r20_families.json")))["families"]["0.4"][:3]
    scopes = {"all_targets": sorted(x for f in fams for x in f), "pick_place_family": sorted(fams[0])}
    res = dict(families=fams, arms=ARMS)
    for scope, tasks in scopes.items():
        table = {str(t): {k: fin[(str(t), tag)] for k, tag in ARMS.items() if (str(t), tag) in fin} for t in tasks}
        out = {}
        for k in ARMS:
            ts = [t for t in table if k in table[t]]
            kk = sum(sum(table[t][k].values()) for t in ts)
            n = sum(len(table[t][k]) for t in ts)
            out[k] = dict(k=int(kk), n=int(n), tasks=len(ts), rate=kk / n if n else float("nan"))
        for x, y in PAIRS:
            pairs = stats.paired(table, x, y, list(table))
            if pairs:
                d = stats.bootstrap_diff(pairs, B=a.B)
                b_, c_ = stats.discordant(pairs)
                d.pop("samples", None)
                d.update(discordant=[b_, c_], sign_p=stats.sign_test(b_, c_), n=int(sum(len(v[0]) for v in pairs.values())))
                out[f"{x}-{y}"] = d
        out["per_task"] = {t: {k: int(sum(v.values())) for k, v in table[t].items()} for t in table}
        # m3_prereg tag family2: T1 (fam - onedemo) and T2 (fam - rnd), Holm-adjusted together
        ps = sorted((out[k]["sign_p"], k) for k in ("fam-onedemo", "fam-rnd") if k in out)
        run = 0.0
        for r, (p_, k) in enumerate(ps):
            run = max(run, min(1.0, (len(ps) - r) * p_))
            out[k]["holm_p"] = run
        res[scope] = out
        print(f"[{scope}] tasks {tasks}")
        for k in ARMS:
            v = out[k]
            print(f"   {k:8s} {v['k']}/{v['n']} = {v['rate']:.3f} over {v['tasks']} tasks")
        for x, y in PAIRS:
            d = out.get(f"{x}-{y}")
            if d:
                print(f"   {x}-{y}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] over {d['n']} pairs, discordant {d['discordant']}, "
                      f"sign p {d['sign_p']:.4f}" + (f" (Holm {d['holm_p']:.4f})" if "holm_p" in d else ""))
        print(f"   per task: {out['per_task']}")
    out_json = os.path.join(R, "r20_family_stats.json" if a.prefix == "r20" else f"{a.prefix}_family_stats.json")
    json.dump(res, open(out_json, "w"), indent=1)
    if a.record:
        ledger.record("r20_family", bench="libero_90", tag="read" if a.prefix == "r20" else f"read_{a.prefix}", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=vars(a),
                      script="experiments/r20_family_stats.py", command=" ".join(sys.argv),
                      artifacts=[out_json])
        print("[r20] ledger row written")


if __name__ == "__main__":
    main()
