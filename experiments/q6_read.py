"""The deployable scheme's read (ledger m3_prereg tag q6_main): zero-shot vs the form (attention binding with the demo's object
prototype) vs the token alone, init states 30-49, tasks 43 and 52.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q6_read.py [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TASKS = [43, 52]
ARMS = {"zs": "zs", "form": "bindw4_rho0.7_approach", "token": "ftoken_self_w2"}


def main():
    tab, missing = {}, []
    for t in TASKS:
        for k, tag in ARMS.items():
            f = os.path.join(R, f"q6_main_{t}_{tag}.json")
            if not os.path.exists(f):
                missing.append(f"{t}:{k}")
                continue
            tab.setdefault(str(t), {})[k] = {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values()
                                             for e in v["episodes"]}
    res = dict(missing=missing, per_task={t: {k: int(sum(v.values())) for k, v in r.items()} for t, r in tab.items()})
    hyp = {}
    for h, x in (("H1", "token"), ("H2", "form")):
        pairs = stats.paired(tab, x, "zs", list(tab))
        if not pairs:
            continue
        d = stats.bootstrap_diff(pairs, B=10000)
        d.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    order = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run < 0.05)
    res["hypotheses"] = hyp
    for h, v in hyp.items():
        print(f"[q6] {h}: {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}, Holm "
              f"{v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("[q6] per task", res["per_task"], ("missing " + str(missing)) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "q6_read.json"), "w"), indent=1)
    if "--record" in sys.argv and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q6_main", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=2 * 18432, tree=0, codes=2 * 4096, samples=0, teacher_copies=0, total=2 * (18432 + 4096)),
                            config={}, script="experiments/q6_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
