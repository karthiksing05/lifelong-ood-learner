"""The grammar read (ledger m3_prereg tag q24_grammar): the instance the qualifier names, from a type prototype's peaks, against
zero-shot on init states 30-49 (noise seed 6).  H1: the named object is moved first; H2: success; Holm over both.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q24_read.py [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def main():
    from lifelong_ood.policy import bind_hooks as BH
    from lifelong_ood.policy import factors as F
    d = {t.index: t for t in F.load() if t.suite == "libero_90"}
    plan = json.load(open(os.path.join(R, "q24_plan.json")))
    tab = {"bound": {}, "success": {}}
    missing, per = [], {}
    for t, v in plan.items():
        named = BH.referents(d[int(t)])["open"][0]
        arms = {"grammar": f"bindw4_rho0.7_approach_type_qual_npk{v['npk']}_seed6", "zs": "zs_seed6"}
        eps = {}
        for k, tg in arms.items():
            f = os.path.join(R, f"q24_main_{t}_{tg}.json")
            if not os.path.exists(f):
                missing.append(f"{t}:{k}")
                continue
            eps[k] = [e for x in json.load(open(f))["tasks"].values() for e in x["episodes"]]
        if len(eps) < 2:
            continue
        for m in ("bound", "success"):
            tab[m][t] = {k: {int(e["trial"]): int((e.get("bound") == named) if m == "bound" else bool(e["success"])) for e in es}
                         for k, es in eps.items()}
        per[t] = dict(word=v["word"], **{f"{m}_{k}": int(sum(tab[m][t][k].values())) for m in tab for k in ("grammar", "zs")})
    hyp = {}
    for h, m in (("H1", "bound"), ("H2", "success")):
        pairs = stats.paired(tab[m], "grammar", "zs", list(tab[m]))
        d_ = stats.bootstrap_diff(pairs, B=10000)
        d_.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(metric=m, x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d_, discordant=[int(b), int(c)],
                      sign_p=float(stats.sign_test(b, c)))
    order = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run < 0.05)
    res = dict(missing=missing, per_task=per, hypotheses=hyp)
    for h, v in hyp.items():
        print(f"[q24] {h} ({v['metric']}): {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p "
              f"{v['sign_p']:.3g}, Holm {v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    for t, v in per.items():
        print(f"[q24]   {t:>3} {v['word']:6s} bound {v['bound_grammar']} vs {v['bound_zs']}  success {v['success_grammar']} vs "
              f"{v['success_zs']}", flush=True)
    json.dump(res, open(os.path.join(R, "q24_read.json"), "w"), indent=1)
    if "--record" in sys.argv and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q24_grammar", bench="libero_90", tag="read", seed=6, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config={}, script="experiments/q24_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
