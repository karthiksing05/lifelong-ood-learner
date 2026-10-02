"""The acceptance test on the grammar arm (ledger m3_prereg tag q26_grammar_accept): keep the grammar arm only if it has more
successes than zero-shot on init states 0-9 (q26_accept, seed 6); score the kept arm with q24_main's episodes (states 30-49, seed 6).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q26_read.py [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def eps(exp, t, tag):
    f = os.path.join(R, f"{exp}_{t}_{tag}.json")
    return None if not os.path.exists(f) else [e for x in json.load(open(f))["tasks"].values() for e in x["episodes"]]


def main():
    from lifelong_ood.policy import bind_hooks as BH
    from lifelong_ood.policy import factors as F
    d = {t.index: t for t in F.load() if t.suite == "libero_90"}
    plan = json.load(open(os.path.join(R, "q24_plan.json")))
    tab, kept, acc, missing = {"success": {}, "bound": {}}, {}, {}, []
    for t, v in plan.items():
        g = f"bindw4_rho0.7_approach_type_qual_npk{v['npk']}_seed6"
        a_g, a_z = eps("q26_accept", t, g), eps("q26_accept", t, "zs_seed6")
        t_g, t_z = eps("q24_main", t, g), eps("q24_main", t, "zs_seed6")
        if None in (a_g, a_z, t_g, t_z):
            missing.append(t)
            continue
        acc[t] = dict(grammar=int(sum(bool(e["success"]) for e in a_g)), zs=int(sum(bool(e["success"]) for e in a_z)))
        kept[t] = "grammar" if acc[t]["grammar"] > acc[t]["zs"] else "zs"
        named = BH.referents(d[int(t)])["open"][0]
        x = t_g if kept[t] == "grammar" else t_z
        for m in tab:
            f = (lambda e: int(bool(e["success"]))) if m == "success" else (lambda e: int(e.get("bound") == named))
            tab[m][t] = {"kept": {int(e["trial"]): f(e) for e in x}, "zs": {int(e["trial"]): f(e) for e in t_z}}
    hyp = {}
    for h, m in (("H1", "success"), ("H2", "bound")):
        pairs = stats.paired(tab[m], "kept", "zs", list(tab[m]))
        d_ = stats.bootstrap_diff(pairs, B=10000)
        d_.pop("samples", None)
        b, c = stats.discordant(pairs)
        hyp[h] = dict(metric=m, x=int(sum(sum(v[0]) for v in pairs.values())), control=int(sum(sum(v[1]) for v in pairs.values())),
                      n=int(sum(len(v[0]) for v in pairs.values())), diff=d_, discordant=[int(b), int(c)], sign_p=float(stats.sign_test(b, c)))
    order = sorted(hyp, key=lambda h: hyp[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * hyp[h]["sign_p"]))
        hyp[h]["holm_p"] = run
        hyp[h]["passes"] = bool(hyp[h]["x"] > hyp[h]["control"] and run < 0.05)
    res = dict(missing=missing, acceptance=acc, kept=kept, hypotheses=hyp,
               per_task={t: [int(sum(v["kept"].values())), int(sum(v["zs"].values()))] for t, v in tab["success"].items()})
    for h, v in hyp.items():
        print(f"[q26] {h} ({v['metric']}): {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p "
              f"{v['sign_p']:.3g}, Holm {v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("[q26] kept", {t: k for t, k in kept.items() if k != "zs"}, "acceptance", acc, flush=True)
    json.dump(res, open(os.path.join(R, "q26_read.json"), "w"), indent=1)
    if "--record" in sys.argv and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q26_grammar_accept", bench="libero_90", tag="read", seed=6, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config={}, script="experiments/q26_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
