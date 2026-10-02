"""The acceptance-test read (ledger m3_prereg tag q13_main): per task, keep the arm (policy alone, binding form, folded token) with
the most successes on the acceptance states 0-9; score the kept arm, the token and the form against the policy on init states 30-49,
noise seed 1.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q13_read.py [--record]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TOKEN = {t: "ftoken_self_q102" for t in (12, 2, 13, 14, 79, 15, 70)} | {43: "ftoken_self_w2", 52: "ftoken_self_w2"}
ORDER = ("zs", "token", "form")                            # the tie order: the policy, then the token, then the form
BYTES = {"zs": 0, "token": 18432, "form": 4096}


def tags(t):
    return {"zs": "zs_seed1", "form": "bindw4_rho0.7_approach_seed1", "token": TOKEN[t] + "_seed1"}


def load(exp, t, tag):
    f = os.path.join(R, f"{exp}_{t}_{tag}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def main():
    tab, accept, kept, missing = {}, {}, {}, []
    for t in TOKEN:
        acc = {k: load("q13_accept", t, tg) for k, tg in tags(t).items()}
        test = {k: load("q13_main", t, tg) for k, tg in tags(t).items()}
        missing += [f"accept {t}:{k}" for k, v in acc.items() if v is None] + [f"main {t}:{k}" for k, v in test.items() if v is None]
        if any(v is None for v in acc.values()) or any(v is None for v in test.values()):
            continue
        accept[str(t)] = {k: int(sum(v.values())) for k, v in acc.items()}
        kept[str(t)] = max(ORDER, key=lambda k: (accept[str(t)][k], -ORDER.index(k)))
        tab[str(t)] = dict(test, kept=test[kept[str(t)]])
    res = dict(missing=missing, acceptance=accept, kept=kept,
               per_task={t: {k: int(sum(v.values())) for k, v in r.items()} for t, r in tab.items()})
    hyp = {}
    for h, x in (("H1", "kept"), ("H2", "token"), ("H3", "form")):
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
    res["kept_bytes"] = int(sum(BYTES[k] for k in kept.values()))
    for h, v in hyp.items():
        print(f"[q13] {h}: {v['x']} vs {v['control']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}, Holm "
              f"{v['holm_p']:.3g} -> passes {v['passes']}", flush=True)
    print("[q13] acceptance", accept, flush=True)
    print("[q13] kept", kept, "per task", res["per_task"], ("missing " + str(missing)) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "q13_read.json"), "w"), indent=1)
    if "--record" in sys.argv and not missing:
        from lifelong_ood import ledger
        print(ledger.record("q13_main", bench="libero_90", tag="read", seed=1, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=res["kept_bytes"], samples=0, teacher_copies=0, total=res["kept_bytes"]),
                            config={}, script="experiments/q13_read.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
