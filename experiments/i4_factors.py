"""INTERPRETATION 4: what plan E's two factors are, in weight space (exploratory, offline).  The factored library holds an OBJECT
primitive per object key (approach frames) and a GOAL primitive per goal key (transport frames); each is a low-rank update
dW = B A on each Linear of the action expert, compared exactly in rank space (experiments/i2_primitives.py).  Reports
  where      -- each factor's share of its squared update norm by layer (0-17) and by module (attention q/k/v/o, MLP gate/up/down);
  similarity -- cosine within object primitives, within goal primitives, and between the two factors;
  stability  -- an object prototype refit without one task's demonstration (E0 leave-one-out, Oloo<t>) against the prototype with
                every demonstration (E0 all-in, Oall<i>): is a prototype a summary of its demonstrations, or one demonstration?
Figure: figures/robot/i4_factors.png.

    sbatch --export=ALL,SCRIPT="experiments/i4_factors.py --record" slurm/jobs/cpu.sbatch
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load(path):
    import torch
    z = torch.load(path, map_location="cpu")
    return {n: (w["A"].float().numpy(), w["B"].float().numpy()) for n, w in z["banks"].items()}


def inner(p, q):
    return float(sum(np.sum((B1.T @ q[n][1]) * (A1 @ q[n][0].T)) for n, (A1, B1) in p.items() if n in q))


def where(p):
    sq = {n: max(float(np.sum((B.T @ B) * (A @ A.T))), 0.0) for n, (A, B) in p.items()}
    tot = sum(sq.values())
    lay, mod = np.zeros(18), {m: 0.0 for m in MODULES}
    for n, v in sq.items():
        m = re.search(r"layers\.(\d+)\.", n)
        if m:
            lay[int(m.group(1))] += v / tot
        for mm in MODULES:
            if n.endswith(mm):
                mod[mm] += v / tot
    return lay, mod


def cos(a, b, sq):
    return inner(a[1], b[1]) / np.sqrt(sq[a[0]] * sq[b[0]])


def main():
    e1 = json.load(open(os.path.join(R, "e1_plan.json")))
    e0 = json.load(open(os.path.join(R, "e0_plan.json")))
    objs = {v["obj_name"]: v["object_key"] for v in e1.values()}
    goals = {v["goal_name"]: v["goal_key"] for v in e1.values()}
    P = {n: load(os.path.join(R, f"prim_{n}_0.pt")) for n in list(objs) + list(goals)}
    sq = {n: inner(p, p) for n, p in P.items()}
    on, gn = sorted(objs), sorted(goals)
    pair = lambda xs, ys, same: [cos((a, P[a]), (b, P[b]), sq) for i, a in enumerate(xs)        # noqa: E731
                                 for b in (xs[i + 1:] if same else ys)]
    sim = dict(object_object=float(np.mean(pair(on, on, True))), goal_goal=float(np.mean(pair(gn, gn, True))),
               object_goal=float(np.mean(pair(on, gn, False))))
    W = {n: where(p) for n, p in P.items()}
    wh = {f: dict(layers=np.mean([W[n][0] for n in ns], 0).tolist(),
                  modules={m: float(np.mean([W[n][1][m] for n in ns])) for m in MODULES})
          for f, ns in (("object", on), ("goal", gn))}
    # stability: E0's leave-one-out object primitives against E0's all-in primitive of the same object, and against other objects'
    stab, other = [], []
    allin = {v["obj_all_name"]: v["object_key"] for v in e0.values()}
    A0 = {n: load(os.path.join(R, f"prim_{n}_0.pt")) for n in allin}
    sq0 = {n: inner(p, p) for n, p in A0.items()}
    for t, v in e0.items():
        if not v["loo_obj"]:
            continue
        n = v["obj_loo_name"]
        p = load(os.path.join(R, f"prim_{n}_0.pt"))
        s = inner(p, p)
        for m, q in A0.items():
            c = inner(p, q) / np.sqrt(s * sq0[m])
            (stab if m == v["obj_all_name"] else other).append(dict(task=int(t), loo=n, allin=m, cos=float(c),
                                                                     donors=len(v["loo_obj"])))
    by_donors = {}
    for x in stab:
        by_donors.setdefault(str(min(x["donors"], 3)), []).append(x["cos"])
    res = dict(similarity=sim, where=wh, stability=dict(same_object=float(np.mean([x["cos"] for x in stab])),
                                                         other_objects=float(np.mean([x["cos"] for x in other])),
                                                         by_donors={k: float(np.mean(v)) for k, v in sorted(by_donors.items())},
                                                         n_same=len(stab), rows=stab),
               objects=objs, goals=goals)
    late = lambda f: float(sum(wh[f]["layers"][9:]))                                              # noqa: E731
    attn = lambda f: float(sum(wh[f]["modules"][m] for m in ("q_proj", "k_proj", "v_proj", "o_proj")))   # noqa: E731
    res["summary"] = {f: dict(late_layers=late(f), attention=attn(f)) for f in ("object", "goal")}
    print(json.dumps({k: res[k] for k in ("similarity", "summary")}, indent=1), flush=True)
    print(json.dumps({k: v for k, v in res["stability"].items() if k != "rows"}, indent=1), flush=True)
    figure(res)
    json.dump(res, open(os.path.join(R, "i4_factors.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("i4_factors", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i4_factors.py", command=" ".join(sys.argv)))


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"object": "#2a78d6", "goal": "#eb6834"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for f in ("object", "goal"):
        ax[0].plot(range(18), res["where"][f]["layers"], marker="o", ms=3, color=col[f], label=f"{f} primitives")
    ax[0].set_xlabel("action-expert layer")
    ax[0].set_ylabel("share of update norm$^2$")
    ax[0].set_xticks(range(0, 18, 2))
    ax[0].legend(frameon=False)
    x = np.arange(len(MODULES))
    for k, f in enumerate(("object", "goal")):
        ax[1].bar(x + (k - 0.5) * 0.38, [res["where"][f]["modules"][m] for m in MODULES], 0.38, color=col[f], label=f)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([m.replace("_proj", "") for m in MODULES])
    ax[1].set_ylabel("share of update norm$^2$")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures", "robot"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "robot", "i4_factors.png"), dpi=160)


if __name__ == "__main__":
    main()
