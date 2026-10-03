"""The fresh-state benchmark (ledger m3_prereg tag f1_fresh): the scheme and its baselines on 20 never-used start states per task
(results/robot/fresh_states), noise seed 9, experiment f1_main.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/f1_fresh.py --launch Q27     # zs, lora, q27 kept arm
    ~/flash/miniconda3/envs/chunks/bin/python experiments/f1_fresh.py --launch C2      # zs, lora, bcl joint/seq/er, c2 kept, c2 final
    ~/flash/miniconda3/envs/chunks/bin/python experiments/f1_fresh.py --read [--record]
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import stats                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED = 9


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def c2_arms():
    st = json.load(open(os.path.join(R, "c2_state.json")))
    out = {}
    for k, s in st.items():
        if k.startswith("_"):
            continue
        final = f"prims@{s['prim']}.0" if s["prim"] else "zs"
        for a in s["arrivals"]:
            out[str(a["task"])] = dict(key=k, kept=a["kept"], final=final, demo=a["demo"],
                                       bcl={m: f"bcl@{m}_{k}" for m in ("joint", "seq", "er")})
    return out, st


def q27_arms():
    return {t: a for t, a in json.load(open(os.path.join(R, "q27_kept.json")))["kept"].items()}


def launch(which):
    per = {}
    if which == "Q27":
        for t, a in q27_arms().items():
            per[t] = sorted({"zs", "lora", a})
    else:
        arms, _ = c2_arms()
        for t, v in arms.items():
            per[t] = sorted({"zs", "lora", v["kept"], v["final"], *v["bcl"].values()})
    jobs = []
    for t, arms in per.items():
        out = subprocess.run([PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms), "--experiment", "f1_main",
                              "--first", "0", "--trials", "20", "--seed", str(SEED), "--fresh", "--bind", "track", "--partition",
                              "overcap", "--per", "2"], capture_output=True, text=True, cwd=ROOT).stdout
        print(out.strip()[-300:], flush=True)


def load(t, a):
    f = os.path.join(R, f"f1_main_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return {int(e["trial"]): int(bool(e["success"])) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]}


def compare(pairs_of, label):
    tab = {}
    for t, (x, y) in pairs_of.items():
        if x is not None and y is not None:
            tab[t] = {"x": x, "y": y}
    pairs = stats.paired(tab, "x", "y", list(tab))
    if not pairs:
        return None
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    return dict(label=label, x=int(sum(sum(v[0]) for v in pairs.values())), y=int(sum(sum(v[1]) for v in pairs.values())),
                n=int(sum(len(v[0]) for v in pairs.values())), tasks=len(pairs), diff=d, discordant=[int(b), int(c)],
                sign_p=float(stats.sign_test(b, c)))


def read(record):
    res, missing = {}, []
    c2, st = c2_arms()
    q27 = q27_arms()
    L = lambda t, a: load(t, a)                                         # noqa: E731
    for t, v in c2.items():
        for a in ("zs", "lora", v["kept"], v["final"], *v["bcl"].values()):
            if load(t, a) is None:
                missing.append(f"C2 {t}:{a}")
    for t, a in q27.items():
        for x in ("zs", "lora", a):
            if load(t, x) is None:
                missing.append(f"Q27 {t}:{x}")
    hyp = {"H1": compare({t: (L(t, v["kept"]), L(t, "zs")) for t, v in c2.items()}, "C2 kept vs zs"),
           "H2": compare({t: (L(t, a), L(t, "zs")) for t, a in q27.items()}, "Q27 kept vs zs")}
    ok = {h: v for h, v in hyp.items() if v}
    order = sorted(ok, key=lambda h: ok[h]["sign_p"])
    run = 0.0
    for r_, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - r_) * ok[h]["sign_p"]))
        ok[h]["holm_p"] = run
        ok[h]["passes"] = bool(ok[h]["x"] > ok[h]["y"] and run < 0.05)
    desc = {}
    for name, f in (("c2_vs_lora", lambda t, v: (L(t, v["kept"]), L(t, "lora"))),
                    ("c2_vs_joint", lambda t, v: (L(t, v["kept"]), L(t, v["bcl"]["joint"]))),
                    ("c2_vs_seq", lambda t, v: (L(t, v["kept"]), L(t, v["bcl"]["seq"]))),
                    ("c2_vs_er", lambda t, v: (L(t, v["kept"]), L(t, v["bcl"]["er"]))),
                    ("c2_final_vs_zs", lambda t, v: (L(t, v["final"]), L(t, "zs"))),
                    ("lora_vs_zs", lambda t, v: (L(t, "lora"), L(t, "zs"))),
                    ("joint_vs_zs", lambda t, v: (L(t, v["bcl"]["joint"]), L(t, "zs"))),
                    ("seq_vs_lora", lambda t, v: (L(t, v["bcl"]["seq"]), L(t, "lora"))),
                    ("er_vs_zs", lambda t, v: (L(t, v["bcl"]["er"]), L(t, "zs")))):
        desc[name] = compare({t: f(t, v) for t, v in c2.items()}, name)
    desc["q27_vs_lora"] = compare({t: (L(t, a), L(t, "lora")) for t, a in q27.items()}, "q27_vs_lora")
    demos = {k: len(s["library"]) for k, s in st.items() if not k.startswith("_")}
    practice = sum(10 * (1 + (a.get("prims1") is not None) + (a.get("prims2") is not None))
                   for k, s in st.items() if not k.startswith("_") for a in s["arrivals"])
    res = dict(missing=missing, hypotheses=hyp, descriptive=desc,
               costs=dict(c2_demos=demos, c2_demos_total=sum(demos.values()), c2_arrivals=len(c2), c2_practice_episodes=practice,
                          baseline_demos=len(c2)))
    for h, v in hyp.items():
        if v:
            print(f"[f1] {h} {v['label']}: {v['x']} vs {v['y']} of {v['n']}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}"
                  + (f", Holm {v['holm_p']:.3g} -> passes {v['passes']}" if "holm_p" in v else ""), flush=True)
    for k, v in desc.items():
        if v:
            print(f"[f1]   {k}: {v['x']} vs {v['y']} of {v['n']} ({v['tasks']} tasks), discordant {v['discordant']}, p {v['sign_p']:.3g}",
                  flush=True)
    print("[f1] costs", res["costs"], ("missing " + str(len(missing))) if missing else "", flush=True)
    json.dump(res, open(os.path.join(R, "f1_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("f1_fresh", bench="libero_90", tag="read", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                            config={}, script="experiments/f1_fresh.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch(sys.argv[sys.argv.index("--launch") + 1])
    if "--read" in sys.argv:
        read("--record" in sys.argv)
