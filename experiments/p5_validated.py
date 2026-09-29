"""P5 amendment 2 (EXPLORATORY; ledger m3_prereg tag p5_stream_amend2): the stream with VALIDATED composition.

An arriving task is composed from the table's first sources only if that composed program succeeds on >= 0.5 of the practice
placements in simulation with no policy (experiments/p38_generate.py --combos: results/robot/g29_generate.json); otherwise it requests
its demo, whose keys enter the table when its own program passes the same check (amendment 1).  Scored from the existing runs on init
states 30-49 (composed: streamc_<g>_<p>; demo: phasebind2_rjpred).

    python experiments/p5_validated.py plan          # -> results/robot/p5_stream_plan_validated.json; lists compositions not yet run
    ~/flash/miniconda3/envs/chunks/bin/python experiments/p5_validated.py read [--record]
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")


def load(f):
    return json.load(open(os.path.join(R, f)))


def plan():
    pool = load("p5_pool.json")
    demo_ok = {int(t): v["k"] / v["n"] >= 0.5 for f in ("g27_generate.json", "g28_generate.json") for t, v in load(f).items()}
    comp_ok = {tuple(map(int, k.split("_"))): v["k"] / v["n"] >= 0.5 for k, v in load("g29_generate.json").items() if k.count("_") == 2}
    a = sorted(int(t) for t in pool)
    random.Random(0).shuffle(a)
    out = {}
    for name, order in {"A": a, "B": a[::-1]}.items():
        grasp, place, steps = {}, {}, []
        for t in order:
            info = pool[str(t)]
            if info["composable"] and info["object_type"] in grasp and info["goal_key"] in place:
                g, p = grasp[info["object_type"]], place[info["goal_key"]]
                if comp_ok.get((t, g, p), False):
                    steps.append(dict(task=t, action="compose", g=g, p=p))
                    continue
                steps.append(dict(task=t, action="demo", rejected=[g, p]))
            else:
                steps.append(dict(task=t, action="demo"))
            if info["composable"] and demo_ok.get(t, False):
                grasp.setdefault(info["object_type"], t)
                place.setdefault(info["goal_key"], t)
        demos = sum(s["action"] == "demo" for s in steps)
        half = len(order) // 2
        out[name] = dict(order=order, steps=steps, demos=demos, composed=len(order) - demos,
                         rejected=sum("rejected" in s for s in steps),
                         composed_first_half=sum(s["action"] == "compose" for s in steps[:half]),
                         composed_second_half=sum(s["action"] == "compose" for s in steps[half:]))
        print(f"order {name}: {demos} demos ({out[name]['rejected']} compositions rejected by the check); composed "
              f"{out[name]['composed_first_half']} + {out[name]['composed_second_half']}")
    json.dump(out, open(os.path.join(R, "p5_stream_plan_validated.json"), "w"), indent=1)
    need = sorted({(s["task"], s["g"], s["p"]) for o in out.values() for s in o["steps"] if s["action"] == "compose"
                   and not os.path.exists(os.path.join(R, f"m3_final_{s['task']}_streamc_{s['g']}_{s['p']}.json"))})
    print("compositions to run before the read:", need)
    return out


def read(record=False):
    import importlib.util
    from lifelong_ood import ledger, stats
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    P = load("p5_stream_plan_validated.json")
    own = lambda t: fin[(str(t), "phasebind2_rjpred")]
    res, tab = {"orders": {}}, {}
    for o, pl in P.items():
        ks = kall = n = 0
        for s in pl["steps"]:
            e_own = own(s["task"])
            e = fin[(str(s["task"]), f"streamc_{s['g']}_{s['p']}")] if s["action"] == "compose" else e_own
            ks += sum(e.values()); kall += sum(e_own.values()); n += len(e)
            if s["action"] == "compose":
                tab[f"{o}:{s['task']}"] = {"c": e, "own": e_own}
        res["orders"][o] = dict(stream=int(ks), all_demo=int(kall), n=int(n), demos=pl["demos"], rejected=pl["rejected"],
                                composed_first_half=pl["composed_first_half"], composed_second_half=pl["composed_second_half"])
        print(f"order {o}: stream {ks}/{n} with {pl['demos']} demos ({pl['rejected']} compositions rejected); every task its own demo {kall}/{n}")
    pairs = stats.paired(tab, "c", "own", list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    kc = sum(sum(v[0]) for v in pairs.values()); ko = sum(sum(v[1]) for v in pairs.values()); nn = sum(len(v[0]) for v in pairs.values())
    res["composed_vs_own"] = dict(x=int(kc), control=int(ko), n=int(nn), diff=d, discordant=[b, c], sign_p=stats.sign_test(b, c),
                                  non_inferior=bool(d["lo90"] >= -0.10))
    v = res["composed_vs_own"]
    print(f"composed arrivals vs their own demos: {kc} vs {ko} of {nn}, diff {d['diff']:+.3f}, lower bound {d['lo90']:+.3f}, "
          f"sign p {v['sign_p']:.3g} -> non-inferior {v['non_inferior']}")
    if record:
        print(ledger.record("p5_stream", bench="libero_90", tag="validated", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/p5_validated.py", command=" ".join(sys.argv)))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan()
    else:
        read("--record" in sys.argv)
