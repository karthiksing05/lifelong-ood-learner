"""P5 amendment 2 (EXPLORATORY; ledger m3_prereg tag p5_stream_amend2): the stream with VALIDATED composition.

An arriving task is composed from the table's first sources only if that composed program succeeds on >= 0.5 of the practice
placements in simulation with no policy (experiments/p38_generate.py --combos: results/robot/g29_generate.json); otherwise it requests
its demo, whose keys enter the table when its own program passes the same check (amendment 1).  Scored from the existing runs on init
states 30-49 (composed: streamc_<g>_<p>; demo: phasebind2_rjpred).

    python experiments/p5_validated.py plan [--multi]  # -> results/robot/p5_stream_plan_validated[_multi].json; lists compositions not yet run
    (--multi: the table keeps every validated source per key, and an arriving task takes the first source pair whose composition passes)
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


KIND_EXP = {"streamc": "dev_p5", "streamh": "dev_handoff", "streamte": "dev_carry", "streamhe": "dev_carry", "streamtle": "dev_carry"}   # kind -> its practice runs (states 0-9)


def practice_rate(kind):
    import glob
    import re
    ex = KIND_EXP[kind]
    out = {}
    for f in glob.glob(os.path.join(R, f"{ex}_*_{kind}_*_*.json")):
        t, g, p = map(int, re.match(rf".*{ex}_(\d+)_{kind}_(\d+)_(\d+)\.json", f).groups())
        e = json.load(open(f))["tasks"][str(t)]["episodes"]
        out[(t, g, p)] = sum(x["success"] for x in e) / len(e)
    return out


CARRIES = ("streamc", "streamh")


def practice_best(thr=0.8, carries=None):
    """kind best: each composition practised on states 0-9 with every carry in `carries` (guided streamc, the prior's streamh, the
    arrival-gated transport streamte, ...); it passes if the best reaches thr and executes with that one (ties: the earlier carry).
    -> {(t, g, p): kind or None} for the compositions practised with every carry"""
    carries = carries or CARRIES
    rates = [practice_rate(k) for k in carries]
    keys = set.intersection(*[set(r) for r in rates])
    out = {}
    for key in keys:
        i = max(range(len(carries)), key=lambda j: (rates[j][key], -j))
        out[key] = carries[i] if rates[i][key] >= thr else None
    return out


def practice_ok(thr=0.8, kind="streamc"):
    """{(task, g, p): passes} from the practice runs of composed programs on init states 0-9 (streamc: experiment dev_p5; streamh,
    the prior carrying between the bindings: dev_handoff)."""
    import glob
    import re
    out = {}
    ex = KIND_EXP[kind]
    for f in glob.glob(os.path.join(R, f"{ex}_*_{kind}_*_*.json")):
        t, g, p = map(int, re.match(rf".*{ex}_(\d+)_{kind}_(\d+)_(\d+)\.json", f).groups())
        e = json.load(open(f))["tasks"][str(t)]["episodes"]
        out[(t, g, p)] = sum(x["success"] for x in e) / len(e) >= thr
    return out


def scenes():
    """task -> its LIBERO-90 scene (KITCHEN_SCENE10, ...), from the benchmark's task map."""
    import re
    src = open("/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py").read()
    block = src[src.index('"libero_90"'):]
    names = re.findall(r'"([^"]+)"', block[block.index("[") + 1:block.index("]")])
    return {t: re.match(r"([A-Z_]+SCENE\d+)", n).group(1) for t, n in enumerate(names)}


def plan(multi=False, practice=False, order_seed=0, kind="streamc", rank=False):
    pool = load("p5_pool.json")
    demo_ok = {int(t): v["k"] / v["n"] >= 0.5 for f in ("g27_generate.json", "g28_generate.json") for t, v in load(f).items()}
    comp_ok = {tuple(map(int, k.split("_"))): v["k"] / v["n"] >= 0.5 for k, v in load("g29_generate.json").items() if k.count("_") == 2}
    unpractised = set()
    if practice:                                              # amendment 4: the gate is practice through the stream method
        prac = ({k: v is not None for k, v in practice_best().items()} if kind == "best" else practice_ok(kind=kind))
        chosen = practice_best() if kind == "best" else {}
        comp_ok = {}

        class Gate(dict):
            def get(self, k, default=False):
                if k not in prac:
                    unpractised.add(k)
                    return False
                return prac[k]
        comp_ok = Gate()
    a = sorted(int(t) for t in pool)
    random.Random(order_seed).shuffle(a)
    out = {}
    names = {0: ("A", "B"), 1: ("C", "D"), 2: ("E", "F"), 3: ("G", "H")}[order_seed]   # C, D: m3_prereg p5_confirm; E-H: fresh
    for name, order in {names[0]: a, names[1]: a[::-1]}.items():
        grasp, place, steps = {}, {}, []                      # key -> the validated sources, in arrival order
        for t in order:
            info = pool[str(t)]
            if info["composable"] and info["object_type"] in grasp and info["goal_key"] in place:
                gs, ps = grasp[info["object_type"]], place[info["goal_key"]]
                if rank:                                      # the interpretable lookup (docs/GRAMMAR.md 3): grasp sources from the same
                    sc = scenes()                             # scene first; place sources of the same object type, then the same scene
                    gs = sorted(gs, key=lambda g: sc[g] != sc[t])
                    ps = sorted(ps, key=lambda p: (pool[str(p)]["object_type"] != info["object_type"], sc[p] != sc[t]))
                pairs = [(g, p) for g in gs for p in ps]
                if not multi:
                    pairs = pairs[:1]                         # the first sources only (the registered table)
                hit = next(((g, p) for g, p in pairs if comp_ok.get((t, g, p), False)), None)
                if hit:
                    steps.append(dict(task=t, action="compose", g=hit[0], p=hit[1], tried=len(pairs),
                                      kind=chosen.get((t,) + hit, kind if kind != "best" else "streamc")))
                    continue
                steps.append(dict(task=t, action="demo", rejected=[list(x) for x in pairs]))
            else:
                steps.append(dict(task=t, action="demo"))
            if info["composable"] and demo_ok.get(t, False):
                grasp.setdefault(info["object_type"], []).append(t)
                place.setdefault(info["goal_key"], []).append(t)
        demos = sum(s["action"] == "demo" for s in steps)
        half = len(order) // 2
        out[name] = dict(order=order, steps=steps, demos=demos, composed=len(order) - demos,
                         rejected=sum("rejected" in s for s in steps),
                         composed_first_half=sum(s["action"] == "compose" for s in steps[:half]),
                         composed_second_half=sum(s["action"] == "compose" for s in steps[half:]))
        print(f"order {name}: {demos} demos ({out[name]['rejected']} compositions rejected by the check); composed "
              f"{out[name]['composed_first_half']} + {out[name]['composed_second_half']}")
    name = "p5_stream_plan_practice.json" if practice else ("p5_stream_plan_validated_multi.json" if multi else "p5_stream_plan_validated.json")
    if order_seed:
        name = name.replace(".json", f"_orders{order_seed}.json")
    if kind != "streamc":
        name = name.replace(".json", f"_{kind}.json")
    if rank:
        name = name.replace(".json", "_rank.json")
    json.dump(out, open(os.path.join(R, name), "w"), indent=1)
    if practice:
        print("candidates without practice yet (run them on states 0-9, then replan):", sorted(unpractised))
        json.dump(sorted(unpractised), open(os.path.join(R, "p5_practice_todo.json" if kind == "streamc" else f"p5_practice_todo_{kind}.json"), "w"))
    need = sorted({(s["task"], s["g"], s["p"], s["kind"]) for o in out.values() for s in o["steps"] if s["action"] == "compose"
                   and not os.path.exists(os.path.join(R, f"m3_final_{s['task']}_{s['kind']}_{s['g']}_{s['p']}.json"))})
    print("compositions to run before the read:", need)
    return out


def read(record=False, multi=False, practice=False, order_seed=0, score_seed=0, kind="streamc"):
    import importlib.util
    from lifelong_ood import ledger, stats
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    name = "p5_stream_plan_practice.json" if practice else ("p5_stream_plan_validated_multi.json" if multi else "p5_stream_plan_validated.json")
    if order_seed:
        name = name.replace(".json", f"_orders{order_seed}.json")
    if kind != "streamc":
        name = name.replace(".json", f"_{kind}.json")
    P = load(name)
    sfx = f"_seed{score_seed}" if score_seed else ""                  # the confirmation scores with a fresh seed (m3_prereg p5_confirm)
    own0 = lambda t: fin[(str(t), "phasebind2_rjpred")]
    own = lambda t: fin.get((str(t), "phasebind2_rjpred" + sfx), own0(t))
    res, tab = {"orders": {}}, {}
    for o, pl in P.items():
        ks = kall = n = 0
        for s in pl["steps"]:
            e_own = own(s["task"]) if s["action"] == "compose" else own0(s["task"])
            e = fin[(str(s["task"]), f"{s.get('kind', kind)}_{s['g']}_{s['p']}{sfx}")] if s["action"] == "compose" else e_own
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
        print(ledger.record("p5_stream", bench="libero_90", tag=("handoff" if kind in ("streamh", "best") else "confirm" if order_seed else ("practice" if practice else ("validated_multi" if multi else "validated"))), seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/p5_validated.py", command=" ".join(sys.argv)))
    return res


def read_scheme(order_seeds=(2, 3), score_seed=3, record=False):
    """The compositional scheme's confirmation (m3_prereg p5_scheme): the ranked lookup, practice choosing the carry (kind best), fresh
    orders.  Each UNIQUE composition (task, g, p, carry) arriving in any order is one unit, scored once at `score_seed` on init states
    30-49 and paired with the task's own demo at the same seed; units are the bootstrap's clusters."""
    import importlib.util
    from lifelong_ood import ledger, stats
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    sfx = f"_seed{score_seed}"
    units, res = {}, {"orders": {}, "carries": list(CARRIES), "score_seed": score_seed}
    for sd in order_seeds:
        name = "p5_stream_plan_practice.json".replace(".json", (f"_orders{sd}" if sd else "") + "_best_rank.json")
        for o, pl in load(name).items():
            comp = [(x["task"], x["g"], x["p"], x["kind"]) for x in pl["steps"] if x["action"] == "compose"]
            res["orders"][o] = dict(demos=pl["demos"], composed=len(comp), rejected=pl["rejected"], compositions=[list(u) for u in comp],
                                    composed_first_half=pl["composed_first_half"], composed_second_half=pl["composed_second_half"])
            units.update(dict.fromkeys(comp))
    tab = {f"{t}_{g}_{p}_{k}": {"c": fin[(str(t), f"{k}_{g}_{p}{sfx}")], "own": fin[(str(t), f"phasebind2_rjpred{sfx}")]}
           for (t, g, p, k) in units}
    pairs = stats.paired(tab, "c", "own", list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    kc = sum(sum(v[0]) for v in pairs.values()); ko = sum(sum(v[1]) for v in pairs.values()); nn = sum(len(v[0]) for v in pairs.values())
    res["per_unit"] = {k: dict(composed=int(sum(v["c"].values())), own=int(sum(v["own"].values()))) for k, v in tab.items()}
    res["H1"] = dict(x=int(kc), control=int(ko), n=int(nn), units=len(tab), diff=d, discordant=[b, c], sign_p=stats.sign_test(b, c),
                     non_inferior=bool(d["lo90"] >= -0.10))
    for o, v in res["orders"].items():
        print(f"order {o}: {v['demos']} demos, {v['composed']} composed ({v['composed_first_half']} + {v['composed_second_half']}), "
              f"{v['rejected']} rejected")
    v = res["H1"]
    print(f"H1 unique compositions ({v['units']}) vs their own demos: {kc} vs {ko} of {nn}, diff {d['diff']:+.3f}, lower bound "
          f"{d['lo90']:+.3f}, sign p {v['sign_p']:.3g} -> non-inferior {v['non_inferior']}")
    if record:
        print(ledger.record("p5_stream", bench="libero_90", tag="scheme", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/p5_validated.py", command=" ".join(sys.argv)))
    return res


if __name__ == "__main__":
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "streamc"
    if "--carries" in sys.argv:                               # kind best: the carry set practice chooses from
        CARRIES = tuple(sys.argv[sys.argv.index("--carries") + 1].split(","))
    if sys.argv[1] == "read_scheme":
        osd = tuple(int(x) for x in sys.argv[sys.argv.index("--order-seeds") + 1].split(",")) if "--order-seeds" in sys.argv else (2, 3)
        sc = int(sys.argv[sys.argv.index("--score-seed") + 1]) if "--score-seed" in sys.argv else 3
        read_scheme(osd, sc, "--record" in sys.argv)
        sys.exit(0)
    if sys.argv[1] == "plan":
        seed = int(sys.argv[sys.argv.index("--order-seed") + 1]) if "--order-seed" in sys.argv else 0
        plan("--multi" in sys.argv, "--practice" in sys.argv, seed, kind, "--rank" in sys.argv)
    else:
        seed = int(sys.argv[sys.argv.index("--order-seed") + 1]) if "--order-seed" in sys.argv else 0
        sc = int(sys.argv[sys.argv.index("--score-seed") + 1]) if "--score-seed" in sys.argv else 0
        read("--record" in sys.argv, "--multi" in sys.argv, "--practice" in sys.argv, seed, sc, kind)
