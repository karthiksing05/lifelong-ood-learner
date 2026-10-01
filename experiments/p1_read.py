"""P1 and P6 reads (archive/2026-10-program/PLAN_TRANSFER.md; ledger m3_prereg tags p1, p1_amend1, p6): phase programs with object-relative
bindings executed through the frozen policy, against the current program (dimfold_s1_last2; dimcode_s1 where the demo grasps
nothing), paired on init states 30-49 of m3_final.

    primary    the 9 concentrated-failure tasks, arms A (phasebind: bindings + the 258 KB fold) and B (phasebind_nofold), Holm
               over the two arms; pass = diff >= +0.11 and Holm p < 0.05; kill = one-sided 95% lower bound < -0.05
    secondary  the same over all 21 held-out tasks (420 paired)
    p6         the learned pose reader (phasebind_reader) on the 9 tasks: non-inferior to the oracle arm B at -0.10 (one-sided
               95% lower bound), and above the control (diff >= +0.11, exact sign test p < 0.05)
    v2         rules v2 (ledger m3_prereg tag v2) on the 21 tasks: V1 v2 > v1, V2 v2 > LoRA + failure negatives, V4 the v2 generator's
               LoRA > 3.8's (sign tests, Holm over the three); V3 the learned reader non-inferior to the simulator's poses at -0.10

    ~/flash/miniconda3/envs/chunks/bin/python experiments/p1_read.py primary|secondary|p6|v2|v2_synth|p3c|p3c_reader|p3c_reader2|alloc|p5|yaw|seed1|demo1|compress [--record]
"""
import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger, stats                                   # noqa: E402

P1_TASKS = [42, 32, 1, 35, 40, 36, 52, 62, 41]
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
ZERO = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)


def table(tasks, arms):
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    out = {}
    for t in tasks:
        row = {k: fin[(str(t), tag)] for k, tag in arms.items() if (str(t), tag) in fin}
        if "ctrl" not in row and "ctrl_nofold" in row:          # tasks whose demo does not grasp run borrowing alone
            row["ctrl"] = row["ctrl_nofold"]
        missing = [k for k in arms if k != "ctrl_nofold" and k not in row]
        if missing:
            raise SystemExit(f"task {t}: missing {missing}")
        out[str(t)] = row
    return out


def compare(tab, x, y):
    pairs = stats.paired(tab, x, y, list(tab))
    d = stats.bootstrap_diff(pairs, B=10000)
    d.pop("samples", None)
    b, c = stats.discordant(pairs)
    kx = sum(sum(v[0]) for v in pairs.values())
    ky = sum(sum(v[1]) for v in pairs.values())
    n = sum(len(v[0]) for v in pairs.values())
    return dict(x=int(kx), control=int(ky), n=int(n), diff=d, discordant=[b, c], sign_p=stats.sign_test(b, c))


def p1(tasks):
    arms = {"A": "phasebind", "B": "phasebind_nofold", "ctrl": "dimfold_s1_last2", "ctrl_nofold": "dimcode_s1"}
    tab = table(tasks, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    for arm in ("A", "B"):
        res[arm] = dict(arm=arms[arm], **compare(tab, arm, "ctrl"))
        res[arm]["killed"] = bool(res[arm]["diff"]["lo90"] < -0.05)
    ps = sorted((res[a]["sign_p"], a) for a in ("A", "B"))
    run = 0
    for r, (p, a) in enumerate(ps):
        run = max(run, min(1, (2 - r) * p))
        res[a]["holm_p"] = run
    for a in ("A", "B"):
        v = res[a]
        v["passes"] = bool(v["diff"]["diff"] >= 0.11 and v["holm_p"] < 0.05 and v["x"] > v["control"])
        print(f"{a} {v['arm']}: {v['x']} vs {v['control']} of {v['n']}, diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f},"
              f"{v['diff']['hi']:+.3f}] lo90 {v['diff']['lo90']:+.3f}, discordant {v['discordant']}, sign p {v['sign_p']:.4g} "
              f"Holm {v['holm_p']:.4g} -> passes {v['passes']}, killed {v['killed']}")
    return res


def p6():
    arms = {"R": "phasebind_reader", "B": "phasebind_nofold", "ctrl": "dimfold_s1_last2", "ctrl_nofold": "dimcode_s1"}
    tab = table(P1_TASKS, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    ni = compare(tab, "R", "B")
    ni["non_inferior"] = bool(ni["diff"]["lo90"] >= -0.10)
    res["reader_vs_oracle"] = ni
    ab = compare(tab, "R", "ctrl")
    ab["passes"] = bool(ab["diff"]["diff"] >= 0.11 and ab["sign_p"] < 0.05 and ab["x"] > ab["control"])
    res["reader_vs_control"] = ab
    for k, v in (("reader vs oracle", ni), ("reader vs control", ab)):
        print(f"{k}: {v['x']} vs {v['control']} of {v['n']}, diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f},{v['diff']['hi']:+.3f}] "
              f"lo90 {v['diff']['lo90']:+.3f}, discordant {v['discordant']}, sign p {v['sign_p']:.4g}")
    print(f"non-inferior at -0.10: {ni['non_inferior']}; above the control: {ab['passes']}")
    return res


def v2():
    """The v2 read (ledger m3_prereg tag v2): V1 v2 > v1, V2 v2 > LoRA + failure negatives, V4 r27gen > r25gen (sign tests, Holm);
    V3 the learned reader non-inferior to the simulator's poses at -0.10."""
    arms = {"v2": "phasebind2_nofold", "v2r": "phasebind2_reader", "v1": "phasebind_nofold", "nft": "lora_r17bal",
            "g27": "lora_r27gen", "g25": "lora_r25gen", "ctrl": "dimfold_s1_last2", "ctrl_nofold": "dimcode_s1"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    for h, (x, y) in {"V1": ("v2", "v1"), "V2": ("v2", "nft"), "V4": ("g27", "g25")}.items():
        res[h] = dict(x_arm=arms[x], y_arm=arms[y], **compare(tab, x, y))
    ps = sorted((res[h]["sign_p"], h) for h in ("V1", "V2", "V4"))
    run = 0
    for r, (p, h) in enumerate(ps):
        run = max(run, min(1, (3 - r) * p))
        res[h]["holm_p"] = run
    for h in ("V1", "V2", "V4"):
        v = res[h]
        v["passes"] = bool(v["holm_p"] < 0.05 and v["x"] > v["control"])
    res["V3"] = dict(x_arm=arms["v2r"], y_arm=arms["v2"], **compare(tab, "v2r", "v2"))
    res["V3"]["non_inferior"] = bool(res["V3"]["diff"]["lo90"] >= -0.10)
    for k, (x, y) in {"reader_vs_nft": ("v2r", "nft"), "g27_vs_nft": ("g27", "nft"), "v2_vs_ctrl": ("v2", "ctrl"),
                      "reader_vs_ctrl": ("v2r", "ctrl")}.items():
        res[k] = dict(x_arm=arms[x], y_arm=arms[y], **compare(tab, x, y))
    for k in ("V1", "V2", "V4", "V3", "reader_vs_nft", "g27_vs_nft", "v2_vs_ctrl", "reader_vs_ctrl"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f", Holm {v['holm_p']:.3g} -> passes {v['passes']}" if "holm_p" in v else "")
              + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else ""))
    return res


def v2_synth():
    """ledger m3_prereg tag v2_synth: V5 the rendered-scene reader non-inferior to the simulator's poses at -0.10; V6 > LoRA + negatives."""
    arms = {"s": "phasebind2_reader_spatial4_synth", "v2": "phasebind2_nofold", "v2r": "phasebind2_reader", "nft": "lora_r17bal",
            "ctrl": "dimfold_s1_last2", "ctrl_nofold": "dimcode_s1"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["V5"] = dict(x_arm=arms["s"], y_arm=arms["v2"], **compare(tab, "s", "v2"))
    res["V5"]["non_inferior"] = bool(res["V5"]["diff"]["lo90"] >= -0.10)
    res["V6"] = dict(x_arm=arms["s"], y_arm=arms["nft"], **compare(tab, "s", "nft"))
    res["V6"]["passes"] = bool(res["V6"]["sign_p"] < 0.05 and res["V6"]["x"] > res["V6"]["control"])
    res["vs_30_placement_reader"] = dict(x_arm=arms["s"], y_arm=arms["v2r"], **compare(tab, "s", "v2r"))
    for k in ("V5", "V6", "vs_30_placement_reader"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else "")
              + (f" -> passes {v['passes']}" if "passes" in v else ""))
    return res


def p3c():
    """ledger m3_prereg tag p3c: C1 composed-from-other-tasks non-inferior to the own demo at -0.10; C2 composed > sibling's whole demo."""
    import json
    tasks = sorted(int(t) for t in json.load(open(os.path.join(ROOT, "results", "robot", "p3c_table.json")))["table"])
    arms = {"c": "compose2", "sib": "sibling2", "own": "phasebind2_nofold", "nft": "lora_r17bal", "ctrl": "dimfold_s1_last2",
            "ctrl_nofold": "dimcode_s1"}
    tab = table(tasks, arms)
    res = {"tasks": tasks, "per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["C1"] = dict(x_arm=arms["c"], y_arm=arms["own"], **compare(tab, "c", "own"))
    res["C1"]["non_inferior"] = bool(res["C1"]["diff"]["lo90"] >= -0.10)
    res["C2"] = dict(x_arm=arms["c"], y_arm=arms["sib"], **compare(tab, "c", "sib"))
    res["C2"]["passes"] = bool(res["C2"]["sign_p"] < 0.05 and res["C2"]["x"] > res["C2"]["control"])
    for k, y in (("vs_ctrl", "ctrl"), ("vs_nft", "nft")):
        res[k] = dict(x_arm=arms["c"], y_arm=arms[y], **compare(tab, "c", y))
    for k in ("C1", "C2", "vs_ctrl", "vs_nft"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else "")
              + (f" -> passes {v['passes']}" if "passes" in v else ""))
    return res


def p3c_reader():
    """ledger m3_prereg tag p3c_reader: R1 composed + rendered-scene reader non-inferior to composed with the simulator's poses at -0.10."""
    import json
    tasks = sorted(int(t) for t in json.load(open(os.path.join(ROOT, "results", "robot", "p3c_table.json")))["table"])
    arms = {"cr": "compose2_reader", "c": "compose2", "own": "phasebind2_nofold", "nft": "lora_r17bal"}
    tab = table(tasks, arms)
    res = {"tasks": tasks, "per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["R1"] = dict(x_arm=arms["cr"], y_arm=arms["c"], **compare(tab, "cr", "c"))
    res["R1"]["non_inferior"] = bool(res["R1"]["diff"]["lo90"] >= -0.10)
    for k, y in (("vs_own", "own"), ("vs_nft", "nft")):
        res[k] = dict(x_arm=arms["cr"], y_arm=arms[y], **compare(tab, "cr", y))
    for k in ("R1", "vs_own", "vs_nft"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else ""))
    return res


def p3c_reader2():
    """ledger m3_prereg tag p3c_reader2: R2 composed + rendered-scene reader + predicted carry non-inferior to composed with the
    simulator's poses at -0.10."""
    import json
    tasks = sorted(int(t) for t in json.load(open(os.path.join(ROOT, "results", "robot", "p3c_table.json")))["table"])
    arms = {"crp": "compose2_reader_pred", "cr": "compose2_reader", "c": "compose2", "own": "phasebind2_nofold", "nft": "lora_r17bal"}
    tab = table(tasks, arms)
    res = {"tasks": tasks, "per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["R2"] = dict(x_arm=arms["crp"], y_arm=arms["c"], **compare(tab, "crp", "c"))
    res["R2"]["non_inferior"] = bool(res["R2"]["diff"]["lo90"] >= -0.10)
    for k, y in (("vs_identity_carry", "cr"), ("vs_own", "own"), ("vs_nft", "nft")):
        res[k] = dict(x_arm=arms["crp"], y_arm=arms[y], **compare(tab, "crp", y))
    for k in ("R2", "vs_identity_carry", "vs_own", "vs_nft"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else ""))
    return res


def alloc():
    """ledger m3_prereg tag alloc_v2: per task, the candidate with the most practice successes on init states 0-9 (ties A, B, C),
    scored by its existing m3_final episodes on states 30-49."""
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    prac, _ = m3s.episodes("alloc_practice")
    cand = {"A": ["phasebind2_reader_spatial4_synth"], "B": ["dimfold_s1_last2", "dimcode_s1"], "C": ["lora_r27gen"]}
    arms = {"A": "phasebind2_reader_spatial4_synth", "Bf": "dimfold_s1_last2", "Bc": "dimcode_s1", "C": "lora_r27gen", "nft": "lora_r17bal"}
    fin, _ = m3s.episodes("m3_final")
    tab, choice, practice = {}, {}, {}
    for t in HELD:
        pk = {}
        for c, tags in cand.items():
            tag = next(g for g in tags if (str(t), g) in prac)
            pk[c] = (int(sum(prac[(str(t), tag)].values())), tag)
        best = max(pk, key=lambda c: (pk[c][0], -"ABC".index(c)))
        choice[str(t)] = dict(choice=best, arm=pk[best][1])
        practice[str(t)] = {c: v[0] for c, v in pk.items()}
        row = {"A": fin[(str(t), arms["A"])], "nft": fin[(str(t), arms["nft"])], "alloc": fin[(str(t), pk[best][1])]}
        tab[str(t)] = row
    res = {"choice": choice, "practice": practice, "per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    spec = importlib.util.spec_from_file_location("m34", os.path.join(ROOT, "experiments", "m34_report.py"))
    m34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m34)
    res["bytes"] = {t: int(m34.arm_bytes(int(t), c["arm"])) for t, c in choice.items()}
    res["bytes_total"] = int(sum(res["bytes"].values()))
    res["bytes_A_total"] = int(sum(m34.arm_bytes(t, arms["A"]) for t in HELD))
    print(f"bytes: alloc {res['bytes_total'] / 1e6:.1f} MB, A alone {res['bytes_A_total'] / 1e6:.1f} MB")
    res["AL1"] = dict(x_arm="alloc", y_arm=arms["A"], **compare(tab, "alloc", "A"))
    res["AL1"]["passes"] = bool(res["AL1"]["sign_p"] < 0.05 and res["AL1"]["x"] > res["AL1"]["control"])
    res["vs_nft"] = dict(x_arm="alloc", y_arm=arms["nft"], **compare(tab, "alloc", "nft"))
    for k in ("AL1", "vs_nft"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> passes {v['passes']}" if "passes" in v else ""))
    print("switched from A:", {t: (c["arm"], res["practice"][t]) for t, c in choice.items() if c["choice"] != "A"})
    return res


def p5():
    """ledger m3_prereg tag p5_stream: S1 structural (from the plan), S2 composed arrivals non-inferior to the same tasks' own demos at
    -0.10, S3 each order's stream success vs the all-demo stream."""
    import json
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    P = json.load(open(os.path.join(ROOT, "results", "robot", "p5_stream_plan_gated.json")))       # amendment 1: validated bindings
    U = json.load(open(os.path.join(ROOT, "results", "robot", "p5_stream_plan.json")))             # the ungated plan (descriptive)
    own = lambda t: fin[(str(t), "phasebind2_rjpred")]
    comp = lambda s: fin[(str(s["task"]), f"streamc_{s['g']}_{s['p']}")]
    tab, res = {}, {"orders": {}}
    for o, pl in P.items():
        k_stream = k_all = n = 0
        rows = []
        for i, s in enumerate(pl["steps"]):
            e_own = own(s["task"])
            e = comp(s) if s["action"] == "compose" else e_own
            k_stream += sum(e.values()); k_all += sum(e_own.values()); n += len(e)
            rows.append(dict(pos=i, task=s["task"], action=s["action"], k=int(sum(e.values())), own=int(sum(e_own.values()))))
            if s["action"] == "compose":
                tab[f"{o}:{s['task']}"] = {"c": e, "own": e_own}
        res["orders"][o] = dict(stream=int(k_stream), all_demo=int(k_all), n=int(n), demos=pl["demos"], arrivals=rows,
                                composed_first_half=pl["composed_first_half"], composed_second_half=pl["composed_second_half"])
        print(f"order {o}: stream {k_stream}/{n} with {pl['demos']} demos; every task its own demo {k_all}/{n}")
    ung = {}
    for o, pl in U.items():
        ks = [sum((comp(s) if s["action"] == "compose" else own(s["task"])).values()) for s in pl["steps"]]
        ung[o] = dict(stream=int(sum(ks)), demos=pl["demos"],
                      poisoned={f"{s['task']}<-{s['g']}+{s['p']}": int(sum(comp(s).values())) for s in pl["steps"]
                                if s["action"] == "compose" and (s["task"], s["g"], s["p"]) not in
                                {(x["task"], x.get("g"), x.get("p")) for x in P[o]["steps"]}})
        print(f"ungated order {o}: stream {ung[o]['stream']} with {pl['demos']} demos; compositions the gate removes: {ung[o]['poisoned']}")
    res["ungated"] = ung
    res["S2"] = dict(x_arm="composed arrivals", y_arm="phasebind2_rjpred (own demo)", **compare(tab, "c", "own"))
    res["S2"]["non_inferior"] = bool(res["S2"]["diff"]["lo90"] >= -0.10)
    late = early = both = 0
    pos = {o: {s["task"]: i for i, s in enumerate(P[o]["steps"])} for o in P}
    cmp_ = {o: {s["task"] for s in P[o]["steps"] if s["action"] == "compose"} for o in P}
    half = len(P["A"]["steps"]) // 2
    for t in pos["A"]:
        a, b = t in cmp_["A"], t in cmp_["B"]
        if a and b:
            both += 1
        elif a or b:
            late += int(pos["A" if a else "B"][t] >= half)
            early += int(pos["A" if a else "B"][t] < half)
    res["S1"] = dict(composed_both=both, only_late=late, only_early=early, sign_p=stats.sign_test(late, early))
    print(f"S1: composed only when late {late}, only when early {early}, both {both}: sign p {res['S1']['sign_p']:.3g}")
    v = res["S2"]
    print(f"S2 {v['x_arm']} vs {v['y_arm']}: {_fmt(v)} -> non-inferior {v['non_inferior']}")
    return res


def yaw():
    """ledger m3_prereg tag yaw: Y1 the own-demo program reading the books' yaw > the same with a fixed yaw (sign test), 11 book tasks."""
    tasks = [73, 74, 75, 78, 80, 81, 83, 86, 87, 88, 89]
    tab = table(tasks, {"y": "phasebind2_rjpred_yaw", "n": "phasebind2_rjpred"})
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["Y1"] = dict(x_arm="phasebind2_rjpred_yaw", y_arm="phasebind2_rjpred", **compare(tab, "y", "n"))
    res["Y1"]["passes"] = bool(res["Y1"]["sign_p"] < 0.05 and res["Y1"]["x"] > res["Y1"]["control"])
    v = res["Y1"]
    print(f"Y1 {v['x_arm']} vs {v['y_arm']}: {_fmt(v)} -> passes {v['passes']}")
    return res


def _pearson(tab, x, y):
    import numpy as np
    a = np.array([sum(v[x].values()) for v in tab.values()], float)
    b = np.array([sum(v[y].values()) for v in tab.values()], float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def seed1():
    """ledger m3_prereg tag seed1: R1 v2 (simulator poses) > LoRA + negatives and R3 allocator > LoRA + negatives at seed 1 (Holm);
    R2 the reader arm non-inferior to v2 at seed 1; test-retest against seed 0."""
    import json
    choice = [r for r in ledger.load("p1_bindings", tag="alloc_v2") if not r.get("retired")][-1]["metrics"]["choice"]
    arms = {"v2": "phasebind2_nofold", "rd": "phasebind2_reader_spatial4_synth", "nft": "lora_r17bal"}
    arms.update({k + "1": v + "_seed1" for k, v in list(arms.items())})
    tab = table(HELD, arms)
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m3s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m3s)
    fin, _ = m3s.episodes("m3_final")
    for t in HELD:
        arm = choice[str(t)]["arm"]
        tab[str(t)]["al"] = fin[(str(t), arm)]
        tab[str(t)]["al1"] = fin[(str(t), arm + "_seed1")]
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["R1"] = dict(x_arm="phasebind2_nofold_seed1", y_arm="lora_r17bal_seed1", **compare(tab, "v21", "nft1"))
    res["R3"] = dict(x_arm="allocator_seed1", y_arm="lora_r17bal_seed1", **compare(tab, "al1", "nft1"))
    ps = sorted((res[h]["sign_p"], h) for h in ("R1", "R3"))
    run = 0
    for r, (p, h) in enumerate(ps):
        run = max(run, min(1, (2 - r) * p))
        res[h]["holm_p"] = run
    for h in ("R1", "R3"):
        res[h]["passes"] = bool(res[h]["holm_p"] < 0.05 and res[h]["x"] > res[h]["control"])
    res["R2"] = dict(x_arm="phasebind2_reader_spatial4_synth_seed1", y_arm="phasebind2_nofold_seed1", **compare(tab, "rd1", "v21"))
    res["R2"]["non_inferior"] = bool(res["R2"]["diff"]["lo90"] >= -0.10)
    res["retest"] = {k: dict(seed0=int(sum(sum(v[k].values()) for v in tab.values())), seed1=int(sum(sum(v[k + "1"].values()) for v in tab.values())),
                             pearson_per_task=_pearson(tab, k, k + "1")) for k in ("v2", "rd", "nft", "al")}
    for k in ("R1", "R3", "R2"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f", Holm {v['holm_p']:.3g} -> passes {v['passes']}" if "holm_p" in v else "")
              + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else ""))
    print("test-retest (seed 0 -> seed 1, per-task r):", json.dumps(res["retest"]))
    return res


def demo1():
    """ledger m3_prereg tag demo1_v2: D1 v2 (demo 1) > the one-demo LoRA (demo 1); D2 the reader arm (demo 1) non-inferior to v2 (demo 1)."""
    arms = {"v2d1": "phasebind2_nofold_d1", "rdd1": "phasebind2_reader_spatial4_synth_d1", "lorad1": "lora_d1",
            "v2": "phasebind2_nofold", "rd": "phasebind2_reader_spatial4_synth", "lora": "lora"}
    tab = table([t for t in HELD if t != 42], arms)                  # amendment 1: task 42's demo 1 has no raw match
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["D1"] = dict(x_arm=arms["v2d1"], y_arm=arms["lorad1"], **compare(tab, "v2d1", "lorad1"))
    res["D1"]["passes"] = bool(res["D1"]["sign_p"] < 0.05 and res["D1"]["x"] > res["D1"]["control"])
    res["D2"] = dict(x_arm=arms["rdd1"], y_arm=arms["v2d1"], **compare(tab, "rdd1", "v2d1"))
    res["D2"]["non_inferior"] = bool(res["D2"]["diff"]["lo90"] >= -0.10)
    res["demo0_vs_demo1"] = {k: dict(demo0=int(sum(sum(v[k].values()) for v in tab.values())),
                                     demo1=int(sum(sum(v[k + "d1"].values()) for v in tab.values())),
                                     pearson_per_task=_pearson(tab, k, k + "d1")) for k in ("v2", "rd", "lora")}
    for k in ("D1", "D2"):
        v = res[k]
        print(f"{k} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> passes {v['passes']}" if "passes" in v else "")
              + (f" -> non-inferior {v['non_inferior']}" if "non_inferior" in v else ""))
    print("demo 0 -> demo 1:", res["demo0_vs_demo1"])
    return res


def compress():
    """ledger m3_prereg tag reader_compress: RC1 the shared-basis reader head (pca512) non-inferior to the full reader at -0.05, 50 tasks."""
    import json
    pool = sorted(int(t) for t in json.load(open(os.path.join(ROOT, "results", "robot", "p5_pool.json"))))
    arms = {"pca": "phasebind2_rjpred_pca512_rj", "full": "phasebind2_rjpred"}
    tab = table(pool, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["RC1"] = dict(x_arm=arms["pca"], y_arm=arms["full"], **compare(tab, "pca", "full"))
    res["RC1"]["non_inferior_005"] = bool(res["RC1"]["diff"]["lo90"] >= -0.05)
    res["RC1"]["non_inferior_010"] = bool(res["RC1"]["diff"]["lo90"] >= -0.10)
    v = res["RC1"]
    print(f"RC1 {v['x_arm']} vs {v['y_arm']}: {_fmt(v)} -> non-inferior at -0.05 {v['non_inferior_005']} (at -0.10 {v['non_inferior_010']})")
    return res


def sota():
    """ledger m3_prereg tag baselines_sota: B1-B3 ours (bindings, 7 KB reader head) vs STRAP, DSRL, experience replay; Holm."""
    arms = {"ours": "phasebind2_rjpred_pca512_rj", "strap": "lora_strap", "dsrl": "dsrl", "er": "lora_er", "nft": "lora_r17bal"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    ps = {}
    for h, b in (("B1", "strap"), ("B2", "dsrl"), ("B3", "er")):
        res[h] = dict(x_arm=arms["ours"], y_arm=arms[b], **compare(tab, "ours", b))
        ps[h] = res[h]["sign_p"]
    order = sorted(ps, key=ps.get)
    for i, h in enumerate(order):                                    # Holm
        res[h]["holm_p"] = min(1.0, max(ps[o] * (len(order) - j) for j, o in enumerate(order[:i + 1])))
        res[h]["ours_better"] = bool(res[h]["x"] > res[h]["control"] and res[h]["holm_p"] < 0.05)
    res["vs_nft"] = {b: compare(tab, b, "nft") for b in ("strap", "dsrl", "er")}
    for h in ("B1", "B2", "B3"):
        v = res[h]
        print(f"{h} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}, Holm p {v['holm_p']:.3g} -> ours better {v['ours_better']}")
    return res


def pi0():
    """ledger m3_prereg tag pi0_generality: Z the bindings on pi0 > pi0 zero-shot."""
    arms = {"b": "phasebind2_rjpred_pca512_rj_pi0", "z": "zeroshot_pi0", "b05": "phasebind2_rjpred_pca512_rj"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["Z"] = dict(x_arm=arms["b"], y_arm=arms["z"], **compare(tab, "b", "z"))
    res["Z"]["passes"] = bool(res["Z"]["sign_p"] < 0.05 and res["Z"]["x"] > res["Z"]["control"])
    res["vs_pi05"] = compare(tab, "b", "b05")
    print(f"Z {_fmt(res['Z'])} -> passes {res['Z']['passes']}; vs the same method on pi0.5: {_fmt(res['vs_pi05'])}")
    return res


def smolvla():
    """ledger m3_prereg tag smolvla_generality: Z the bindings on SmolVLA > SmolVLA zero-shot."""
    arms = {"b": "phasebind2_rjpred_pca512_rj_smolvla", "z": "zeroshot_smolvla", "b05": "phasebind2_rjpred_pca512_rj"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["Z"] = dict(x_arm=arms["b"], y_arm=arms["z"], **compare(tab, "b", "z"))
    res["Z"]["passes"] = bool(res["Z"]["sign_p"] < 0.05 and res["Z"]["x"] > res["Z"]["control"])
    res["vs_pi05"] = compare(tab, "b", "b05")
    print(f"Z {_fmt(res['Z'])} -> passes {res['Z']['passes']}; vs the same method on pi0.5: {_fmt(res['vs_pi05'])}")
    return res


def nopolicy():
    """ledger m3_prereg tag nopolicy_control: N the program executed with no policy non-inferior to the bindings arm at -0.05."""
    arms = {"n": "nopolicy_rjpred_pca512_rj", "b": "phasebind2_rjpred_pca512_rj", "z": "zeroshot"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["N"] = dict(x_arm=arms["n"], y_arm=arms["b"], **compare(tab, "n", "b"))
    res["N"]["non_inferior_005"] = bool(res["N"]["diff"]["lo90"] >= -0.05)
    res["vs_zeroshot"] = compare(tab, "n", "z")
    v = res["N"]
    print(f"N {v['x_arm']} vs {v['y_arm']}: {_fmt(v)} -> non-inferior at -0.05 {v['non_inferior_005']}; vs zero-shot: {_fmt(res['vs_zeroshot'])}")
    return res


ZC_TASKS = (18, 26, 48, 49, 52, 53, 58, 59, 62, 65, 66, 71)


def zc_main():
    """ledger m3_prereg tag zc_main: H1 fcomp@8 > zero-shot (sign p < 0.05 and lower bound > +0.05); H1b-c vs cfg@8 and
    placebo@8 (Holm)."""
    arms = {"f": "fcomp_8", "z": "zeroshot", "c": "cfg_8", "p": "placebo_8"}
    tab = table(ZC_TASKS, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    res["H1"] = dict(x_arm=arms["f"], y_arm=arms["z"], **compare(tab, "f", "z"))
    res["H1"]["passes"] = bool(res["H1"]["sign_p"] < 0.05 and res["H1"]["diff"]["lo90"] > 0.05)
    ps = {}
    for h, y in (("H1b", "c"), ("H1c", "p")):
        res[h] = dict(x_arm=arms["f"], y_arm=arms[y], **compare(tab, "f", y))
        ps[h] = res[h]["sign_p"]
    order = sorted(ps, key=ps.get)
    for i, h in enumerate(order):
        res[h]["holm_p"] = min(1.0, max(ps[o] * (len(order) - j) for j, o in enumerate(order[:i + 1])))
        res[h]["passes"] = bool(res[h]["x"] > res[h]["control"] and res[h]["holm_p"] < 0.05)
    for h in ("H1", "H1b", "H1c"):
        v = res[h]
        print(f"{h} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)} -> passes {v['passes']}")
    return res


def zc_fold():
    """ledger m3_prereg tag zc_fold: H2a token (demo 0) vs one-demo LoRA, H2b token (demo 1) vs LoRA demo 1, both
    non-inferior at -0.10; Z token vs zero-shot."""
    arms = {"tok": "ftoken_self", "lora": "lora", "tok1": "ftoken_self_d1", "lora1": "lora_d1", "z": "zeroshot"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    for h, (x, y) in (("H2a", ("tok", "lora")), ("H2b", ("tok1", "lora1")), ("Z", ("tok", "z"))):
        res[h] = dict(x_arm=arms[x], y_arm=arms[y], **compare(tab, x, y))
    for h in ("H2a", "H2b"):
        res[h]["non_inferior_010"] = bool(res[h]["diff"]["lo90"] >= -0.10)
    res["claim"] = bool(res["H2a"]["non_inferior_010"] and res["H2b"]["non_inferior_010"])
    res["Z"]["passes"] = bool(res["Z"]["sign_p"] < 0.05 and res["Z"]["x"] > res["Z"]["control"])
    for h in ("H2a", "H2b", "Z"):
        v = res[h]
        print(f"{h} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}" + (f" -> non-inferior {v['non_inferior_010']}" if h != "Z" else
                                                                  f" -> passes {v['passes']}"))
    print("claim (both demos):", res["claim"])
    return res


def sota_amend1():
    """ledger m3_prereg tag baselines_sota_amend1: B1' ours vs STRAP (authors' retrieval), B2' ours vs DSRL (authors' config), B3 vs ER."""
    arms = {"ours": "phasebind2_rjpred_pca512_rj", "strap": "lora_strap2", "dsrl": "dsrlo", "er": "lora_er", "nft": "lora_r17bal"}
    tab = table(HELD, arms)
    res = {"per_task": {t: {k: int(sum(v[k].values())) for k in v} for t, v in tab.items()}}
    ps = {}
    for h, b in (("B1", "strap"), ("B2", "dsrl"), ("B3", "er")):
        res[h] = dict(x_arm=arms["ours"], y_arm=arms[b], **compare(tab, "ours", b))
        ps[h] = res[h]["sign_p"]
    order = sorted(ps, key=ps.get)
    for i, h in enumerate(order):
        res[h]["holm_p"] = min(1.0, max(ps[o] * (len(order) - j) for j, o in enumerate(order[:i + 1])))
        res[h]["ours_better"] = bool(res[h]["x"] > res[h]["control"] and res[h]["holm_p"] < 0.05)
    res["vs_nft"] = {b: compare(tab, b, "nft") for b in ("strap", "dsrl", "er")}
    for h in ("B1", "B2", "B3"):
        v = res[h]
        print(f"{h} {v['x_arm']} vs {v['y_arm']}: {_fmt(v)}, Holm p {v['holm_p']:.3g} -> ours better {v['ours_better']}")
    return res


def _fmt(v):
    return (f"{v['x']} vs {v['control']} of {v['n']}, diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f},{v['diff']['hi']:+.3f}] lo90 "
            f"{v['diff']['lo90']:+.3f}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("read", choices=("primary", "secondary", "p6", "v2", "v2_synth", "p3c", "p3c_reader", "p3c_reader2", "alloc", "p5", "yaw", "seed1", "demo1", "compress", "sota", "sota_amend1", "pi0", "smolvla", "nopolicy", "zc_fold", "zc_main"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.read == "pi0":
        res = pi0()
        exp, tag, what = "p1_bindings", "pi0", "the unchanged method on pi0 (another flow VLA) vs pi0 zero-shot, 21 tasks"
    elif a.read == "zc_main":
        res = zc_main()
        exp, tag, what = "p1_bindings", "zc_main", "factor-composed generation (the policy's own conditionals) vs zero-shot and controls"
    elif a.read == "zc_fold":
        res = zc_fold()
        exp, tag, what = "p1_bindings", "zc_fold", "factor tokens (18.4 KB, frozen policy) vs the one-demo LoRA, two demos"
    elif a.read == "nopolicy":
        res = nopolicy()
        exp, tag, what = "p1_bindings", "nopolicy", "the phase program executed with no policy vs the bindings arm through pi0.5, 21 tasks"
    elif a.read == "smolvla":
        res = smolvla()
        exp, tag, what = "p1_bindings", "smolvla", "the unchanged method on SmolVLA (a non-pi flow VLA) vs SmolVLA zero-shot, 21 tasks"
    elif a.read == "sota_amend1":
        res = sota_amend1()
        exp, tag, what = "p1_bindings", "sota_amend1", "bindings vs STRAP and DSRL with the authors' settings, and experience replay, 21 tasks"
    elif a.read == "sota":
        res = sota()
        exp, tag, what = "p1_bindings", "sota", "bindings (7 KB reader head) vs STRAP, DSRL and experience replay, 21 tasks"
    elif a.read == "compress":
        res = compress()
        exp, tag, what = "p6_reader", "compress_read", "own-demo programs with the shared-basis (pca512) reader head vs the full reader, 50 tasks"
    elif a.read == "seed1":
        res = seed1()
        exp, tag, what = "p1_bindings", "seed1", "seed-1 replicate of the headline arms and the allocator, 21 tasks"
    elif a.read == "demo1":
        res = demo1()
        exp, tag, what = "p1_bindings", "demo1_v2", "rules v2 with demo 1 (IPEC episode 1) vs the demo-1 LoRA, 21 tasks"
    elif a.read == "yaw":
        res = yaw()
        exp, tag, what = "p6_reader", "yaw_read", "own-demo programs reading the books' yaw vs a fixed yaw, 11 book tasks"
    elif a.read == "p5":
        res = p5()
        exp, tag, what = "p5_stream", "read", "P5 stream: 50 tasks, two mirrored orders, binding table, no simulator state at test"
    elif a.read == "alloc":
        res = alloc()
        exp, tag, what = "p1_bindings", "alloc_v2", "practice allocator over v2 phases (reader), prototype borrowing and the v2-generated LoRA"
    elif a.read == "p3c_reader2":
        res = p3c_reader2()
        exp, tag, what = "p3_compose", "p3c_reader2", "P3c reader arm with the in-hand pose predicted from the grasp binding"
    elif a.read == "p3c_reader":
        res = p3c_reader()
        exp, tag, what = "p3_compose", "p3c_reader", "P3c reader arm: composed programs with a reader trained only on rendered scenes"
    elif a.read == "p3c":
        res = p3c()
        exp, tag, what = "p3_compose", "p3c", "P3c: tasks composed from other tasks' phases (rules v2, through the prior) vs own demo and sibling"
    elif a.read == "v2_synth":
        res = v2_synth()
        exp, tag, what = "p1_bindings", "v2_synth", "v2 with the rendered-scene reader vs the simulator's poses and vs LoRA + negatives, 21 tasks"
    elif a.read == "v2":
        res = v2()
        exp, tag, what = "p1_bindings", "v2", "v2 read: rules v2 (sim poses, learned reader) and the v2 generator's LoRA on 21 tasks, states 30-49"
    elif a.read == "p6":
        res = p6()
        exp, tag, what = "p6_reader", "read", "P6 read: phasebind_reader vs phasebind_nofold (oracle) and vs dimfold_s1_last2 on 9 tasks, states 30-49"
    else:
        res = p1(P1_TASKS if a.read == "primary" else HELD)
        exp, tag = "p1_bindings", ("read" if a.read == "primary" else "secondary")
        what = f"P1 {a.read} read: phasebind / phasebind_nofold vs dimfold_s1_last2 on {9 if a.read == 'primary' else 21} tasks, states 30-49"
    print(res.get("per_task", ""))
    if a.record:
        print(ledger.record(exp, bench="libero_90", tag=tag, seed=0, metrics=res, bytes=ZERO, config=vars(a),
                            script="experiments/p1_read.py", command=" ".join(sys.argv) + "  # " + what))


if __name__ == "__main__":
    main()
