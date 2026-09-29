"""P1 and P6 reads (docs/PLAN_TRANSFER.md; ledger m3_prereg tags p1, p1_amend1, p6): phase programs with object-relative
bindings executed through the frozen policy, against the current program (dimfold_s1_last2; dimcode_s1 where the demo grasps
nothing), paired on init states 30-49 of m3_final.

    primary    the 9 concentrated-failure tasks, arms A (phasebind: bindings + the 258 KB fold) and B (phasebind_nofold), Holm
               over the two arms; pass = diff >= +0.11 and Holm p < 0.05; kill = one-sided 95% lower bound < -0.05
    secondary  the same over all 21 held-out tasks (420 paired)
    p6         the learned pose reader (phasebind_reader) on the 9 tasks: non-inferior to the oracle arm B at -0.10 (one-sided
               95% lower bound), and above the control (diff >= +0.11, exact sign test p < 0.05)
    v2         rules v2 (ledger m3_prereg tag v2) on the 21 tasks: V1 v2 > v1, V2 v2 > LoRA + failure negatives, V4 the v2 generator's
               LoRA > 3.8's (sign tests, Holm over the three); V3 the learned reader non-inferior to the simulator's poses at -0.10

    ~/flash/miniconda3/envs/chunks/bin/python experiments/p1_read.py primary|secondary|p6|v2|v2_synth|p3c|p3c_reader|p3c_reader2|alloc [--record]
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


def _fmt(v):
    return (f"{v['x']} vs {v['control']} of {v['n']}, diff {v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f},{v['diff']['hi']:+.3f}] lo90 "
            f"{v['diff']['lo90']:+.3f}, discordant {v['discordant']}, sign p {v['sign_p']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("read", choices=("primary", "secondary", "p6", "v2", "v2_synth", "p3c", "p3c_reader", "p3c_reader2", "alloc"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.read == "alloc":
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
    print(res["per_task"])
    if a.record:
        print(ledger.record(exp, bench="libero_90", tag=tag, seed=0, metrics=res, bytes=ZERO, config=vars(a),
                            script="experiments/p1_read.py", command=" ".join(sys.argv) + "  # " + what))


if __name__ == "__main__":
    main()
