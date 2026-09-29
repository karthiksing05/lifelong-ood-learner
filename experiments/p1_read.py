"""P1 and P6 reads (docs/PLAN_TRANSFER.md; ledger m3_prereg tags p1, p1_amend1, p6): phase programs with object-relative
bindings executed through the frozen policy, against the current program (dimfold_s1_last2; dimcode_s1 where the demo grasps
nothing), paired on init states 30-49 of m3_final.

    primary    the 9 concentrated-failure tasks, arms A (phasebind: bindings + the 258 KB fold) and B (phasebind_nofold), Holm
               over the two arms; pass = diff >= +0.11 and Holm p < 0.05; kill = one-sided 95% lower bound < -0.05
    secondary  the same over all 21 held-out tasks (420 paired)
    p6         the learned pose reader (phasebind_reader) on the 9 tasks: non-inferior to the oracle arm B at -0.10 (one-sided
               95% lower bound), and above the control (diff >= +0.11, exact sign test p < 0.05)

    ~/flash/miniconda3/envs/chunks/bin/python experiments/p1_read.py primary|secondary|p6 [--record]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("read", choices=("primary", "secondary", "p6"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.read == "p6":
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
