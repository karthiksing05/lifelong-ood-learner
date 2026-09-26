"""M3 / M4 report: success per LIBERO-90 task and arm (Wilson 95%), with the bytes each arm stores.

    python experiments/m34_report.py            # markdown table to stdout
    python experiments/m34_report.py --json results/robot/m34_report.json

Arms come from the rollout rows (m3_compose: zeroshot, raw_replay, code_replay, lora; m4_fold: hybrid, expert_on).
Bytes: program arms carry their program's bytes in the rollout row; lora = the r3_fewshot adapter; hybrid = the
r4_fold row's total (expert + new nodes + program codes); expert_on = the r4_fold expert alone.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402


R = os.path.join(ROOT, "results", "robot")


def fold_bytes(t, tag):
    """A fold's expert + its new library nodes (the r4_fold row)."""
    r = [x for x in ledger.load("r4_fold") if x["tag"] == tag and str(x["metrics"]["task"]) == str(t)]
    return (r[-1]["bytes"]["experts"] + r[-1]["bytes"]["tree"]) if r else 0


def dim_program(t, s, tag=None):
    """The dimension-wise program at stride s; with a fold tag, carrying that fold's windows."""
    from lifelong_ood.policy.compose import DemoProgram
    if tag is None:
        return DemoProgram.load(f"{R}/r7_dim_{t}_s{s}.npz")
    if s == 1:
        prog = DemoProgram.load(f"{R}/r7_dim_{t}_s1.npz")
        prog.fold = DemoProgram.load(f"{R}/r4_{t}{tag}_program.npz").fold
        return prog
    return DemoProgram.load(f"{R}/r7_dimfold_{t}_s{s}.npz")


def arm_bytes(t, arm):
    """HONEST bytes an arm stores for task t, recomputed from its artifacts (rows keep what the launcher computed at the
    time: code programs were under-counted before the 2026-09-25 retraction, replayed demos over-counted before
    2026-09-26).  zeroshot stores nothing; lora = the r3_fewshot adapter (fp16)."""
    from lifelong_ood.policy.compose import DemoProgram
    if arm == "zeroshot":
        return 0
    if arm == "lora":
        r = [x for x in ledger.load("r3_fewshot") if str(x["metrics"]["task"]) == str(t) and not x.get("tag")]
        return int(r[-1]["bytes"]["experts"]) if r else 0
    return int({"raw_replay": lambda: DemoProgram.load(f"{R}/r3_prog_{t}_raw_lib40.npz").bytes(),
                "code_replay": lambda: DemoProgram.load(f"{R}/r3_prog_{t}_code_lib40.npz").bytes(),
                "hybrid_approach": lambda: DemoProgram.load(f"{R}/r4_{t}_approach_program.npz").bytes() + fold_bytes(t, "_approach"),
                "dimcode_s5": lambda: dim_program(t, 5).bytes(), "dimcode_s1": lambda: dim_program(t, 1).bytes(),
                "dimfold_full": lambda: dim_program(t, 5, "_approach").bytes() + fold_bytes(t, "_approach"),
                "dimfold_last2": lambda: dim_program(t, 5, "_approach_last2").bytes() + fold_bytes(t, "_approach_last2"),
                "dimfold_s1_full": lambda: dim_program(t, 1, "_approach").bytes() + fold_bytes(t, "_approach"),
                "dimfold_s1_last2": lambda: dim_program(t, 1, "_approach_last2").bytes() + fold_bytes(t, "_approach_last2"),
                "leaf_s5": lambda: DemoProgram.load(f"{R}/r9_leaf_{t}_s5.npz").bytes(),
                "leaf_s1": lambda: DemoProgram.load(f"{R}/r9_leaf_{t}_s1.npz").bytes(),
                "leaffold_s1_last2": lambda: with_fold(f"{R}/r9_leaf_{t}_s1.npz", t, "_approach_last2").bytes() + fold_bytes(t, "_approach_last2"),
                "shared_s1": lambda: DemoProgram.load(f"{R}/r9_shared_{t}_s1.npz").bytes(),
                "seg_s5": lambda: DemoProgram.load(f"{R}/r10_seg_{t}_s5.npz").bytes(),
                "dimshared_all": lambda: dim_program(t, 1, "_approach_last2").bytes() + shared_fold_share("shared_all"),
                "dimshared_xfit": lambda: dim_program(t, 1, "_approach_last2").bytes()
                + shared_fold_share("xfit_B" if int(t) in XFIT_A else "xfit_A", users=len(XFIT_A if int(t) in XFIT_A else XFIT_B)),
                "playback": lambda: 2 * (len(DemoProgram.load(f"{R}/r3_prog_{t}_raw_lib40.npz").mu) + 9) * 7,
                "vinn": lambda: 2 * (len(DemoProgram.load(f"{R}/r3_prog_{t}_raw_lib40.npz").mu) + 9) * 7
                + 2 * 1024 * len(DemoProgram.load(f"{R}/r3_prog_{t}_raw_lib40.npz").mu),
                "promptret": lambda: 1, "loraret": lambda: 0,
                **{f"{g}_d1": (lambda g=g: demo_arm_bytes(t, g, 1)) for g in ("raw_replay", "dimcode_s1", "dimfold_s1_last2", "lora", "seg_s5")}}[arm]())


XFIT_A = {36, 39, 42, 1, 34, 37, 40, 52, 53, 62}                        # grasping tasks of group A (their fold: xfit_B)
XFIT_B = {3, 4, 5, 32, 41, 43, 48, 58, 63}


def shared_fold_share(tag, users=None):
    """A shared fold is stored once: charge each task that uses it an equal share."""
    r = [x for x in ledger.load("r13_shared_fold") if x["tag"] == tag]
    if not r:
        return 0
    return int(round(r[-1]["bytes"]["experts"] / (users or len(r[-1]["metrics"]["tasks"]))))


def demo_arm_bytes(t, arm, k):
    """Bytes of the arms built from demo k (tags *_d<k>)."""
    from lifelong_ood.policy.compose import DemoProgram
    d = f"_d{k}"
    if arm == "raw_replay":
        return DemoProgram.load(f"{R}/r3_prog_{t}_raw_lib40{d}.npz").bytes()
    if arm == "dimcode_s1":
        return DemoProgram.load(f"{R}/r7_dim_{t}_s1{d}.npz").bytes()
    if arm == "seg_s5":
        return DemoProgram.load(f"{R}/r10_seg_{t}_s5{d}.npz").bytes()
    if arm == "lora":
        r = [x for x in ledger.load("r3_fewshot") if str(x["metrics"]["task"]) == str(t) and x.get("tag") == d]
        return int(r[-1]["bytes"]["experts"]) if r else 0
    prog = DemoProgram.load(f"{R}/r7_dim_{t}_s1{d}.npz")
    prog.fold = DemoProgram.load(f"{R}/r4_{t}_approach_last2{d}_program.npz").fold
    return prog.bytes() + fold_bytes(t, f"_approach_last2{d}")


def with_fold(path, t, tag):
    from lifelong_ood.policy.compose import DemoProgram
    prog = DemoProgram.load(path)
    prog.fold = DemoProgram.load(f"{R}/r4_{t}{tag}_program.npz").fold
    return prog


SIZED = ("raw_replay", "code_replay", "hybrid_approach", "dimcode_s5", "dimcode_s1", "dimfold_full", "dimfold_last2",
         "dimfold_s1_full", "dimfold_s1_last2", "leaf_s5", "leaf_s1", "leaffold_s1_last2", "shared_s1", "seg_s5",
         "dimshared_all", "dimshared_xfit", "playback", "vinn", "promptret", "loraret", "raw_replay_d1", "dimcode_s1_d1", "dimfold_s1_last2_d1", "lora_d1", "seg_s5_d1")

ARMS = ["zeroshot", "raw_replay", "code_replay", "lora", "hybrid_lib40", "expert_on_lib40"]
NAMES = {"zeroshot": "zero-shot", "raw_replay": "demo replay", "code_replay": "library codes", "lora": "few-shot LoRA",
         "hybrid_lib40": "compose + fold", "expert_on_lib40": "fold expert, always on"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def collect(experiments=("m3_compose", "m4_fold")):
    cells = defaultdict(lambda: [0, 0, 0])            # (task, arm) -> k, n, bytes
    lang = {}
    for exp in experiments:
        for r in ledger.load(exp):
            arm = r["tag"]
            for tid, t in r["metrics"]["tasks"].items():
                k = round(t["success_rate"] * t["n"])
                c = cells[(tid, arm)]
                c[0] += k; c[1] += t["n"]; c[2] = max(c[2], int(r["bytes"]["total"]))
                lang[tid] = t["language"]
    for r in ledger.load("r3_fewshot"):
        tid = str(r["metrics"]["task"])
        if (tid, "lora") in cells:
            cells[(tid, "lora")][2] = int(r["bytes"]["experts"])
    for r in ledger.load("r4_fold"):
        tid = str(r["metrics"]["task"])
        tag = r["tag"] or ""
        if (tid, "hybrid" + tag) in cells:
            cells[(tid, "hybrid" + tag)][2] = int(r["bytes"]["total"])
        if (tid, "expert_on" + tag) in cells:
            cells[(tid, "expert_on" + tag)][2] = int(r["bytes"]["experts"])
    return cells, lang


def fmt_bytes(b):
    if b <= 0:
        return "0"
    return f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--heldout", action="store_true", help="the held-out init states (m3_heldout) + the practice rule")
    ap.add_argument("--practice", type=float, default=0.5, help="keep the prior if its practice success >= this")
    ap.add_argument("--record", action="store_true", help="write the derived m3_summary ledger row")
    ap.add_argument("--final", action="store_true", help="the FINAL states (m3_final, 30-49): our method vs every option")
    ap.add_argument("--tasks", default="19,35,36,39,42,67", help="--final: the tasks of the summary (default: the first six)")
    a = ap.parse_args()
    if a.final:
        return final(a)
    cells, lang = collect(("m3_heldout",) if a.heldout else ("m3_compose", "m4_fold"))
    if a.heldout:
        # PRACTICE-THEN-COMPOSE: the prior's success on the development init states (m3_compose zeroshot, states 0-9)
        # decides per task; the chosen arm is scored on the held-out states (10-29).  Bytes: the program only if chosen.
        dev, _ = collect(("m3_compose",))
        for t in list(lang):
            k0, n0, _ = dev.get((t, "zeroshot"), (0, 0, 0))
            keep = n0 > 0 and k0 / n0 >= a.practice
            src = cells.get((t, "zeroshot" if keep else "code_replay"))
            if src:
                cells[(t, "practice")] = [src[0], src[1], 0 if keep else src[2]]
        ARMS.append("practice")
        NAMES["practice"] = f"practice, then compose (prior kept if >= {a.practice:.0%} of 10)"
    tasks = sorted(lang, key=int)
    arms = [x for x in ARMS if any((t, x) in cells for t in tasks)]
    print("| task | " + " | ".join(NAMES.get(x, x) for x in arms) + " |")
    print("|---|" + "---|" * len(arms))
    out = {}
    for t in tasks:
        row = []
        for x in arms:
            if (t, x) not in cells:
                row.append("-")
                continue
            k, n, b = cells[(t, x)]
            lo, hi = wilson(k, n)
            row.append(f"{k}/{n} ({fmt_bytes(b)})")
            out[f"{t}/{x}"] = dict(k=k, n=n, bytes=b, wilson=[lo, hi])
        print(f"| {t} {lang[t][:42]} | " + " | ".join(row) + " |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if a.record and a.heldout:
        # a DERIVED row: aggregates of the m3_heldout rows (and the practice rule), so documents can quote them
        fail = [t for t in tasks if (t, "zeroshot") in cells and cells[(t, "zeroshot")][0] / max(cells[(t, "zeroshot")][1], 1) < 0.5]
        agg = {}
        for x in arms:
            for name, ts in (("all", tasks), ("prior_fails", fail)):
                ks = [cells[(t, x)][0] for t in ts if (t, x) in cells]
                ns = [cells[(t, x)][1] for t in ts if (t, x) in cells]
                if ns:
                    agg[f"{x}.{name}"] = dict(k=int(sum(ks)), n=int(sum(ns)), rate=float(sum(ks) / sum(ns)), tasks=len(ns))
        prog_b = [cells[(t, "code_replay")][2] for t in tasks if (t, "code_replay") in cells]
        lora_b = max([cells[(t, "lora")][2] for t in tasks if (t, "lora") in cells] or [0])
        m = dict(arms=agg, cells=out, prior_fails_tasks=fail, practice_threshold=a.practice,
                 code_bytes_min=int(min(prog_b)), code_bytes_max=int(max(prog_b)), lora_bytes=int(lora_b), lora_mb=round(lora_b / 1e6, 1))
        ledger.record("m3_summary", bench="libero_90", tag="heldout", seed=7, metrics=m,
                      bytes=dict(experts=0, tree=0, codes=int(sum(prog_b)), samples=0, teacher_copies=0, total=int(sum(prog_b))),
                      config=vars(a), script="experiments/m34_report.py", command="python experiments/m34_report.py --heldout --record",
                      notes="derived from ledger/m3_heldout (init states 10-29) and ledger/m3_compose zero-shot (states 0-9, the practice)")
        print("[report] m3_summary row written")


def final(a):
    """Our method = practise, then compose: keep the prior where its 10 practice episodes (development states 0-9)
    succeed >= a.practice; elsewhere follow the demo's program of LIBRARY codes, with the rank-4 fold switched on in the
    approach / contact windows when the demo grasps or releases.  Every alternative gets the same practice step."""
    cells, lang = collect(("m3_final",))
    keep_tasks = {x for x in a.tasks.split(",") if x}
    cells = {k: v for k, v in cells.items() if k[0] in keep_tasks}
    lang = {k: v for k, v in lang.items() if k in keep_tasks}
    dev, _ = collect(("m3_compose",))
    for r in ledger.load("r3_fewshot"):
        t = str(r["metrics"]["task"])
        if (t, "lora") in cells:
            cells[(t, "lora")][2] = int(r["bytes"]["experts"])
    tasks = sorted(lang, key=int)
    # HONEST sizes, recomputed from the artifacts for EVERY arm (arm_bytes)
    for t in tasks:
        for arm in SIZED:
            if (t, arm) in cells:
                cells[(t, arm)][2] = arm_bytes(t, arm)
    keep = {t: (lambda k, n: n > 0 and k / n >= a.practice)(*dev.get((t, "zeroshot"), (0, 0, 0))[:2]) for t in tasks}
    def pick(t, arm):
        if keep[t]:
            return cells.get((t, "zeroshot")), "prior"
        if arm == "ours":
            return (cells.get((t, "hybrid_approach")) or cells.get((t, "code_replay"))), ("codes+fold" if (t, "hybrid_approach") in cells else "codes")
        if arm in ("dim", "dim_last2", "dim_full", "dim1", "dim1_last2", "dim1_full"):
            base = "dimcode_s1" if arm.startswith("dim1") else "dimcode_s5"
            fold = {"dim": None, "dim_last2": "dimfold_last2", "dim_full": "dimfold_full",
                    "dim1": None, "dim1_last2": "dimfold_s1_last2", "dim1_full": "dimfold_s1_full"}[arm]
            c = cells.get((t, fold)) if fold else None
            return (c or cells.get((t, base))), ("dim+fold" if c else "dim")
        return cells.get((t, arm)), arm
    rows = {}
    for t in tasks:
        rows[t] = {x: cells.get((t, x)) for x in ("zeroshot", "raw_replay", "code_replay", "lora", "hybrid_approach", "dimcode_s5",
                                                 "dimfold_full", "dimfold_last2", "dimcode_s1", "dimfold_s1_full", "dimfold_s1_last2")}
        for name, arm in (("ours", "ours"), ("practice_lora", "lora"), ("practice_replay", "raw_replay"), ("practice_codes", "code_replay"),
                          ("ours_dim", "dim"), ("ours_dim_last2", "dim_last2"), ("ours_dim_full", "dim_full"),
                          ("ours_dim1", "dim1"), ("ours_dim1_last2", "dim1_last2"), ("ours_dim1_full", "dim1_full")):
            c, how = pick(t, arm)
            rows[t][name] = None if c is None else [c[0], c[1], 0 if keep[t] else c[2], how]
    cols = ["zeroshot", "practice_replay", "practice_codes", "practice_lora", "ours", "ours_dim", "ours_dim_last2", "ours_dim_full",
            "ours_dim1", "ours_dim1_last2", "ours_dim1_full"]
    names = {"zeroshot": "zero-shot", "raw_replay": "demo replay", "code_replay": "library codes", "lora": "few-shot LoRA",
             "ours": "practise, codes + 4.7 MB fold", "practice_lora": "practise, LoRA", "practice_replay": "practise, replay",
             "practice_codes": "practise, codes", "ours_dim": "practise, dim-borrow", "ours_dim_last2": "practise, dim-borrow + 258 KB fold",
             "ours_dim_full": "practise, dim-borrow + 4.7 MB fold", "ours_dim1": "practise, dim-borrow (every frame)",
             "ours_dim1_last2": "practise, dim-borrow (every frame) + 258 KB fold", "ours_dim1_full": "practise, dim-borrow (every frame) + 4.7 MB fold"}
    cols = [c for c in cols if any(rows[t].get(c) is not None for t in tasks)]
    print("| task | " + " | ".join(names[c] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    tot = {c: [0, 0] for c in cols}
    fail_tot = {c: [0, 0] for c in cols}
    fails = [t for t in tasks if not keep[t]]
    for t in tasks:
        out = []
        for c in cols:
            v = rows[t][c]
            if v is None:
                out.append("-"); continue
            out.append(f"{v[0]}/{v[1]} ({fmt_bytes(v[2])})")
            tot[c][0] += v[0]; tot[c][1] += v[1]
            if t in fails:
                fail_tot[c][0] += v[0]; fail_tot[c][1] += v[1]
        print(f"| {t} {lang[t][:40]} | " + " | ".join(out) + " |")
    print("| all | " + " | ".join(f"{k}/{n}" for k, n in (tot[c] for c in cols)) + " |")
    print("| prior fails | " + " | ".join(f"{k}/{n}" for k, n in (fail_tot[c] for c in cols)) + " |")
    if a.record:
        agg = {c: dict(k=tot[c][0], n=tot[c][1], rate=tot[c][0] / max(tot[c][1], 1),
                       fail_k=fail_tot[c][0], fail_n=fail_tot[c][1], fail_rate=fail_tot[c][0] / max(fail_tot[c][1], 1)) for c in cols}
        store = {c: sum((rows[t][c][2] if rows[t][c] is not None else 0) for t in tasks) for c in cols}
        m = dict(arms=agg, stored_bytes=store, kept_prior=[t for t in tasks if keep[t]], rows={t: {c: rows[t][c] for c in cols} for t in tasks},
                 practice_threshold=a.practice)
        ledger.record("m3_summary", bench="libero_90", tag="final", seed=7, metrics=m,
                      bytes=dict(experts=0, tree=0, codes=int(store["ours"]), samples=0, teacher_copies=0, total=int(store["ours"])),
                      config=vars(a), script="experiments/m34_report.py", command="python experiments/m34_report.py --final --record",
                      notes="derived from ledger/m3_final (states 30-49); practice = ledger/m3_compose zero-shot (states 0-9)")
        print("[report] m3_summary (final) row written")


if __name__ == "__main__":
    main()
