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
    dev, _ = collect(("m3_compose",))
    for r in ledger.load("r3_fewshot"):
        t = str(r["metrics"]["task"])
        if (t, "lora") in cells:
            cells[(t, "lora")][2] = int(r["bytes"]["experts"])
    tasks = sorted(lang, key=int)
    keep = {t: (lambda k, n: n > 0 and k / n >= a.practice)(*dev.get((t, "zeroshot"), (0, 0, 0))[:2]) for t in tasks}
    def pick(t, arm):
        if keep[t]:
            return cells.get((t, "zeroshot")), "prior"
        if arm == "ours":
            return (cells.get((t, "hybrid_approach")) or cells.get((t, "code_replay"))), ("codes+fold" if (t, "hybrid_approach") in cells else "codes")
        return cells.get((t, arm)), arm
    rows = {}
    for t in tasks:
        rows[t] = {x: cells.get((t, x)) for x in ("zeroshot", "raw_replay", "code_replay", "lora", "hybrid_approach")}
        for name, arm in (("ours", "ours"), ("practice_lora", "lora"), ("practice_replay", "raw_replay"), ("practice_codes", "code_replay")):
            c, how = pick(t, arm)
            rows[t][name] = None if c is None else [c[0], c[1], 0 if keep[t] else c[2], how]
    cols = ["zeroshot", "raw_replay", "code_replay", "lora", "ours", "practice_lora", "practice_replay"]
    names = {"zeroshot": "zero-shot", "raw_replay": "demo replay", "code_replay": "library codes", "lora": "few-shot LoRA",
             "ours": "OURS: practise, compose (+ fold)", "practice_lora": "practise, then LoRA", "practice_replay": "practise, then replay"}
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
