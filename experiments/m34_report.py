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
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
