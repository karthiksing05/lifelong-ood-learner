"""M3 FINAL statistics from the per-episode outcomes (init states 30-49 of every held-out task in ledger/m3_final).

Every method gets the same practice step: keep the prior where its 10 practice episodes (development states 0-9,
ledger/m3_compose zero-shot) succeed >= 50%, else use the method.  Then, on the paired init states:
  * pooled success per method: Wilson 95% and the two-level cluster bootstrap (tasks, then states);
  * pooled differences between methods: the paired two-level bootstrap and the exact sign test on discordant pairs;
  * NON-INFERIORITY (margin fixed 2026-09-26, before wave A landed): a method is "at least as successful" as another if
    the one-sided 95% lower bound of their pooled difference (5th bootstrap percentile) is above -0.10.
Bytes per method = the HONEST sizes of experiments/m34_report.arm_bytes (0 where the prior is kept).

    python experiments/m3_stats.py [--record] [--json results/robot/m3_stats.json]
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger, stats                                  # noqa: E402

spec = importlib.util.spec_from_file_location("m34", os.path.join(ROOT, "experiments", "m34_report.py"))
m34 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m34)

MARGIN = 0.10
# method -> (the arm used where the practice fails, a fallback arm when the task has no fold)
METHODS = {"zeroshot": None, "practice_replay": ("raw_replay", None), "practice_lora": ("lora", None),
           "ours_dim": ("dimcode_s5", None), "ours_dim1": ("dimcode_s1", None),
           "ours_dim1_last2": ("dimfold_s1_last2", "dimcode_s1"), "ours_codes_full": ("hybrid_approach", "code_replay"),
           "ours_leaf": ("leaf_s5", None), "ours_leaf1_last2": ("leaffold_s1_last2", "leaf_s1"), "ours_shared1": ("shared_s1", None),
           "ours_seg": ("seg_s5", None)}
NAMES = {"zeroshot": "zero-shot", "practice_replay": "demo replay", "practice_lora": "few-shot LoRA (rank 16)",
         "ours_dim": "borrowing, every 5th frame", "ours_dim1": "borrowing, every frame",
         "ours_dim1_last2": "borrowing + 258 KB fold", "ours_codes_full": "PoE codes + 4.7 MB fold",
         "ours_leaf": "leaf borrowing, every 5th frame", "ours_leaf1_last2": "leaf borrowing + 258 KB fold",
         "ours_shared1": "shared-prototype borrowing", "ours_seg": "segment borrowing"}
COMPARE = [("ours_dim1_last2", "practice_lora"), ("ours_dim1_last2", "practice_replay"), ("ours_dim1_last2", "zeroshot"),
           ("ours_dim", "practice_lora"), ("ours_dim", "practice_replay"), ("ours_dim1", "practice_replay"),
           ("practice_lora", "practice_replay"), ("ours_leaf1_last2", "practice_lora"), ("ours_leaf1_last2", "practice_replay"),
           ("ours_leaf", "ours_dim"), ("ours_leaf1_last2", "ours_dim1_last2"), ("ours_shared1", "ours_dim1"),
           ("ours_seg", "practice_replay"), ("ours_seg", "practice_lora"), ("ours_seg", "ours_dim")]


def episodes(experiment):
    """(task, tag) -> {trial: 0/1} from the rows' result files (the last row wins for a repeated trial)."""
    out = defaultdict(dict)
    lang = {}
    for r in ledger.load(experiment):
        arts = [a for a in (r.get("artifacts") or []) if a.endswith(".json")]
        if not arts or not os.path.exists(arts[0]):
            continue
        res = json.load(open(arts[0]))
        for tid, t in res["tasks"].items():
            lang[tid] = t["language"]
            for e in t["episodes"]:
                if e.get("error") is None:
                    out[(tid, r["tag"])][int(e["trial"])] = int(bool(e["success"]))
    return out, lang


def build(practice=0.5):
    fin, lang = episodes("m3_final")
    dev, _ = episodes("m3_compose")
    tasks = sorted({t for t, _ in fin}, key=int)
    table, keep, how, nbytes = {}, {}, {}, {}
    for t in tasks:
        d = dev.get((t, "zeroshot"), {})
        keep[t] = len(d) > 0 and sum(d.values()) / len(d) >= practice
        table[t], how[t], nbytes[t] = {}, {}, {}
        for m, arms in METHODS.items():
            if m == "zeroshot" or keep[t]:
                src = "zeroshot"
            else:
                src = arms[0] if (t, arms[0]) in fin else arms[1]
            if src is None or (t, src) not in fin:
                continue
            table[t][m] = dict(fin[(t, src)])
            how[t][m] = src
            nbytes[t][m] = 0 if src == "zeroshot" else m34.arm_bytes(t, src)
    return table, keep, how, nbytes, lang, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", type=float, default=0.5)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "robot", "m3_stats.json"))
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    table, keep, how, nbytes, lang, dev = build(a.practice)
    tasks = list(table)
    fails = [t for t in tasks if not keep[t]]
    # the methods every task has: comparisons run on those tasks only
    common = [m for m in METHODS if all(m in table[t] for t in tasks)]
    print(f"[stats] {len(tasks)} tasks ({len(fails)} where the practice fails); methods on every task: {common}")
    res = dict(tasks=tasks, prior_fails=fails, kept_prior=[t for t in tasks if keep[t]], methods={}, compare={},
               margin=MARGIN, practice_threshold=a.practice, language={t: lang[t] for t in tasks},
               per_task={t: {m: dict(k=int(sum(v.values())), n=len(v), arm=how[t][m], bytes=int(nbytes[t][m]))
                             for m, v in table[t].items()} for t in tasks},
               practice={t: dict(k=int(sum(dev.get((t, "zeroshot"), {}).values())), n=len(dev.get((t, "zeroshot"), {}))) for t in tasks})
    for scope, ts in (("all", tasks), ("prior_fails", fails)):
        for m in METHODS:
            tm = [t for t in ts if m in table[t]]
            if not tm:
                continue
            b = stats.bootstrap_rate(table, m, tm, B=a.B)
            lo, hi = stats.wilson(b["k"], b["n"])
            b.update(wilson=[lo, hi], bytes=int(sum(nbytes[t][m] for t in tm)), complete=len(tm) == len(ts))
            res["methods"][f"{m}.{scope}"] = b
        for x, y in COMPARE:
            pairs = stats.paired(table, x, y, ts)
            if not pairs:
                continue
            d = stats.bootstrap_diff(pairs, B=a.B)
            bb, cc = stats.discordant(pairs)
            d.update(discordant=[bb, cc], sign_p=stats.sign_test(bb, cc), noninferior=bool(d["lo90"] > -MARGIN))
            res["compare"][f"{x}-{y}.{scope}"] = d
    # test-retest: the same (task, arm) on two DISJOINT init-state sets (held-out 10-29 in m3_heldout vs final 30-49)
    def cells(exp):
        c = {}
        for r in ledger.load(exp):
            for t, v in r["metrics"]["tasks"].items():
                x = c.setdefault((t, r["tag"]), [0, 0])
                x[0] += round(v["success_rate"] * v["n"]); x[1] += v["n"]
        return c
    h, f = cells("m3_heldout"), cells("m3_final")
    common = sorted(set(h) & set(f))
    if len(common) >= 3:
        x = np.array([h[k][0] / h[k][1] for k in common]); y = np.array([f[k][0] / f[k][1] for k in common])
        res["retest"] = dict(cells=len(common), pearson=float(np.corrcoef(x, y)[0, 1]), mean_abs_diff=float(np.abs(x - y).mean()))
        print(f"[stats] test-retest over {len(common)} (task, arm) cells on disjoint init states: r = {res['retest']['pearson']:.3f}, "
              f"mean |difference| {res['retest']['mean_abs_diff']:.3f}")
    json.dump(res, open(a.json, "w"), indent=1)
    for scope in ("all", "prior_fails"):
        print(f"\n[{scope}]  method: pooled success (Wilson 95%) [cluster bootstrap 95%], bytes stored")
        for m in METHODS:
            r = res["methods"].get(f"{m}.{scope}")
            if r:
                print(f"  {NAMES[m]:30s} {r['k']:3d}/{r['n']:3d} = {r['rate']:.3f} ({r['wilson'][0]:.2f}-{r['wilson'][1]:.2f}) "
                      f"[{r['lo']:.2f}-{r['hi']:.2f}] over {r['tasks']} tasks, {r['bytes'] / 1e3:.1f} KB{'' if r['complete'] else ' (not every task)'}")
        for x, y in COMPARE:
            d = res["compare"].get(f"{x}-{y}.{scope}")
            if d:
                print(f"  {NAMES[x]} - {NAMES[y]}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], one-sided lower {d['lo90']:+.3f} "
                      f"({'non-inferior' if d['noninferior'] else 'not shown non-inferior'} at -{MARGIN:.2f}); discordant "
                      f"{d['discordant'][0]} vs {d['discordant'][1]}, sign test p = {d['sign_p']:.3f}; {d['n_pairs']} pairs, {d['n_tasks']} tasks")
    if a.record:
        ledger.record("m3_stats", bench="libero_90", tag=f"final_{len(tasks)}tasks", seed=0, metrics=res,
                      bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0),
                      config=vars(a), script="experiments/m3_stats.py", command="python experiments/m3_stats.py --record",
                      notes="derived from the per-episode result files of ledger/m3_final (states 30-49) and ledger/m3_compose "
                            "zero-shot (states 0-9, the practice); bootstrap = tasks, then init states, paired across methods")
        print("[stats] m3_stats row written")


if __name__ == "__main__":
    main()
