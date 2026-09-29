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
           "ours_seg": ("seg_s5", None), "ours_alloc": "practice",
           "ours_shared_all": ("dimshared_all", "dimcode_s1"), "ours_shared_xfit": ("dimshared_xfit", "dimcode_s1"),
           "base_playback": ("playback", None), "base_vinn": ("vinn", None), "base_promptret": ("promptret", None),
           "base_loraret": ("loraret", None),
           "ours_learned": ("learned", None), "ours_learned_mem0": ("learned_mem0", None),
           "ours_learned_origfold": ("learned_origfold", None), "base_lora_learn": ("lora_learn", None),
           "base_demodiff": ("demodiff", None), "base_demodiff_state": ("demodiff_state", None),
           "base_lora_eq": ("lora_eq", None), "base_lora_r64": ("lora_r64", None), "base_fullft": ("fullft", None),
           "loop_eq1": ("lora_r16eq1", None), "loop_L2": ("lora_r16L2", None), "loop_eq2": ("lora_r16eq2", None),
           "loop_L3": ("lora_r16L3", None), "loop_eq3": ("lora_r16eq3", None),
           "nft_bal": ("lora_r17bal", None), "nft_pool": ("lora_r17pool", None), "nft_fold": ("lora_r17fold", None),
           "rec_bal": ("lora_r18rec", None), "fam_pool": ("lora_r20fam", None), "rnd_pool": ("lora_r20rnd", None),
           "fam_2stage": ("lora_r21fam", None), "rnd_2stage": ("lora_r21rnd", None), "nft_phase": ("lora_r24phase", None),
           "ours_grow": ("growleaf", None), "gen_lora": ("lora_r25gen", None), "ours_phasebind": ("phasebind", None),
           "ours_phasebind_nofold": ("phasebind_nofold", None), "ours_phasebind_reader": ("phasebind_reader", None),
           "ours_phasebind2": ("phasebind2_nofold", None), "ours_phasebind2_reader": ("phasebind2_reader", None),
           "gen_lora_v2": ("lora_r27gen", None), "ours_phasebind2_synth": ("phasebind2_reader_spatial4_synth", None)}
NAMES = {"zeroshot": "zero-shot", "practice_replay": "demo replay", "practice_lora": "few-shot LoRA (rank 16)",
         "ours_dim": "borrowing, every 5th frame", "ours_dim1": "borrowing, every frame",
         "ours_dim1_last2": "borrowing + 258 KB fold", "ours_codes_full": "PoE codes + 4.7 MB fold",
         "ours_leaf": "leaf borrowing, every 5th frame", "ours_leaf1_last2": "leaf borrowing + 258 KB fold",
         "ours_shared1": "shared-prototype borrowing", "ours_seg": "segment borrowing",
         "ours_alloc": "practice-allocated (program, else LoRA)",
         "ours_shared_all": "borrowing + one shared fold", "ours_shared_xfit": "borrowing + the other half's fold",
         "base_playback": "open-loop demo playback", "base_vinn": "VINN (nearest demo frame)",
         "base_promptret": "instruction retrieval", "base_loraret": "adapter retrieval (nearest task's LoRA)",
         "ours_learned": "compose, practise, consolidate", "ours_learned_mem0": "consolidated fold, no memory",
         "ours_learned_origfold": "memory, the demo's fold", "base_lora_learn": "LoRA on the demo + practice successes",
         "base_demodiff": "DemoDiffusion (s* 0.2)", "base_demodiff_state": "DemoDiffusion, state-aligned",
         "base_lora_eq": "equal-bytes adapter (258 KB, whole demo)", "base_lora_r64": "LoRA rank 64", "base_fullft": "full fine-tuning",
         "loop_eq1": "258 KB learner, round 1", "loop_L2": "LoRA, round 2 (success-only)", "loop_eq2": "258 KB learner, round 2",
         "loop_L3": "LoRA, round 3", "loop_eq3": "258 KB learner, round 3",
         "nft_bal": "LoRA + failure negatives (balanced)", "nft_pool": "LoRA + failure negatives (pooled)",
         "nft_fold": "258 KB + failure negatives", "rec_bal": "LoRA + failure negatives + recovery branches",
         "fam_pool": "one demo + the family's data (t = 0.4)", "rnd_pool": "one demo + random tasks' data",
         "fam_2stage": "family pretraining, then the one demo", "rnd_2stage": "random-task pretraining, then the one demo",
         "nft_phase": "LoRA + failure negatives at the failed phase", "ours_grow": "leaf borrowing over the grown library + fold",
         "gen_lora": "LoRA on the demo + generated demonstrations (no practice)", "ours_phasebind": "phase bindings through the prior + fold (sim poses)",
         "ours_phasebind_nofold": "phase bindings through the prior (sim poses)",
         "ours_phasebind_reader": "phase bindings through the prior (learned pose reader)",
         "ours_phasebind2": "phase bindings v2 through the prior (sim poses)",
         "ours_phasebind2_reader": "phase bindings v2 through the prior (learned pose reader)",
         "gen_lora_v2": "LoRA on the demo + v2-generated demonstrations (no practice)",
         "ours_phasebind2_synth": "phase bindings v2 through the prior (reader trained with rendered scenes)"}
COMPARE = [("ours_dim1_last2", "practice_lora"), ("ours_dim1_last2", "practice_replay"), ("ours_dim1_last2", "zeroshot"),
           ("ours_dim", "practice_lora"), ("ours_dim", "practice_replay"), ("ours_dim1", "practice_replay"),
           ("practice_lora", "practice_replay"), ("ours_leaf1_last2", "practice_lora"), ("ours_leaf1_last2", "practice_replay"),
           ("ours_leaf", "ours_dim"), ("ours_leaf1_last2", "ours_dim1_last2"), ("ours_shared1", "ours_dim1"),
           ("ours_seg", "practice_replay"), ("ours_seg", "practice_lora"), ("ours_seg", "ours_dim"),
           ("ours_dim1_last2", "ours_dim1"), ("ours_alloc", "practice_lora"), ("ours_alloc", "ours_dim1_last2"),
           ("ours_shared_all", "ours_dim1_last2"), ("ours_shared_xfit", "ours_dim1_last2"), ("ours_shared_all", "practice_lora"),
           ("ours_shared_xfit", "practice_lora"), ("ours_shared_xfit", "ours_dim1"),
           ("ours_dim1_last2", "base_playback"), ("ours_dim1_last2", "base_vinn"), ("ours_dim1_last2", "base_promptret"),
           ("ours_dim1_last2", "base_loraret"), ("practice_replay", "base_playback"),
           ("ours_learned", "ours_dim1_last2"), ("ours_learned", "base_lora_learn"), ("ours_learned", "practice_lora"),
           ("ours_learned_mem0", "ours_dim1_last2"), ("ours_learned_origfold", "ours_dim1_last2"), ("base_lora_learn", "practice_lora"),
           ("ours_dim1_last2", "base_demodiff"), ("ours_dim1_last2", "base_demodiff_state"), ("ours_seg", "base_demodiff"),
           ("ours_dim1_last2", "base_lora_eq"), ("ours_dim1_last2", "base_lora_r64"), ("ours_dim1_last2", "base_fullft"),
           ("loop_L2", "base_lora_learn"), ("loop_L3", "base_lora_learn"), ("loop_L3", "practice_lora"), ("loop_eq3", "loop_L3"),
           ("loop_eq1", "base_lora_learn"), ("nft_bal", "loop_L2"), ("nft_pool", "loop_L2"), ("nft_fold", "loop_eq2"),
           ("rec_bal", "nft_bal"), ("rec_bal", "loop_L3"),
           ("fam_pool", "rnd_pool"), ("fam_pool", "practice_lora"), ("rnd_pool", "practice_lora"),
           ("fam_2stage", "practice_lora"), ("fam_2stage", "rnd_2stage"), ("rnd_2stage", "practice_lora"),
           ("nft_phase", "nft_bal"), ("nft_phase", "loop_L2"), ("ours_grow", "ours_leaf1_last2"),
           ("gen_lora", "practice_lora"), ("gen_lora", "base_lora_learn"), ("ours_phasebind", "ours_dim1_last2"),
           ("ours_phasebind_nofold", "ours_dim1_last2"), ("ours_phasebind_nofold", "nft_bal"),
           ("ours_phasebind_reader", "ours_phasebind_nofold"), ("ours_phasebind_reader", "ours_dim1_last2"),
           ("ours_phasebind2", "ours_phasebind_nofold"), ("ours_phasebind2", "ours_dim1_last2"), ("ours_phasebind2", "nft_bal"),
           ("ours_phasebind2_reader", "ours_phasebind2"), ("ours_phasebind2_reader", "ours_dim1_last2"),
           ("ours_phasebind2_reader", "nft_bal"), ("gen_lora_v2", "gen_lora"), ("gen_lora_v2", "nft_bal"),
           ("ours_phasebind2_synth", "ours_phasebind2"), ("ours_phasebind2_synth", "nft_bal"),
           ("ours_phasebind2_synth", "ours_phasebind2_reader")]


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
    prac, _ = episodes("m3_practice")
    tasks = sorted({t for t, _ in fin}, key=int)
    table, keep, how, nbytes = {}, {}, {}, {}
    for t in tasks:
        d = dev.get((t, "zeroshot"), {})
        keep[t] = len(d) > 0 and sum(d.values()) / len(d) >= practice
        table[t], how[t], nbytes[t] = {}, {}, {}
        for m, arms in METHODS.items():
            if m == "ours_alloc":
                continue                                                  # built below from the practice runs
            if m == "zeroshot" or keep[t]:
                src = "zeroshot"
            else:
                src = arms[0] if (t, arms[0]) in fin else arms[1]
            if src is None or (t, src) not in fin:
                continue
            table[t][m] = dict(fin[(t, src)])
            how[t][m] = src
            nbytes[t][m] = 0 if src == "zeroshot" else m34.arm_bytes(t, src)
        # ALLOCATE BYTES BY PRACTICE (pre-registered, ledger m3_prereg tag alloc): the program if its practice on states
        # 0-9 succeeds >= 0.5, else the LoRA
        if keep[t]:
            src = "zeroshot"
        else:
            prog = "dimfold_s1_last2" if (t, "dimfold_s1_last2") in prac else "dimcode_s1"
            p = prac.get((t, prog))
            src = None if not p else (prog if sum(p.values()) / len(p) >= practice else "lora")
        if src and (t, src) in fin:
            table[t]["ours_alloc"] = dict(fin[(t, src)])
            how[t]["ours_alloc"] = src
            nbytes[t]["ours_alloc"] = 0 if src == "zeroshot" else m34.arm_bytes(t, src)
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
    # DEMONSTRATION ROBUSTNESS: the same arms built from a second demo (IPEC episode 1; tags *_d1), same init states
    fin_all, _ = episodes("m3_final")
    dtasks = sorted({t for (t, g) in fin_all if g.endswith("_d1")}, key=int)
    if dtasks:
        rob = {}
        for name, tags in (("replay", ("raw_replay",)), ("borrowing (+ fold where it grasps)", ("dimfold_s1_last2", "dimcode_s1")),
                           ("few-shot LoRA", ("lora",)), ("segment borrowing", ("seg_s5",))):
            row = {}
            for d, sfx in (("demo0", ""), ("demo1", "_d1")):
                k = n = 0
                for t in dtasks:
                    for g in tags:
                        if (t, g + "_d1") in fin_all and (t, g + sfx) in fin_all:
                            v = fin_all[(t, g + sfx)]
                            k += sum(v.values()); n += len(v)
                            break
                row[d] = dict(k=int(k), n=int(n))
            rob[name] = row
        res["demo_robustness"] = dict(tasks=dtasks, arms=rob)
        print(f"[stats] demonstration robustness on tasks {dtasks} (states 30-49): " + "; ".join(
            f"{k} {v['demo0']['k']}/{v['demo0']['n']} (demo 0) vs {v['demo1']['k']}/{v['demo1']['n']} (demo 1)" for k, v in rob.items()))
    # SECOND DEMONSTRATION (pre-registered, ledger m3_prereg tag demo1): D1 = borrowing + fold vs the LoRA with demo 1;
    # D2 = both demos pooled (per task, the episodes of both demos; resample tasks, then (demo, state) pairs)
    d1t = [t for t in fails if (t, "lora_d1") in fin_all and ((t, "dimfold_s1_last2_d1") in fin_all or (t, "dimcode_s1_d1") in fin_all)
           and (t, "raw_replay_d1") in fin_all]
    if len(d1t) >= 3:
        tb = {}
        for t in d1t:
            o1 = fin_all[(t, "dimfold_s1_last2_d1")] if (t, "dimfold_s1_last2_d1") in fin_all else fin_all[(t, "dimcode_s1_d1")]
            tb[t] = {"ours_d1": o1, "lora_d1": fin_all[(t, "lora_d1")], "replay_d1": fin_all[(t, "raw_replay_d1")],
                     "ours_both": {**{(0, i): v for i, v in table[t]["ours_dim1_last2"].items()}, **{(1, i): v for i, v in o1.items()}},
                     "lora_both": {**{(0, i): v for i, v in table[t]["practice_lora"].items()},
                                   **{(1, i): v for i, v in fin_all[(t, "lora_d1")].items()}},
                     "replay_both": {**{(0, i): v for i, v in table[t]["practice_replay"].items()},
                                     **{(1, i): v for i, v in fin_all[(t, "raw_replay_d1")].items()}}}
        dem = dict(tasks=d1t, rates={}, compare={})
        for m in ("ours_d1", "lora_d1", "replay_d1", "ours_both", "lora_both", "replay_both"):
            k = sum(sum(tb[t][m].values()) for t in d1t); n = sum(len(tb[t][m]) for t in d1t)
            dem["rates"][m] = dict(k=int(k), n=int(n))
        for x, y in (("ours_d1", "lora_d1"), ("ours_d1", "replay_d1"), ("ours_both", "lora_both"), ("ours_both", "replay_both")):
            pr = stats.paired(tb, x, y, d1t)
            d = stats.bootstrap_diff(pr, B=a.B)
            bb, cc = stats.discordant(pr)
            d.update(discordant=[bb, cc], sign_p=stats.sign_test(bb, cc), noninferior=bool(d["lo90"] > -MARGIN))
            dem["compare"][f"{x}-{y}"] = d
        res["second_demo"] = dem
        print(f"[stats] SECOND DEMO over {len(d1t)} tasks: " + "; ".join(f"{m} {v['k']}/{v['n']}" for m, v in dem["rates"].items()))
        for key, d in dem["compare"].items():
            print(f"  {key}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], one-sided lower {d['lo90']:+.3f} "
                  f"({'non-inferior' if d['noninferior'] else 'not shown'}), discordant {d['discordant']}, sign p {d['sign_p']:.3f}")
    # EXPLORATORY (not pre-registered): does the library's novelty read (the demo frames it leaves unexplained,
    # r3_prepare tag _lib40) predict where composition succeeds, or where it trails LoRA?  Spearman, permutation p.
    nov = {}
    for r in ledger.load("r3_prepare"):
        if r.get("tag") == "_lib40":
            nov[str(r["metrics"]["task_id"])] = float(r["metrics"]["unexplained_frac"])
    ts = [t for t in fails if t in nov and "ours_dim1_last2" in table[t] and "practice_lora" in table[t]]
    if len(ts) >= 6:
        def rate(t, m):
            v = table[t][m]
            return sum(v.values()) / len(v)
        def spearman(x, y):
            rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
            return float(np.corrcoef(rx, ry)[0, 1])
        rng = np.random.default_rng(0)
        x = np.array([nov[t] for t in ts])
        ex = {}
        for name, y in (("ours", [rate(t, "ours_dim1_last2") for t in ts]), ("lora", [rate(t, "practice_lora") for t in ts]),
                        ("ours_minus_lora", [rate(t, "ours_dim1_last2") - rate(t, "practice_lora") for t in ts])):
            y = np.array(y)
            rho = spearman(x, y)
            null = np.array([spearman(x, rng.permutation(y)) for _ in range(5000)])
            ex[name] = dict(rho=rho, perm_p=float((1 + (np.abs(null) >= abs(rho)).sum()) / 5001))
        res["exploratory_novelty"] = dict(tasks=ts, unexplained=[nov[t] for t in ts], **ex)
        print(f"[stats] EXPLORATORY: unexplained fraction vs success over {len(ts)} tasks: ours rho {ex['ours']['rho']:+.2f} "
              f"(p {ex['ours']['perm_p']:.3f}), LoRA rho {ex['lora']['rho']:+.2f} (p {ex['lora']['perm_p']:.3f}), "
              f"ours - LoRA rho {ex['ours_minus_lora']['rho']:+.2f} (p {ex['ours_minus_lora']['perm_p']:.3f})")
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
