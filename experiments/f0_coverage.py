"""P0a of the overhaul (plan 2026-10-01): is pi0.5's zero-shot failure on LIBERO-90 a failure to READ a factor in context?

Joins every LIBERO-90 task's factor coverage (lifelong_ood/policy/factors.py, against the 40 training tasks) with pi0.5's
zero-shot success (ledger/zeroshot90, init states 0-4, 5 episodes per task), and reports:
  - per feature, success when it holds vs when it does not, and the AUC of the feature (or rule) for success;
  - the held-out tasks' coverage rows and group (A: every factor named in this scene family; B: named elsewhere only;
    C: a factor never named in training);
  - per held-out task, the TASK PROTOTYPES that cover it (the plan's decomposition): the readable bases in its scene (tasks
    the prior solves there), the sources of its object (tasks that move that object type) and of its relation.
Exploratory and derived (no new rollout): it fixes the factor parse, whose hash goes into every later pre-registration.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/f0_coverage.py [--record]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import factors as F                                   # noqa: E402

R = os.path.join(ROOT, "results", "robot")
HELD = (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63)
SOLVED = 0.6


def zeroshot():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "ledger", "zeroshot90", "*.json")):
        for t, v in json.load(open(f))["metrics"]["tasks"].items():
            out[int(t)] = float(v.get("success_rate", 0.0))
    return out


def auc(score, label):
    """Probability that a random success scores above a random failure (ties count half)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def main():
    tasks = F.load()
    train = [t for t in tasks if t.suite != "libero_90"]
    l90 = [t for t in tasks if t.suite == "libero_90"]
    zs = zeroshot()
    rows = {}
    for t in l90:
        c = F.coverage(t, train)
        rows[t.index] = dict(language=t.language, scene=t.scene, family=t.family, slots=[list(s) for s in t.slots],
                             qualifiers=list(t.qualifiers), zeroshot=zs.get(t.index), **c)
    feats = ["scene_seen", "obj_same_scene", "obj_same_family", "obj_any", "rel_same_scene", "rel_same_family", "rel_any",
             "pred_any", "quals_seen"]
    ids = [i for i in rows if rows[i]["zeroshot"] is not None]
    y = [rows[i]["zeroshot"] >= SOLVED for i in ids]
    feature_read = {}
    for f in feats + ["contrast_obj"]:
        v = [float(rows[i][f]) for i in ids]
        on = [rows[i]["zeroshot"] for i in ids if rows[i][f]]
        off = [rows[i]["zeroshot"] for i in ids if not rows[i][f]]
        feature_read[f] = dict(auc=auc(v, y), n_on=len(on), n_off=len(off), success_on=float(np.mean(on)) if on else None,
                               success_off=float(np.mean(off)) if off else None)
    # the rule the plan states: every factor named in this scene family (object moved there, relation there or anywhere)
    rule = {i: bool(rows[i]["obj_same_family"] or not rows[i]["slots"] or rows[i]["slots"][0][0] not in ("In", "On"))
            and bool(rows[i]["rel_any"]) and bool(rows[i]["quals_seen"]) for i in ids}
    feature_read["rule_family_named"] = dict(auc=auc([float(rule[i]) for i in ids], y), n_on=sum(rule.values()),
                                             success_on=float(np.mean([rows[i]["zeroshot"] for i in ids if rule[i]])),
                                             success_off=float(np.mean([rows[i]["zeroshot"] for i in ids if not rule[i]])))
    # groups and covering task prototypes for the held-out tasks
    readable = {t.index for t in l90 if (zs.get(t.index) or 0) >= 0.8}
    by_id = {t.index: t for t in l90}
    held = {}
    for i in HELD:
        t, r = by_id[i], rows[i]
        grp = "A" if (r["obj_same_family"] or not t.moved) and r["rel_any"] and r["quals_seen"] else \
              ("B" if (r["obj_any"] or not t.moved) and r["pred_any"] else "C")
        bases = [f"libero_90:{j}" for j in sorted(readable) if by_id[j].scene == t.scene] + \
                [x.tid for x in train if x.scene == t.scene]
        obj_src = [x.tid for x in train if set(t.moved) & set(x.moved)] + \
                  [f"libero_90:{j}" for j in sorted(readable) if set(t.moved) & set(by_id[j].moved)]
        rel_src = [x.tid for x in train if any(s[:1] + s[2:] == u[:1] + u[2:] for s in x.slots for u in t.slots)] + \
                  [f"libero_90:{j}" for j in sorted(readable)
                   if any(s[:1] + s[2:] == u[:1] + u[2:] for s in by_id[j].slots for u in t.slots)]
        held[i] = dict(group=grp, language=t.language, scene=t.scene, zeroshot=r["zeroshot"], bases=bases[:6],
                       object_sources=obj_src[:6], relation_sources=rel_src[:6],
                       **{k: r[k] for k in feats + ["contrast_obj"]})
    res = dict(parser_hash=F.parser_hash(), solved_threshold=SOLVED, n_tasks=len(ids), n_solved=int(sum(y)),
               features=feature_read, held_out=held, tasks=rows,
               groups={g: sorted(i for i in held if held[i]["group"] == g) for g in "ABC"})
    json.dump(res, open(os.path.join(R, "f0_coverage.json"), "w"), indent=1)
    print(f"parser {res['parser_hash']}: {len(ids)} LIBERO-90 tasks, {int(sum(y))} solved zero-shot (>= {SOLVED})")
    for f, v in feature_read.items():
        print(f"  {f:22s} AUC {v['auc']:.2f}  on {v['n_on']:2d} -> {v['success_on'] if v['success_on'] is not None else float('nan'):.2f}"
              f"  off -> {v['success_off'] if v['success_off'] is not None else float('nan'):.2f}")
    for g in "ABC":
        print(f"  group {g}: {res['groups'][g]}")
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("f0_coverage", bench="libero_90", tag="read", seed=0,
                            metrics={k: res[k] for k in ("parser_hash", "solved_threshold", "n_tasks", "n_solved", "features",
                                                         "held_out", "groups")},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/f0_coverage.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "f0_coverage.json")]))


if __name__ == "__main__":
    main()
