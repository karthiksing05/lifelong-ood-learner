"""The qualifier grammar (exploratory, offline): in every LIBERO-90 task whose instruction picks one of several objects of a type by a
qualifier (front, back, middle, left, right), where does the named instance sit among its same-type siblings in the policy's own
image (the client's flipped agent view, 16 x 16 tokens)?  Simulator segmentation of the first frames of init states 0-9
(experiments/f0b_firstframes_seg.py) gives each instance's token centroid; the named instance's rank along the image rows and
columns is counted per task.  A consistent rank per word is a rule the learner can use to pick ONE instance of a type prototype's
map instead of trying each.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q22_grammar.py [--record]
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import bind_hooks as BH      # noqa: E402
from lifelong_ood.policy import factors as F          # noqa: E402
from lifelong_ood.policy import where as W            # noqa: E402

FR = os.path.join(ROOT, "results", "robot", "f0b_firstframes")
QUAL = ("front", "back", "middle", "left", "right")
# the rule read off the counts below: (axis, which end) in the client image; left/right are mirrored (the camera faces the robot)
RULE = {"back": ("row", "first"), "front": ("row", "last"), "middle": ("row", "middle"), "left": ("col", "last"),
        "right": ("col", "first")}


def qualifier(lang, word_pos):
    """The qualifier that names the MOVED object: the first qualifier word in the instruction (the moved object is named first)."""
    hits = sorted((m.start(), q) for q in QUAL for m in re.finditer(rf"\b{q}\b", lang))
    return hits[0][1] if hits else None


def pick_of(by_r, by_c, axis, end):
    order = by_r if axis == "row" else by_c
    return order[0] if end == "first" else order[-1] if end == "last" else order[len(order) // 2]


def main():
    tasks = sorted((x for x in F.load() if x.suite == "libero_90"), key=lambda x: x.index)
    per, agree, total, frames, word_of = {}, 0, 0, {}, {}
    for t in tasks:
        q = qualifier(t.language, 0)
        mv = BH.referents(t)["open"]
        if q is None or not mv:
            continue
        named, ty = mv[0], t.objects.get(mv[0])
        sibs = [n for n, v in t.objects.items() if v == ty]
        f = os.path.join(FR, f"libero_90_{t.index}.npz")
        if len(sibs) < 2 or not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        rows, cols, ok = [], [], []
        for k in range(len(z["states"])):
            cen = {}
            for n in sibs:
                m = W.oracle_map(z["seg"][k], W.geoms_of(z, name=n)).reshape(16, 16)
                if (m > 0.1).any():
                    cen[n] = np.argwhere(m > 0.1).mean(0)
            if named not in cen or len(cen) < 2:
                continue
            by_r = sorted(cen, key=lambda n: cen[n][0])
            by_c = sorted(cen, key=lambda n: cen[n][1])
            rows.append(by_r.index(named))
            cols.append(by_c.index(named))
            axis, end = RULE[q]
            ok.append(int(pick_of(by_r, by_c, axis, end) == named))
            frames.setdefault(t.index, []).append((by_r, by_c, named))
        word_of[t.index] = q
        per[str(t.index)] = dict(language=t.language, qualifier=q, instances=len(sibs), frames=len(ok),
                                 row_rank=np.bincount(rows, minlength=len(sibs)).tolist(),
                                 col_rank=np.bincount(cols, minlength=len(sibs)).tolist(), rule_hits=int(sum(ok)))
        agree += sum(ok)
        total += len(ok)
        print(f"[q22] {t.index:>3} {q:6s} n={len(sibs)} row-rank {per[str(t.index)]['row_rank']} col-rank "
              f"{per[str(t.index)]['col_rank']} rule {sum(ok)}/{len(ok)} | {t.language}", flush=True)
    # leave one task out: each word's rule learned from the OTHER tasks with that word, applied to the held-out task
    rules = [(a, e) for a in ("row", "col") for e in ("first", "middle", "last")]
    loo, loo_n = 0, 0
    for t, fr in frames.items():
        others = [x for u, f_ in frames.items() if u != t and word_of[u] == word_of[t] for x in f_]
        if not others:
            continue
        best = max(rules, key=lambda r: sum(pick_of(a, b, *r) == n for a, b, n in others))
        hit = sum(pick_of(a, b, *best) == n for a, b, n in fr)
        per[str(t)]["loo_rule"], per[str(t)]["loo_hits"] = list(best), int(hit)
        loo += hit
        loo_n += len(fr)
    print(f"[q22] leave-one-task-out: {loo} of {loo_n} first frames", flush=True)
    res = dict(rule={k: list(v) for k, v in RULE.items()}, per_task=per, tasks=len(per), frames=total, rule_hits=agree,
               loo_hits=int(loo), loo_frames=int(loo_n))
    print(f"[q22] the rule picks the named instance in {agree} of {total} first frames over {len(per)} tasks", flush=True)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("q22_grammar", bench="libero_90", tag="offline", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/q22_grammar.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
