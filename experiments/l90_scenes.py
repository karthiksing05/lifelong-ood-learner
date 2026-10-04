"""Each LIBERO-90 task's OWN demonstrations.  The IPEC port merges tasks by language: 12 instructions are shared by 28 tasks in
different scenes, and every read before 2026-10-04 took the text's first episode as "the task's demonstration" (r3_prepare.py),
so a task sharing its instruction could be given another scene's demonstration, and a leave-one-out fit could contain the scored
task's own demonstration through a same-text sibling.

A merged text's episodes form one contiguous block per original demonstration file (one scene); each of the 61 unshared texts is
one block.  Each block is assigned to the task whose rendered first frames (results/robot/f0_firstframes, the client's
orientation) are nearest to the block's first video frames (mean squared pixel difference), the assignment that minimises the
summed distance within the group; it must agree at two resolutions and two sample sizes.  Writes
results/robot/l90_scene_episodes.json {task: [episode indices]} and ledger row l90_scenes tag map.  Needs PyAV (openpi venv):

    /coc/flash10/ksingara3/robot/openpi/.venv/bin/python experiments/l90_scenes.py
"""
import collections
import importlib.util
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import data as D                     # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def runs(ix):
    out, s = [], ix[0]
    for a, b in zip(ix, ix[1:]):
        if b != a + 1:
            out.append((s, a))
            s = b
    out.append((s, ix[-1]))
    return out


def main():
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, names = r3p.libero90_texts()
    d90 = D.Demos90()
    by = collections.defaultdict(list)
    for t, x in texts.items():
        by[x].append(t)
    first = {}
    out, evidence = {}, {}
    for x, ts in by.items():
        eps = sorted(d90.episodes_of_text(x))
        if not eps:
            for t in ts:
                out[t] = []
            continue
        R_ = runs(eps)
        if len(ts) == 1:
            assert len(R_) == 1, (ts, R_)
            out[ts[0]] = eps
            continue
        assert len(R_) == len(ts), (ts, R_)
        picks = []
        for res, n_ep, n_st in ((56, 3, 5), (112, 6, 10)):
            def small(img):
                return np.asarray(Image.fromarray(img).resize((res, res), Image.BILINEAR), np.float32) / 255.0
            ref = {t: np.mean([small(i) for i in np.load(os.path.join(R, "f0_firstframes", f"libero_90_{t}.npz"))["img"][:n_st]], 0)
                   for t in ts}
            feats = []
            for a, b in R_:
                fr = []
                for e in range(a, min(b + 1, a + n_ep)):
                    if e not in first:
                        first[e] = d90.frames(e)[0][0]
                    fr.append(small(first[e]))
                feats.append(np.mean(fr, 0))
            Dm = np.array([[float(np.mean((f - ref[t]) ** 2)) for t in ts] for f in feats])
            best = min(itertools.permutations(range(len(ts))), key=lambda p: sum(Dm[i, p[i]] for i in range(len(R_))))
            picks.append(([ts[p] for p in best], Dm))
        assert picks[0][0] == picks[1][0], (ts, picks)
        owner = picks[0][0]
        for (a, b), t in zip(R_, owner):
            out[t] = list(range(a, b + 1))
        Dm = picks[1][1]
        ratio = min(min(Dm[i, j] for j in range(len(ts)) if ts[j] != owner[i]) / Dm[i, ts.index(owner[i])] for i in range(len(R_)))
        evidence[x] = dict(tasks=ts, blocks=[[int(a), int(b)] for a, b in R_], owners=owner, dist=Dm.round(5).tolist(),
                           min_ratio=float(ratio), old_demo=int(eps[0]), old_demo_owner=owner[0])
        print(f"[l90] {ts}: blocks {[a for a, _ in R_]} -> {owner}; nearest-other / own distance >= {ratio:.2f}", flush=True)
    wrong = sorted(t for v in evidence.values() for t in v["tasks"] if t != v["old_demo_owner"])
    json.dump({str(t): v for t, v in sorted(out.items())}, open(os.path.join(R, "l90_scene_episodes.json"), "w"))
    print(f"[l90] tasks given another scene's demonstration by the first-episode rule: {wrong}", flush=True)
    from lifelong_ood import ledger
    print(ledger.record("l90_scenes", bench="libero_90", tag="map", seed=0,
                        metrics=dict(groups=evidence, shared_texts=len(evidence), shared_tasks=sum(len(v["tasks"]) for v in evidence.values()),
                                     wrong_scene_tasks=wrong, episodes={str(t): len(v) for t, v in out.items()}),
                        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                        script="experiments/l90_scenes.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
