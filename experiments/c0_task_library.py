"""The task-prototype library (overhaul plan, user's note): chains of the tasks the policy performs, and the covering task
prototypes of every held-out task (lifelong_ood/policy/tasks.py).

  training tasks       chains of their demonstrations (physical-intelligence/libero, 3 episodes each)
  solved LIBERO-90     chains of the policy's own successful rollouts where recorded (experiments/f1_binding's f1_bind runs)

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/c0_task_library.py" slurm/jobs/robot_gpu.sbatch   [--record]
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import data as D                                      # noqa: E402
from lifelong_ood.policy import factors as F                                   # noqa: E402
from lifelong_ood.policy import tasks as TK                                    # noqa: E402

R = os.path.join(ROOT, "results", "robot")
HELD = (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63)


def main():
    tasks = F.load()
    by_lang = {}
    for t in tasks:
        if t.suite != "libero_90":
            by_lang.setdefault(t.language, t)
    cov = json.load(open(os.path.join(R, "f0_coverage.json")))
    zs = {int(k): v["zeroshot"] for k, v in cov["tasks"].items()}
    lab = TK.Labeller()
    names = TK.verb_names()
    lib = []
    demos = D.Demos()
    seqs, owners = [], []
    for ti, text in sorted(demos.tasks.items()):
        t = by_lang.get(text)
        if t is None:
            print(f"[lib] no factor parse for training text '{text}'", flush=True)
            continue
        for e in demos.episodes(ti)[:3]:
            seqs.append(np.asarray(demos.load(e)["actions"], np.float32))
            owners.append(("train", t))
    for d in sorted(glob.glob(os.path.join(R, "f1_bind_*_zeroshot_rec"))):
        m = re.match(r"f1_bind_(\d+)_zeroshot_rec$", os.path.basename(d))
        t = next(x for x in tasks if x.suite == "libero_90" and x.index == int(m.group(1)))
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            if bool(z["success"]):
                seqs.append(np.asarray(z["action"], np.float32))
                owners.append(("zeroshot", t))
    L = lab(seqs)
    entries = {}
    for i, (src, t) in enumerate(owners):
        ch, co = TK.collapse(L[0.4][i]), TK.collapse(L[0.55][i])
        e = entries.get(t.tid)
        if e is None:
            e = entries[t.tid] = TK.from_task(t, src, success=1.0 if src == "train" else zs.get(t.index, 1.0))
            e.chain, e.coarse = ch, co
    lib = list(entries.values())
    held = {}
    for i in HELD:
        t = next(x for x in tasks if x.suite == "libero_90" and x.index == i)
        sel = TK.select(t, lib, K=3)
        held[i] = dict(language=t.language, selected=[dict(tid=p.tid, language=p.language, covers=[list(c) for c in cov_],
                                                           chain=[names.get(c, str(c)) for c in p.chain][:12])
                                                      for p, cov_ in sel],
                       uncovered=sorted([list(f) for f in TK.from_task(t, "q").factors()
                                         - set().union(*[set(map(tuple, c)) if c else set() for _, c in sel] or [set()])]))
        print(f"[lib] {i:>2} {t.language}", flush=True)
        for p, c in sel:
            print(f"        <- {p.tid:16s} {p.language:60s} covers {c}", flush=True)
    res = dict(n_prototypes=len(lib), library=[dict(tid=p.tid, language=p.language, source=p.source, success=p.success,
                                                    chain=p.chain, coarse=p.coarse, verbs=[names.get(c, str(c)) for c in p.chain])
                                               for p in lib], held_out=held)
    json.dump(res, open(os.path.join(R, "task_library.json"), "w"), indent=1)
    print(f"[lib] {len(lib)} task prototypes", flush=True)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("f0_task_library", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/c0_task_library.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "task_library.json")]))


if __name__ == "__main__":
    main()
