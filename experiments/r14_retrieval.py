"""Retrieval baselines' choices, fixed from the ONE demo before any rollout.

  instruction retrieval  the training task (of pi05_libero's 40) owning most of the demo chunks' nearest library chunks
  adapter retrieval      the OTHER held-out task whose demo is nearest (symmetric chamfer distance between chunk sets); its
                         per-task LoRA is reused with no training (an IsCiL / DMPEL-style adapter library with retrieval)
Writes results/robot/r14_retrieval.json and a ledger row r14_retrieval.
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402

HELD = [19, 67, 35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]


def main():
    os.chdir(ROOT)
    z = np.load("results/robot/r2b_libero_40.npz")
    lib, task = z["demo"].astype(np.float32), z["task"]
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    train_text = D.Demos().tasks
    d90, norm = D.Demos90(), D.Norm.load()
    rows = {}
    for t in HELD:
        ep = d90.load(d90.episodes_of_text(texts[t])[0])
        T = len(ep["actions"])
        rows[t] = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm).astype(np.float32)

    def nn_dist(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.sqrt(np.maximum(d, 0) / A.shape[1])

    out = {}
    for t in HELD:
        near = task[nn_dist(rows[t], lib).argmin(1)]
        vals, cnt = np.unique(near, return_counts=True)
        p = int(vals[cnt.argmax()])
        cham = {}
        for u in HELD:
            if u != t:
                d = nn_dist(rows[t], rows[u])
                cham[u] = float(0.5 * (d.min(1).mean() + d.min(0).mean()))
        a = min(cham, key=cham.get)
        out[str(t)] = dict(text=texts[t], prompt_task=p, prompt_text=train_text[p], prompt_share=float(cnt.max() / len(near)),
                           adapter_task=int(a), adapter_text=texts[a], adapter_dist=cham[a])
        print(f"[r14] {t:3d} '{texts[t][:40]}' -> instruction of '{train_text[p][:40]}' ({cnt.max() / len(near):.0%} of chunks); "
              f"adapter of {a} '{texts[a][:34]}' ({cham[a]:.3f})", flush=True)
    json.dump(out, open("results/robot/r14_retrieval.json", "w"), indent=1)
    ledger.record("r14_retrieval", bench="libero_90", tag="", seed=0, metrics=dict(choices=out),
                  bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                  script="experiments/r14_retrieval.py", command="python experiments/r14_retrieval.py",
                  artifacts=["results/robot/r14_retrieval.json"])


if __name__ == "__main__":
    main()
