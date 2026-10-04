"""pi0.5's own chunk on every frame of each task's ONE demo (the prior expert's training data for the composer).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r6b_prior_proposals.py --tasks 39,35,36,42,67,19" slurm/jobs/robot_gpu.sbatch
For each demo frame: the task's instruction and the demo's images and state -> K samples of the frozen policy -> their
mean (normalised 70-D rows).  Writes results/robot/r6_prior_<task>.npz (prior [T, 70], demo rows [T, 70]).
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import data as D                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="39,35,36,42,67,19")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90, norm = D.Demos90(), D.Norm.load()
    for tid in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
        t0 = time.time()
        ep = d90.load(d90.episodes_of_task(tid)[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, bk.H)[0]], norm)
        mu = np.zeros_like(rows)
        for b0 in range(0, T, a.batch):
            sl = list(range(b0, min(T, b0 + a.batch)))
            pre = bk.prefix(bk.observations([d90.observation(ep, i) for i in sl]))
            S = bk.sample(len(sl) * a.K, bk.repeat(pre, a.K), seed=a.seed + b0).reshape(len(sl), a.K, bk.dim)
            mu[sl] = S.mean(1)
        np.savez_compressed(f"results/robot/r6_prior_{tid}.npz", prior=mu, demo=rows, state=ep["state"])
        print(f"[r6b] task {tid} '{texts[tid]}': prior chunk {np.sqrt(((mu - rows) ** 2).mean()):.3f} rms from the demo chunk "
              f"over {T} frames ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
