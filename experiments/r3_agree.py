"""M3 refinement: steer only where the prior DISAGREES with the demo.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r3_agree.py --tasks 67,19,36,42,35,39" slurm/jobs/robot_gpu.sbatch

A constant-weight guide overrides pi0.5 at every frame, which hurts a task the prior already solves (put the white mug
on the left plate: zero-shot 9/10, library codes 3/10).  For each demo frame f, the policy's own chunk on the demo's
observation (mean of K samples) is compared with the demo's chunk: d_f = rms(mu_policy(o_f) - a_f).  The threshold is
the q-quantile of the same gap on TRAINED tasks (results/robot/r2b_libero_40.npz: the policy's conditional mean vs the
demo chunk on 10k library frames), so d_f above it means "the prior would not do what the demo does here".  The steer
mask (dilated by +-`dilate` frames so the guide also covers the approach) is written into copies of the task's
programs (tag <src>_agree); the hooks guide only on steer frames and let the prior act elsewhere.
Ledger row r3_agree per task: the gap profile, the steered fraction, the threshold.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--library", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--dilate", type=int, default=5)
    ap.add_argument("--sources", default="_lib40,_demo", help="program tags to copy with the steer mask")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    lib = np.load(a.library)
    d_lib = np.sqrt(((lib["mu"] - lib["demo"]) ** 2).mean(1))
    thr = float(np.quantile(d_lib, a.q))
    print(f"[agree] threshold q{a.q} of the policy-demo gap on {len(d_lib)} trained frames: {thr:.3f}", flush=True)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90, norm = D.Demos90(), D.Norm.load()
    for tid in [int(x) for x in a.tasks.split(",")]:
        t0 = time.time()
        ep = d90.load(d90.episodes_of_text(texts[tid])[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, bk.H)[0]], norm)
        mu = np.zeros_like(rows)
        for b0 in range(0, T, a.batch):
            sl = list(range(b0, min(T, b0 + a.batch)))
            pre = bk.prefix(bk.observations([d90.observation(ep, i) for i in sl]))
            S = bk.sample(len(sl) * a.K, bk.repeat(pre, a.K), seed=a.seed + b0).reshape(len(sl), a.K, bk.dim)
            mu[sl] = S.mean(1)
        d = np.sqrt(((mu - rows) ** 2).mean(1))
        raw = d > thr
        steer = raw.copy()
        for f in np.where(raw)[0]:
            steer[max(0, f - a.dilate): f + a.dilate + 1] = True
        for src in a.sources.split(","):
            for variant in ("code", "raw"):
                p = f"results/robot/r3_prog_{tid}_{variant}{src}.npz"
                if not os.path.exists(p):
                    continue
                prog = DemoProgram.load(p)
                if len(prog.mu) != T:
                    print(f"[agree] {p}: {len(prog.mu)} frames != demo {T}; skipped", flush=True)
                    continue
                prog.steer = steer
                prog.save(f"results/robot/r3_prog_{tid}_{variant}{src}_agree.npz")
        m = dict(task=tid, text=texts[tid], n_frames=int(T), threshold=thr, gap_median=float(np.median(d)),
                 gap_q90=float(np.quantile(d, 0.9)), disagree_frac=float(raw.mean()), steer_frac=float(steer.mean()),
                 gap_profile=[round(float(x), 3) for x in d[::5]])
        print(f"[agree] task {tid} '{texts[tid]}': policy-demo gap median {m['gap_median']:.3f}, disagrees on {m['disagree_frac']:.2f} "
              f"of frames (steered {m['steer_frac']:.2f} after dilation) [{time.time() - t0:.0f}s]", flush=True)
        ledger.record("r3_agree", bench="libero_90", tag="", seed=a.seed, keys=dict(task=tid), metrics=m,
                      bytes=dict(experts=0, tree=0, codes=(T + 7) // 8, samples=0, teacher_copies=0, total=(T + 7) // 8),
                      config=vars(a), script="experiments/r3_agree.py", command=" ".join(sys.argv))


if __name__ == "__main__":
    main()
