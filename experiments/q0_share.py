"""P0 of the prototype-PoE plan: does the policy's share of the action follow the expert weight at the LAST Euler steps?

Offline on demonstration frames of the 21 held-out LIBERO-90 tasks (demo 0, 10 frames each): two experts, the policy under the
task's instruction (row 0) and a Gaussian at the frame's own demo chunk q with the policy's calibrated error variance.  The
prototype's weight is w_early for t > 0.3 and w_late for t <= 0.3 (the last three of ten steps).  Predicted output
a ~= (1 - w_late) a_pi + w_late q.  Reported: the deviation from the prediction and the effect of w_early (both relative to
||q - a_pi||), and the observation-sensitivity ratio ||a(o_i) - a(o_j)|| / ||a_pi(o_i) - a_pi(o_j)|| for two frames 10 steps apart
under the SAME expert (the bypass gave ~0.001).  Also checks that a one-hot weight on the policy is the policy bit for bit, and
writes the calibrated per-dim policy variance (results/robot/poe_policy_var.npy, from the policy's error against the 40 training
tasks' demos, results/robot/r2b_libero_40.npz).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q0_share.py --record" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import data as D             # noqa: E402
from lifelong_ood.policy import poe                    # noqa: E402

R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
ARMS = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0), (0.0, 0.3), (0.0, 0.7)]


def calibrated_var():
    z = np.load(os.path.join(R, "r2b_libero_40.npz"))
    e = ((z["mu"] - z["demo"]) ** 2).reshape(-1, 10, 7).mean((0, 1))
    v = np.tile(e, 10).astype(np.float32)                    # [70] step-major
    np.save(os.path.join(R, "poe_policy_var.npy"), v)
    return v


def main():
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    var = calibrated_var()
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    rows = [poe.PolicyRow(prompt="")]
    res = {f"{we}_{wl}": dict(dev=[], early=[], sens=[]) for we, wl in ARMS}
    exact = []
    for tid in HELD:
        ep = d90.load(d90.episodes_of_task(tid)[0])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        Q = D.to_rows(ep["actions"][idx], norm)                         # [T, 70]
        frames = np.linspace(10, max(11, T - 25), 10).astype(int)
        for n, i in enumerate(frames):
            j = min(i + 10, T - 1)
            outs = {}
            for f in (i, j):
                obs = d90.observation(ep, int(f))
                pre = bk.prefix(bk.observations([obs]))
                g = np.random.default_rng(1000 * tid + n)
                noise = g.standard_normal(bk.dim_full).astype(np.float32)
                proto = [poe.ProtoExpert(Q[i], var, "demo_i")]
                for we, wl in ARMS:
                    beta = (lambda we_, wl_: (lambda t: np.stack([np.full(70, 1 - (wl_ if t <= 0.3 + 1e-9 else we_)),
                                                                  np.full(70, wl_ if t <= 0.3 + 1e-9 else we_)])))(we, wl)
                    outs[(f, we, wl)] = poe.sample_poe(bk, pre, rows, proto, beta, noise=noise)[0]
                if f == i and n == 0:
                    ref = bk.sample(1, pre, noise=noise[None])[0]
                    exact.append(float(np.abs(ref - outs[(f, 0.0, 0.0)]).max()))
            a_pi_i, a_pi_j = outs[(i, 0.0, 0.0)], outs[(j, 0.0, 0.0)]
            gap = max(np.linalg.norm(Q[i] - a_pi_i), 1e-6)
            dpi = max(np.linalg.norm(a_pi_i - a_pi_j), 1e-6)
            for we, wl in ARMS:
                a = outs[(i, we, wl)]
                res[f"{we}_{wl}"]["dev"].append(float(np.linalg.norm(a - ((1 - wl) * a_pi_i + wl * Q[i])) / gap))
                res[f"{we}_{wl}"]["early"].append(float(np.linalg.norm(a - outs[(i, 0.0, wl)]) / gap))
                res[f"{we}_{wl}"]["sens"].append(float(np.linalg.norm(a - outs[(j, we, wl)]) / dpi))
        print(f"[q0] task {tid}: done ({len(frames)} frames)", flush=True)
    summary = {k: {m: float(np.median(v[m])) for m in ("dev", "early", "sens")} for k, v in res.items()}
    for k, v in summary.items():
        print(f"[q0] w_early_w_late {k:9s} deviation {v['dev']:.3f} early effect {v['early']:.3f} obs sensitivity {v['sens']:.3f}", flush=True)
    early_max = max(summary[f"1.0_{wl}"]["early"] for wl in (0.0, 0.5, 1.0))
    out = dict(var=var[:7].tolist(), exact_max_abs=max(exact), summary=summary, early_effect_max=early_max,
               kill=bool(early_max > 0.2), frames=int(sum(len(v["dev"]) for v in res.values()) // len(ARMS)))
    print(f"[q0] one-hot policy vs sample: max |diff| {max(exact):.3g}; early effect max {early_max:.3f} -> kill {out['kill']}", flush=True)
    json.dump(out, open(os.path.join(R, "q0_share.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("q0_share", bench="libero_90", tag="read", seed=0, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/q0_share.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
