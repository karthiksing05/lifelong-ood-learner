"""M1 smoke (openpi env, GPU): load pi05_libero through PolicyBackbone, check the wrapper against openpi's own
sampler and transforms, the flow identities, and run a first mode ascent on real demo chunks.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r1_policy_smoke.py" slurm/jobs/robot_gpu.sbatch

Checks (each a ledger metric):
  parity      our Euler sampler with the same noise == model.sample_actions (max abs diff, full 320-D)
  norm        data.to_rows (numpy quantile normalisation) inverts through the policy's own output transform
  identities  (1 - t) E[a|x_t] + t E[eps|x_t] == x_t (algebraic)
  finite-diff score vs the finite-difference gradient of log p_t of the padded dims' Gaussian marginal is not
              available (no log-density), so the score's SIGN is checked instead: one ascent step raises
              the Tweedie-consistency (|eps_hat| falls)
  ascent      Zekun's mode ascent from noised demo chunks at t in {0.1, 0.3, 0.55}: score norm start -> end,
              the number of distinct Tweedie modes among 64 starts (tol 0.05 rms), and the RMS distance of
              the modes to their starting chunks
  timing      seconds per denoise_step at batch 64 (the budget of discovery)
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.backbones.score_ops import mode_ascent               # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy.backbone import PolicyBackbone                # noqa: E402
from lifelong_ood.tree.update import dedup                             # noqa: E402

ZERO = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bk = PolicyBackbone("pi05_libero", device=dev)
    print(f"[r1] {bk.describe()}", flush=True)
    demos = D.Demos()
    eps_goal = demos.episodes("libero_goal")
    ep = demos.load(eps_goal[0])
    obs = bk.observation(demos.observation(ep, 0))
    pre = bk.prefix(obs)
    bk.bind(pre)
    print(f"[r1] prefix from episode {ep['episode_index']} frame 0: '{ep['task']}', kv batch {pre.n}", flush=True)
    m = {}
    # parity with openpi's sampler on the same noise
    g = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((1, bk.H, bk.A), generator=g).to(dev)
    with torch.no_grad():
        ref = bk.model.sample_actions(dev, obs, noise=noise, num_steps=10).reshape(1, -1).float().cpu().numpy()
    ours = bk.sample(1, pre, num_steps=10, noise=noise.reshape(1, -1), full=True)
    m["parity_max_abs_diff"] = float(np.abs(ours - ref).max())
    act = bk.unnormalize(ours[:, bk.real])[0]
    print(f"[r1] parity max|diff| {m['parity_max_abs_diff']:.2e}; first action (raw) {np.round(act[0], 3)}", flush=True)
    # normalisation parity: numpy rows -> policy's output transform == raw demo actions
    cs = D.chunkset(demos, eps_goal[:4], H=bk.H)
    m["norm_max_abs_err"] = float(np.abs(bk.unnormalize(cs.rows[:50]) - cs.raw[:50]).max())
    print(f"[r1] norm roundtrip through the policy's Unnormalize: max err {m['norm_max_abs_err']:.2e} on {len(cs)} chunks", flush=True)
    # identities and timing
    X = torch.as_tensor(cs.rows[:64], device=dev)
    t = 0.5
    Xt = (1 - t) * X + t * torch.randn(X.shape, generator=g).to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    v = bk.v(Xt, t, pre)
    if dev == "cuda":
        torch.cuda.synchronize()
    m["secs_per_step_b64"] = time.time() - t0
    a_hat, e_hat = bk.tweedie(Xt, t, pre), bk.eps(Xt, t, pre)
    m["identity_err"] = float(((1 - t) * a_hat + t * e_hat - Xt).abs().max())
    m["tweedie_rms_to_demo"] = float(((a_hat - X) ** 2).mean().sqrt())
    m["tweedie_rms_to_noisy"] = float(((Xt - X) ** 2).mean().sqrt())
    print(f"[r1] identity err {m['identity_err']:.2e}; {m['secs_per_step_b64']*1000:.0f} ms per step at batch 64; "
          f"Tweedie at t=0.5 is {m['tweedie_rms_to_demo']:.3f} rms from the demo chunk (noisy input {m['tweedie_rms_to_noisy']:.3f})", flush=True)
    # score sign: one small step along the score should shrink |eps_hat|
    s = bk.score(Xt, t, pre)
    e1 = bk.eps(Xt + 1e-2 * s / s.norm(dim=1, keepdim=True) * Xt.norm(dim=1, keepdim=True), t, pre)
    m["eps_norm_before"], m["eps_norm_after_step"] = float(e_hat.norm(dim=1).mean()), float(e1.norm(dim=1).mean())
    print(f"[r1] |eps_hat| {m['eps_norm_before']:.3f} -> {m['eps_norm_after_step']:.3f} after a 1% step along the score", flush=True)
    # mode ascent on demo chunks (Zekun's discovery, through the generic score_ops on the flow path)
    res = {}
    for u in (0.1, 0.3, 0.55):
        t1 = time.time()
        xt, x0 = mode_ascent(bk, cs.rows[:64], u, iters=100, lr=0.02, seed=1)
        # score_ops.mode_ascent calls backbone.eps(X, u, cls=, emb=, gate=, batch=): PolicyBackbone.eps needs
        # the prefix, bound below
        s0 = float(bk.score(torch.as_tensor(xt, device=dev), u, pre).norm(dim=1).mean())
        kept, _ = dedup(x0, tol=0.05)
        res[str(u)] = dict(n_distinct=len(kept), score_norm_end=s0, mode_rms_to_start=float(np.sqrt(((x0 - cs.rows[:64]) ** 2).mean())),
                           secs=time.time() - t1)
        print(f"[r1] ascent t={u}: {len(kept)}/64 distinct Tweedie modes (tol 0.05 rms), |score| at end {s0:.2f}, "
              f"modes {res[str(u)]['mode_rms_to_start']:.3f} rms from their demo chunks, {res[str(u)]['secs']:.0f}s", flush=True)
    m["ascent"] = res
    m["ok"] = bool(m["parity_max_abs_diff"] < 1e-3 and m["identity_err"] < 1e-4 and m["norm_max_abs_err"] < 1e-4)
    ledger.record("r1_policy_smoke", bench="libero_goal", tag="", seed=0, metrics=m, bytes=ZERO,
                  config=dict(config="pi05_libero", episode=int(ep["episode_index"]), task=ep["task"]),
                  script="experiments/r1_policy_smoke.py", command=" ".join(sys.argv),
                  backbone=dict(config="pi05_libero", n_params=bk.n_params, H=bk.H, A=bk.A, dim=bk.dim))
    print("[r1] PASS" if m["ok"] else "[r1] FAIL", flush=True)
    return 0 if m["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
