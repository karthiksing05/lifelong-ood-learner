"""M1 smoke (openpi env, GPU): load pi05_libero through PolicyBackbone, check the flow identities and the
wrapper's parity with openpi's own sampler, and run a first mode ascent on the action expert.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r1_policy_smoke.py" slurm/jobs/robot_gpu.sbatch

Checks (each a ledger metric):
  parity      our Euler sampler with the same noise == model.sample_actions (max abs diff)
  identities  E[a|x_t] = x_t - t v and E[eps|x_t] = x_t + (1-t) v recombine to x_t (algebraic, ~1e-6)
  score-sign  moving along the score raises the (approximate) log-density: the eps norm shrinks
  ascent      Zekun's mode ascent from noised demo-free chunks at t in {0.1, 0.3, 0.55}: the score norm at
              the end vs the start, and the Tweedie modes' spread (are there several distinct modes?)
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
from lifelong_ood.accounting import charge                             # noqa: E402
from lifelong_ood.policy.backbone import PolicyBackbone                # noqa: E402

ZERO = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bk = PolicyBackbone("pi05_libero", device=dev)
    print(f"[r1] {bk.describe()}", flush=True)
    from openpi.policies import libero_policy
    ex = libero_policy.make_libero_example()
    ex["prompt"] = "put the bowl on the plate"
    obs = bk.observation(ex)
    pre = bk.prefix(obs)
    m = {}
    # parity with openpi's sampler on the same noise
    g = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((1, bk.H, bk.A), generator=g).to(dev)
    with torch.no_grad():
        ref = bk.model.sample_actions(dev, obs, noise=noise, num_steps=10).reshape(1, -1).float().cpu().numpy()
    ours = bk.sample(1, pre, num_steps=10, noise=noise.reshape(1, -1))
    m["parity_max_abs_diff"] = float(np.abs(ours - ref).max())
    print(f"[r1] parity max|diff| {m['parity_max_abs_diff']:.2e}; chunk range [{ref.min():.2f}, {ref.max():.2f}]", flush=True)
    # identities and timing
    X = torch.randn((64, bk.dim), generator=g).to(dev)
    t = 0.5
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    v = bk.v(X, t, pre)
    torch.cuda.synchronize() if dev == "cuda" else None
    m["secs_per_step_b64"] = time.time() - t0
    a_hat, e_hat = bk.tweedie(X, t, pre), bk.eps(X, t, pre)
    recon = (1 - t) * a_hat + t * e_hat
    m["identity_err"] = float((recon - X).abs().max())
    print(f"[r1] identity |(1-t)a + t eps - x| {m['identity_err']:.2e}; {m['secs_per_step_b64']*1000:.0f} ms per step at batch 64", flush=True)
    # score sign and mode ascent
    from lifelong_ood.backbones.score_ops import mode_ascent
    class _B:                       # score_ops calls backbone.eps(X, u, cls=, emb=, gate=, batch=) and reads .dim/.device
        dim, device = bk.dim, bk.device
        def eps(self, X, u, cls=None, emb=None, gate=None, batch=256):
            return bk.eps(X, u, pre, gate)
    b = _B()
    res = {}
    for u in (0.1, 0.3, 0.55):
        from lifelong_ood.backbones.schedule import cosine_abar   # score_ops uses the cosine abar for the noise; on the flow
        # path the noise std is t itself, so we run the ascent by hand here with the flow's own geometry
        x = torch.randn((32, bk.dim), generator=g).to(dev) * u + (1 - u) * 0.0
        s0 = bk.score(x, u, pre).norm(dim=1).mean().item()
        mom = torch.zeros_like(x); vel = torch.zeros_like(x)
        for i in range(1, 101):
            gsc = bk.score(x, u, pre)
            mom = 0.9 * mom + 0.1 * gsc
            vel = 0.999 * vel + 0.001 * gsc * gsc
            x = x + 0.02 * (mom / (1 - 0.9 ** i)) / (torch.sqrt(vel / (1 - 0.999 ** i)) + 1e-8)
        s1 = bk.score(x, u, pre).norm(dim=1).mean().item()
        modes = bk.tweedie(x, u, pre).float().cpu().numpy()
        d = np.sqrt(((modes[:, None] - modes[None]) ** 2).mean(-1))
        n_distinct = int(len({int(np.argmin(np.where(d[i] < 0.05, np.arange(len(d)), len(d)))) for i in range(len(d))}))
        res[str(u)] = dict(score_norm_start=s0, score_norm_end=s1, n_distinct=n_distinct, mode_rms=float(np.sqrt((modes ** 2).mean())))
        print(f"[r1] ascent t={u}: score norm {s0:.2f} -> {s1:.2f}; distinct Tweedie modes {n_distinct}/32 (tol 0.05 rms)", flush=True)
    m["ascent"] = res
    m["ok"] = bool(m["parity_max_abs_diff"] < 1e-3 and m["identity_err"] < 1e-4)
    ledger.record("r1_policy_smoke", bench="libero_goal", tag="", seed=0, metrics=m, bytes=ZERO, config=dict(config="pi05_libero"),
                  script="experiments/r1_policy_smoke.py", command=" ".join(sys.argv),
                  backbone=dict(config="pi05_libero", n_params=bk.n_params, H=bk.H, A=bk.A))
    print("[r1] PASS" if m["ok"] else "[r1] FAIL", flush=True)
    return 0 if m["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
