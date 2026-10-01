"""GPU checks of the composed sampler (lifelong_ood/policy/conditionals.py) on real first observations (f0_firstframes):

  identity_K1     a one-prompt composition is PolicyBackbone.sample bit for bit (same noise)
  identity_K3     a three-prompt composition with coefficients e_0 matches the base alone (batched VLM pass: ~1e-5)
  in_the_loop     composed chunks at two unrelated observations stay as far apart as the policy's own (the guide toward a plan
                  gave 0.0003 against 0.39: ledger p0_audit guide_override)
  arithmetic      the composed sample sits near base + lam (plus - minus) of the prompts' one-step conditional means

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/f2_compose_check.py" slurm/jobs/robot_gpu.sbatch
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")

R = os.path.join(ROOT, "results", "robot")


def first_obs(name, k=0):
    z = np.load(os.path.join(R, "f0_firstframes", name + ".npz"), allow_pickle=True)
    return {"observation/image": z["img"][k], "observation/wrist_image": z["wimg"][k],
            "observation/state": np.asarray(z["state8"][k], np.float32)}, str(z["language"])


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy import conditionals as C
    bk = PolicyBackbone("pi05_libero", device="cuda")
    out = {}
    o48, _ = first_obs("libero_90_48")
    oG, _ = first_obs("libero_goal_1")
    base48 = "pick up the alphabet soup and put it in the basket"
    plus, minus = "pick up the ketchup and put it in the basket", "pick up the alphabet soup and put it in the basket"
    noise = np.random.default_rng(0).standard_normal(bk.dim_full).astype(np.float32)
    # identity, K = 1
    pre1 = C.prefixes(bk, o48, [base48])
    a = C.sample_composed(bk, pre1, np.array([1.0], np.float32), noise=noise)
    b = bk.sample(1, pre1, noise=noise[None])
    out["identity_K1_maxabs"] = float(np.abs(a - b).max())
    # identity, K = 3 with e_0
    pre3 = C.prefixes(bk, o48, [base48, plus, minus])
    a3 = C.sample_composed(bk, pre3, np.array([1.0, 0.0, 0.0], np.float32), noise=noise)
    out["identity_K3_maxabs"] = float(np.abs(a3 - b).max())
    # in the loop: the same composition at two unrelated observations
    comp = C.Composition.factor(base48, [(plus, minus)], [2.0])
    preG = C.prefixes(bk, oG, comp.prompts)
    c48 = C.sample_composed(bk, C.prefixes(bk, o48, comp.prompts), comp.coef, noise=noise)
    cG = C.sample_composed(bk, preG, comp.coef, noise=noise)
    own48 = bk.sample(1, C.prefixes(bk, o48, [base48]), noise=noise[None])
    ownG = bk.sample(1, C.prefixes(bk, oG, [base48]), noise=noise[None])
    out["in_the_loop_composed_rms"] = float(np.sqrt(np.mean((c48 - cG) ** 2)))
    out["in_the_loop_own_rms"] = float(np.sqrt(np.mean((own48 - ownG) ** 2)))
    # arithmetic: composed sample vs the affine combination of the prompts' conditional means
    m = C.tweedie_means(bk, C.prefixes(bk, o48, comp.prompts), noise)
    pred = (np.asarray(comp.coef)[:, None] * m).sum(0, keepdims=True)
    out["arithmetic_rms"] = float(np.sqrt(np.mean((c48 - pred) ** 2)))
    out["factor_difference_rms"] = float(np.sqrt(np.mean((m[1] - m[2]) ** 2)))
    out["composed_vs_base_rms"] = float(np.sqrt(np.mean((c48 - own48) ** 2)))
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(os.path.join(R, "f2_compose_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
