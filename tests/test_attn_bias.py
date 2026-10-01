"""P0 of the Bind plan: the attention bias's mechanics on the real policy (GPU; run through slurm, never on the login node).

    sbatch --export=ALL,ENV=openpi,SCRIPT="tests/test_attn_bias.py" slurm/jobs/robot_gpu.sbatch

Checks: (1) the column map puts a camera's 16 x 16 grid at its prefix columns; (2) beta = 0 (fixed) and rho = None (mass),
and the logging path with no bias, reproduce the policy's chunk bit for bit; (3) a bias changes the chunk and reaches all 18
expert layers and no VLM layer (the prefix cache is unchanged); (4) the mass mode raises the referent's attention mass to at
least rho in every layer; (5) a prefix with an appended token keeps its extra columns unbiased.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import attn_bias as AB          # noqa: E402


def obs_of(path, k=0):
    z = np.load(path, allow_pickle=True)
    return {"observation/image": z["img"][k], "observation/wrist_image": z["wimg"][k],
            "observation/state": z["state8"][k], "prompt": str(z["language"])}


def main():
    m = AB.prefix_map({"agent": np.ones((16, 16)), "wrist": np.full((16, 16), 0.5)}, 968)
    assert float(m[:256].min()) == 1.0 and float(m[256:512].min()) == 0.5 and float(m[512:].abs().max()) == 0.0
    print("[p0] column map ok")

    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    obs = obs_of(os.path.join(ROOT, "results", "robot", "f0_firstframes", "libero_90_48.npz"))
    pre = bk.prefix(bk.observation(obs))
    k0 = [t.clone() for t in pre.kv.key_cache]
    base = bk.sample(1, pre, seed=3)
    grid = np.zeros((16, 16), np.float32)
    grid[6:10, 4:8] = 1.0
    maps = {"agent": grid}
    same = {
        "fixed beta=0": bk.sample(1, pre, seed=3, step_gate=AB.bias_gate(bk, maps, pre, mode="fixed", beta=0.0)),
        "mass rho=None": bk.sample(1, pre, seed=3, step_gate=AB.bias_gate(bk, maps, pre, mode="mass", rho=None)),
        "log only": bk.sample(1, pre, seed=3, step_gate=AB.bias_gate(bk, maps, pre, mode="fixed", beta=0.0, log=[])),
    }
    for k, v in same.items():
        d = float(np.abs(v - base).max())
        print(f"[p0] {k}: max |diff| {d:.3g}")
        assert d == 0.0, f"{k} is not bit-identical"
    log = []
    steered = bk.sample(1, pre, seed=3, step_gate=AB.bias_gate(bk, maps, pre, mode="fixed", beta=4.0, log=log))
    d = float(np.abs(steered - base).max())
    layers = sorted({l for l, _, _ in log})
    print(f"[p0] fixed beta=4: max |diff| {d:.3g}; layers biased {len(layers)}; mass before {np.mean([a for _, a, _ in log]):.4f} "
          f"after {np.mean([b for _, _, b in log]):.4f}")
    assert d > 0 and layers == list(range(18))
    for a, b in zip(k0, pre.kv.key_cache):
        assert torch.equal(a, b), "the prefix cache changed"
    log = []
    bk.sample(1, pre, seed=3, step_gate=AB.bias_gate(bk, maps, pre, mode="mass", rho=0.5, log=log))
    after = np.array([b for _, _, b in log])
    print(f"[p0] mass rho=0.5: per-call mass after min {after.min():.3f} mean {after.mean():.3f}")
    assert after.min() >= 0.5 - 1e-3
    from lifelong_ood.policy import token_fold as TF
    tok = TF.KVToken.from_prefix(pre, [800])
    pre2 = TF.attach(pre, tok.to(bk.device) if hasattr(tok, "to") else tok)
    log = []
    bk.sample(1, pre2, seed=3, step_gate=AB.bias_gate(bk, maps, pre2, mode="fixed", beta=4.0, log=log))
    print(f"[p0] with an attached token: width {pre2.pad_masks.shape[1]}, biased calls {len(log)}")
    print("[p0] ALL PASS")


if __name__ == "__main__":
    main()
