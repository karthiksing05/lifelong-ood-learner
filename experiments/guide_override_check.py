"""How much does the frozen policy contribute to a chunk sampled through the bindings' guide (w = 1 constant, variance 1e-3)?

With the guide at weight 1, the sampler sets the real dims' noise prediction to the guide Gaussian's own (backbone.sample:
e = e0 + w (e_guide - e0)) on every step but the first, so the sampled chunk should be the guide's mean up to O(sqrt(var)),
whatever the policy predicts.  This measures it: for two observations with different images and instructions, and three noise
seeds, the distance of the guided sample from the guide's mean, and of the two observations' guided samples from each other,
against the spread of the policy's own unguided samples.  Normalised action units.

    sbatch ... <venv python> experiments/guide_override_check.py --backbone pi05_libero | smolvla_libero
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="pi05_libero")
    a = ap.parse_args()
    from lifelong_ood.operators.decompose import guide
    if a.backbone.startswith("smolvla"):
        from lifelong_ood.policy.backbone_smolvla import SmolVLABackbone
        bk = SmolVLABackbone()
    else:
        from lifelong_ood.policy.backbone import PolicyBackbone
        ck = None if a.backbone == "pi05_libero" else f"/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/{a.backbone}_pytorch"
        bk = PolicyBackbone(a.backbone, checkpoint_dir=ck)
    r = np.random.default_rng(0)
    obs = []
    for prompt in ("put the black bowl on the plate", "open the top drawer of the cabinet"):
        obs.append({"observation/state": r.normal(size=8).astype(np.float32), "prompt": prompt,
                    "observation/image": r.integers(256, size=(224, 224, 3), dtype=np.uint8),
                    "observation/wrist_image": r.integers(256, size=(224, 224, 3), dtype=np.uint8)})
    pres = [bk.prefix(bk.observation(o)) for o in obs]
    m = np.clip(r.normal(scale=0.5, size=bk.dim), -1.5, 1.5).astype(np.float32)     # a target chunk in normalised units
    g = guide(m, np.full_like(m, 1e-3), wmax=1.0, schedule="const", path="flow")
    out = {"backbone": a.backbone, "H": bk.H, "dim": bk.dim}
    guided = [[bk.sample(1, p, seed=s, guide=g)[0] for s in range(3)] for p in pres]
    own = [[bk.sample(1, p, seed=s)[0] for s in range(3)] for p in pres]
    out["guided_minus_target_max"] = float(max(np.abs(x - m).max() for xs in guided for x in xs))
    out["guided_minus_target_rms"] = float(np.sqrt(np.mean([np.mean((x - m) ** 2) for xs in guided for x in xs])))
    out["guided_obs0_minus_obs1_rms"] = float(np.sqrt(np.mean([np.mean((guided[0][s] - guided[1][s]) ** 2) for s in range(3)])))
    out["own_obs0_minus_obs1_rms"] = float(np.sqrt(np.mean([np.mean((own[0][s] - own[1][s]) ** 2) for s in range(3)])))
    out["own_minus_target_rms"] = float(np.sqrt(np.mean([np.mean((x - m) ** 2) for xs in own for x in xs])))
    out["guided_real_actions_max_diff"] = float(np.abs(bk.unnormalize(np.stack(guided[0])) - bk.unnormalize(np.stack([m] * 3))).max())
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "robot", f"guide_override_{a.backbone}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
