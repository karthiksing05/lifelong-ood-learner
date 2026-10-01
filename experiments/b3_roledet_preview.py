"""Preview of the DEPLOYABLE role editor (lifelong_ood/policy/role_hooks.py) on first frames (experiments/f0b_firstframes_seg.py):
what the policy would see, and whether the detector's boxes land on the right objects (scored against the oracle masks).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/b3_roledet_preview.py --tasks 48+49+52+53" slurm/jobs/robot_gpu.sbatch
-> results/robot/b3_roledet_preview_<task>.png (rows: states; columns: real | edited, agent view) and b3_roledet_boxes.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import factors as F             # noqa: E402
from lifelong_ood.policy import roles as RO              # noqa: E402
from lifelong_ood.policy import where as W               # noqa: E402
from lifelong_ood.policy.role_hooks import RoleHooks     # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def main():
    tids = [int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].split("+")] if "--tasks" in sys.argv else [48, 49, 52, 53]
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    zs = {}
    for f in glob.glob(os.path.join(ROOT, "ledger", "zeroshot90", "*.json")):
        zs.update({int(k): v["success_rate"] for k, v in json.load(open(f))["metrics"]["tasks"].items()})
    bind = json.load(open(os.path.join(R, "f1_bind_libero_90_binding.json")))
    out = {}
    hooks = None
    from PIL import Image
    for tid in tids:
        t = by[("libero_90", tid)]
        code = RO.role_code(t, tasks, zs, bind)
        ph = F.scene_phrases(t, tasks)
        phrases = {n: ph.get(t.objects.get(n), str(t.objects.get(n, n)).replace("_", " ")) for p in code["pairs"] for n in p}
        if hooks is None:
            hooks = RoleHooks(code, phrases)
        else:
            hooks.code, hooks.prompt, hooks.pairs, hooks.phrases = code, code["prompt"], code["pairs"], phrases
        z = np.load(os.path.join(R, "f0b_firstframes", f"libero_90_{tid}.npz"), allow_pickle=True)
        rows, hits = [], []
        for k in range(len(z["states"])):
            img = z["img"][k]
            rec = {}
            for inst in code["pairs"][0]:
                b, sc = hooks._box(img, inst)
                truth = W.oracle_map(z["seg"][k], W.geoms_of(z, name=inst))
                if b is None:
                    rec[inst] = dict(hit=0, score=0.0)
                    continue
                s = 256 / 224.0
                ci = min(15, int(0.5 * (b[1] + b[3]) * s) // 16)
                cj = min(15, int(0.5 * (b[0] + b[2]) * s) // 16)
                rec[inst] = dict(hit=int(truth[ci, cj] > 0.1), score=sc, box=b)
            hits.append(rec)
            obs = hooks.rewrite({"observation/image": img, "observation/wrist_image": z["wimg"][k], "prompt": t.language}, {})
            rows.append(np.concatenate([img, obs["observation/image"]], axis=1))
        Image.fromarray(np.concatenate(rows[:4], axis=0)).save(os.path.join(R, f"b3_roledet_preview_{tid}.png"))
        acc = {inst: float(np.mean([h[inst]["hit"] for h in hits])) for inst in code["pairs"][0]}
        out[str(tid)] = dict(code=code, phrases=phrases, box_on_object=acc, states=hits)
        print(f"[b3d] {tid} {t.language}: box centre on the object {acc}", flush=True)
    json.dump(out, open(os.path.join(R, "b3_roledet_boxes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
