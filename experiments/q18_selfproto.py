"""SELF-PROTOTYPES: a 4 KB object prototype from the learner's OWN successful attempts, no demonstration (ledger m3_prereg tag
q18_self).  In a successful episode the end effector sits on the moved object when the commanded gripper first closes; that position,
projected into the episode's first frame (of the closes sustained for 5
records, the lowest: the commanded gripper flickers shut during the approach and some episodes close it at the start) (the fixed agent camera of experiments/f0b_firstframes_seg.py), marks the object's patch,
and the policy's own SigLIP features there and at its 4 neighbours, averaged over the task's successful episodes, are the prototype
(the same construction as experiments/q4_template.py, with the policy's own success in place of the demonstration).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q18_selfproto.py --rec q18_explore --tag zs_seed4" slurm/jobs/robot_gpu.sbatch
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import where as W           # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FR = os.path.join(R, "f0b_firstframes")
OUT = os.path.join(R, "q18_self_protos.npz")
DEMO = os.path.join(R, "q4_object_protos.npz")
RUN = 5


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("q4", os.path.join(ROOT, "experiments", "q4_template.py"))
    q4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q4)
    out = os.path.join(R, sys.argv[sys.argv.index("--out") + 1] + ".npz") if "--out" in sys.argv else OUT
    demo = np.load(DEMO)
    from lifelong_ood.policy import bind_hooks as BH
    from lifelong_ood.policy import factors as F
    by = {t.index: t for t in F.load() if t.suite == "libero_90"}
    exp = sys.argv[sys.argv.index("--rec") + 1]
    tag = sys.argv[sys.argv.index("--tag") + 1]
    bk = PolicyBackbone("pi05_libero", device="cuda")
    protos, per = {}, {}
    for d in sorted(glob.glob(os.path.join(R, f"{exp}_*_{tag}_rec"))):
        tid = int(os.path.basename(d)[len(exp) + 1:].split("_")[0])
        z0 = np.load(os.path.join(FR, f"libero_90_{tid}.npz"), allow_pickle=True)
        vecs, toks = [], []
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            e = np.load(f)
            if not bool(e["success"]):
                continue
            c = e["action"][:, 6] > 0                                   # the grasp: of the SUSTAINED closes (>= 5 records; single-
            run = np.convolve(c.astype(int), np.ones(RUN, int), "valid") == RUN   # step closes flicker during the approach, and
            starts = [i for i in np.where(run)[0] if i == 0 or not c[i - 1]]      # some episodes close at the start), the one
            if not starts:                                                        # that begins lowest: the bottom of the approach
                continue
            g = int(min(starts, key=lambda i: e["state"][i, 2]))
            rc = W.project(e["state"][g][:3], z0["K"], z0["E"][0])
            i, j = int(np.clip(rc[0] // 16, 0, 15)), int(np.clip(rc[1] // 16, 0, 15))
            f0 = q4.feats(bk, e["image"][0], e["wrist"][0], e["state"][0])
            v = np.mean([f0[a * 16 + b] for a, b in q4.neighbours(i, j)], 0)
            vecs.append(v / np.linalg.norm(v))
            toks.append([i, j])
        n_eps = len(glob.glob(os.path.join(d, "*.npz")))
        per[str(tid)] = dict(episodes=n_eps, successes=len(vecs), tokens=toks)
        if vecs:
            p = np.mean(vecs, 0)
            protos[str(tid)] = (p / np.linalg.norm(p)).astype(np.float32)
            moved = BH.referents(by[tid])["open"][0]                    # validity: the best patch on the scoring task's first
            hits = []                                                   # frames (states 0-9) lies on the moved object
            for k in range(len(z0["states"])):
                fk = q4.feats(bk, z0["img"][k], z0["wimg"][k], z0["state8"][k])
                truth = W.oracle_map(z0["seg"][k], W.geoms_of(z0, name=moved))
                hits.append(int(truth.reshape(-1)[int(np.argmax(fk @ protos[str(tid)]))] > 0.1))
            per[str(tid)]["pointing"] = float(np.mean(hits))
            per[str(tid)]["cos_demo"] = float(protos[str(tid)] @ demo[str(tid)]) if str(tid) in demo.files else None
        print(f"[q18] task {tid}: {len(vecs)} successes of {n_eps} -> {'prototype' if vecs else 'none'} {toks[:3]} "
              f"pointing {per[str(tid)].get('pointing')} cos to demo {per[str(tid)].get('cos_demo')}", flush=True)
    np.savez(out, **protos)
    json.dump(dict(per_task=per, n_protos=len(protos)), open(out[:-4] + ".json", "w"), indent=1)
    print(f"[q18] {len(protos)} self-prototypes -> {out}", flush=True)


if __name__ == "__main__":
    main()
