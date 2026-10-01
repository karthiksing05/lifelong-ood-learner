"""Interpretability of role composition: does a role-rebound episode run the BASE task prototype's verb sentence?

Every recorded episode's executed actions are labelled with the policy's verb prototypes (lifelong_ood/policy/tasks.py
Labeller: basins of the t = 0.4 level of the marginal tree, collapsed over repeats; names from grammar_readout.json).  For each
role-coded task, the chains of its role episodes are compared (normalised edit distance) with the chains of the base task's
own zero-shot episodes, and with those of the other tasks' zero-shot episodes in the same scene (a reference distance).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/b4_role_chains.py --pairs 52:50+53:50 --role-rec b_k1_practice" \
        slurm/jobs/robot_gpu.sbatch
-> results/robot/b4_role_chains.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import tasks as TK                # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def edit(a, b):
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return d[n, m] / max(n, m, 1)


def episodes(pattern, success_only=False):
    out = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if success_only and not bool(z["success"]):
            continue
        out.append(dict(file=os.path.relpath(f, ROOT), actions=np.asarray(z["action"], np.float32), success=bool(z["success"])))
    return out


def main():
    pairs = [tuple(int(x) for x in p.split(":")) for p in sys.argv[sys.argv.index("--pairs") + 1].split("+")]
    role_exp = sys.argv[sys.argv.index("--role-rec") + 1] if "--role-rec" in sys.argv else "b_k1_practice"
    lab = TK.Labeller(levels=(0.4,))
    names = TK.verb_names("0.4")
    res = {}
    for tid, base in pairs:
        role = episodes(os.path.join(R, f"{role_exp}_{tid}_role_rec", "*.npz"), success_only=True)
        base_eps = episodes(os.path.join(R, f"f1_bind_{base}_zeroshot_rec", "*.npz"), success_only=True)
        others = {}
        for g in sorted(glob.glob(os.path.join(R, "f1_bind_*_zeroshot_rec"))):
            t2 = int(os.path.basename(g).split("_")[2])
            if t2 not in (tid, base):
                others[t2] = episodes(os.path.join(g, "*.npz"), success_only=True)
        allx = [e["actions"] for e in role + base_eps] + [e["actions"] for v in others.values() for e in v]
        if not role or not base_eps:
            print(f"[b4] {tid}: role {len(role)} / base {len(base_eps)} successful episodes; skipped", flush=True)
            continue
        labs = lab(allx, every=5)[0.4]
        chains = [TK.collapse(l) for l in labs]
        nr, nb = len(role), len(base_eps)
        cr, cb = chains[:nr], chains[nr:nr + nb]
        co = chains[nr + nb:]
        d_rb = float(np.mean([edit(a, b) for a in cr for b in cb]))
        d_bb = float(np.mean([edit(a, b) for i, a in enumerate(cb) for j, b in enumerate(cb) if i < j])) if nb > 1 else None
        d_ro = float(np.mean([edit(a, b) for a in cr for b in co])) if co else None
        ex = cr[0]
        res[str(tid)] = dict(base=base, n_role=nr, n_base=nb, n_other=len(co), d_role_base=d_rb, d_base_base=d_bb,
                             d_role_other=d_ro, example_role_chain=[names.get(int(x), str(x)) for x in ex],
                             example_base_chain=[names.get(int(x), str(x)) for x in cb[0]])
        print(f"[b4] {tid} vs base {base}: role-base {d_rb:.3f}, base-base {d_bb}, role-other {d_ro} | role chain "
              f"{' > '.join(res[str(tid)]['example_role_chain'][:8])}", flush=True)
    json.dump(res, open(os.path.join(R, "b4_role_chains.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
