"""CONTINUAL-LEARNING BASELINES for plan C (one demonstration per task, every task requests its demo): ONE LoRA on the action expert
trained on the stream's tasks
    joint -- multi-task: all tasks' demos at once (the "consolidate everything into one adapter" upper reference),
    seq   -- sequential fine-tuning: task after task in the stream order, each on its own demo only (forgetting),
    er    -- sequential with experience replay: each batch half the current task's demo, half all earlier demos.
Each demo frame is prompted with its task's own instruction.  Saves results/robot/bcl_<mode><tag>.pt; ledger row b_continual.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/b_continual.py --mode joint --tasks 10+15+..." slurm/jobs/robot_gpu.sbatch
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.experts.lora import add_node                         # noqa: E402
from lifelong_ood.policy import adapt                                  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("joint", "seq", "er"))
    ap.add_argument("--tasks", required=True, help="t1+t2+... in stream order")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500, help="per task (seq, er)")
    ap.add_argument("--joint-steps", type=int, default=2000, help="joint: total steps over all the tasks' demos")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tasks = [int(x) for x in a.tasks.split("+")]
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    obs, rows, owner = [], [], []
    skipped = [t for t in tasks if not d90.episodes_of_text(texts[t])]
    tasks = [t for t in tasks if t not in skipped]                       # no demonstration in the dataset (task 51)
    if skipped:
        print(f"[bcl] skipped (no demonstration in the dataset): {skipped}", flush=True)
    for t in tasks:
        ep = d90.load(d90.episodes_of_text(texts[t])[0])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        rows.append(D.pad_rows(D.to_rows(ep["actions"][idx], norm), bk.H))
        obs += [d90.observation(ep, i) for i in range(T)]                  # prompt = the task's own instruction
        owner += [t] * T
    full = np.concatenate(rows, 0)
    owner = np.asarray(owner)
    bank = adapt.LazyPrefixBank(bk, obs)                                    # joint / er: prefixes computed per batch
    state, banks = adapt.inject(bk)
    key = f"bcl_{a.mode}"
    params = add_node(banks, key, a.rank)
    t0 = time.time()
    curves = []
    if a.mode == "joint":
        curves.append(adapt.train(bk, state, key, params, bank, full, steps=a.joint_steps, lr=a.lr, batch=a.batch,
                                  log=lambda x: print(x, flush=True)))
    else:
        for i, t in enumerate(tasks):
            cur = np.where(owner == t)[0]
            if a.mode == "seq":                                             # the current demo alone: its prefixes cached
                tb = adapt.PrefixBank(bk, [obs[j] for j in cur])
                curves.append(adapt.train(bk, state, key, params, tb, full[cur], steps=a.steps, lr=a.lr, batch=a.batch, seed=i,
                                          log=lambda x: print(x, flush=True)))
                del tb
            else:
                groups = [(cur, a.batch)]
                if i > 0:
                    past = np.where(np.isin(owner, tasks[:i]))[0]
                    groups = [(cur, a.batch // 2), (past, a.batch - a.batch // 2)]
                curves.append(adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=a.lr, batch=a.batch, seed=i,
                                          groups=groups, log=lambda x: print(x, flush=True)))
            print(f"[bcl] {a.mode}: task {t} ({i + 1}/{len(tasks)}) done at {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(R, f"bcl_{a.mode}{a.tag}.pt")
    nb = adapt.save(banks, key, path, meta=dict(mode=a.mode, tasks=tasks, rank=a.rank, scope="expert"))
    print(f"[bcl] saved {path} ({nb / 1e6:.1f} MB) in {time.time() - t0:.0f}s", flush=True)
    print(ledger.record("b_continual", bench="libero_90", tag=f"{a.mode}{a.tag}", seed=0,
                        metrics=dict(mode=a.mode, tasks=tasks, frames=int(len(obs)), final_losses=[c["loss"][-1] for c in curves],
                                     secs=time.time() - t0),
                        bytes=dict(experts=nb, tree=0, codes=0, samples=0, teacher_copies=0, total=nb), config=vars(a),
                        script="experiments/b_continual.py", command=" ".join(sys.argv), artifacts=[path]))


if __name__ == "__main__":
    main()
