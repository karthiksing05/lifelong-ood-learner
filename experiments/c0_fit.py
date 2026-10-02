"""Plan C (docs/PLAN_PRIMITIVES.md): fit one motion PRIMITIVE per step of a task from its one demonstration.

The instruction's steps (primitives.steps_of) cut the demo at its first sustained release (primitives.segment); each segment's
frames, prompted with the step's clause, fit a keyed LoRA on the action expert by flow matching (adapt.train, the one-demo LoRA's
settings).  Saves results/robot/prim_<task>_<step><tag>.pt (meta: clause, step key, frames); ledger row c0_fit.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/c0_fit.py --task 3" slurm/jobs/robot_gpu.sbatch
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
from lifelong_ood.policy import primitives as P                        # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--scope", default="expert")
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    text = texts[a.task]
    steps = P.steps_of(text)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    ep_idx = d90.episodes_of_text(text)[a.demo]
    ep = d90.load(ep_idx)
    segs = P.segment(ep["actions"], len(steps))
    if segs is None:
        print(f"[c0] task {a.task}: the demo has no sustained release to cut {len(steps)} steps at; fitting one primitive", flush=True)
        steps, segs = [text], [(0, len(ep["actions"]))]
    norm = D.Norm.load()
    state, banks = adapt.inject(bk, scope=a.scope)                       # once: every step's primitive is its own key
    out, total = [], 0
    for s, (clause, (lo, hi)) in enumerate(zip(steps, segs)):
        t0 = time.time()
        obs = []
        for i in range(lo, hi):
            o = d90.observation(ep, i)
            o["prompt"] = clause
            obs.append(o)
        bank = adapt.PrefixBank(bk, obs)
        idx, _ = D.chunk_indices(len(ep["actions"]), bk.H)                # a chunk near the step's end runs into the demo's real
        full = D.pad_rows(D.to_rows(ep["actions"][idx[lo:hi]], norm), bk.H)  # next actions (its release); padding at the step's end
        #                                                                   taught the first step never to open the gripper
        key = f"prim_{a.task}_{s}{a.tag}"
        params = add_node(banks, key, a.rank)
        hist = adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=a.lr, batch=a.batch,
                           log=lambda x: print(x, flush=True))
        path = os.path.join(R, f"prim_{a.task}_{s}{a.tag}.pt")
        meta = dict(task=a.task, step=s, clause=clause, step_key=P.step_key(clause), frames=[int(lo), int(hi)], rank=a.rank,
                    scope=a.scope, demo_episode=int(ep_idx))
        nb = adapt.save(banks, key, path, meta=meta)
        total += nb
        out.append(dict(meta, bytes=nb, first_loss=hist["loss"][0], final_loss=hist["loss"][-1], secs=time.time() - t0, path=path))
        print(f"[c0] task {a.task} step {s} '{clause}' key {meta['step_key']}: frames {lo}-{hi}, {nb / 1e6:.1f} MB, flow "
              f"{hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}", flush=True)
        del bank
    print(ledger.record("c0_fit", bench="libero_90", tag=f"task{a.task}{a.tag}", seed=0, keys=dict(task=a.task),
                        metrics=dict(text=text, steps=out), bytes=dict(experts=total, tree=0, codes=0, samples=0, teacher_copies=0,
                                                                         total=total),
                        config=vars(a), script="experiments/c0_fit.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
