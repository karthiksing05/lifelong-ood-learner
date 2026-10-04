"""ONE gated fold shared by several held-out tasks.

Consolidation (plan M6): the fold's arrival set is the union of the contact windows (-pre/+contact frames around each gripper
event) of every listed task's demo; the anchor half is the same as r4_fold.py (library frames, target = the base velocity with
the expert off).  Trained on some tasks and applied to OTHERS (cross-fitting), it tests whether a new task can reuse the
library's existing fold, i.e. cost only its program.  At rollout each task switches the shared expert on in its own windows
(the fold mask of its r4 program).  Same size as one per-task fold (rank 2 on the last two layers: 258 KB).

    python experiments/r13_shared_fold.py --tasks 1,34,37 --tag shared_all
Writes results/robot/r13_<tag>_expert.pt and a ledger row r13_shared_fold.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import adapt                                   # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--library", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stride", type=int, default=2, help="every k-th frame of the contact windows")
    ap.add_argument("--pre", type=int, default=30)
    ap.add_argument("--contact", type=int, default=10)
    ap.add_argument("--n-anchor", type=int, default=384)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--scope", default="last2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    log = lambda s: print(s, flush=True)                                  # noqa: E731
    rng = np.random.default_rng(a.seed)
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r4 = load_module("r4", os.path.join(ROOT, "experiments", "r4_fold.py"))
    texts, _ = r3p.libero90_texts()
    demos, d90, norm = D.Demos(), D.Demos90(), D.Norm.load()
    lib = np.load(a.library)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    obs, chunks, per_task = [], [], {}
    for t in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
        ep = d90.load(d90.episodes_of_task(t)[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        win = np.zeros(T, bool)
        for e in D.gripper_events(ep["actions"]):
            win[max(0, e - a.pre): e + a.contact + 1] = True
        idx = np.where(win)[0][::a.stride]
        per_task[t] = int(len(idx))
        obs += [d90.observation(ep, int(i)) for i in idx]
        chunks.append(rows[idx])
        log(f"[r13] task {t} '{texts[t][:40]}': {int(win.sum())} window frames, {len(idx)} used")
    t0 = time.time()
    arr_bank = adapt.PrefixBank(bk, obs)
    arr_chunks = D.pad_rows(np.concatenate(chunks), bk.H)
    aj = rng.choice(len(lib["episode"]), size=min(a.n_anchor, len(lib["episode"])), replace=False)
    anc_bank = adapt.PrefixBank(bk, r4.library_obs(demos, lib["episode"][aj], lib["frame"][aj]))
    anc_chunks = D.pad_rows(lib["demo"][aj], bk.H)
    log(f"[r13] prefixes: {arr_bank.n} arrival ({arr_bank.bytes() / 1e9:.1f} GB) + {anc_bank.n} anchor in {time.time() - t0:.0f}s")
    state, banks = adapt.inject(bk, scope=a.scope)
    key = f"fold_{a.tag}"
    params = r4.add_node(banks, key, a.rank)
    hist = adapt.train(bk, state, key, params, arr_bank, arr_chunks, steps=a.steps, lr=a.lr, seed=a.seed,
                       anchor=(anc_bank, anc_chunks), anchor_weight=a.anchor_weight, log=log)
    path = f"results/robot/r13_{a.tag}_expert.pt"
    eb = adapt.save(banks, key, path, meta=dict(tasks=list(per_task), rank=a.rank, scope=a.scope, shared=True))
    m = dict(tasks=list(per_task), frames_per_task={str(k): v for k, v in per_task.items()}, n_arrival=int(arr_bank.n),
             expert_params=adapt.n_params(params), curve=hist)
    ledger.record("r13_shared_fold", bench="libero_90", tag=a.tag, seed=a.seed, metrics=m,
                  bytes=dict(experts=eb, tree=0, codes=0, samples=0, teacher_copies=0, total=eb),
                  config=vars(a), script="experiments/r13_shared_fold.py", command=" ".join(sys.argv), artifacts=[path])
    log(f"[r13] done: {path}, {eb / 1e3:.0f} KB shared by {len(per_task)} tasks")


if __name__ == "__main__":
    main()
