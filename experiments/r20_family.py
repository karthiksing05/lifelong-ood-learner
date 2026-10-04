"""Family transfer (pre-registered: ledger m3_prereg tag family): do the coarse prototype levels predict which OTHER tasks' data
teach a held-out task?

Families = the clusters of the 21 held-out tasks by the edit distance between their demos' label sequences at t = 0.4
(experiments/r19_families.py, average linkage, cut 0.5; results/robot/r20_families.json): pick-and-place into a receptacle (7
tasks), put in the top drawer and close it (3), and {34, 42}.  For a target task in a family of k tasks:

    own   the target's ONE demo, every frame (as the one-demo LoRA); the target's own practice is NOT used
    pool  --arm fam: the k - 1 other family members; --arm rnd: k - 1 tasks drawn (seed 1000 + task) from the held-out tasks
          outside the family.  Per source task: its demo, every frame, + every 5th frame of its successful practice rollouts
          (m5_practice, m6_round1, m6_round2: init states 0-29), under its OWN instruction; --pool-cap frames in all, an equal
          share per source, uniformly subsampled

LoRA rank 16 on the action expert, 1000 steps, lr 1e-4, batch 32 = --own-per-batch rows of the own demo + the rest from the pool,
seed 0 (the practice learners' settings).  Scored on init states 30-49 (launch_pack.py --arms lora@r20fam,lora@r20rnd).

    python experiments/r20_family.py --task 36 --arm fam
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import adapt                                   # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402

R = os.path.join(ROOT, "results", "robot")
TASKS = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
PRACTICE = ("m5_practice", "m6_round1", "m6_round2")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sources(task, arm, level):
    fams = json.load(open(os.path.join(R, "r20_families.json")))["families"][level]
    fam = next(f for f in fams if task in f)
    others = [m for m in fam if m != task]
    if arm == "fam":
        return fam, others
    rng = np.random.default_rng(1000 + task)
    outside = [t for t in TASKS if t not in fam]
    return fam, sorted(int(x) for x in rng.choice(outside, size=len(others), replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--arm", choices=["fam", "rnd"], required=True)
    ap.add_argument("--name", default=None, help="default r20<arm>")
    ap.add_argument("--level", default="0.4")
    ap.add_argument("--pool-cap", type=int, default=1000)
    ap.add_argument("--own-per-batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bank-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="list the data only (no model)")
    ap.add_argument("--two-stage", action="store_true", help="pre-registered tag family2: stage 1 = the pool ALONE (uniform batches), "
                    "stage 2 = the one-demo LoRA's recipe (r3_fewshot.py: every demo frame, 1000 steps, lr 1e-4, batch 32, seed 0) "
                    "continued from the stage-1 adapter with a fresh optimizer")
    a = ap.parse_args()
    os.chdir(ROOT)
    name = a.name or (f"r21{a.arm}" if a.two_stage else f"r20{a.arm}")
    log = lambda s: print(s, flush=True)                                  # noqa: E731
    t = a.task
    fam, src = sources(t, a.arm, a.level)
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    # OWN: the target's one demo, every frame
    ep = d90.load(d90.episodes_of_task(t)[0])
    T0 = len(ep["actions"])
    own_rows = D.to_rows(ep["actions"][D.chunk_indices(T0, 10)[0]], norm)
    own = [("demo", t, ep, i) for i in range(T0)]
    # POOL: per source, its demo (every frame) + every stride-th frame of its practice successes, an equal share of the cap
    def build(srcs, n_succ):
        share = a.pool_cap // max(len(srcs), 1)
        out = []
        for s in srcs:
            eps_ = d90.load(d90.episodes_of_task(s)[0])
            cand = [("demo", s, eps_, i) for i in range(len(eps_["actions"]))]
            files = sorted(f for p in PRACTICE for f in glob.glob(f"{R}/{p}_{s}_*_rec/task{s}_trial*.npz"))
            succ = [f for f in files if bool(np.load(f)["success"])]
            n_succ[s] = len(succ)
            for f in succ:
                T = len(np.load(f)["action"])
                cand += [("rec", s, f, i) for i in range(0, T, a.stride)]
            keep = np.linspace(0, len(cand) - 1, min(share, len(cand))).round().astype(int)
            out += [cand[i] for i in keep]
        return out
    n_succ = {}
    pool = build(src, n_succ)
    if a.arm == "rnd":                   # size-matched: never more frames than the family's own pool for this target
        n_fam = len(build(sources(t, "fam", a.level)[1], {}))
        if len(pool) > n_fam:
            keep = np.linspace(0, len(pool) - 1, n_fam).round().astype(int)
            pool = [pool[i] for i in keep]
    log(f"[r20] task {t} '{texts[t]}', arm {a.arm}: family {fam}; pool sources {src} ({[n_succ[s] for s in src]} successes); "
        f"{len(own)} own frames + {len(pool)} pool frames")
    if a.dry:
        return
    # observations and chunk rows (each recording loaded once)
    obs, rows = [], []
    for _, _, e, i in own:
        obs.append(d90.observation(e, int(i))); rows.append(own_rows[i])
    cache = {}
    for kind, s, e, i in pool:
        if kind == "demo":
            key_ = ("demo", s)
            if key_ not in cache:
                cache[key_] = D.to_rows(e["actions"][D.chunk_indices(len(e["actions"]), 10)[0]], norm)
            obs.append(d90.observation(e, int(i))); rows.append(cache[key_][i])
        else:
            if e not in cache:
                z = dict(np.load(e))
                z["rows"] = D.to_rows(z["action"][D.chunk_indices(len(z["action"]), 10)[0]], norm)
                cache = {k: v for k, v in cache.items() if k[0] == "demo"}       # keep one recording in memory at a time
                cache[e] = z
            z = cache[e]
            obs.append({"observation/image": z["image"][i], "observation/wrist_image": z["wrist"][i],
                        "observation/state": z["state"][i], "prompt": texts[s]})
            rows.append(z["rows"][i])
    r4 = load_module("r4", os.path.join(ROOT, "experiments", "r4_fold.py"))
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    t0 = time.time()
    state, banks = adapt.inject(bk, scope="expert")
    key = f"{name}_{t}"
    params = r4.add_node(banks, key, a.rank)
    if a.two_stage:
        # stage 1: the pool alone -- the family's (or random tasks') grammar and bindings
        n_own = len(own)
        pool_bank = adapt.PrefixBank(bk, obs[n_own:], store=a.bank_device)
        h1 = adapt.train(bk, state, key, params, pool_bank, D.pad_rows(np.stack(rows[n_own:]), bk.H), steps=a.steps, lr=1e-4,
                         batch=32, seed=a.seed, log=log)
        del pool_bank
        # stage 2: exactly the one-demo LoRA's call (r3_fewshot.py), from the stage-1 adapter: the target's own bindings
        own_bank = adapt.PrefixBank(bk, obs[:n_own])
        h2 = adapt.train(bk, state, key, params, own_bank, D.pad_rows(np.stack(rows[:n_own]), bk.H), steps=a.steps, lr=1e-4,
                         batch=32, seed=a.seed, log=log)
        hist = dict(stage1=h1, stage2=h2)
    else:
        bank = adapt.PrefixBank(bk, obs, store=a.bank_device)
        groups = [(np.arange(len(own)), a.own_per_batch), (np.arange(len(own), len(obs)), 32 - a.own_per_batch)]
        hist = adapt.train(bk, state, key, params, bank, D.pad_rows(np.stack(rows), bk.H), steps=a.steps, lr=1e-4, batch=32,
                           seed=a.seed, log=log, groups=groups)
    path = f"{R}/{name}_{t}_lora.pt"
    nb = adapt.save(banks, key, path, meta=dict(task=t, rank=a.rank, arm=a.arm, family=fam, sources=src, two_stage=a.two_stage))
    log(f"[r20] saved {path} ({nb} B) in {time.time() - t0:.0f}s")
    ledger.record("r20_family", bench="libero_90", tag=name, seed=a.seed, keys=dict(task=t),
                  metrics=dict(task=t, text=texts[t], arm=a.arm, family=fam, sources=src, successes=n_succ, own=len(own),
                               pool=len(pool), curve=hist),
                  bytes=dict(experts=nb, tree=0, codes=0, samples=0, teacher_copies=0, total=nb), config=vars(a),
                  script="experiments/r20_family.py", command=" ".join(sys.argv), artifacts=[path])


if __name__ == "__main__":
    main()
