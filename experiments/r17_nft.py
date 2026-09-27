"""Experiment 1 of reports/RL from practice for composition.md: learn from FAILED practice rollouts too.

Same data as the success-only round-2 learner (r16L2: rounds 0 and 1, experiments m5_practice + m6_round1), same positives
(every demo frame + every 5th frame of every success, capped at 1000 by uniform subsampling), same adapter, steps and learning
rate.  NEW: negatives.  In every FAILED rollout the demo-alignment pointer (the composition's own state alignment, recomputed
offline from the recorded 8-d states at every policy query) locates the stall: the first query at which the pointer reaches
its final maximum.  The executed chunks at the last W queries up to the stall are negatives; the rest of the failed rollout
stays unlabelled.  Loss: negative-aware flow matching (adapt.train_nft, DiffusionNFT with beta = 1; equal to the control's
loss on positives).

    python experiments/r17_nft.py --task 39 --name r17bal                       # rank 16, 8 negatives per batch of 32
    python experiments/r17_nft.py --task 39 --name r17pool --pooled             # rank 16, natural frequency
    python experiments/r17_nft.py --task 39 --name r17fold --rank 2 --scope last2
    python experiments/r17_nft.py --task 42 --name r18rec --recovery            # Exp. 2: + the stall-point branches' recovery data
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
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import adapt                                   # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy.compose import ComposeHooks, DemoProgram       # noqa: E402

R = "results/robot"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stall_queries(prog, states, W, every=5):
    """The composition's alignment pointer at every policy query of a recorded rollout; -> (query steps of the W queries up
    to the stall, pointer trace)."""
    h = ComposeHooks(prog, key_mode="pose", window=20)
    h.reset()
    trace = []
    for step in range(0, len(states), every):
        h.rewrite({"observation/state": states[step]}, {})
        trace.append(int(h._j))
    trace = np.asarray(trace)
    q = int(np.argmax(trace == trace.max()))                        # first query at the final maximum
    return [every * k for k in range(max(0, q - W + 1), q + 1)], trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--practice", default="m5_practice+m6_round1")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scope", default="expert")
    ap.add_argument("--window", type=int, default=2, help="W: negative queries up to the stall (pre-registered: 2)")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--neg-batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lora-stride", type=int, default=5)
    ap.add_argument("--max-pos", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recovery", action="store_true", help="Exp. 2 (m3_prereg tag branch): + every 5th post-branch step of each "
                    "RESCUED stall-point branch as positives (<= --max-rec per task, after the cap) + the last W queries up to the "
                    "own stall of each FAILED post-branch segment as negatives (prefix-gated branches of r18_branch_stats.json)")
    ap.add_argument("--max-rec", type=int, default=300)
    ap.add_argument("--bank-device", default=None, help="cpu: keep the cached prefixes in host memory (same values; for banks "
                    "larger than the GPU, e.g. with the recovery data)")
    a = ap.parse_args()
    os.chdir(ROOT)
    log = lambda s: print(s, flush=True)                                  # noqa: E731
    t = a.task
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r4 = load_module("r4", os.path.join(ROOT, "experiments", "r4_fold.py"))
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    ep = d90.load(d90.episodes_of_text(texts[t])[0])
    T0 = len(ep["actions"])
    demo_rows = D.to_rows(ep["actions"][D.chunk_indices(T0, 10)[0]], norm)
    demo_prog = DemoProgram.load(f"{R}/r3_prog_{t}_raw_lib40.npz")
    recs = sorted(f for p in a.practice.replace("+", ",").split(",") for f in glob.glob(f"{R}/{p}_{t}_*_rec/task{t}_trial*.npz"))
    eps = [dict(np.load(f)) for f in recs]
    text = texts[t]
    # POSITIVES: exactly the success-only learner's frames (r15_consolidate.py --lora)
    pos_obs, pos_rows = [], []
    for i in range(0, T0):
        pos_obs.append(d90.observation(ep, int(i))); pos_rows.append(demo_rows[i])
    neg_obs, neg_rows, stalls = [], [], []
    for e in eps:
        acts, sts = e["action"], e["state"]
        T = len(acts)
        rows = D.to_rows(acts[D.chunk_indices(T, 10)[0]], norm)
        mk = lambda i, e=e, sts=sts: {"observation/image": e["image"][i], "observation/wrist_image": e["wrist"][i],   # noqa: E731
                                      "observation/state": sts[i], "prompt": text}
        if bool(e["success"]):
            for i in range(0, T, a.lora_stride):
                pos_obs.append(mk(int(i))); pos_rows.append(rows[i])
        else:
            steps, trace = stall_queries(demo_prog, sts, a.window)
            stalls.append(dict(trial=int(e["trial"]), stall_frame=int(trace.max()), demo_frames=len(demo_prog.mu),
                               stall_query=int(steps[-1] // 5), queries=len(trace)))
            for i in steps:
                if i < T:
                    neg_obs.append(mk(int(i))); neg_rows.append(rows[i])
    if len(pos_obs) > a.max_pos:
        keep = np.linspace(0, len(pos_obs) - 1, a.max_pos).round().astype(int)
        pos_obs, pos_rows = [pos_obs[i] for i in keep], [pos_rows[i] for i in keep]
    rec_pos = rec_neg = 0
    if a.recovery:
        brs = [b for b in json.load(open(f"{R}/r18_branch_stats.json"))["branches"]
               if b["task"] == t and "_full_" not in b["id"] and b.get("prefix_ok")]
        rp_obs, rp_rows = [], []
        for b in brs:
            f = next(iter(glob.glob(f"{R}/r18_branch_{t}_*_rec/task{t}_trial{b['trial']}_{b['id']}.npz")), None)
            if f is None:
                continue
            e = dict(np.load(f))
            acts, sts = e["action"], e["state"]
            T = len(acts)
            rows = D.to_rows(acts[D.chunk_indices(T, 10)[0]], norm)
            mk = lambda i, e=e, sts=sts: {"observation/image": e["image"][i], "observation/wrist_image": e["wrist"][i],   # noqa: E731
                                          "observation/state": sts[i], "prompt": text}
            if b["success"]:
                for i in range(5 * b["at"], T, a.lora_stride):
                    rp_obs.append(mk(int(i))); rp_rows.append(rows[i])
            else:
                steps, _ = stall_queries(demo_prog, sts, a.window)
                if steps[-1] // 5 >= b["at"]:                      # a new stall after the branch point (else: the recorded one)
                    for i in steps:
                        if i < T:
                            neg_obs.append(mk(int(i))); neg_rows.append(rows[i]); rec_neg += 1
        if len(rp_obs) > a.max_rec:
            keep = np.linspace(0, len(rp_obs) - 1, a.max_rec).round().astype(int)
            rp_obs, rp_rows = [rp_obs[i] for i in keep], [rp_rows[i] for i in keep]
        pos_obs += rp_obs; pos_rows += rp_rows; rec_pos = len(rp_obs)
        log(f"[r17] recovery data: {len(brs)} branches, {rec_pos} positive frames from rescued segments, {rec_neg} negatives")
    n_fail = sum(1 for e in eps if not bool(e["success"]))
    log(f"[r17] task {t} '{text}': {len(eps)} rollouts ({n_fail} failed); {len(pos_obs)} positive frames, {len(neg_obs)} negative "
        f"frames (W = {a.window}); stall at a median {np.median([s['stall_frame'] / s['demo_frames'] for s in stalls]) if stalls else float('nan'):.2f} "
        f"of the demo")
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    t0 = time.time()
    pos_bank = adapt.PrefixBank(bk, pos_obs, store=a.bank_device)
    neg_bank = adapt.PrefixBank(bk, neg_obs, store=a.bank_device) if neg_obs else None
    state, banks = adapt.inject(bk, scope=a.scope)
    key = f"nft_{t}"
    params = r4.add_node(banks, key, a.rank)
    hist = adapt.train_nft(bk, state, key, params, pos_bank, D.pad_rows(np.stack(pos_rows), bk.H),
                           neg_bank, D.pad_rows(np.stack(neg_rows), bk.H) if neg_obs else None, steps=a.steps, lr=1e-4,
                           batch=32, neg_batch=a.neg_batch, pooled=a.pooled, beta=1.0, seed=a.seed, log=log)
    path = f"{R}/{a.name}_{t}_lora.pt"
    nb = adapt.save(banks, key, path, meta=dict(task=t, rank=a.rank, scope=a.scope, nft=True, window=a.window, pooled=a.pooled))
    log(f"[r17] saved {path} ({nb} B) in {time.time() - t0:.0f}s")
    ledger.record("r17_nft", bench="libero_90", tag=a.name, seed=a.seed, keys=dict(task=t),
                  metrics=dict(task=t, text=text, rollouts=len(eps), failed=n_fail, positives=len(pos_obs), negatives=len(neg_obs), recovery_positives=rec_pos, recovery_negatives=rec_neg,
                               stalls=stalls, curve=hist),
                  bytes=dict(experts=nb, tree=0, codes=0, samples=0, teacher_copies=0, total=nb), config=vars(a),
                  script="experiments/r17_nft.py", command=" ".join(sys.argv), artifacts=[path])


if __name__ == "__main__":
    main()
