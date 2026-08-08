"""N4 — batch addition, benchmarked as continual learning.

Two questions, one run.

**Is batch addition worth it?**  The same K OOD concepts are added to the manifold two ways:
  `sequential`  K calls to learn_concept -- discover, sample and distil one concept at a time
  `batch`       ceil(K/B) calls to learn_concepts_batch, B concepts at a time
Both must end with the same K concepts in the manifold and the same preservation guarantee; the
question is what batching costs and what it buys.  Batching should win on wall clock for three
separate reasons (query-agnostic modes discovered once instead of K times, one reverse-diffusion run
instead of K, one optimizer instead of K) and should additionally produce genuinely SHARED group
nodes, which a sequential run cannot: a sequential concept can only inherit a frozen warm start,
while a batch trains the shared node on all of its members at once.

**How does it score as a continual learner?**  The full accuracy matrix `acc[i][t]` -- concept i
evaluated after step t -- is recorded, giving the standard continual-learning metrics:

  ACC   mean final accuracy over all concepts                       (higher better)
  BWT   mean_i acc[i][T] - acc[i][i]   -- backward transfer         (0 = no forgetting, <0 = forgetting)
  FGT   mean_i max_t acc[i][t] - acc[i][T]  -- forgetting measure   (0 = none)

Reported against arms that are the standard baselines of that literature under different names:
  `unfrozen` = sequential fine-tuning of shared parameters (the catastrophic-forgetting control)
  `flat`     = one independent adapter per task, stored (parameter-isolation, no sharing;
               this is also exactly the poe-diffusion paper's per-query LoRA, merely kept)
  `hier`     = ours
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import composition_accuracy, fit_probes, save_grid, save_json  # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, fingerprint, load_field,  # noqa: E402
                                 to_data, to_model)


def cl_metrics(acc):
    """acc[i][t] = accuracy of concept i after step t (t >= i).  -> ACC / BWT / FGT.

    ACC and BWT follow the GEM convention (Lopez-Paz & Ranzato, NeurIPS 2017); FGT is Chaudhry
    et al.'s forgetting measure (RWalk, ECCV 2018), i.e. the drop from each concept's best
    historical value rather than from its value at birth.
    """
    K = len(acc)
    final = [acc[i][-1] for i in range(K)]
    bwt = [acc[i][-1] - acc[i][0] for i in range(K - 1)]          # [0] is "right after learning i"
    fgt = [np.nanmax(acc[i]) - acc[i][-1] for i in range(K - 1)]
    # nan-aware throughout.  On CelebA the `target` matrix legitimately contains nan for concepts
    # whose distinguishing attribute has no usable probe, and a plain mean lets ONE such entry blank
    # the entire column -- which it did: every tgtACC/tgtFGT in the first full run came out nan while
    # the per-step prints showed real values (0.250, 0.516, 0.550).
    with np.errstate(invalid="ignore"):
        return dict(ACC=float(np.nanmean(final)) if len(final) else float("nan"),
                    BWT=float(np.nanmean(bwt)) if bwt else 0.0,
                    FGT=float(np.nanmean(fgt)) if fgt else 0.0)


def efficiency_metrics(acc, params, stored, n_train_total, params0=None):
    """Diaz-Rodriguez et al. (NeurIPS-W 2018) normalized CL metrics, all in [0,1], higher better.

        A   = sum_{i>=j} R_ij / (N(N+1)/2)     the WHOLE lower triangle, not just the final row
        REM = 1 - |min(BWT, 0)|                remembering
        MS  = min(1, mean_i Mem(theta_1)/Mem(theta_i))     model-size efficiency
        SSS = 1 - min(1, mean_i Mem(M_i)/Mem(D))           sample-storage efficiency

    These are here because the accuracy columns systematically flatter methods that pay for their
    retention somewhere the table does not look.  `A` rewards holding a concept for the WHOLE stream
    rather than recovering by the end -- final-row ACC cannot tell "never dropped" from "dropped and
    came back".  `SSS` is the axis on which every rehearsal method loses to an insert-only bank and
    on which ours is exactly 1.0, because it keeps no data at all.  `MS` is the axis on which OUR
    method pays: a bank grows, so this is the column a reviewer should be pointed at rather than
    away from.

    `stored` = real images retained forever; `n_train_total` = the size of the stream, so SSS is
    measured against what a full-replay method would have to keep.

    arXiv:1810.13166.
    """
    K = len(acc)
    if not K:
        return {}
    # lower triangle: acc[i][t] exists only for steps at or after concept i was learned
    tri = [v for row in acc for v in row]
    A = float(np.nanmean(tri)) if tri else float("nan")
    bwt = np.nanmean([acc[i][-1] - acc[i][0] for i in range(K - 1)]) if K > 1 else 0.0
    rem = float(1.0 - abs(min(float(bwt), 0.0)))
    p0 = float(params0 if params0 else (params[0] if np.ndim(params) else params))
    pk = np.atleast_1d(np.asarray(params, float))
    ms = float(min(1.0, float(np.mean(p0 / np.maximum(pk, 1e-9)))))
    sss = float(1.0 - min(1.0, float(stored) / max(float(n_train_total), 1.0)))
    return dict(A=A, REM=rem, MS=ms, SSS=sss)


def quality_metrics(q, lower_better=True):
    """CLoG's generative-CL protocol applied to a per-concept quality matrix.

    `q[i][t]` = generation quality of concept i measured after step t.  Returns

        AFQ = mean_i q[i][-1]                     Average Final Quality
        AIQ = mean_t mean_{i seen by t} q[i][t]   Average Incremental Quality
        FR  = mean_i (q[i][-1] - q[i][0])         Forgetting Rate, signed so >0 is degradation

    This exists because joint probe accuracy is FIDELITY-BLIND: a method can keep the right
    attributes legible while its images degrade badly, and the accuracy column will not show it.
    That is not hypothetical -- Masip et al. (CoLLAs 2025) measure generative replay on a DDPM at
    HIGHER classifier accuracy than plain finetuning (0.50 vs 0.17) while its FID is three times
    worse (211.8 vs 68.8).  Reporting only accuracy would report that arm as the better one.

    `lower_better` matches a distance (Frechet); the FR sign is flipped for a similarity so that
    positive always means "got worse".

    Zhang et al., "CLoG: Benchmarking Continual Learning of Image Generation Models," 2024,
    arXiv:2406.04584.
    """
    K = len(q)
    if not K:
        return dict(AFQ=float("nan"), AIQ=float("nan"), FR=float("nan"))
    s = 1.0 if lower_better else -1.0
    afq = float(np.mean([q[i][-1] for i in range(K)]))
    # column t averages every concept already learned at that point; rows have different lengths
    T = max(len(r) for r in q)
    cols = []
    for t in range(T):
        vals = [q[i][t] for i in range(K) if t < len(q[i])]
        if vals:
            cols.append(float(np.mean(vals)))
    fr = [s * (q[i][-1] - q[i][0]) for i in range(K - 1)]
    return dict(AFQ=afq, AIQ=float(np.mean(cols)) if cols else float("nan"),
                FR=float(np.mean(fr)) if fr else 0.0)


def evaluate_all(L, field, probes, learned, n_samples, seed):
    """Accuracy of EVERY concept learned so far, each with its own pinned seed."""
    out = {}
    for name, q, s in learned:
        S = to_data(L.sample_concept(name, n=n_samples, seed=seed + 500 + s)).astype(np.float32)
        out[name] = composition_accuracy(probes, S, q["d"], q["f"], q["b"])["joint"]
    return out


def run_arm(args, data, probes, qs, mode, arm, dev):
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, rank=args.rank, steps=args.steps, seed=args.seed, device=dev,
                              flat=(arm == "flat"), freeze_ancestors=(arm != "unfrozen"))
    names = [f"ood_{q['slot']}" for q in qs]
    xqs = [to_model(data["X"][q["query"]]).ravel() for q in qs]
    disc = dict(u_grid=(0.1, 0.2, 0.3, 0.4), n_starts=args.ascent_starts, iters=args.ascent_iters)

    learned, marks, acc_rows, exact_rows = [], {}, {n: [] for n in names}, []
    t0 = time.time()
    steps = list(range(0, len(qs), args.batch_size)) if mode == "batch" else list(range(len(qs)))
    for si, i0 in enumerate(steps):
        sl = slice(i0, i0 + args.batch_size) if mode == "batch" else slice(i0, i0 + 1)
        if mode == "batch":
            teachers, pools = L.learn_concepts_batch(xqs[sl], names[sl], n_pool=args.n_pool,
                                                     epochs=args.epochs, disc=disc)
        else:
            teachers, pools = [], []
            for xq, nm in zip(xqs[sl], names[sl]):
                t, p = L.learn_concept(xq, nm, n_pool=args.n_pool, epochs=args.epochs, disc=None)
                teachers.append(t); pools.append(p)
        for nm, q in zip(names[sl], qs[sl]):
            learned.append((nm, q, len(learned)))
            marks[nm] = fingerprint(field, nm, seed=args.seed)
        # every concept so far, so the full acc[i][t] matrix is available
        cur = evaluate_all(L, field, probes, learned, args.n_samples, args.seed)
        for nm, v in cur.items():
            acc_rows[nm].append(v)
        exact_rows.append(float(np.mean([
            float(np.array_equal(fingerprint(field, nm, seed=args.seed), marks[nm]))
            for nm, _, _ in learned])))
        print(f"[n4:{arm}:{mode}] step {si + 1}/{len(steps)} "
              f"({len(learned)} concepts): mean acc {np.mean(list(cur.values())):.2f}, "
              f"exact retention {exact_rows[-1]:.2f}, {L.bank.n_params() / 1e6:.2f}M params, "
              f"{time.time() - t0:.0f}s", flush=True)
        if si == 0 and mode == "batch" and arm == "hier":
            save_grid(np.concatenate([to_data(pools[0])[:8], to_data(pools[-1])[:8]]),
                      os.path.join(RESULTS, "n4_batch_pools.png"),
                      title="two concepts from one batched sampling run")
    wall = time.time() - t0
    acc = [acc_rows[n] for n in names if acc_rows[n]]
    m = cl_metrics(acc)
    s = L.bank.summary()
    return dict(arm=arm, mode=mode, wall_seconds=wall, params=s["adapter_params"],
                n_nodes=s["n_nodes"], shared=len(s["shared_ancestors"]),
                retention_exact=exact_rows[-1], acc_matrix=acc, **m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-pool", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ascent-starts", type=int, default=16)
    p.add_argument("--ascent-iters", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["hier"])
    p.add_argument("--modes", nargs="+", default=["batch", "sequential"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_concepts, args.batch_size, args.n_pool = 4, 2, 16
        args.n_samples, args.n_per_slot, args.epochs = 8, 20, 4
        args.ascent_starts, args.ascent_iters, args.steps = 4, 20, 10
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)
    print(f"[n4] {len(qs)} concepts, batch size {args.batch_size}, arms {args.arms}, "
          f"modes {args.modes}, device {dev}", flush=True)

    rows = []
    for arm in args.arms:
        for mode in args.modes:
            rows.append(run_arm(args, data, probes, qs, mode, arm, dev))
            print(f"[n4] -> {arm}/{mode}: {rows[-1]['wall_seconds']:.0f}s, "
                  f"ACC {rows[-1]['ACC']:.3f}, BWT {rows[-1]['BWT']:+.3f}, "
                  f"FGT {rows[-1]['FGT']:.3f}\n", flush=True)
    save_json(dict(rows=rows, config=vars(args)), os.path.join(RESULTS, "n4_batch.json"),
              seed=args.seed)

    print(f"\n[n4] {args.n_concepts} OOD concepts, batch size {args.batch_size}")
    print(f"    {'arm':8s}{'mode':>11s}{'wall(s)':>9s}{'speedup':>9s}{'ACC':>7s}"
          f"{'BWT':>8s}{'FGT':>7s}{'exact':>7s}{'params':>9s}{'shared':>8s}")
    base = {r["arm"]: r["wall_seconds"] for r in rows if r["mode"] == "sequential"}
    for r in rows:
        sp = base.get(r["arm"], r["wall_seconds"]) / max(r["wall_seconds"], 1e-9)
        print(f"    {r['arm']:8s}{r['mode']:>11s}{r['wall_seconds']:9.0f}{sp:9.2f}x"
              f"{r['ACC']:7.3f}{r['BWT']:+8.3f}{r['FGT']:7.3f}{r['retention_exact']:7.2f}"
              f"{r['params'] / 1e6:8.2f}M{r['shared']:8d}")
    print("\n  ACC final mean accuracy | BWT backward transfer (0 = no forgetting) | "
          "FGT forgetting measure\n  exact = fraction of concepts still bit-identical to birth")
    print(f"[n4] wrote {RESULTS}/n4_batch.json")


if __name__ == "__main__":
    main()
