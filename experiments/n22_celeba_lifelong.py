"""N22 — class-incremental CelebA: N5's protocol on spatially overlapping attributes.

N5 is the retention result the paper rests on: streaming REAL held-out images one batch at a time,
the arms that structurally REMOVE shared parameters sit at exactly 0.000 forgetting while every arm
that tries to PROTECT shared parameters forgets (EWC 0.058, finetune 0.082, replay 0.163).  It was
measured only on ColorMNIST.

The worry that makes re-running it worthwhile is specific, not generic.  Preservation here is
claimed to be *structural* -- a frozen ancestor path plus a private conditioning vector cannot be
written by a later concept -- and a structural argument should be indifferent to whether the factors
overlap in pixel space.  If forgetting appears on CelebA, the argument was not structural after all
and something else was carrying it.  So this is a test the method can genuinely fail.

Same six arms as N5, same accuracy matrix, same CL metrics.  What changes is the data (held-out
attribute combinations, real images) and the score: `target` -- the single attribute separating each
held-out combination from its nearest trained one -- alongside the mean-attribute accuracy, for the
reason given in n20.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                       # noqa: E402
from n4_batch import cl_metrics                                        # noqa: E402
from n19_celeba_train import build_probes                              # noqa: E402
from n20_celeba_composition import score                               # noqa: E402

from lifelong_ood.celeba import ATTRS, as_float, load_celeba_comp, nearest_seen  # noqa: E402
from lifelong_ood.evals import save_json                               # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, fingerprint, load_field,  # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.baselines import (CLoRA, DERPlusPlus, EWC, MAS, SI,  # noqa: E402
                                           test_time_train)

# see n5_classinc.py for what each arm holds fixed; the arm set is identical so the ColorMNIST and
# CelebA retention tables can be read against each other directly
ARMS = {
    "ours":        (False, True,  False, False, False),
    "independent": (True,  True,  False, False, False),
    "finetune":    (True,  False, False, False, False),
    "replay":      (True,  False, True,  False, False),
    "gen-distill": (True,  False, True,  False, False),
    "der++":       (True,  False, False, False, False),
    "ewc":         (True,  False, False, True,  False),
    "si":          (True,  False, False, False, False),
    "mas":         (True,  False, False, False, False),
    "ttt":         (True,  False, False, False, True),
    "c-lora":      (True,  False, False, False, False),
}
REGULARIZERS = {"ewc": EWC, "si": SI, "mas": MAS, "c-lora": CLoRA}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba_factored.pt"))
    p.add_argument("--n-tasks", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--n-train", type=int, default=64, help="REAL images per task")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--replay-n", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--tag", default="", help="suffix for the output file; --quick runs use one "
                                             "so a smoke can never overwrite a full result")
    p.add_argument("--ewc-lambda", type=float, default=5000.0)
    p.add_argument("--clora-lambda", type=float, default=1e4,
                   help="C-LoRA self-regularisation strength (Smith et al., TMLR 2024)")
    p.add_argument("--si-lambda", type=float, default=1.0)
    p.add_argument("--mas-lambda", type=float, default=1.0)
    p.add_argument("--buffer-size", type=int, default=200)
    p.add_argument("--der-alpha", type=float, default=0.5)
    p.add_argument("--der-beta", type=float, default=0.5)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_tasks, args.batch_size, args.n_train = 4, 2, 24
        args.n_samples, args.epochs, args.replay_n, args.steps = 8, 4, 8, 10
        args.arms = ["ours", "finetune"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img = ck["img"]
    import lifelong_ood.neural.ddpm as _d
    _d.IMG, _d.DIM = img, 3 * img * img
    # every split argument comes FROM the checkpoint, and verify_hold_ids then fails loudly if the
    # recomputed split differs -- an evaluation scoring against combinations the backbone actually
    # trained on would otherwise look like a very good result
    data = load_celeba_comp(img=img, min_count=ck.get("min_count", 60),
                            n_holdout=ck.get("n_holdout", len(ck["hold_ids"])),
                            holdout_pair=ck.get("holdout_pair"),
                            holdout_attr=ck.get("holdout_attr"),
                            seed=ck.get("seed", 0), cache_dir=CACHE,
                            verify_hold_ids=ck["hold_ids"])
    sp = data["split"]
    probes = build_probes(data, args.seed, 8 if args.quick else 30)

    rng = np.random.default_rng(args.seed)
    picks = rng.permutation(sp["hold_ids"])[:args.n_tasks]
    tasks = []
    for j in picks:
        idx = np.where(sp["combo_of"] == j)[0]
        R = as_float(data["X8"][idx])
        _, diff = nearest_seen(sp["all_combos"][j], sp)
        tasks.append(dict(name=f"task_{int(j)}", combo=sp["all_combos"][j], ref=R,
                          diff=diff, X=to_model(R[:args.n_train])))
    print(f"[n22] {len(tasks)} held-out CelebA combinations x {args.n_train} REAL images, "
          f"batches of {args.batch_size}, arms {args.arms}, device {dev}", flush=True)

    rows = []
    for arm in args.arms:
        flat, freeze, use_replay, use_ewc, is_ttt = ARMS[arm]
        field = load_field(args.ckpt, device=dev, batch=64)
        L = NeuralLifelongLearner(field, rank=args.rank, steps=args.steps, seed=args.seed,
                                  device=dev, flat=flat, freeze_ancestors=freeze)
        # every regularizer shares EWC's (begin_task, penalty, consolidate) interface
        ewc = None
        if arm in REGULARIZERS:
            lam = {"ewc": args.ewc_lambda, "c-lora": args.clora_lambda,
                   "si": args.si_lambda, "mas": args.mas_lambda}[arm]
            ewc = REGULARIZERS[arm](lam=lam)
        buffer = (DERPlusPlus(size=args.buffer_size, alpha=args.der_alpha, beta=args.der_beta,
                              seed=args.seed) if arm == "der++" else None)
        acc_rows = {t["name"]: [] for t in tasks}
        tgt_rows = {t["name"]: [] for t in tasks}
        marks, learned = {}, []
        t0 = time.time()
        for i0 in range(0, len(tasks), args.batch_size):
            grp = tasks[i0:i0 + args.batch_size]
            replay = None
            if use_replay and learned:
                replay = [(nm, L.sample_concept(nm, n=args.replay_n, seed=args.seed + 900))
                          for nm, _ in learned]
            L.consolidate_batch([(t["name"], t["X"], None) for t in grp],
                                epochs=(0 if is_ttt else args.epochs),
                                replay=replay, ewc=ewc, buffer=buffer,
                                replay_distill=(arm == "gen-distill"))
            for t in grp:
                learned.append((t["name"], t))
                marks[t["name"]] = fingerprint(field, t["name"], seed=args.seed)
            for si, (nm, t) in enumerate(learned):
                if is_ttt:
                    restore = test_time_train(L, t["X"], nm, epochs=args.epochs, seed=args.seed)
                    S = to_data(L.sample_concept(nm, n=args.n_samples,
                                                 seed=args.seed + 500 + si)).astype(np.float32)
                    restore()
                else:
                    S = to_data(L.sample_concept(nm, n=args.n_samples,
                                                 seed=args.seed + 500 + si)).astype(np.float32)
                sc = score(probes, S, t["combo"], t["ref"], t["diff"])
                acc_rows[nm].append(sc["mean_attr"])
                tgt_rows[nm].append(sc["target"])
            exact = float(np.mean([float(np.array_equal(fingerprint(field, nm, seed=args.seed),
                                                        marks[nm])) for nm, _ in learned]))
            print(f"[n22:{arm}] {len(learned)}/{len(tasks)}: "
                  f"mean-attr {np.nanmean([acc_rows[nm][-1] for nm, _ in learned]):.3f}, "
                  f"target {np.nanmean([tgt_rows[nm][-1] for nm, _ in learned]):.3f}, "
                  f"exact {exact:.2f}, {L.bank.n_params() / 1e6:.2f}M, "
                  f"{time.time() - t0:.0f}s", flush=True)
        acc = [acc_rows[t["name"]] for t in tasks if acc_rows[t["name"]]]
        tgt = [tgt_rows[t["name"]] for t in tasks if tgt_rows[t["name"]]]
        m = cl_metrics(acc)
        mt = cl_metrics(tgt)
        stored = (len(buffer) if buffer is not None else
                  (sum(len(t["X"]) for t in tasks) if is_ttt else 0))
        rows.append(dict(stored_images=stored,
                         arm=arm, wall_seconds=time.time() - t0, params=L.bank.n_params(),
                         retention_exact=exact, acc_matrix=acc, target_matrix=tgt,
                         ACC_target=mt["ACC"], BWT_target=mt["BWT"], FGT_target=mt["FGT"], **m))
        print(f"[n22] -> {arm}: ACC {m['ACC']:.3f} BWT {m['BWT']:+.3f} FGT {m['FGT']:.3f} | "
              f"target ACC {mt['ACC']:.3f} FGT {mt['FGT']:.3f}\n", flush=True)

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n22_celeba_lifelong{args.tag}.json"), seed=args.seed)
    print(f"\n[n22] class-incremental CelebA, {len(tasks)} held-out combinations of REAL images")
    print(f"    {'arm':13s}{'ACC':>7s}{'BWT':>9s}{'FGT':>8s}{'tgtACC':>8s}{'tgtFGT':>8s}"
          f"{'exact':>8s}{'params':>9s}")
    for r in rows:
        print(f"    {r['arm']:13s}{r['ACC']:7.3f}{r['BWT']:+9.3f}{r['FGT']:8.3f}"
              f"{r['ACC_target']:8.3f}{r['FGT_target']:8.3f}{r['retention_exact']:8.2f}"
              f"{r['params'] / 1e6:8.2f}M")
    print("\n  ACC/FGT on mean-attribute accuracy; tgtACC/tgtFGT on the ONE attribute that "
          "separates\n  each held-out combination from its nearest trained one.")
    print(f"[n22] wrote {RESULTS}/n22_celeba_lifelong.json")


if __name__ == "__main__":
    main()
