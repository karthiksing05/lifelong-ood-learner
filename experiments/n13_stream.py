"""N13 — the full loop on a stream of genuinely new vocabulary.

N11 established the two regimes separately: a new COMBINATION of known factor values is free via
basis extension and beats an adapter (F1 0.755 vs 0.658), while a new VALUE cannot be conditioned
into existence at all (background accuracy 0.000) and needs weights driven by real data (0.833).

This runs them together, as a stream, on a backbone that has never seen FOUR factor values --
background 3, foreground 3, digits 8 and 9 (56 seen classes, conditional accuracy 0.953).  Each new
value arrives, is absorbed, and then has to survive everything that arrives after it.

Per value the loop is:
  1. absorb the value into WEIGHTS from real images of a few combinations  (the vocabulary step)
  2. fit a basis direction for it now that the model can render it         (the composition step)
  3. score combinations of it that were never trained, via the basis alone (free, no parameters)

Arms:
  `ours`            steps 1-3 above: weights per VALUE, basis for every combination of it
  `adapter-per-combo` an adapter for every combination, which is what the method did before N11
  `basis-only`      step 3 without step 1, the control N11 phase B predicts must fail

Retention is measured over every value absorbed so far, so the question "can this go on forever"
gets a curve rather than an assertion.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                          # noqa: E402

from lifelong_ood.data.datasets import load_colormnist                          # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, precision_recall,  # noqa: E402
                                save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field, to_data, to_model)  # noqa: E402
from lifelong_ood.neural.idealwords import (compose_with, extend_basis, factorize,  # noqa: E402
                                            sample_factored_pool, slot_factors)

# factor index -> (name, held-out values arriving as a stream)
STREAM = [(2, 3), (1, 3), (0, 8), (0, 9)]           # bg3, fg3, digit8, digit9
FNAME = {0: "digit", 1: "fg", 2: "bg"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist_stream.pt"))
    p.add_argument("--n-teach", type=int, default=4, help="combinations used to absorb a value")
    p.add_argument("--n-eval", type=int, default=8, help="never-trained combinations scored")
    p.add_argument("--n-per-combo", type=int, default=48)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--ext-steps", type=int, default=600)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rehearse-w", type=float, default=5.0,
                   help="off-concept no-op weight while absorbing vocabulary (tuned in 5r)")
    p.add_argument("--arms", nargs="+", default=["ours", "adapter-per-combo", "basis-only"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_teach, args.n_eval, args.n_samples = 2, 3, 8
        args.epochs, args.ext_steps, args.steps, args.n_per_slot = 4, 60, 10, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    remap = ck["remap"]
    slots_seen = np.array(sorted(remap))
    C = ck["ema"]["cemb.weight"].numpy()[[remap[int(x)] for x in slots_seen]]
    labels = slot_factors(slots_seen)
    seen_idx = np.where(data["seen"])[0]
    REAL = to_model(data["X"][seen_idx[:256]])          # real images of compositions already known
    print(f"[n13] backbone {os.path.basename(args.ckpt)}: {len(slots_seen)} seen classes; "
          f"stream {[(FNAME[f], v) for f, v in STREAM]}; device {dev}", flush=True)

    def combos_for(hf, hv, rng):
        """(other-factor assignments) for a held-out value, spanning the other factors."""
        others = [f for f in (0, 1, 2) if f != hf]
        sizes = {0: 10, 1: 4, 2: 4}
        pool = [(a, b) for a in range(sizes[others[0]]) for b in range(sizes[others[1]])]
        # only combinations whose other factors are themselves SEEN by this backbone
        ok = []
        for (a, b) in pool:
            asg = {hf: hv, others[0]: a, others[1]: b}
            if all(asg[f] in set(labels[:, f]) for f in others):
                ok.append((a, b))
        rng.shuffle(ok)
        span, used = [], set()
        for (a, b) in ok:
            if len(span) >= args.n_teach:
                break
            if b not in used or len(used) >= 4:
                span.append((a, b)); used.add(b)
        rest = [c for c in ok if c not in span][:args.n_eval]
        return span, rest, others

    def images(hf, hv, others, a, b, n):
        m = ((data["digit" if hf == 0 else ("fg" if hf == 1 else "bg")] == hv)
             & (data["digit" if others[0] == 0 else ("fg" if others[0] == 1 else "bg")] == a)
             & (data["digit" if others[1] == 0 else ("fg" if others[1] == 1 else "bg")] == b))
        idx = np.where(m)[0]
        return data["X"][idx[:n]]

    def triple(hf, hv, others, a, b):
        t = {hf: hv, others[0]: a, others[1]: b}
        return t[0], t[1], t[2]

    rows = []
    for arm in args.arms:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
        fac = factorize(C, labels)
        vdirs = {}                                   # (factor, value) -> learned direction
        absorbed = []
        t0 = time.time()
        print(f"\n[n13:{arm}] basis R^2 {fac['r2']:.3f}", flush=True)
        for si, (hf, hv) in enumerate(STREAM):
            rng = np.random.default_rng(args.seed + si)
            span, rest, others = combos_for(hf, hv, rng)
            if not rest:
                continue
            # 1. absorb the VALUE into weights, from real images of a few combinations
            #
            # For `ours` the delta must be MERGED into the base.  A first version left it as a
            # per-concept adapter and the arm scored identically to `basis-only` -- correctly, since
            # an adapter is only active while its own path is, and composing through the basis
            # activates no path at all.  New vocabulary has to land in the always-on weights or the
            # composition step cannot see it.  Merging is exact for the concept just learned
            # (test_merge.py) and rehearsal against the frozen base is what keeps repeated folds
            # survivable (RESULTS 5r), so both are used here.
            if arm in ("ours", "adapter-per-combo"):
                for (a, b) in span:
                    X = to_model(images(hf, hv, others, a, b, args.n_per_combo))
                    X = X.reshape(len(X), -1)
                    if len(X) == 0:
                        continue
                    nm = f"v{si}_{a}_{b}"
                    reh = None
                    if arm == "ours":
                        reh = dict(x0=REAL, conds=[C[i] for i in range(0, len(C), 4)][:12],
                                   w=args.rehearse_w)
                    L.consolidate(X, nm, mu=X.mean(0), epochs=args.epochs, rehearse=reh)
                    if arm == "ours":
                        L.bank.merge_concept(nm)      # vocabulary goes into the always-on weights
            # 2. fit a direction for it, now that the model can render it
            if arm in ("ours", "basis-only"):
                tgt, lab = [], []
                for (a, b) in span:
                    Xc = images(hf, hv, others, a, b, args.n_per_combo)
                    if len(Xc) == 0:
                        continue
                    tgt.append(to_model(Xc))
                    d, f_, bg = triple(hf, hv, others, a, b)
                    lab.append(np.tile([d, f_, bg], (len(Xc), 1)))
                tgt = np.concatenate(tgt).reshape(-1, field.dim)
                lab = np.concatenate(lab)
                v, _ = extend_basis(field, tgt, lab, fac, hf, n_steps=args.ext_steps,
                                    seed=args.seed, norm_match=True)
                vdirs[(hf, hv)] = v
            absorbed.append((hf, hv, others, rest))

            # 3. score every value absorbed so far on combinations never trained
            for (ahf, ahv, aoth, arest) in absorbed:
                accs, f1s = [], []
                for (a, b) in arest:
                    d, f_, bg = triple(ahf, ahv, aoth, a, b)
                    R = images(ahf, ahv, aoth, a, b, 100)
                    if len(R) < 4:
                        continue
                    if arm == "adapter-per-combo":
                        nm = f"c{ahf}_{ahv}_{a}_{b}"
                        if nm not in L.bank.concepts:
                            Xc = to_model(R[:args.n_per_combo]).reshape(-1, field.dim)
                            L.consolidate(Xc, nm, mu=Xc.mean(0), epochs=args.epochs)
                        S = to_data(L.sample_concept(nm, n=args.n_samples,
                                                     seed=args.seed)).astype(np.float32)
                    else:
                        asg = [0, 0, 0]
                        asg[ahf], asg[aoth[0]], asg[aoth[1]] = ahv, a, b
                        c = compose_with(fac, asg, ahf, vdirs[(ahf, ahv)])
                        S = to_data(sample_factored_pool(
                            field, torch.as_tensor(c, dtype=torch.float32, device=dev),
                            n=args.n_samples, steps=args.steps, seed=args.seed)).astype(np.float32)
                    accs.append(composition_accuracy(probes, S, d, f_, bg)["joint"])
                    f1s.append(precision_recall(S, R)["f1"])
                rows.append(dict(arm=arm, step=si + 1, value=f"{FNAME[ahf]}={ahv}",
                                 acc=float(np.mean(accs)) if accs else float("nan"),
                                 f1=float(np.mean(f1s)) if f1s else float("nan"),
                                 n_combos=len(accs), params=L.bank.n_params()))
                print(f"[n13:{arm}] after {si + 1} value(s): {FNAME[ahf]}={ahv} on "
                      f"{len(accs)} never-trained combos -> acc "
                      f"{np.mean(accs) if accs else float('nan'):.2f} "
                      f"F1 {np.mean(f1s) if f1s else float('nan'):.2f}, "
                      f"{L.bank.n_params() / 1e6:.2f}M, {time.time() - t0:.0f}s", flush=True)

    save_json(dict(rows=rows, config=vars(args)), os.path.join(RESULTS, "n13_stream.json"),
              seed=args.seed)
    print(f"\n[n13] final state after the whole stream")
    print(f"    {'arm':20s}{'value':12s}{'acc':>7s}{'F1':>7s}{'params':>10s}")
    last = max(r["step"] for r in rows) if rows else 0
    for arm in args.arms:
        for r in [x for x in rows if x["arm"] == arm and x["step"] == last]:
            print(f"    {arm:20s}{r['value']:12s}{r['acc']:7.2f}{r['f1']:7.2f}"
                  f"{r['params'] / 1e6:9.2f}M")
    print(f"[n13] wrote {RESULTS}/n13_stream.json")


if __name__ == "__main__":
    main()
