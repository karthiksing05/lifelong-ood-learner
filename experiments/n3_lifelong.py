"""N3 — the headline: continually adding OOD compositions to a NEURAL manifold, without decay.

A stream of held-out compositions arrives one at a time.  Each is composed from discovered
prototypes, sampled through the manifold, and consolidated as a new leaf in the hierarchical adapter
bank.  After every consolidation we re-measure everything that could have been damaged.

WHAT IS MEASURED, and why each one is the right test:

  base_drift        epsilon-prediction error of the FROZEN base on held-out SEEN data. The base is
                    never optimized, so this must be exactly flat; it is the control that proves the
                    harness itself is not leaking updates.
  retention_exact   the fraction of previously consolidated concepts whose FINGERPRINT -- the raw
                    epsilon output on fixed noise at fixed noise levels -- is BIT-IDENTICAL to when
                    they were first consolidated.  This is the strong form of "no forgetting": not
                    "accuracy stayed similar" but "the function did not move at all". With frozen
                    ancestors it should be 1.00 at every step.
  retention_acc     composition accuracy of every earlier concept, re-sampled under the current
                    model. The behavioural version of the same claim.
  new_acc           composition accuracy of the concept just learned (plasticity).
  transfer          the distillation loss reached by concept k after a FIXED budget of epochs. If
                    the hierarchy shares anything useful, later concepts that land near existing
                    relatives should start lower and end lower than early isolated ones.
  params            adapter parameters added so far (cost of the mechanism).

ARMS.  `hier` is the method. Two ablations isolate its two ingredients:
  `flat`     every concept is a child of the root, so nothing is shared -- this is the paper's
             per-query adapter, merely stored rather than discarded. Isolates the value of sharing.
  `unfrozen` the hierarchy is kept but ancestors stay trainable, so consolidating concept k
             rewrites deltas that earlier concepts depend on. Isolates the value of freezing, and is
             expected to be the arm where retention_exact collapses.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                         # noqa: E402

from lifelong_ood.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes, sample_geometry,  # noqa: E402
                                save_grid, save_json)
from lifelong_ood.neural import (NeuralLifelongLearner, fingerprint, load_field,  # noqa: E402
                                 to_data, to_model)


def base_error(field, Xm, u_grid=(0.2, 0.5), seed=0):
    """Epsilon error of the frozen base (empty adapter path) on held-out seen data."""
    rng = np.random.default_rng(seed)
    from lifelong_ood.score import cosine_abar
    tot = 0.0
    for u in u_grid:
        a = float(cosine_abar(u))
        e = rng.standard_normal(Xm.shape)
        xt = np.sqrt(a) * Xm + np.sqrt(1 - a) * e
        tot += float(np.mean((field.eps(xt, u, None) - e) ** 2))
    return tot / len(u_grid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=12)
    p.add_argument("--n-pool", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["hier", "flat", "unfrozen"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_concepts, args.n_pool, args.n_samples = 3, 16, 8
        args.n_per_slot, args.epochs, args.n_eval = 20, 3, 32
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    seen_idx = np.where(data["seen"])[0]
    Xev = to_model(data["X"][rng.choice(seen_idx, args.n_eval, replace=False)])
    print(f"[n3] {len(qs)} OOD concepts, arms {args.arms}, device {dev}", flush=True)

    results = {}
    for arm in args.arms:
        field = load_field(args.ckpt, device=dev)
        L = NeuralLifelongLearner(field, rank=args.rank, steps=args.steps, seed=args.seed,
                                  device=dev, flat=(arm == "flat"),
                                  freeze_ancestors=(arm != "unfrozen"))
        e0 = base_error(field, Xev, seed=args.seed)
        steps = [dict(step=0, base_err=e0, base_drift=0.0, params=0,
                      retention_exact=1.0, retention_acc=None)]
        print(f"\n[n3:{arm}] base epsilon error {e0:.4f}", flush=True)
        marks, learned = {}, []
        for k, q in enumerate(qs, 1):
            name = f"ood_{q['slot']}"
            xq01 = data["X"][q["query"]]
            teacher, pool = L.learn_concept(to_model(xq01).ravel(), name, n_pool=args.n_pool,
                                            epochs=args.epochs, seed=args.seed + k)
            marks[name] = fingerprint(field, name, seed=args.seed)
            learned.append((name, q))

            S = to_data(L.sample_concept(name, n=args.n_samples, seed=args.seed + k)).astype(np.float32)
            rec = dict(step=k, slot=q["slot"], name=slot_name(q["d"], q["f"], q["b"]),
                       base_err=base_error(field, Xev, seed=args.seed),
                       params=L.bank.n_params(),
                       new_acc=composition_accuracy(probes, S, q["d"], q["f"], q["b"]),
                       geometry=sample_geometry(S, xq01, data["X"][q["rest"][:100]]))
            rec["base_drift"] = rec["base_err"] - e0
            # the strong no-forgetting test: are earlier concepts' functions untouched?
            exact, accs = [], []
            for pname, pq in learned[:-1]:
                exact.append(float(np.array_equal(fingerprint(field, pname, seed=args.seed),
                                                  marks[pname])))
                Sp = to_data(L.sample_concept(pname, n=args.n_samples,
                                              seed=args.seed + 500 + k)).astype(np.float32)
                accs.append(composition_accuracy(probes, Sp, pq["d"], pq["f"], pq["b"])["joint"])
            rec["retention_exact"] = float(np.mean(exact)) if exact else 1.0
            rec["retention_acc"] = float(np.mean(accs)) if accs else None
            rec["depth"] = len(L.bank.tree.path_of(L.bank.concepts[name]["leaf"])) - 1
            steps.append(rec)
            if k <= 2 and arm == "hier":
                save_grid(np.concatenate([[xq01], to_data(pool)[:15]]),
                          os.path.join(RESULTS, f"n3_pool_{name}.png"),
                          title=f"consolidated pool — {rec['name']}")
                save_grid(S[:16], os.path.join(RESULTS, f"n3_cond_{name}.png"),
                          title=f"conditional samples from adapter {name}")
            ra = "n/a" if rec["retention_acc"] is None else f"{rec['retention_acc']:.2f}"
            print(f"[n3:{arm}] step {k} (+{rec['name']}): new {rec['new_acc']['joint']:.2f}, "
                  f"retention exact {rec['retention_exact']:.2f} acc {ra}, "
                  f"base drift {rec['base_drift']:+.5f}, depth {rec['depth']}, "
                  f"params {rec['params'] / 1e6:.2f}M", flush=True)
        results[arm] = dict(steps=steps, bank=L.bank.summary())

    save_json(results, os.path.join(RESULTS, "n3_lifelong.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for arm, r in results.items():
        st = r["steps"]
        ks = [s["step"] for s in st]
        axes[0].plot(ks, [s["base_drift"] for s in st], marker="o", label=arm)
        axes[1].plot(ks, [s["retention_exact"] for s in st], marker="o", label=arm)
        axes[2].plot([s["step"] for s in st if s.get("retention_acc") is not None],
                     [s["retention_acc"] for s in st if s.get("retention_acc") is not None],
                     marker="o", label=arm)
        axes[3].plot([s["step"] for s in st if "new_acc" in s],
                     [s["new_acc"]["joint"] for s in st if "new_acc" in s], marker="o", label=arm)
    for ax, t in zip(axes, ("frozen-base drift (want 0)", "retention: fraction bit-identical",
                            "retention: composition accuracy", "plasticity: new-concept accuracy")):
        ax.set_title(t, fontsize=9); ax.set_xlabel("consolidations"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "n3_lifelong.png"), dpi=140)

    print("\n[n3] summary after the full stream:")
    for arm, r in results.items():
        last = r["steps"][-1]
        print(f"    {arm:9s} retention_exact {last['retention_exact']:.2f}  "
              f"retention_acc {last['retention_acc'] if last['retention_acc'] is not None else float('nan'):.2f}  "
              f"base_drift {last['base_drift']:+.5f}  adapter {last['params'] / 1e6:.2f}M  "
              f"shared ancestors {len(r['bank']['shared_ancestors'])}")
    print(f"[n3] wrote {RESULTS}/n3_lifelong.json and n3_lifelong.png")


if __name__ == "__main__":
    main()
