"""N46 -- Liu et al. (ICCV 2023) unsupervised concept discovery as a baseline, on ColorMNIST.

Which Liu et al.  This repository already runs `composable-diff` / `composable-mcmc`, which are
**Liu et al. 2022** (Composable Diffusion): they compose concepts they are GIVEN, using the class
embeddings the backbone already has.  This is the **2023** follow-up, which DISCOVERS the concepts
from unlabelled pixels.  That makes it the closest published baseline to what this project actually
claims, because it lands on the same shape of answer we do:

    them   a shared library of K concept embeddings + a per-image weight vector on the simplex
    us     a shared factor basis + a per-concept address (factor coordinates)

Both store a small coordinate against a shared library; both leave the backbone frozen.  So the
comparison is not "who generates better pictures" in the abstract -- it is a direct test of the
thesis, on two axes at once:

  1. QUALITY   can their discovered library express a held-out composition it never saw?
  2. STORAGE   what does one held-out concept cost, once the library is paid for?

Their per-query cost is K floats (the weight vector); ours is 72 bytes.  Whichever wins, the
comparison is finally like-for-like, which the 2022 arm is not: Composable Diffusion is handed the
ground-truth factor labels at composition time, and this method is handed nothing.

Two arms deliberately separated, because they answer different questions:

  `liu-unsup`         library discovered on SEEN images only, then weights fitted to the held-out
                      query.  This is the honest lifelong setting -- the library predates the query.
  `liu-unsup-oracle`  library discovery is allowed to SEE the query image too.  Not a fair lifelong
                      baseline, but it upper-bounds what the method can do here and separates
                      "the library cannot express this" from "the weights could not find it".

    python experiments/n46_unsup_concepts.py --K 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                                  # noqa: E402

from lifelong_ood.datasets import load_colormnist, slot_name                    # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,               # noqa: E402
                                precision_recall, save_json)
from lifelong_ood.neural import load_field, to_data                             # noqa: E402
from lifelong_ood.neural.unsup_concepts import (discover_concepts, fit_weights,  # noqa: E402
                                                sample_unsup)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=8,
                   help="library size.  Their K is a design choice per dataset; ColorMNIST has "
                        "10 digits x 4 fg x 4 bg, so K=8 is well under the 18 values a factored "
                        "description needs and K=18 is the matched-capacity setting.")
    p.add_argument("--n-cases", type=int, default=6)
    p.add_argument("--n-control", type=int, default=6,
                   help="SEEN compositions used as the port-works control")
    p.add_argument("--identity-check", action="store_true",
                   help="sanity-test the composition machinery: library = the true class\n"
                        "embedding, w one-hot, so Eq. 11 reduces to the class conditional")
    p.add_argument("--oracle", action="store_true",
                   help="let discovery see the held-out queries.  Not a fair baseline; it separates a library that cannot express the target from weights that cannot find it.")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-pool", type=int, default=512, help="unlabelled images for discovery")
    p.add_argument("--disc-steps", type=int, default=600)
    p.add_argument("--fit-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-slot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_cases, args.n_samples, args.n_pool, args.n_control = 2, 8, 64, 2
        args.disc_steps, args.fit_steps, args.steps, args.n_per_slot = 40, 20, 20, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    field = load_field(os.path.join(CACHE, "ddpm_colormnist.pt"), device=dev)
    qs = ood_queries(data, n_slots=args.n_cases, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # The discovery pool is UNLABELLED and drawn from SEEN slots only, so the library cannot have
    # been built with knowledge of the held-out compositions.
    seen = np.where(data["seen"])[0]
    pool = data["X"][seen[rng.permutation(len(seen))[:args.n_pool]]]
    if args.oracle:
        # NOT a fair lifelong baseline -- the library is allowed to see the held-out queries, so it
        # cannot be run as a competitor.  It upper-bounds the method here, which is what separates
        # "the discovered library cannot express this" from "the weights could not find it".
        pool = np.concatenate([pool, data["X"][[q["query"] for q in qs]]], axis=0)
    print(f"[n46] discovering K={args.K} concepts from {len(pool)} UNLABELLED "
          f"{'seen + HELD-OUT (oracle)' if args.oracle else 'seen'} images "
          f"({args.disc_steps} steps)", flush=True)
    t0 = time.time()
    lib = discover_concepts(field, pool, K=args.K, n_steps=args.disc_steps, lr=args.lr,
                            seed=args.seed, verbose=True)
    print(f"[n46] library learned in {(time.time() - t0) / 60:.1f} min; "
          f"final loss {np.mean(lib['hist'][-20:]):.4f}", flush=True)

    d_emb = lib["C"].shape[1]
    rows, panels = [], {}
    # defined before BOTH the identity check and the control, since each draws from it
    seen_slots = np.unique(data["slot"][seen])

    # ── IDENTITY CHECK: is the COMPOSITION MACHINERY itself correct? ─────────
    # Independent of whether discovery works.  Put the backbone's OWN class embeddings in as the
    # library and make w one-hot; then Eq. 11 collapses to eps_unsup = eps(x, t | c_class), the
    # ordinary class conditional, so the samples must reproduce that class.  If THIS is at chance,
    # the plumbing (score composition, sampling hook, layout) is broken and nothing else here can be
    # interpreted.  If it passes, the plumbing is right and any failure is discovery or the method.
    if args.identity_check:
        import torch as _t
        ck = _t.load(os.path.join(CACHE, "ddpm_colormnist.pt"), map_location="cpu",
                     weights_only=False)
        remap = ck["remap"]
        cemb = ck["ema"]["cemb.weight"].numpy()
        ident = []
        for s_ in seen_slots[rng.permutation(len(seen_slots))[:args.n_control]]:
            rows_s = np.where(data["slot"] == s_)[0]
            i0 = rows_s[0]
            cid = remap[int(s_)]
            C1 = cemb[[cid]]                      # a library of exactly one: the true class
            S = to_data(sample_unsup(field, C1, np.array([1.0]), n=args.n_samples,
                                     steps=args.steps, seed=args.seed,
                                     clip=(-1, 1))).astype(np.float32)
            a = composition_accuracy(probes, S, int(data["digit"][i0]), int(data["fg"][i0]),
                                     int(data["bg"][i0]))
            ident.append(float(a["joint"]))
        print(f"[n46] IDENTITY CHECK (library = the true class embedding, w one-hot): "
              f"mean joint {np.mean(ident):.3f}  {[round(v,2) for v in ident]}")
        print(f"[n46]   this MUST be high.  If it is at chance the fault is in the composition "
              f"machinery, not in concept discovery.\n", flush=True)
        rows.append(dict(slot=-1, arm="identity-check", K=1, joint=float(np.mean(ident)),
                         digit=float("nan"), fg=float("nan"), bg=float("nan"),
                         recall=float("nan"), bytes_per_concept=0, library_bytes=0))

    # ── CONTROL, and it runs FIRST ───────────────────────────────────────────
    # A baseline reported at 0.000 is worthless without evidence that the port works at all.  The
    # same library and the same weight-fitting, applied to SEEN compositions, must reproduce them:
    # those are in the discovery pool's own distribution, so anything near chance here indicts this
    # implementation rather than the method, and no held-out number below should be believed.
    ctrl = []
    for s_ in seen_slots[rng.permutation(len(seen_slots))[:args.n_control]]:
        rows_s = np.where(data["slot"] == s_)[0]
        xs = data["X"][rows_s[0]]
        i0 = rows_s[0]
        w = fit_weights(field, xs, lib["C"], n_steps=args.fit_steps, lr=args.lr, seed=args.seed)
        S = to_data(sample_unsup(field, lib["C"], w, n=args.n_samples, steps=args.steps,
                                 seed=args.seed, clip=(-1, 1))).astype(np.float32)
        a = composition_accuracy(probes, S, int(data["digit"][i0]), int(data["fg"][i0]),
                                 int(data["bg"][i0]))
        ctrl.append(float(a["joint"]))
        rows.append(dict(slot=int(s_), arm="liu-unsup-SEEN-control", K=args.K,
                         joint=float(a["joint"]), digit=float(a["digit"]), fg=float(a["fg"]),
                         bg=float(a["bg"]), recall=float("nan"),
                         bytes_per_concept=int(args.K * 4), library_bytes=int(args.K * d_emb * 4)))
    print(f"[n46] CONTROL on {len(ctrl)} SEEN compositions: mean joint {np.mean(ctrl):.3f}  "
          f"{[round(v, 2) for v in ctrl]}")
    print(f"[n46]   chance is ~{1/10 * 1/4 * 1/4:.3f}.  If this is at chance the port is broken and "
          f"the held-out numbers below mean nothing.\n", flush=True)

    for q in qs:
        xq = data["X"][q["query"]]
        R = data["X"][q["rest"][:100]].astype(np.float32)
        w = fit_weights(field, xq, lib["C"], n_steps=args.fit_steps, lr=args.lr, seed=args.seed)
        S = to_data(sample_unsup(field, lib["C"], w, n=args.n_samples, steps=args.steps,
                                 seed=args.seed, clip=(-1, 1))).astype(np.float32)
        a = composition_accuracy(probes, S, q["d"], q["f"], q["b"])
        pr = precision_recall(S, R)
        rows.append(dict(slot=int(q["slot"]), arm=("liu-unsup-oracle" if args.oracle else "liu-unsup"), K=args.K,
                         joint=float(a["joint"]), digit=float(a["digit"]), fg=float(a["fg"]),
                         bg=float(a["bg"]), recall=float(pr["recall"]),
                         bytes_per_concept=int(args.K * 4),
                         library_bytes=int(args.K * d_emb * 4)))
        panels[int(q["slot"])] = (S, q)
        print(f"[n46] slot {q['slot']:3d} {slot_name(q['d'], q['f'], q['b']):32s} "
              f"joint {a['joint']:.3f}  (d {a['digit']:.2f} fg {a['fg']:.2f} bg {a['bg']:.2f})  "
              f"recall {pr['recall']:.2f}  w_max {w.max():.2f}", flush=True)

    j = [r["joint"] for r in rows]
    print(f"\n[n46] liu-unsup mean joint {np.mean(j):.3f} over {len(j)} held-out compositions; "
          f"{args.K * 4} bytes per concept against a {args.K * d_emb * 4 / 1024:.1f} KB library")
    save_json(dict(rows=rows, config=vars(args),
                   library=dict(K=args.K, d_emb=int(d_emb),
                                final_loss=float(np.mean(lib["hist"][-20:])))),
              os.path.join(RESULTS, f"n46_unsup_concepts{args.tag}.json"), seed=args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_show = min(6, args.n_samples)
    fig, axes = plt.subplots(len(panels) * 2, n_show, figsize=(1.4 * n_show, 1.5 * len(panels) * 2))
    axes = np.atleast_2d(axes)
    r = 0
    for slot, (S, q) in panels.items():
        for name, imgs in (("real\n" + slot_name(q["d"], q["f"], q["b"]),
                            data["X"][q["rest"][:n_show]].astype(np.float32)),
                           ("liu-unsup (ICCV 23)", S)):
            for c in range(n_show):
                axes[r][c].axis("off")
                if c < len(imgs):
                    axes[r][c].imshow(np.clip(np.asarray(imgs[c], np.float32).reshape(32, 32, 3),
                                              0, 1))
            axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
            axes[r][0].axis("on")
            axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for s_ in axes[r][0].spines.values():
                s_.set_visible(False)
            r += 1
    fig.suptitle(f"Liu et al. (ICCV 2023) unsupervised concept discovery, K={args.K}, on "
                 f"ColorMNIST held-out compositions.\nLibrary discovered from {args.n_pool} "
                 f"UNLABELLED seen images; each query then costs a {args.K}-float weight vector.  "
                 f"Mean joint {np.mean(j):.3f}.", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG, f"f22_liu_unsup{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[n46] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
