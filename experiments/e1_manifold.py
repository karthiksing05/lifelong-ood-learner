"""E1 — Is the Cobweb tree a valid diffusion manifold?

Three checks, one claim: the analytic score field read off the tree behaves the way a trained DDPM's
learned score behaves, INCLUDING its hierarchical mode structure.

  (a) GENERATION.  Reverse-diffusion (DDIM) sampling from pure noise on the analytic field must
      produce coherent ColorMNIST images whose (digit, fg, bg) triple -- read by the probes -- lands
      in SEEN compositional slots.  Reported: valid-composition rate + sample grids for both the
      leaf-mixture field and the depth-annealed frontier field.

  (b) MODE HIERARCHY.  The poe-diffusion paper shows (their Appendix A) that mode ascent on a DDPM's
      p_t recovers prototypes that coarsen as t grows.  Here mode ascent on OUR p_t should recover
      the tree's OWN nodes: for each ascended mode we report the distance to the nearest tree node
      and that node's depth.  If matched depth falls as u rises while match distance stays small, the
      incrementally-built hierarchy IS the density-mode hierarchy of the diffusion marginals.

  (c) FIDELITY.  Tweedie denoising R^2 and mean held-out log-likelihood, so later experiments have a
      baseline for what "no damage" means.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, base_parser, load_all                      # noqa: E402

from lifelong_ood.evals import composition_accuracy, per_slot_metrics, save_grid, save_json  # noqa: E402
from lifelong_ood.score import ScoreField, cosine_abar                 # noqa: E402
from lifelong_ood.datasets import slot_id                              # noqa: E402


def valid_rate(probes, samples, seen_slots):
    d = probes["digit"].predict(samples)
    f = probes["fg"].predict(samples)
    b = probes["bg"].predict(samples)
    sl = np.array([slot_id(dd, ff, bb) for dd, ff, bb in zip(d, f, b)])
    return float(np.isin(sl, seen_slots).mean())


def main():
    args = base_parser().parse_args()
    data, learner, probes = load_all(args)
    snap = learner.tree.snapshot(args.min_count)
    seen_slots = np.unique(data["slot"][data["seen"]])
    out = dict(n_nodes=learner.tree.n_nodes, n_kept=len(snap["M"]),
               n_leaves=int(snap["is_leaf"].sum()), max_depth=int(snap["depth"].max()))
    print(f"[e1] snapshot: {out['n_kept']} kept / {out['n_leaves']} effective leaves / depth {out['max_depth']}")
    n_samp = 16 if args.quick else 64

    # (a) generation from pure noise, both field readings
    for mode in ("leaves", "frontier"):
        field = ScoreField(snap, mode=mode)
        S = field.sample(n=n_samp, steps=args.steps, seed=args.seed)
        out[f"valid_rate_{mode}"] = valid_rate(probes, S, seen_slots)
        save_grid(S[:32], os.path.join(RESULTS, f"e1_samples_{mode}.png"),
                  title=f"unconditional DDIM samples — {mode} field")
        print(f"[e1] {mode}: valid seen-composition rate {out[f'valid_rate_{mode}']:.3f}")

    # (b) the mode hierarchy: ascent at growing noise levels vs the tree's own nodes
    field = ScoreField(snap, mode="leaves")
    rng = np.random.default_rng(args.seed)
    seen_idx = np.where(data["seen"])[0]
    queries = data["X"][rng.choice(seen_idx, 4 if args.quick else 8, replace=False)]
    u_grid = (0.1, 0.25, 0.4, 0.6, 0.8)
    hier = []
    panel = []
    for u in u_grid:
        _, x0 = field.mode_ascent(queries, u, n_starts=len(queries),
                                  iters=40 if args.quick else 120, lr=0.05, seed=args.seed)
        d2 = ((x0[:, None, :] - snap["M"][None]) ** 2).sum(2)
        j = np.argmin(d2, 1)
        hier.append(dict(u=u, match_rmse=float(np.sqrt(d2[np.arange(len(j)), j] / x0.shape[1]).mean()),
                         matched_depth=float(snap["depth"][j].mean()),
                         matched_leaf_frac=float(snap["is_leaf"][j].mean())))
        panel.extend(list(np.clip(x0[:4], 0, 1)))
        print(f"[e1] u={u:.2f}: matched depth {hier[-1]['matched_depth']:.2f}, "
              f"rmse {hier[-1]['match_rmse']:.4f}, leaf-frac {hier[-1]['matched_leaf_frac']:.2f}")
    out["mode_hierarchy"] = hier
    save_grid(panel, os.path.join(RESULTS, "e1_mode_hierarchy.png"), ncol=4,
              title="modes of p_t by ascent — rows: u = " + ", ".join(f"{u}" for u in u_grid))

    # (c) fidelity numbers on held-out seen data
    ev = rng.choice(seen_idx, 200 if not args.quick else 60, replace=False)
    m = per_slot_metrics(field, data["X"], data["slot"], np.unique(data["slot"][ev])[:12])
    out["seen_loglik_mean"] = float(np.mean([v["loglik"] for v in m.values()]))
    out["seen_r2_mean"] = float(np.mean([v["r2"] for v in m.values()]))
    print(f"[e1] seen: loglik {out['seen_loglik_mean']:.1f}, denoise R2 {out['seen_r2_mean']:.3f}")

    save_json(out, os.path.join(RESULTS, "e1_manifold.json"))
    print(f"[e1] wrote {RESULTS}/e1_manifold.json")


if __name__ == "__main__":
    main()
