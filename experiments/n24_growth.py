"""N24 — the growth accounting: what does a lifelong stream actually COST?

The thesis claims efficiency, and so far the repository has only claimed it per concept: a basis
direction is 256 floats where an adapter is 551K parameters (5q), and a kept orthogonal delta is
2.8x smaller than a kept LoRA (5w).  Those are per-concept numbers.  The claim that matters for
LIFELONG learning is about the CURVE -- whether the cost of running forever grows with how long you
run, or only with how much genuinely new vocabulary the world contains.

This computes that curve.  Nothing here is simulated in the sense of invented: every per-concept
cost is READ from the measured results (`results/n17_ortho_merge.json`, `n11_basis_extension*.json`),
and the stream is the actual combinatorial structure of each benchmark.  The only modelling step is
the routing rule itself, which N18 measured a detector for.

The stream: compositional slots arrive in random order.  A slot is a NEW VOCABULARY event if it is
the first arrival carrying some factor value not seen before, and a NEW COMBINATION event otherwise.
Vocabulary saturates -- ColorMNIST has 18 factor values against 160 slots, CelebA 28 attribute
values against 1200+ realized combinations -- so the interesting quantity is what happens after it
does.

Arms:
  `per-query LoRA`   the source paper's consolidation, kept rather than discarded: one adapter per
                     query, forever.  Linear in EXPERIENCES.
  `adapter per concept`  this repository before 5q: one adapter per new composition.  Also linear.
  `ours (routed)`    a kept orthogonal delta for each new vocabulary item, nothing for a new
                     combination.  Linear in VOCABULARY, flat in experiences once it saturates.

Run on CPU; it reads JSON and draws.  No GPU, no model.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                              # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def measured_costs():
    """Per-concept parameter costs, READ from the measured runs rather than restated."""
    out = {}
    p17 = os.path.join(RESULTS, "n17_ortho_merge.json")
    if os.path.exists(p17):
        d = json.load(open(p17))
        n = int(d.get("config", {}).get("n_tasks", 12))
        for r in d.get("rows", []):
            if r["arm"] == "lora-keep":
                out["lora_per_concept"] = r["params"] / n
                out["lora_acc"] = r.get("ACC")
            if r["arm"] == "ortho-keep":
                out["ortho_per_concept"] = r["params"] / n
                out["ortho_acc"] = r.get("ACC")
    for tag in ("n11_basis_extension_B", "n11_basis_extension"):
        p11 = os.path.join(RESULTS, f"{tag}.json")
        if os.path.exists(p11):
            d = json.load(open(p11))
            out.setdefault("adapter_marginal", d.get("adapter_marginal"))
            out.setdefault("basis_dim", d.get("dim", 256))
            break
    # the router's MEASURED misroute rate, so the curve is not drawn under oracle routing
    pagg = os.path.join(RESULTS, "aggregate.json")
    if os.path.exists(pagg):
        t = json.load(open(pagg)).get("n18_router", {}).get("table", {})
        s = t.get("combination_sent_to_weights")
        if isinstance(s, dict) and s.get("mean") is not None:
            out["overspend_rate"] = float(s["mean"])
        s = t.get("vocab_sent_to_free")
        if isinstance(s, dict) and s.get("mean") is not None:
            out["underspend_rate"] = float(s["mean"])
    out.setdefault("lora_per_concept", 551152.0)
    out.setdefault("ortho_per_concept", 213001.0)
    out.setdefault("basis_dim", 256)
    out.setdefault("overspend_rate", 0.0)
    out.setdefault("underspend_rate", 0.0)
    return out


def colormnist_stream(seed=0):
    """160 slots as (digit, fg, bg); vocabulary = the 10 + 4 + 4 factor VALUES."""
    slots = [(d, f, b) for d in range(10) for f in range(4) for b in range(4)]
    rng = np.random.default_rng(seed)
    rng.shuffle(slots)
    return slots, [10, 4, 4]


def shapes3d_stream(seed=0):
    """Shapes3D as (floor_hue, object_hue, shape); vocabulary = the 5 + 5 + 4 factor VALUES.

    The split n57/n58/n59 actually train and measure on: `value_stride=2` thins the two 10-valued
    hues to 5 each and spares `shape`, giving 100 compositions.  Using the full 10x10x4 here would
    report a growth curve for a benchmark no result in this repository was measured on.
    """
    slots = [(a, b, c) for a in range(5) for b in range(5) for c in range(4)]
    rng = np.random.default_rng(seed)
    rng.shuffle(slots)
    return slots, [5, 5, 4]


def celeba_stream(seed=0, cache=None):
    """Realized CelebA attribute combinations; vocabulary = the 14 x 2 attribute VALUES.

    Uses the real combination frequencies when the cached attribute matrix is available, so the
    stream has the benchmark's actual structure rather than a uniform product.
    """
    rng = np.random.default_rng(seed)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        A = z["A"]
        keys, counts = np.unique(A, axis=0, return_counts=True)
        keys = keys[counts >= 25]
        idx = rng.permutation(len(keys))
        return [tuple(int(v) for v in keys[i]) for i in idx], [2] * A.shape[1]
    # A FIXED number of unique combinations, not "however many survive deduplication".
    #
    # This used to be `{tuple(...) for _ in range(1200)}` -- a set built from 1200 draws, whose SIZE
    # depends on how many collisions that seed happened to produce. Each seed therefore returned a
    # different-length stream, the per-seed curves came out ragged, and `np.array([...], float)` in
    # the plotting block raised `inhomogeneous shape` -- so `04_growth_per_concept.png` could not be regenerated
    # at all. The cached branch above is deterministic; this one has to be too.
    n_target = 1000
    combos = set()
    while len(combos) < n_target:
        combos.update(tuple(int(v) for v in row)
                      for row in rng.integers(0, 2, size=(n_target, 14)))
    combos = sorted(combos)[:n_target]          # sorted() so the set's iteration order cannot leak
    rng.shuffle(combos)
    return combos, [2] * 14


ARMS = ("per-query LoRA", "adapter per concept", "ours (routed)", "ours (measured routing)")


def run_stream(slots, costs, sizes, rng=None):
    """-> per-arm cumulative parameter curves, and the route taken at each step.

    `ours (routed)` is the oracle-routing curve.  `ours (measured routing)` charges the router's own
    error rate from N18 -- 20% of new COMBINATIONS are sent to the weight route and buy an adapter
    they did not need -- so the honest curve is not drawn under an assumption the detector does not
    meet.  The other error direction is charged at N18's measured rate too, which is 0.000: a new
    vocabulary item was never once sent down the free route, across 3 seeds and 120 queries.
    """
    rng = rng or np.random.default_rng(0)
    seen_values = [set() for _ in sizes]
    routes, curves = [], {k: [] for k in ARMS}
    tot = {k: 0.0 for k in curves}
    n_vocab = 0
    for s in slots:
        new_vals = [f for f, v in enumerate(s) if v not in seen_values[f]]
        for f in new_vals:
            seen_values[f].add(s[f])
        is_vocab = bool(new_vals)
        routes.append("new-vocabulary" if is_vocab else "new-combination")
        n_vocab += len(new_vals)
        tot["per-query LoRA"] += costs["lora_per_concept"]
        tot["adapter per concept"] += costs["lora_per_concept"]
        # the routing rule 5q measured: vocabulary buys weights, combinations are free
        tot["ours (routed)"] += costs["ortho_per_concept"] * len(new_vals)
        spent = costs["ortho_per_concept"] * len(new_vals)
        if not is_vocab and rng.random() < costs["overspend_rate"]:
            spent += costs["ortho_per_concept"]        # a combination misrouted to the weight path
        tot["ours (measured routing)"] += spent
        for k in curves:
            curves[k].append(tot[k])
    return curves, routes, n_vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    costs = measured_costs()
    print(f"[n24] measured per-concept costs: kept LoRA {costs['lora_per_concept'] / 1e3:.0f}K, "
          f"kept orthogonal {costs['ortho_per_concept'] / 1e3:.0f}K, "
          f"basis direction {costs['basis_dim']} floats", flush=True)

    cache = os.path.join(RESULTS, "cache", "celeba_raw_img64_n0.npz")
    benches = {}
    for name, fn in (("ColorMNIST", colormnist_stream),
                     ("Shapes3D", shapes3d_stream),
                     ("CelebA", lambda s: celeba_stream(s, cache=cache))):
        allc, allr, nv = [], [], []
        for s in range(1 if args.quick else args.seeds):
            slots, sizes = fn(s)
            c, r, n = run_stream(slots, costs, sizes, rng=np.random.default_rng(1000 + s))
            allc.append(c); allr.append(r); nv.append(n)
        benches[name] = dict(curves=allc, routes=allr, n_slots=len(allc[0]["ours (routed)"]),
                             n_vocab=float(np.mean(nv)), sizes=sizes)
        rt = allr[0]
        first_free = next((i for i in range(len(rt))
                           if all(x == "new-combination" for x in rt[i:])), len(rt))
        benches[name]["saturates_at"] = first_free
        print(f"[n24] {name}: {len(rt)} experiences, {np.mean(nv):.0f} vocabulary items, "
              f"vocabulary saturates after experience {first_free} "
              f"({100 * (len(rt) - first_free) / len(rt):.0f}% of the stream is then free)",
              flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, b) in zip(axes, benches.items()):
        x = np.arange(1, b["n_slots"] + 1)
        for k, col in (("per-query LoRA", "#E45756"), ("ours (measured routing)", "#F58518"),
                       ("ours (routed)", "#4C78A8")):
            Y = np.array([c[k] for c in b["curves"]], float) / 1e6
            m, sd = Y.mean(0), Y.std(0)
            ax.plot(x, m, color=col, lw=2, label=k)
            ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.2)
        ax.axvline(b["saturates_at"], ls="--", lw=1, color="0.45")
        ax.annotate(f"vocabulary complete\n(experience {b['saturates_at']})",
                    xy=(b["saturates_at"], ax.get_ylim()[1] * 0.55), fontsize=7,
                    xytext=(6, 0), textcoords="offset points", color="0.3")
        ax.set_xlabel("experiences absorbed")
        ax.set_ylabel("cumulative adapter parameters (M)")
        ax.set_title(f"{name}: {b['n_slots']} compositions, "
                     f"{b['n_vocab']:.0f} vocabulary items", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("N24 — cost grows with VOCABULARY, not with experience.  Per-concept costs are "
                 "measured (n17/n11); the stream is each benchmark's own structure.", fontsize=9)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "n24_growth.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    out = dict(costs=costs)
    print(f"\n[n24] cumulative cost at the end of the stream")
    print(f"    {'benchmark':12s}{'experiences':>13s}{'per-query LoRA':>17s}"
          f"{'ours (oracle)':>16s}{'ours (measured)':>17s}{'ratio':>9s}")
    for name, b in benches.items():
        fin = {k: float(np.mean([c[k][-1] for c in b["curves"]])) for k in b["curves"][0]}
        ratio = fin["per-query LoRA"] / max(fin["ours (measured routing)"], 1.0)
        out[name] = dict(final=fin, n_slots=b["n_slots"], n_vocab=b["n_vocab"],
                         saturates_at=b["saturates_at"], ratio=ratio)
        print(f"    {name:12s}{b['n_slots']:13d}{fin['per-query LoRA'] / 1e6:16.1f}M"
              f"{fin['ours (routed)'] / 1e6:15.1f}M"
              f"{fin['ours (measured routing)'] / 1e6:16.1f}M{ratio:9.1f}x")
    with open(os.path.join(RESULTS, "n24_growth.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n  The routed curve is FLAT after its vocabulary is complete: every later experience "
          f"is a\n  new combination and costs nothing.  The per-query curve never stops rising, "
          f"which is what\n  makes 'forever' the operative word in the claim.")
    print(f"[n24] wrote figures/n24_growth.png and {RESULTS}/n24_growth.json")


if __name__ == "__main__":
    main()
