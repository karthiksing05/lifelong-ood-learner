"""E4 — is composition CLOSED under consolidation?  (the distinctive lifelong claim)

The poe-diffusion paper closes on exactly this gap: "human-level intelligence also requires learning
new primitives and incorporating them into future compositions ... A full mechanism for detecting,
consolidating, and reusing genuinely new primitives remains an important direction for future work."
E1-E3 show we can compose, sample and consolidate without degrading. E4 asks the remaining question:
once consolidated, is a new concept REUSABLE as a primitive by later compositions?

The design exploits the structure of the ColorMNIST held-out set. Each OOD slot is a (fg, bg) pair
that never co-occurs in training, crossed with all ten digits. So:

  * TEACH on some digits of one held-out pair, e.g. (cyan, deep_red) for digits 0-4: each is composed
    from base prototypes, sampled, and consolidated as an adapter.
  * TEST on the REMAINING digits of the SAME pair (5-9). Before teaching, composing one of these
    requires assembling the colour scheme from two different seen slots (a cyan-on-something expert
    and a something-on-deep_red expert) while also getting the digit right. After teaching, the
    adapters contain concepts that already carry the whole (cyan, deep_red) scheme, so the composer
    can spend its K experts on shape instead. If composition is closed under consolidation, test
    accuracy RISES and the selected experts start coming from adapter branches.

  * CONTROL: an arm that consolidates the SAME NUMBER of concepts from a DIFFERENT held-out pair.
    It adds just as much density and just as many components, but nothing reusable for the test
    pair. If the treatment arm gains and the control arm does not, the gain is reuse of the specific
    primitive rather than a generic benefit of having more stuff in the mixture.

Reported per arm, before vs after: composition accuracy on the test digits, the geometry triple, and
the fraction of selected experts drawn from adapter branches (the direct evidence of reuse).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, base_parser, fresh_learner, load_all      # noqa: E402

from lifelong_ood.datasets import FG_NAMES, BG_NAMES, OOD_PAIRS       # noqa: E402
from lifelong_ood.evals import composition_accuracy, sample_geometry, save_grid, save_json  # noqa: E402


def pair_slots(data, fg_name, bg_name):
    """The ten OOD slots (one per digit) of a held-out (fg, bg) pair, as query indices."""
    fi, bi = FG_NAMES.index(fg_name), BG_NAMES.index(bg_name)
    rng = np.random.default_rng(0)
    out = []
    for d in range(10):
        idx = np.where((data["digit"] == d) & (data["fg"] == fi) & (data["bg"] == bi))[0]
        if len(idx) == 0:
            continue
        q = idx[rng.integers(len(idx))]
        out.append(dict(d=d, f=fi, b=bi, query=int(q), rest=idx[idx != q],
                        slot=int(data["slot"][q])))
    return out


def evaluate(learner, data, probes, tests, n_samples, seed, tag, save=None):
    """Compose+sample each test slot under the CURRENT manifold; also record expert provenance."""
    A, G, reuse = [], [], []
    snap = learner.field.snap
    for qi, t in enumerate(tests):
        xq = data["X"][t["query"]]
        teacher = learner.compose(xq, discovery="tree")
        nodes = teacher["meta"]["node"][teacher["sel"]]
        reuse.append(float(np.mean(snap["branch"][nodes] != "base")))
        S = learner.sample_teacher(teacher, n=n_samples, seed=seed + qi)
        A.append(composition_accuracy(probes, S, t["d"], t["f"], t["b"]))
        G.append(sample_geometry(S, xq, data["X"][t["rest"][:100]]))
        if save is not None and qi == 0:
            save_grid(np.concatenate([[xq], S[:15]]), save,
                      title=f"{tag} — digit {t['d']}, {FG_NAMES[t['f']]} on {BG_NAMES[t['b']]}")
    m = lambda k, L: float(np.mean([x[k] for x in L]))   # noqa: E731
    return dict(**{k: m(k, A) for k in ("joint", "digit", "fg", "bg")},
                **{k: m(k, G) for k in ("novelty", "diversity", "gen_rmse")},
                adapter_expert_frac=float(np.mean(reuse)),
                per_slot_joint=[a["joint"] for a in A])


def main():
    p = base_parser()
    p.add_argument("--n-teach", type=int, default=5, help="digits of the pair consolidated")
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--n-samples", type=int, default=24)
    args = p.parse_args()
    if args.quick:
        args.n_teach, args.n_pool, args.n_samples = 3, 16, 8
    data, learner, probes = load_all(args)

    target_fg, target_bg = OOD_PAIRS[0]           # the pair we teach and test
    other_fg, other_bg = OOD_PAIRS[1]             # the control pair
    slots = pair_slots(data, target_fg, target_bg)
    teach, tests = slots[:args.n_teach], slots[args.n_teach:]
    ctrl_teach = pair_slots(data, other_fg, other_bg)[:args.n_teach]
    print(f"[e4] target pair ({target_fg}, {target_bg}): teach digits "
          f"{[t['d'] for t in teach]}, test digits {[t['d'] for t in tests]}")
    print(f"[e4] control pair ({other_fg}, {other_bg}): teach digits {[t['d'] for t in ctrl_teach]}")

    out = dict(target_pair=[target_fg, target_bg], control_pair=[other_fg, other_bg],
               teach_digits=[t["d"] for t in teach], test_digits=[t["d"] for t in tests])

    before = evaluate(learner, data, probes, tests, args.n_samples, args.seed, "before teaching",
                      save=os.path.join(RESULTS, "e4_before.png"))
    out["before"] = before
    print(f"[e4] BEFORE: joint {before['joint']:.3f} (digit {before['digit']:.2f} "
          f"fg {before['fg']:.2f} bg {before['bg']:.2f}), adapter-experts {before['adapter_expert_frac']:.2f}")

    for arm, tset in (("treatment", teach), ("control", ctrl_teach)):
        L = learner if arm == "treatment" else fresh_learner(args, data)
        for k, t in enumerate(tset, 1):
            L.learn_concept(data["X"][t["query"]], f"{arm}_d{t['d']}",
                            n_pool=args.n_pool, seed=args.seed + k)
            print(f"[e4:{arm}] consolidated digit {t['d']} "
                  f"({len(L.adapters)} adapters, {len(L.field.snap['M'])} concepts)", flush=True)
        after = evaluate(L, data, probes, tests, args.n_samples, args.seed, f"after {arm}",
                         save=os.path.join(RESULTS, f"e4_after_{arm}.png"))
        out[f"after_{arm}"] = after
        out[f"delta_{arm}"] = {k: after[k] - before[k]
                               for k in ("joint", "digit", "fg", "bg", "novelty", "diversity")}
        print(f"[e4] AFTER {arm}: joint {after['joint']:.3f} "
              f"(delta {after['joint'] - before['joint']:+.3f}), "
              f"adapter-experts {after['adapter_expert_frac']:.2f}")

    save_json(out, os.path.join(RESULTS, "e4_closure.json"), seed=args.seed)
    dt = out["delta_treatment"]["joint"]
    dc = out["delta_control"]["joint"]
    print(f"\n[e4] composition accuracy on held-out digits of ({target_fg}, {target_bg}):")
    print(f"     before {before['joint']:.3f} | after teaching that pair {out['after_treatment']['joint']:.3f} "
          f"({dt:+.3f}) | after teaching a different pair {out['after_control']['joint']:.3f} ({dc:+.3f})")
    print(f"[e4] reuse: {out['after_treatment']['adapter_expert_frac']:.2f} of selected experts come "
          f"from adapters in the treatment arm, {out['after_control']['adapter_expert_frac']:.2f} in the control")
    verdict = ("CLOSED: consolidated concepts are reused as primitives" if dt > 0 and dt > dc
               else "NOT SHOWN at this scale")
    print(f"[e4] verdict: {verdict}")
    out["verdict"] = verdict
    save_json(out, os.path.join(RESULTS, "e4_closure.json"))


if __name__ == "__main__":
    main()
