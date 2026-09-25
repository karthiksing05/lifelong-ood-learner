"""Extract the un-confounded retention numbers from an N3 run.

N3's per-step `retention_acc` is a MEAN OVER A CHANGING SUBSET of earlier concepts (the oldest plus
the most recent few), so it wanders as concepts of differing quality enter and leave the probe set.
That movement reads as forgetting and is not: some pools were bad, and a concept that never worked
cannot decay.

The honest questions are per concept:
  drift from birth      how much has THIS concept's accuracy changed since it was consolidated?
  range across steps    how much does THIS concept's accuracy move at all over the run?

`range` is the sharp one, because with frozen ancestors it should be exactly 0.000: the concept's
function is bit-identical, so with a pinned evaluation seed its accuracy cannot move.  The
`unfrozen` ablation is where it must break.

    python experiments/analyze_n3.py [results/n3_lifelong.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(HERE), "results", "n3_lifelong.json")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path) as fh:
        r = json.load(fh)
    print(f"{os.path.basename(path)}\n")
    print(f"{'arm':10s}{'concepts':>9s}{'drift/birth':>12s}{'drift (worked)':>16s}"
          f"{'range mean':>12s}{'range max':>11s}{'plasticity':>12s}")
    for arm, blob in r.items():
        steps = blob["steps"]
        born = {f"ood_{s['slot']}": s["new_acc"]["joint"] for s in steps if "new_acc" in s}
        seq = {}
        for s in steps:
            for cname, acc in (s.get("retention_per_concept") or {}).items():
                seq.setdefault(cname, []).append(acc)
        drift = [v[-1] - born[c] for c, v in seq.items() if c in born]
        worked = [v[-1] - born[c] for c, v in seq.items() if born.get(c, 0) > 0.5]
        rng = [max(v) - min(v) for v in seq.values() if len(v) > 1]
        plast = [s["new_acc"]["joint"] for s in steps if "new_acc" in s]
        print(f"{arm:10s}{len(seq):9d}{np.mean(drift):+12.3f}"
              f"{(np.mean(worked) if worked else np.nan):+16.3f}"
              f"{(np.mean(rng) if rng else 0):12.3f}{(max(rng) if rng else 0):11.3f}"
              f"{np.mean(plast):12.3f}")
    print("\n  drift/birth      mean change in a concept's accuracy since it was consolidated")
    print("  drift (worked)   the same, over concepts that scored >0.5 at birth (a concept that")
    print("                   never worked cannot decay, and including it dilutes the signal)")
    print("  range            how far a single concept's accuracy moves across the run;")
    print("                   with frozen ancestors this must be 0.000")

    # pool -> consolidated fidelity, the other thing worth reading off
    print("\nconsolidation fidelity (pool accuracy -> accuracy of the consolidated concept):")
    for arm, blob in r.items():
        pairs = [(s["pool_acc"]["joint"], s["new_acc"]["joint"])
                 for s in blob["steps"] if "pool_acc" in s and "new_acc" in s]
        good = [(p, n) for p, n in pairs if p > 0.5]
        keep = float(np.mean([n / p for p, n in good])) if good else float("nan")
        print(f"  {arm:10s}" + "  ".join(f"{p:.2f}->{n:.2f}" for p, n in pairs[:6])
              + f"   | retained {keep:.0%} on pools that worked")

    for arm, blob in r.items():
        t = blob.get("transfer")
        if t:
            print(f"\n{arm}: distillation starts at {t['early_first']:.4f} (early concepts) / "
                  f"{t['late_first']:.4f} (late), ends {t['early_final']:.4f} / {t['late_final']:.4f}")


if __name__ == "__main__":
    main()
