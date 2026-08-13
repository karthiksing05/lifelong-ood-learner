"""Can the Shapes3D factors be READ from pixels at all?  Run this before scoring anything with them.

RESULTS 5e-sexies is a retracted negative that was entirely a probe artifact: compositions were
scored through probes that could not read their own attribute, and one probe with a degenerate base
rate scored 1.000 by always answering "no".  The lesson is cheap to apply -- check the instrument
before trusting the measurement -- and Shapes3D needs it, because an early check at 4 images/slot put
`shape` at val 0.367 against a base rate of 0.292, i.e. unreadable, while both hues read fine.

Prints validation accuracy and majority-class base rate per factor at the settings n52 actually uses.
A factor that does not clear its base rate by a margin is EXCLUDED from `joint` by
`evals.probe_readable`, so if `shape` fails here then every Shapes3D `joint` is a two-factor joint and
must say so.

    python tests/diag_shapes3d_probes.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.evals import fit_factor_probes, probe_readable      # noqa: E402
from lifelong_ood.factored import load_factored                        # noqa: E402


def main():
    bad = []
    for n_per_slot, epochs in ((40, 30), (40, 80)):
        d = load_factored("shapes3d", n_per_slot=n_per_slot)
        pr = fit_factor_probes(d, epochs=epochs)
        print(f"\n[probe] n_per_slot={n_per_slot} epochs={epochs} "
              f"({int(d['seen'].sum())} seen images)")
        for nm in d["names"]:
            ok = probe_readable(pr, nm)
            print(f"[probe]   {nm:12s} val {pr[nm + '_val']:.3f}  base {pr[nm + '_base']:.3f}  "
                  f"{'READABLE' if ok else 'UNUSABLE -> excluded from joint'}", flush=True)
            if not ok and epochs == 80:
                bad.append(nm)
    if bad:
        print(f"\n[probe] {len(bad)} factor(s) unreadable at the strongest setting: {bad}")
        print("[probe] a softmax probe on raw pixels is the weak link, not necessarily the model. "
              "Any Shapes3D joint accuracy must state which factors it covers.")
    else:
        print("\n[probe] all factors readable -- joint accuracy covers the full composition")
    return 0


if __name__ == "__main__":
    sys.exit(main())
