"""The unified factored-dataset contract must be a DROP-IN, not a parallel truth.

`lifelong_ood.factored` exists so that Shapes3D and CLEVR are a loader rather than a refactor. That
is only safe if its generic encoding agrees exactly with the per-dataset arithmetic already baked
into the repository -- `idealwords.slot_factors` decodes ColorMNIST slots as (s // 16, (s % 16) // 4,
s % 4), and 39 call sites read `data["slot"]`. If the generic version disagreed anywhere, adopting it
would silently relabel every experiment.

Also asserts the property the whole three-tier argument rests on: a held-out COMBINATION is a new
arrangement of known values, so a split must never accidentally remove a value entirely -- that would
turn it into new VOCABULARY and quietly change what the benchmark measures.

    python tests/test_factored.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelong_ood.datasets import load_colormnist                              # noqa: E402
from lifelong_ood.factored import (as_factored, decode_slot, encode_slot,      # noqa: E402
                                   holdout_mask)
from lifelong_ood.neural.idealwords import slot_factors                        # noqa: E402


def main():
    fails = []
    d = as_factored(load_colormnist(n_per_slot=4, seed=0))

    # 1. the generic encoding must reproduce the dataset's own slot ids
    mine = encode_slot(d["factors"], d["sizes"])
    if not np.array_equal(mine, np.asarray(d["slot"])):
        fails.append("encode_slot does not reproduce data['slot']")

    # 2. and must agree with the hardcoded decode the neural stack already uses
    slots = np.unique(np.asarray(d["slot"]))
    if not np.array_equal(decode_slot(slots, d["sizes"]), np.asarray(slot_factors(slots))):
        fails.append("decode_slot disagrees with idealwords.slot_factors")

    # 3. round trip
    if not np.array_equal(decode_slot(mine, d["sizes"]), d["factors"]):
        fails.append("decode(encode(x)) != x")

    # 4. a held-out split must not starve any factor VALUE (combination, not vocabulary)
    for n_hold in (10, 40, 60):
        seen, held = holdout_mask(d["factors"], d["sizes"], n_holdout=n_hold, seed=0)
        F = d["factors"][seen]
        for f in range(F.shape[1]):
            if len(np.unique(F[:, f])) != d["sizes"][f]:
                fails.append(f"holdout of {n_hold} starved a value of factor {f}")
        if len(held) != n_hold:
            fails.append(f"asked for {n_hold} held-out combinations, got {len(held)}")

    # 5. the CelebA shape must map too -- binary attributes are factors of size two
    fake = dict(X=np.zeros((6, 12288), np.float32),
                attrs=np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1],
                                [1, 0, 1], [0, 1, 1], [1, 0, 0]]))
    g = as_factored(fake)
    if g["sizes"] != [2, 2, 2] or g["factors"].shape != (6, 3):
        fails.append(f"binary-attribute mapping wrong: sizes={g['sizes']}, "
                     f"factors={g['factors'].shape}")

    # 6. SHAPES3D -- the third dataset, and the real test of "a loader, not a refactor".
    #    Skipped rather than failed when the 255 MB h5 is absent, so the suite still runs on a box
    #    that has not fetched it; the skip is PRINTED, because a silent skip is how a contract test
    #    stops testing anything.
    n_checks = 5
    try:
        from lifelong_ood.factored import load_factored
        sh = load_factored("shapes3d", n_per_slot=4)
        n_checks = 6
        need = ("X", "factors", "names", "sizes", "slot", "seen", "img")
        missing = [k for k in need if k not in sh]
        if missing:
            fails.append(f"shapes3d missing contract keys {missing}")
        else:
            if not np.array_equal(encode_slot(sh["factors"], sh["sizes"]),
                                  np.asarray(sh["slot"])):
                fails.append("shapes3d: encode_slot does not reproduce its slot ids")
            if not np.array_equal(decode_slot(np.asarray(sh["slot"]), sh["sizes"]),
                                  sh["factors"]):
                fails.append("shapes3d: decode(encode(x)) != x")
            if sh["X"].ndim != 2 or sh["X"].shape[1] != sh["img"] ** 2 * 3:
                fails.append(f"shapes3d: X is {sh['X'].shape}, expected flat rows of "
                             f"{sh['img'] ** 2 * 3}")
            if float(sh["X"].min()) < 0.0 or float(sh["X"].max()) > 1.0:
                fails.append("shapes3d: X outside [0, 1]")
            # the split guarantee has to hold on 3 factors of DIFFERENT sizes than ColorMNIST's
            seen, held = holdout_mask(sh["factors"], sh["sizes"], n_holdout=20, seed=0)
            F = sh["factors"][seen]
            for f_ in range(F.shape[1]):
                if len(np.unique(F[:, f_])) != sh["sizes"][f_]:
                    fails.append(f"shapes3d: holdout starved a value of factor {f_}")
            # and every composition must have MANY images, which is why nuisance factors exist
            cnt = np.bincount(np.asarray(sh["slot"]))
            if cnt[cnt > 0].min() < 2:
                fails.append("shapes3d: a composition has <2 images; nuisance split is wrong")
    except FileNotFoundError as e:
        print(f"[factored] SKIP shapes3d: {str(e).splitlines()[0]}")
    except ImportError as e:
        print(f"[factored] SKIP shapes3d: {e}")

    print(f"[factored] {n_checks - len(set(f.split()[0] for f in fails))}/{n_checks} checks passed"
          if fails else f"[factored] {n_checks}/{n_checks} checks passed")
    for f in fails:
        print(f"  FAIL  {f}")
    print("[factored] " + ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
