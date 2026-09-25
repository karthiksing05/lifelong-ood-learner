"""Shapes3D images must come from the rows their labels describe.

`load_shapes3d` filters the label table (`fix_nuisance`, `value_stride`) and, until 2026-09-25, then
read `f["images"][keep]` with `keep` indexing the FILTERED table -- the wrong rows of the file.  This
pins the fix and documents what the legacy mapping (kept as the default so existing checkpoints load
the data they were trained on) actually delivered:

  * rows="correct": every attached label equals the label of the h5 row that is read, on every split;
  * rows="legacy", stride 1 + pinned wall (the flagship split): a consistent relabelling -- image floor
    always 0, the `floor_hue` label is the image's WALL hue, object/scale/shape/orientation exact;
  * rows="legacy", stride 2: scrambled -- object labels agree with the image at about chance.

Reads only the 23 MB label table (`labels_only=True`); seconds, no torch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lifelong_ood.data.shapes3d import _find, load_shapes3d    # noqa: E402


def main():
    try:
        p = _find()
    except FileNotFoundError:
        print("SKIP  3dshapes.h5 not present")
        return 0
    import h5py
    with h5py.File(p, "r") as f:
        L = f["labels"][:]
    true = np.stack([np.searchsorted(np.unique(L[:, k]), L[:, k]) for k in range(6)], 1)
    kw = dict(n_per_slot=40, seed=0, n_holdout=0, labels_only=True)
    splits = {"s1+pin": dict(value_stride=1, fix_nuisance=["wall_hue"]),
              "s2+pin": dict(value_stride=2, fix_nuisance=["wall_hue"]),
              "s2": dict(value_stride=2), "s1": dict(value_stride=1)}
    for name, sp in splits.items():
        c = load_shapes3d(rows="correct", **kw, **sp)
        assert np.array_equal(true[c["h5_rows"]], c["claimed"]), f"{name}: correct rows disagree"
    leg = load_shapes3d(rows="legacy", **kw, **splits["s1+pin"])
    act, cl = true[leg["h5_rows"]], leg["claimed"]
    assert np.all(act[:, 0] == 0), "legacy s1+pin: image floor should always be hue 0"
    assert np.array_equal(act[:, 1], cl[:, 0]), "legacy s1+pin: floor_hue label should be the WALL"
    assert np.array_equal(act[:, 2:], cl[:, 2:]), "legacy s1+pin: object/scale/shape/orient exact"
    leg2 = load_shapes3d(rows="legacy", **kw, **splits["s2"])
    obj_agree = float(np.mean(true[leg2["h5_rows"]][:, 2] == leg2["claimed"][:, 2]))
    assert obj_agree < 0.2, f"legacy s2 object agreement {obj_agree:.3f} -- expected ~chance"
    base = load_shapes3d(rows="legacy", **kw, **splits["s1"])
    assert np.array_equal(base["h5_rows"], load_shapes3d(rows="correct", **kw,
                                                          **splits["s1"])["h5_rows"])
    print(f"PASS  shapes3d rows: 'correct' matches labels on 4 splits; legacy s1+pin = floor label is "
          f"the wall (100%); legacy s2 object agreement {obj_agree:.3f} (chance 0.1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
