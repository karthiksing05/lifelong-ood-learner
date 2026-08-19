"""Every field family must DECLARE its conventions, and the declaration must match what it does.

Four conventions coexist in this repository and they disagree:

    ScoreField            data  [0,1]   clip (-0.2, 1.2)   the analytic Cobweb manifold
    NeuralScoreField      model [-1,1]  clip (-1, 1)       our trained UNets
    PretrainedScoreField  model [-1,1]  clip (-1, 1)       diffusers UNet2DModel
    ve_bridge._Field      data  [0,1]   clip (0, 1)        the VE->VP Shapes3D bridge

The base class's historical default was `(-0.2, 1.2)` -- a DATA-range clamp, correct for exactly one
of the four, and inherited by the two it is most wrong for. Applied to a model-space trajectory it
clamps away the negative half at every reverse step. That fault has now been paid for twice: a
109-minute retrain on the multi-object backbone (VLA_PATH lesson 2-bis) and a day of CelebA numbers
(RESULTS 7ee), and in both cases the samples still looked like plausible images.

So the convention is now declared per class (`data_space`, optionally `clip_default`) and derived by
`score.default_clip` instead of defaulted. This file is what fails when a FIFTH family arrives
without declaring itself -- adding one is one entry in `FAMILIES`.

No GPU and no checkpoints: the registry uses stub models, so `run_all.py` picks this up for free.

    python tests/test_field_contract.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.instrument import assert_space, to_data_space       # noqa: E402
from lifelong_ood.score import CLIP_BY_SPACE, default_clip            # noqa: E402

FAIL = []


def ok(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


def families():
    """(name, class, expected_space, expected_clip). Classes, not instances -- no weights needed."""
    from lifelong_ood.neural.ddpm import NeuralScoreField
    from lifelong_ood.neural.pretrained import PretrainedScoreField
    from lifelong_ood.score import ScoreField
    return [
        ("ScoreField", ScoreField, "data", (-0.2, 1.2)),
        ("NeuralScoreField", NeuralScoreField, "model", (-1.0, 1.0)),
        ("PretrainedScoreField", PretrainedScoreField, "model", (-1.0, 1.0)),
    ]


class _Stub:
    """Stands in for a field instance; `default_clip` only reads the declared attributes."""

    def __init__(self, space, clip_default=None):
        self.data_space = space
        if clip_default is not None:
            self.clip_default = clip_default


def main():
    print("[contract] every family declares a space, and it resolves to the clip it already used")
    for name, cls, space, want_clip in families():
        got_space = getattr(cls, "data_space", None)
        ok(got_space == space, f"{name:22s} declares data_space={got_space!r}")
        got_clip = default_clip(_Stub(got_space))
        ok(tuple(got_clip) == want_clip,
           f"{name:22s} resolves to clip {got_clip} (was {want_clip} before this change)")

    print("\n[contract] a field may override when neither generic range fits")
    # The VE bridge is defined inside a closure, so it is checked by construction rather than by
    # import -- what matters is that an explicit `clip_default` wins over the space mapping.
    ve = _Stub("data", (0.0, 1.0))
    ok(tuple(default_clip(ve)) == (0.0, 1.0),
       "an explicit clip_default (0.0, 1.0) beats the generic data range (-0.2, 1.2)")
    ok(tuple(default_clip(_Stub("data"))) == CLIP_BY_SPACE["data"], "and without one, the space decides")
    src = open(os.path.join(HERE, "lifelong_ood", "neural", "ve_bridge.py"), encoding="utf-8").read()
    ok('self.data_space = "data"' in src and "self.clip_default = (0.0, 1.0)" in src,
       "ve_bridge._Field declares both (it is a closure class, so this is a source check)")

    print("\n[contract] the base sampler no longer hardcodes a range")
    base = open(os.path.join(HERE, "lifelong_ood", "score.py"), encoding="utf-8").read()
    ok('clip="auto"' in base, "BaseScoreField.sample takes clip='auto'")
    ok("clip = default_clip(self)" in base, "and resolves it from the field")
    ok("clip=(-0.2, 1.2)" not in base.split("CLIP_BY_SPACE")[1],
       "the (-0.2, 1.2) literal survives only inside CLIP_BY_SPACE")

    print("\n[contract] declared space and actual contents must agree")
    data_rows = np.random.default_rng(0).random((16, 48)).astype(np.float32)
    model_rows = data_rows * 2 - 1
    ok(assert_space(data_rows, "data").ok, "data rows pass a data assertion")
    ok(not assert_space(data_rows, "model").ok,
       "and FAIL a model assertion (no negative mass to speak of)")
    ok(assert_space(model_rows, "model").ok, "model rows pass a model assertion")
    ok(not assert_space(model_rows, "data").ok, "and fail a data one")

    print("\n[contract] to_data_space is the single conversion point")
    ok(np.allclose(to_data_space(_Stub("model"), model_rows), data_rows, atol=1e-5),
       "model-space rows convert back to the originals")
    ok(np.allclose(to_data_space(_Stub("data"), data_rows), data_rows, atol=1e-6),
       "data-space rows pass through untouched")

    print(f"\n[contract] {'PASS' if not FAIL else 'FAIL: ' + '; '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
