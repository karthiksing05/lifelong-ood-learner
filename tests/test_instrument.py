"""The gate is a measuring device, so it is tested against instruments whose answers are known.

A gate that is merely plausible is worth nothing -- it would join the four verdicts that already
fired on metrics which could not discriminate. So every check here has a ground truth:

  * a factor whose labels are NOISE must read unreadable;
  * a factor that is a linear function of a pixel must read readable;
  * a target whose real images score at the floor must be refused, and must take `Report.ok` with it
    when it is the only target;
  * the n73 fault -- a [-1,1] array handed to [0,1] probes -- must be caught FROM ARRAYS ALONE, and
    the wrong number it would have produced printed next to the right one;
  * a table with as many free vectors as classes must be marked non-quotable while a
    ColorMNIST-shaped one is quotable.

No GPU and no checkpoint: everything here is numpy, so `tests/run_all.py` picks it up for free.

    python tests/test_instrument.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from lifelong_ood.evals import fit_factor_probes                      # noqa: E402
from lifelong_ood.instrument import (MARGIN, MARGIN_LEGACY, assert_space,  # noqa: E402
                                     check_instrument, min_detectable,
                                     table_report)

IMG = 8
D = IMG * IMG * 3
FAIL = []


def ok(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAIL.append(msg)


def make_data(n_per=60, noise_factor=False, seed=0):
    """A tiny factored dataset whose two factors are INDEPENDENTLY and linearly readable.

    factors: [colour(3), bright(2)]. Colour is which channel is raised; brightness is a global
    offset. They must be independent, or a "held-out combination" is unreadable for a legitimate
    reason and the fixture, not the gate, is what fails -- which is exactly what the first version of
    this file did (brightness encoded inside the colour channel, so the joint read 0.000 on the
    held-out cell and looked like a gate bug).

    `noise_factor=True` scrambles `bright`'s labels so that factor becomes genuinely unreadable.
    """
    rng = np.random.default_rng(seed)
    rows, F = [], []
    for c in range(3):
        for b in range(2):
            for _ in range(n_per):
                x = np.full((IMG, IMG, 3), 0.10, np.float32)
                # SPATIALLY DISJOINT, so a linear probe trained without this combination still
                # generalises to it. A global brightness offset does not: the probe keys on
                # "channel 0 or 1 raised AND bright", and the held-out (colour=2, bright=1) then
                # reads 0.000 for a legitimate reason -- a fixture failure that looks like a gate
                # failure, which is the whole thing this file exists to distinguish.
                x[:IMG // 2, :, c] += 0.55                  # colour lives in the top half
                x[IMG // 2:, :, :] += 0.45 * b              # brightness lives in the bottom half
                x += rng.normal(0, 0.02, x.shape).astype(np.float32)
                rows.append(np.clip(x, 0, 1).ravel())
                F.append([c, b])
    X = np.asarray(rows, np.float32)
    F = np.asarray(F, int)
    slot = F[:, 0] * 2 + F[:, 1]
    if noise_factor:
        F = F.copy()
        F[:, 1] = rng.integers(0, 2, len(F))               # labels now carry no signal
    seen = slot != 5                                        # hold out (colour=2, bright=1)
    return dict(X=X, factors=F, names=["colour", "bright"], sizes=[3, 2],
                slot=slot, seen=seen, img=IMG)


def main():
    print("[instrument] readability, against factors whose answer is known")
    d = make_data()
    probes = fit_factor_probes(d, seed=0)
    ok(probes["colour_val"] > 0.9 and probes["bright_val"] > 0.9,
       f"both probes validate near 1.0 ({probes['colour_val']:.3f}, {probes['bright_val']:.3f})")

    rs = check_instrument(tag="t", real=d, probes=probes, targets=[(0, 0)], n_arm=24, n_gate=120)
    gs = rs.targets[0]
    ok(gs.ceiling > 0.9, f"a SEEN target reads high (ceiling {gs.ceiling:.3f})")
    ok(gs.floor < gs.ceiling, f"the empirical floor sits below it ({gs.floor:.3f})")
    ok(gs.readable_legacy and gs.readable_strict, "and it passes both bars")
    ok(rs.ok, "so the report passes")

    # THE EFFECT THE GATE EXISTS FOR, and it is not hypothetical: probes trained on a design that
    # is MISSING one combination exploit the correlation between factors, so the held-out cell reads
    # far worse than the validation accuracy advertises. Here `colour` validates at 1.000 and then
    # predicts colour=0 for every held-out (colour=2, bright=1) image. This is the mechanism behind
    # RESULTS 7ee -- a global ceiling of 0.524 over cells that themselves read 0.142 -- reproduced
    # in eight lines of numpy.
    rh = check_instrument(tag="t", real=d, probes=probes, targets=[(2, 1)], n_arm=24, n_gate=120)
    gh = rh.targets[0]
    ok(gh.ceiling < gs.ceiling - 0.3,
       f"the HELD-OUT combination reads far worse ({gh.ceiling:.3f} vs {gs.ceiling:.3f}) "
       f"despite both probes validating near 1.0")
    ok(not gh.readable_strict,
       "so the gate refuses it -- which a global per-factor ceiling would never have caught")
    ok(gh.per_factor.get("bright", 0) > 0.9 and gh.per_factor.get("colour", 1) < 0.5,
       f"and it names the factor that failed: {gh.per_factor}")

    dn = make_data(noise_factor=True, seed=1)
    pn = fit_factor_probes(dn, seed=0)
    # A noise-labelled factor does NOT drag the joint down -- `probe_readable` drops it and the
    # joint is then computed over the survivors, so it goes UP. That is the correct behaviour and
    # the trap: a rising joint can mean the instrument narrowed, not that the model improved. What
    # must be visible is that a factor was dropped.
    ok(pn["bright_val"] <= pn["bright_base"] + 0.10,
       f"the noise factor's probe cannot beat its base rate "
       f"({pn['bright_val']:.3f} vs {pn['bright_base']:.3f})")
    ok(probes["bright_val"] > probes["bright_base"] + 0.10,
       f"while the real one can ({probes['bright_val']:.3f} vs {probes['bright_base']:.3f})")
    from lifelong_ood.evals import factor_accuracy
    n_clean = factor_accuracy(probes, d["X"][:24], (0, 0))["n_usable"]
    n_noise = factor_accuracy(pn, dn["X"][:24], (0, 0))["n_usable"]
    ok(n_noise < n_clean,
       f"and the joint is computed over fewer factors ({n_noise} vs {n_clean}) -- a joint that "
       f"RISES because the instrument narrowed")

    print("\n[instrument] targets may be given as a dict keyed by the caller's own class ids")
    rd = check_instrument(tag="t", real=d, probes=probes, targets={7: (0, 0), 9: (1, 1)}, n_arm=24)
    ok(set(rd.targets) == {7, 9}, "the report is indexed by the caller's keys, not by position")
    ok(rd.targets[7].target == (0, 0), "and each keeps its factor row")

    print("\n[instrument] a blind target takes the report down with it")
    db = make_data()
    # relabel the held-out target to a combination no real image carries
    tgt_absent = (2, 1)
    db["factors"] = db["factors"].copy()
    db["factors"][db["slot"] == 5] = [0, 0]                # the held-out cell is now mislabelled
    rb = check_instrument(tag="t", real=db, probes=probes, targets=[tgt_absent], n_arm=24)
    ok(not rb.ok, "a target with no real images at all fails the readout check")
    try:
        rb.require("t")
        ok(False, "require() should have exited")
    except SystemExit as e:
        ok(int(getattr(e, "code", 1)) == 2, "require() exits 2 and prints the numbers")

    print("\n[instrument] the n73 fault, caught from arrays alone")
    data_rows = d["X"][:32]
    model_rows = data_rows * 2.0 - 1.0                     # what the sampler actually returns
    c_bad = assert_space(model_rows, "data", "n73_samples")
    c_good = assert_space(model_rows, "model", "n73_samples")
    ok(not c_bad.ok, f"model-space rows FAIL a data-space assertion ({c_bad.detail})")
    ok(c_good.ok, "and pass a model-space one")
    ok(c_bad.detail["frac_negative"] > 0.3,
       f"the fingerprint is the negative mass: {c_bad.detail['frac_negative']:.3f}")
    # and the number it would have produced, next to the right one
    from lifelong_ood.evals import factor_accuracy
    t0 = tuple(int(v) for v in d["factors"][0])
    j_wrong = factor_accuracy(probes, model_rows[:24], t0)["joint"]
    j_right = factor_accuracy(probes, data_rows[:24], t0)["joint"]
    print(f"       joint would have read {j_wrong:.3f} in the wrong space vs {j_right:.3f} correct")
    ok(True, "both numbers reported rather than one asserted")

    print("\n[instrument] resolution: a verdict bar below the noise floor")
    md24 = min_detectable(0.3, 24)
    ok(md24 > MARGIN,
       f"at n=24, p=0.3 the smallest resolvable margin is {md24:.3f} > the {MARGIN:.2f} bar")
    ok(min_detectable(0.3, 400) < MARGIN, "and n=400 brings it under the bar")

    print("\n[instrument] table ratio decides only whether in-sample R2 is quotable")
    rng = np.random.default_rng(0)
    lab_small = np.array([[a, b] for a in range(3) for b in range(2)])
    C_small = rng.normal(size=(len(lab_small), 32))         # random rows, no additive structure
    t_small = table_report(C_small, lab_small, n_perm=40)
    ok(not t_small["quotable"],
       f"6 classes / {t_small['n_free']} free vectors -> ratio {t_small['ratio']:.1f}, not quotable")
    ok(t_small["r2"] > t_small["r2_cv"],
       f"and in-sample {t_small['r2']:.3f} flatters vs leave-one-out {t_small['r2_cv']:.3f}")

    lab_big = np.array([[a, b, c] for a in range(10) for b in range(4) for c in range(4)])
    mu = rng.normal(size=32)
    V = [rng.normal(size=(s, 32)) for s in (10, 4, 4)]
    C_big = np.stack([mu + V[0][a] + V[1][b] + V[2][c] for a, b, c in lab_big])
    t_big = table_report(C_big, lab_big, n_perm=40)
    ok(t_big["quotable"], f"120 classes / {t_big['n_free']} -> ratio {t_big['ratio']:.1f}, quotable")
    ok(t_big["r2_cv"] > 0.9, f"and a truly additive table cross-validates ({t_big['r2_cv']:.3f})")
    ok(t_big["perm_null"] < 0.5, f"against a permutation null of {t_big['perm_null']:.3f}")

    print("\n[instrument] the two bars are both reported and have not been silently unified")
    ok(MARGIN > MARGIN_LEGACY, f"strict {MARGIN} > legacy {MARGIN_LEGACY}")
    from lifelong_ood import evals
    src = open(evals.__file__, encoding="utf-8").read()
    ok("b + 0.02" in src,
       "evals.probe_readable still uses +0.02 (changing it would move n75's pinned n_readable)")

    print(f"\n[instrument] {'PASS' if not FAIL else 'FAIL: ' + '; '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
