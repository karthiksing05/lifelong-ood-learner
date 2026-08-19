"""The instrument gate: prove the measuring device works BEFORE reporting anything measured with it.

Written 2026-08-19, after a day in which every number that looked good turned out to have been read
off an uncalibrated instrument. Three faults, all silent, all producing plausible numbers:

  1. MODEL-space [-1,1] samples scored by probes trained on DATA-space [0,1] images, and
     `field.sample` called at its default `clip=(-0.2,1.2)` -- a data-range clamp applied to a
     model-space trajectory at every reverse step. 49.3% of sample pixels were negative.
  2. A GLOBAL probe ceiling (the product of per-factor accuracies over ALL classes) used as the
     denominator when the held-out classes were the rarest and hence least readable ones. It read
     0.524 where real images of the held-out cells read 0.142 against a chance of 0.143.
  3. An in-sample basis R^2 quoted as evidence on a table with barely more classes than free
     vectors: 0.875 in-sample against 0.207 leave-one-out.

and a fourth that only became visible while designing this module:

  4. VERDICT THRESHOLDS BELOW THE MEASUREMENT'S RESOLUTION. At n=24 samples and p~0.3 the smallest
     margin the arm can resolve is 2*sqrt(p(1-p)/n) = 0.187, and the experiments declare verdicts at
     +0.10. Several verdicts fired below the noise floor of their own sample size.

The fixes existed as one-off `tests/diag_*.py` files, which `tests/run_all.py` never runs -- it
globs `test_*.py` only. This module makes them one thing an experiment calls before it measures.

DESIGN RULES, each paid for:

  * REPORT NUMBERS, NEVER A BARE VERDICT. Every check carries the value and the bar it was judged
    against, so a reader can disagree with the bar.
  * REFUSAL LIVES AT THE EXPERIMENT BOUNDARY, not inside `evals`. A metric that refuses to compute
    makes the broken case unmeasurable, and diagnostics have to be able to measure the broken case.
  * THE FLOOR IS EMPIRICAL, NEVER `1/n_classes`. Chance ignores probe bias: a probe that always
    answers "brown hair" has a floor well above chance, and 1/n would hide exactly that.
  * THE TABLE CHECK NEVER FAILS A RUN. It gates whether an IN-SAMPLE R^2 may be quoted, nothing
    else; a global ratio bar would kill n64's legitimate basis claim on its own table.

    from lifelong_ood.instrument import check_instrument
    rep = check_instrument(tag="n73", real=d, probes=probes, field=field, targets=..., n_arm=24)
    rep.require("n73")                      # exits if conventions are wrong or nothing is readable
    for h in held:
        if not rep.readable(h):
            continue                        # not sampled, not scored; the reason is in the JSON
"""
from dataclasses import dataclass, field as _dcfield

import numpy as np

# The strict readability margin -- the bar `tests/diag_grid_*.py` already use and call "the bar used
# elsewhere". New claims must clear this one.
MARGIN = 0.10
# What `evals.probe_readable` and n73/n75 use TODAY. NOT changed here: raising it would move the
# pinned `n75_grid_pooled/coverage/n_readable = 16.0` and every readable-only mean with it. Both are
# reported so the divergence is a recorded number rather than an inconsistency.
MARGIN_LEGACY = 0.02
# Below this, no arm can be ranked against another whatever it scores.
MIN_DYNAMIC_RANGE = 0.10
# n73's number. At 200 the readability decision has a standard error near 0.035; at 24 it is 0.09,
# and this number decides whether a whole class is scored at all.
N_GATE = 200
# Classes-per-free-vector below which an in-sample R^2 is mostly arithmetic (measured: ColorMNIST at
# 6.3 gives 0.826/0.772 in-sample/CV, the CelebA grid at 1.1 gives 0.875/0.207).
MIN_RATIO = 3.0


@dataclass(frozen=True)
class Check:
    """One measured judgement. `detail` always carries the competing numbers, never just a verdict."""
    name: str
    ok: bool
    value: float
    bar: float
    detail: dict = _dcfield(default_factory=dict)

    def line(self):
        return (f"{'ok ' if self.ok else 'FAIL'} {self.name:28s} {self.value:9.4f}  bar {self.bar:.4f}"
                + (f"  {self.detail}" if self.detail else ""))


@dataclass
class TargetGate:
    """Whether one held-out target can be measured at all, and by how much."""
    key: object
    target: tuple
    ceiling: float          # REAL images of THIS target, scored by THIS run's probes
    n_gate: int
    floor: float            # the same probes on real images of OTHER (seen) classes
    chance: float           # reported for orientation; never used as the bar
    dynamic_range: float
    min_detectable: float   # the smallest margin `n_arm` samples can resolve at all
    per_factor: dict
    readable_strict: bool
    readable_legacy: bool
    space: str = "pixel"

    def as_dict(self):
        d = dict(self.__dict__)
        d["key"] = int(self.key) if isinstance(self.key, (int, np.integer)) else str(self.key)
        d["target"] = [int(v) for v in self.target]
        return d


@dataclass
class Report:
    ok: bool
    checks: list
    targets: dict
    table: dict
    meta: dict

    def readable(self, key, strict=False):
        """Default `strict=False` reproduces n73/n75's legacy bar; pass True for the +0.10 bar."""
        g = self.targets.get(key)
        if g is None:
            return False
        return g.readable_strict if strict else g.readable_legacy

    def scored(self, strict=False):
        return [k for k in self.targets if self.readable(k, strict)]

    def failed(self, channel=None):
        return [c for c in self.checks
                if not c.ok and (channel is None or c.name.startswith(channel))]

    def require(self, tag, channels=("convention", "readout")):
        """Refuse to proceed. Prints every failing check's NUMBERS, then exits non-zero.

        Asymmetric on purpose: a convention fault makes everything downstream meaningless, but a
        partially-blind readout only removes the blind targets -- which is how n73's 16-of-36 was
        produced and how it should keep being produced.
        """
        bad = [c for c in self.checks if not c.ok and c.name.split(":")[0] in channels]
        if not bad:
            return
        print(f"\n[{tag}] INSTRUMENT GATE FAILED -- refusing to report numbers measured this way.")
        for c in bad:
            print(f"[{tag}]   {c.line()}")
        if any(c.name.startswith("readout") for c in bad):
            print(f"[{tag}]   No target is readable: real images of every held-out class score at "
                  f"the floor under this run's own probes, so NO arm can be judged, ours included.")
        raise SystemExit(2)

    def summary(self, tag=""):
        out = [f"[{tag}] instrument: {'PASS' if self.ok else 'FAIL'} "
               f"({len(self.scored())} of {len(self.targets)} targets readable at the legacy bar, "
               f"{len(self.scored(strict=True))} at +{MARGIN:.2f})"]
        for c in self.checks:
            out.append(f"[{tag}]   {c.line()}")
        if self.table:
            t = self.table
            out.append(f"[{tag}]   table {t['n_classes']} classes / {t['n_free']} free vectors = "
                       f"ratio {t['ratio']:.1f}; R2 {t['r2']:.3f} in-sample, {t['r2_cv']:.3f} "
                       f"leave-one-out{'' if t['quotable'] else '  <- in-sample R2 NOT quotable'}")
        return "\n".join(out)

    def as_dict(self):
        return dict(ok=bool(self.ok), meta=self.meta, table=self.table,
                    checks=[dict(name=c.name, ok=bool(c.ok), value=float(c.value),
                                 bar=float(c.bar), detail=c.detail) for c in self.checks],
                    targets={str(k): g.as_dict() for k, g in self.targets.items()})


# ── the space assertion: fault #1, catchable from arrays alone ───────────────
def assert_space(X, space, name="", tol=0.02):
    """Does this array live where the caller thinks it does?  -> Check.

    On the arrays n73 was scoring, this reports `frac_negative = 0.493` -- the exact fingerprint of
    a model-space sample handed to a data-space probe -- with no GPU and no model.
    """
    X = np.asarray(X, np.float64)
    lo, hi, mean = float(X.min()), float(X.max()), float(X.mean())
    frac_neg = float((X < -1e-6).mean())
    detail = dict(min=round(lo, 4), max=round(hi, 4), mean=round(mean, 4),
                  frac_negative=round(frac_neg, 4), declared=space)
    if space == "model":
        # a model-space array must actually USE the negative half, and must not overflow it
        ok = frac_neg > 0.01 and lo >= -1.05 and hi <= 1.05
        val = frac_neg
    elif space == "data":
        inside = float(((X >= -tol) & (X <= 1.0 + tol)).mean())
        ok = inside >= 0.98
        val = inside
        detail["frac_inside_01"] = round(inside, 4)
    else:
        raise ValueError(f"unknown space {space!r}")
    return Check(f"space:{name or space}", bool(ok), float(val), 0.98 if space == "data" else 0.01,
                 detail)


def to_data_space(field, X):
    """Convert a field's sampler output to DATA space [0,1], whatever that field's convention is.

    The single place `to_data` should be called. Every experiment that scores samples against probes
    goes through here, so the conversion cannot be forgotten at one call site out of thirty.
    """
    from .neural.ddpm import to_data
    space = getattr(field, "data_space", "model")
    X = np.asarray(X, np.float32)
    return np.asarray(to_data(X), np.float32) if space == "model" else np.clip(X, 0.0, 1.0)


# ── conventions ─────────────────────────────────────────────────────────────
def convention_probe(field, X01, img=None, u=0.3, n=4, seed=0):
    """Measure a backbone's layout and range from data, rather than trusting its declaration.

    The arithmetic is lifted from `tests/diag_backbone_conventions.py`, which stays as the
    human-facing per-checkpoint driver -- it holds the institutional memory for three results lost to
    this fault class, and a diag a person runs against a checkpoint is a different job from a
    function an experiment calls.
    """
    import torch

    from .score import cosine_abar

    img = int(img or getattr(field, "img", 64))
    X = np.asarray(X01, np.float32)[:8]
    dev = field.device

    def loss(x_img):
        g = torch.Generator(device=dev).manual_seed(seed)
        a = float(cosine_abar(np.array([u]))[0])
        tot = 0.0
        for _ in range(n):
            e = torch.randn(x_img.shape, generator=g, device=dev)
            xt = (a ** 0.5) * x_img + ((1 - a) ** 0.5) * e
            uu = torch.full((len(x_img),), float(u), device=dev)
            tot += float(((field.model(xt, uu) - e) ** 2).mean())
        return tot / n

    t = torch.as_tensor(X, device=dev)
    ch = X.shape[1] // (img * img)
    hwc = t.reshape(len(X), img, img, ch).permute(0, 3, 1, 2).contiguous()
    chw = t.reshape(len(X), ch, img, img).contiguous()
    l_hwc, l_chw = loss(hwc * 2 - 1), loss(chw * 2 - 1)
    l_01, l_11 = loss(hwc), loss(hwc * 2 - 1)
    return dict(layout="hwc" if l_hwc < l_chw else "permuted",
                range="[0,1]" if l_01 < l_11 else "[-1,1]",
                l_hwc=round(l_hwc, 5), l_chw=round(l_chw, 5),
                l_01=round(l_01, 5), l_11=round(l_11, 5))


# ── readout ceiling, per target ─────────────────────────────────────────────
def readout_ceiling(probes, X, factors, target, n_gate=N_GATE, exclude=None, space="pixel"):
    """What REAL images of THIS target score under THESE probes.  -> dict.

    Never the global product of per-factor validation accuracies: that read 0.524 on a backbone whose
    held-out cells read 0.142 against a chance of 0.143, because held-out combinations are the rarest
    ones and therefore the ones the probes are worst at.
    """
    from .evals import factor_accuracy

    F = np.asarray(factors, int)
    tgt = np.asarray(target, int)
    hit = np.all(F == tgt[None, :], axis=1)
    idx = np.where(hit)[0]
    if exclude is not None:
        idx = np.array([i for i in idx if i not in set(np.asarray(exclude).tolist())], int)
    if not len(idx):
        return dict(ceiling=float("nan"), n=0, per_factor={})
    idx = idx[:n_gate]
    a = factor_accuracy(probes, np.asarray(X, np.float32)[idx], tuple(int(v) for v in tgt))
    j = float(a.get("joint", float("nan")))
    return dict(ceiling=0.0 if j != j else j, n=int(len(idx)),
                per_factor={k: float(v) for k, v in a.items()
                            if k not in ("joint", "n_usable")}, space=space)


def empirical_floor(probes, X, factors, target, seen_mask=None, n=N_GATE, seed=0):
    """What real images of OTHER classes score against this target -- the floor, measured.

    `1/n_classes` is the same class of mistake as the global ceiling: it ignores probe bias. A probe
    that always answers "brown hair" gives a floor well above chance, and chance would hide it.
    """
    from .evals import factor_accuracy

    F = np.asarray(factors, int)
    tgt = np.asarray(target, int)
    other = ~np.all(F == tgt[None, :], axis=1)
    if seen_mask is not None:
        other &= np.asarray(seen_mask, bool)
    idx = np.where(other)[0]
    if not len(idx):
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)[:n]
    a = factor_accuracy(probes, np.asarray(X, np.float32)[idx], tuple(int(v) for v in tgt))
    j = float(a.get("joint", float("nan")))
    return 0.0 if j != j else j


def min_detectable(p, n):
    """The smallest margin `n` samples can resolve at all: 2 * sqrt(p(1-p)/n).

    At n=24 and p=0.3 this is 0.187 -- LARGER than the +0.10 thresholds n73/n75 use to declare a
    verdict. An experiment whose verdict bar is below this number is reporting noise, and should
    either raise n or lower the claim.
    """
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    return float(2.0 * np.sqrt(p * (1 - p) / max(int(n), 1)))


# ── the table ───────────────────────────────────────────────────────────────
def table_report(C, labels, ridge=1e-3, min_ratio=MIN_RATIO, n_perm=200, seed=0):
    """Is this class table big enough for an in-sample R^2 to mean anything?  -> dict.

    Never fails a run. It decides whether `r2` may be QUOTED; `n64`'s basis claim is legitimate on
    its own table and a global ratio bar would kill it.
    """
    from .neural.idealwords import cv_r2, factorize

    C = np.asarray(C, np.float64)
    L = np.asarray(labels, int)
    fac = factorize(C, L, ridge)
    sizes = [int(L[:, f].max()) + 1 for f in range(L.shape[1])]
    n_free = 1 + sum(s - 1 for s in sizes)          # after centring, per cv_r2's docstring
    r2cv, n_cv = cv_r2(C, L, ridge)
    rng = np.random.default_rng(seed)
    null = [factorize(C, L[rng.permutation(len(L))], ridge)["r2"] for _ in range(int(n_perm))]
    ratio = len(C) / max(n_free, 1)
    return dict(n_classes=int(len(C)), n_free=int(n_free), ratio=float(ratio),
                r2=float(fac["r2"]), r2_cv=float(r2cv), n_cv=int(n_cv),
                perm_null=float(np.mean(null)), quotable=bool(ratio >= min_ratio))


# ── the entry point ─────────────────────────────────────────────────────────
def check_instrument(*, tag, real, probes, targets=None, field=None, n_arm=24, n_gate=N_GATE,
                     exclude=None, table=None, labels=None, expect_layout=None, expect_range=None,
                     space="pixel", seen_mask=None, seed=0):
    """Prove the instrument before measuring with it.  -> Report.

    `real` is the `data/factored.py` contract dict (X, factors, names, sizes, seen, slot, img), so
    this works unchanged on ColorMNIST, CelebA, the CelebA grid and Shapes3D.
    """
    X = np.asarray(real["X"], np.float32)
    F = np.asarray(real["factors"], int)
    seen = np.asarray(real["seen"], bool) if seen_mask is None else np.asarray(seen_mask, bool)
    n_cls = int(len(np.unique(np.asarray(real["slot"])[seen]))) if "slot" in real else 0
    chance = 1.0 / max(n_cls, 1)
    checks = []

    # (i) conventions -- only when a field is supplied
    if field is not None:
        conv = convention_probe(field, X, img=real.get("img"))
        want_layout = expect_layout or getattr(field, "layout", None)
        want_range = expect_range or ("[-1,1]" if getattr(field, "data_space", "model") == "model"
                                      else "[0,1]")
        if want_layout:
            checks.append(Check("convention:layout", conv["layout"] == want_layout,
                                1.0 if conv["layout"] == want_layout else 0.0, 1.0,
                                dict(measured=conv["layout"], declared=want_layout,
                                     l_hwc=conv["l_hwc"], l_chw=conv["l_chw"])))
        checks.append(Check("convention:range", conv["range"] == want_range,
                            1.0 if conv["range"] == want_range else 0.0, 1.0,
                            dict(measured=conv["range"], declared=want_range,
                                 l_01=conv["l_01"], l_11=conv["l_11"])))
        # the check the existing diag is missing, and the one that IS fault #1: does the sampler's
        # own output land where the field says it does?
        try:
            S = field.sample(n=4, steps=20, seed=seed)
            checks.append(assert_space(S, getattr(field, "data_space", "model"), "sampler_output"))
        except Exception as e:                                       # noqa: BLE001
            checks.append(Check("convention:sampler", False, 0.0, 1.0, dict(error=repr(e)[:200])))

    # (ii)-(iii) per-target readability and dynamic range
    if targets is None:
        held = np.unique(np.asarray(real["slot"])[~seen]) if "slot" in real else []
        targets = [tuple(int(v) for v in F[np.asarray(real["slot"]) == h][0]) for h in held]
        keys = [int(h) for h in held]
    elif isinstance(targets, dict):
        # {class id -> factor row}: the caller's own keys are kept, so a report can be indexed by
        # the cell id the experiment already uses. This branch has to come BEFORE the sequence one,
        # or iterating the dict yields its keys and the tuple conversion walks integers.
        keys = list(targets)
        targets = [tuple(int(v) for v in t) for t in targets.values()]
    else:
        targets = [tuple(int(v) for v in t) for t in targets]
        keys = list(range(len(targets)))

    gates = {}
    for k, t in zip(keys, targets):
        c = readout_ceiling(probes, X, F, t, n_gate=n_gate, exclude=exclude, space=space)
        fl = empirical_floor(probes, X, F, t, seen_mask=seen, n=n_gate, seed=seed)
        fl = 0.0 if fl != fl else fl
        dr = c["ceiling"] - fl
        gates[k] = TargetGate(
            key=k, target=t, ceiling=c["ceiling"], n_gate=c["n"], floor=fl, chance=chance,
            dynamic_range=dr, min_detectable=min_detectable(max(c["ceiling"], 0.05), n_arm),
            per_factor=c["per_factor"],
            readable_strict=bool(dr >= MIN_DYNAMIC_RANGE and c["ceiling"] > fl + MARGIN),
            readable_legacy=bool(c["ceiling"] > chance + MARGIN_LEGACY), space=space)

    n_ok = sum(1 for g in gates.values() if g.readable_legacy)
    checks.append(Check("readout:any_readable", n_ok > 0, float(n_ok), 1.0,
                        dict(readable_legacy=n_ok,
                             readable_strict=sum(1 for g in gates.values() if g.readable_strict),
                             total=len(gates))))
    if gates:
        md = float(np.mean([g.min_detectable for g in gates.values()]))
        checks.append(Check("readout:resolution", md <= MARGIN, md, MARGIN,
                            dict(n_arm=int(n_arm),
                                 note="smallest margin this sample count can resolve; a verdict "
                                      "threshold below it is reporting noise")))

    # (iv) the table -- reported, never fatal
    tbl = None
    if table is not None and labels is not None:
        tbl = table_report(table, labels)

    ok = all(c.ok for c in checks if c.name.split(":")[0] in ("convention", "readout"))
    meta = dict(tag=tag, space=space, n_arm=int(n_arm), n_gate=int(n_gate), seed=int(seed),
                margin=MARGIN, margin_legacy=MARGIN_LEGACY,
                min_dynamic_range=MIN_DYNAMIC_RANGE, n_classes=n_cls, chance=chance)
    return Report(ok=bool(ok), checks=checks, targets=gates, table=tbl, meta=meta)
