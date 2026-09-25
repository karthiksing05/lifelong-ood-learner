"""COMPOSE for a policy: ONE demonstration of a new task becomes a program over the prototype library, and the
frozen policy is sampled through the product of experts of the prototypes the program names (Zekun's Eq. 8-10,
14-16 in chunk space, on the flow path).

    prog = DemoProgram.from_demo(tree, chunk_rows, states, K=3, levels=(0, 1, 2), variant="code")
    hooks = ComposeHooks(prog, w0=1.2)                   # GuidedPolicy(backbone, hooks=hooks)

A program stores, per demo frame, the CODE of its action chunk (K node ids + weights, 4 B each) and the
alignment key (the demo's eef position + finger qpos, fp16).  At rollout the current state is aligned to the
demo (nearest frame in a forward window), and that frame's product of experts N(mu, var) guides the sampler:
eps <- eps_policy + w (eps_teacher - eps_policy), w by Eq. 16, so the prototype sets the coarse structure (high
noise) and the policy the details (low noise).

Variants: "code" (prototype PoE, the method), "raw" (the demo chunk itself with a floor variance: retrieval
guidance, the upper bound on what a code can carry), and "none" (the policy alone: zero-shot).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

from ..operators.decompose import decompose, guide
from . import data as D

ALIGN_DIMS = (0, 1, 2, 6, 7)          # eef position + finger qpos (axis-angle wraps near pi: not used)


def state_norm(path=D.NORM_STATS):
    d = json.load(open(path))["norm_stats"]["state"]
    q01, q99 = np.asarray(d["q01"], np.float32), np.asarray(d["q99"], np.float32)
    return lambda s: ((np.asarray(s, np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0)


@dataclasses.dataclass
class DemoProgram:
    mu: np.ndarray                 # [T, dim] teacher means (x0 space, normalised chunk rows)
    var: np.ndarray                # [T, dim] teacher variances
    key: np.ndarray                # [T, len(ALIGN_DIMS)] normalised alignment states
    codes: list                    # per frame: list of (node id, weight) (empty for "raw")
    variant: str
    fold: np.ndarray = None        # [T] bool: frames the library does NOT explain (the fold expert's frames)

    @classmethod
    def from_demo(cls, tree, rows, states, K=3, levels=None, variant="code", raw_var=1e-2):
        norm = state_norm()
        key = norm(states)[:, list(ALIGN_DIMS)].astype(np.float16).astype(np.float32)
        if variant == "raw":
            return cls(mu=np.asarray(rows, np.float32), var=np.full_like(np.asarray(rows, np.float32), raw_var), key=key,
                       codes=[], variant=variant)
        mus, vars_, codes = [], [], []
        for r in np.asarray(rows, np.float32):
            c = decompose(tree, r, K=K, levels=levels)
            mus.append(c.mu); vars_.append(c.var); codes.append(c.entries())
        return cls(mu=np.stack(mus), var=np.stack(vars_), key=key, codes=codes, variant=variant)

    def bytes(self):
        """What the program stores: codes (uint16 id + fp16 weight each) or the raw chunks (fp16), + the keys."""
        store = sum(4 * len(c) for c in self.codes) if self.variant == "code" else 2 * int(self.mu.size)
        return int(store + 2 * self.key.size)

    def save(self, path):
        np.savez_compressed(path, mu=self.mu, var=self.var, key=self.key, variant=self.variant,
                            fold=np.zeros(len(self.mu), bool) if self.fold is None else self.fold,
                            codes=np.array([np.asarray(c, np.float32) for c in self.codes], dtype=object), allow_pickle=True)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(mu=z["mu"], var=z["var"], key=z["key"], codes=[list(map(tuple, c)) for c in z["codes"]], variant=str(z["variant"]),
                   fold=z["fold"] if "fold" in z else None)


class ComposeHooks:
    """Per-episode state: a pointer into the program, advanced by nearest-state alignment in a forward window."""

    def __init__(self, program, w0=1.2, wmax=2.0, window=20, lookahead=0, schedule="variance", var=None):
        """schedule: "variance" (Zekun's Eq. 16: strong at high noise, fading as the policy takes over) or
        "const" (w = w0 at every step: w0 = 1 samples the teacher alone, i.e. executes the program open-loop in
        chunk space; pi0.5's conditional is nearly deterministic, so a fading guide may be overwritten late)."""
        self.prog, self.w0, self.wmax, self.window, self.lookahead = program, float(w0), float(wmax), int(window), int(lookahead)
        self.schedule = schedule
        # Mixing eps with weight w is (per noise level) a product of experts p_policy^(1-w) p_teacher^w, whose mean is
        # precision-weighted: pi0.5's conditional std is ~0.024 (r2b), so a teacher looser than that barely moves the
        # sample unless w ~ 1.  `var` overrides the teacher's variance to set its precision explicitly.
        self.var = None if var is None else float(var)
        self.norm = state_norm()
        self.p = 0
        self.trace = []

    def reset(self, meta=None):
        self.p = 0
        self.trace = []

    def align(self, state):
        s = self.norm(state)[list(ALIGN_DIMS)]
        lo, hi = self.p, min(len(self.prog.key), self.p + self.window + 1)
        d = ((self.prog.key[lo:hi] - s[None]) ** 2).sum(1)
        self.p = lo + int(d.argmin())
        return min(self.p + self.lookahead, len(self.prog.key) - 1)

    def guide_at(self, j):
        v = self.prog.var[j] if self.var is None else np.full_like(self.prog.var[j], self.var)
        if self.schedule == "const":
            return guide(self.prog.mu[j], v, wmax=self.w0, schedule="const", path="flow")
        return guide(self.prog.mu[j], v, w0=self.w0, wmax=self.wmax, path="flow")

    def __call__(self, obs, pre, meta=None):
        j = self.align(obs["observation/state"])
        self.trace.append(j)
        return self.guide_at(j), None


class HybridHooks(ComposeHooks):
    """COMPOSE where the library explains the demo, FOLD where it does not: at an aligned frame flagged in
    `program.fold` the gated expert `key` is switched on and no guide is used; elsewhere the prototype code guides
    the frozen policy with the expert off (so everything the library explains runs on the untouched prior)."""

    def __init__(self, program, state, key, **kw):
        super().__init__(program, **kw)
        from ..experts.lora import keys_on
        self.on = keys_on(state, {key})

    def __call__(self, obs, pre, meta=None):
        j = self.align(obs["observation/state"])
        self.trace.append(j)
        if self.prog.fold is not None and bool(self.prog.fold[j]):
            return None, (lambda x, t: self.on)
        return super().guide_at(j), None

    # the plain ComposeHooks call, factored so the hybrid can reuse it
