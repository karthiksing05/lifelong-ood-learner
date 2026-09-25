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
                            codes=np.array([np.asarray(c, np.float32) for c in self.codes], dtype=object), allow_pickle=True)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(mu=z["mu"], var=z["var"], key=z["key"], codes=[list(map(tuple, c)) for c in z["codes"]], variant=str(z["variant"]))


class ComposeHooks:
    """Per-episode state: a pointer into the program, advanced by nearest-state alignment in a forward window."""

    def __init__(self, program, w0=1.2, wmax=2.0, window=20, lookahead=0):
        self.prog, self.w0, self.wmax, self.window, self.lookahead = program, float(w0), float(wmax), int(window), int(lookahead)
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

    def __call__(self, obs, pre, meta=None):
        j = self.align(obs["observation/state"])
        self.trace.append(j)
        return guide(self.prog.mu[j], self.prog.var[j], w0=self.w0, wmax=self.wmax, path="flow"), None
