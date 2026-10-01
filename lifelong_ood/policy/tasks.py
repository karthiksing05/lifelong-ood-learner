"""TASK PROTOTYPES: the overhaul decomposes a new task into prototypes of tasks the policy already performs (user's note in the
plan of 2026-10-01).

Every task the policy reads is a task prototype: its factors (lifelong_ood/policy/factors.py), its scene, and its PROTOTYPE
CHAIN, the sequence of t = 0.4 verb prototypes (mode-finding on the policy's action marginal, results/robot/
r2b_libero_40_tree.npz) its actions pass through, collapsed over repeats.  Chains are read from demonstrations of the 40
training tasks and from recorded rollouts of tasks the policy solves; a composed or folded task enters the library with the
chain of its own successes.  A new task is DECOMPOSED by greedy (submodular) cover of its factors with library entries, the
selection step of Wang et al. moved from pixels to tasks: each selected prototype supplies the factors it shares, and its chain
says in which phase that factor acts.
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results", "robot")
WEIGHTS = {"object": 1.0, "relation": 1.0, "qualifier": 0.5, "scene": 0.5}


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Labeller:
    """Basin labels of 10-step action chunks at the tree's levels (the readout's procedure: experiments/grammar_readout.py)."""

    def __init__(self, levels=(0.4, 0.55), device="cuda"):
        from lifelong_ood.policy import data as D
        from lifelong_ood.tree.nodes import Tree
        self.r19, self.r2b = _mod("r19_invariance"), _mod("r2b_marginal")
        self.lib = np.load(os.path.join(R, "r2b_libero_40.npz"))
        self.tree = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
        self.norm = D.Norm.load()
        self.device = device
        self.levels = {}
        for u in levels:
            li = int(np.argmin([abs(v - u) for v in self.tree.u_grid]))
            ids = self.tree.levels[li]
            self.levels[u] = (float(self.tree.u_grid[li]), np.stack([self.tree.nodes[i].mode for i in ids]).astype(np.float32))

    def __call__(self, actions_list, every=5):
        """[T_i, 7] raw action sequences -> {level: [labels per sequence]} (one label per chunk start, every `every` steps)."""
        X = [self.r19.chunks(a, self.norm, every=every) for a in actions_list]
        cut = np.cumsum([0] + [len(x) for x in X])
        allX = np.concatenate(X)
        out = {}
        for u, (uu, modes) in self.levels.items():
            lab = np.asarray(self.r19.labels(allX, self.lib["mu"], self.lib["var"], modes, uu, self.r2b, self.device))
            out[u] = [lab[cut[i]:cut[i + 1]] for i in range(len(X))]
        return out


def collapse(seq):
    out = []
    for s in seq:
        if not out or out[-1] != int(s):
            out.append(int(s))
    return out


def verb_names(level="0.4"):
    g = json.load(open(os.path.join(R, "grammar_readout.json")))
    return {int(k): v["name"].split(" (")[0] for k, v in g["prototypes"][level].items()}


@dataclass
class TaskPrototype:
    tid: str
    language: str
    scene: str
    family: str
    slots: list
    moved: list
    qualifiers: list
    source: str                         # train | zeroshot | composed | folded
    success: float = 1.0
    chain: list = field(default_factory=list)         # t = 0.4 prototype ids, collapsed
    coarse: list = field(default_factory=list)        # t = 0.55
    code: dict = field(default_factory=dict)          # for composed / folded entries: how to run it

    def factors(self):
        f = {("object", o) for o in self.moved}
        f |= {("relation", tuple(s[:1] + s[2:])) for s in self.slots}
        f |= {("qualifier", q) for q in self.qualifiers}
        f.add(("scene", self.scene))
        return f


def from_task(t, source, success=1.0):
    return TaskPrototype(t.tid, t.language, t.scene, t.family, [list(s) for s in t.slots], list(t.moved), list(t.qualifiers),
                         source, float(success))


def select(target, library, K=3):
    """Greedy cover of the target's factors by library prototypes (weights WEIGHTS; a prototype of the same scene that the
    policy solves scores its scene factor).  -> [(prototype, factors it covers)] in selection order."""
    want = target.factors() if isinstance(target, TaskPrototype) else from_task(target, "query").factors()
    covered, chosen = set(), []
    for _ in range(K):
        best, gain, cov = None, 0.0, set()
        for p in library:
            if p.tid == getattr(target, "tid", None) or p in [c for c, _ in chosen]:
                continue
            c = (p.factors() & want) - covered
            g = sum(WEIGHTS[k] for k, _ in c) * (0.5 + 0.5 * p.success)
            if g > gain:
                best, gain, cov = p, g, c
        if best is None:
            break
        chosen.append((best, sorted(cov)))
        covered |= cov
    return chosen
