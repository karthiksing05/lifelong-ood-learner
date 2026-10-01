"""Codes and server hooks for factor-composed generation (the overhaul's step 3; lifelong_ood/policy/conditionals.py).

A CODE is what a task stores when composition explains it: a base prompt the policy reads in this scene, factor pairs
(plus, minus) and one weight each, and the phase in which each factor acts.  It is built by rule (code_for), never tuned
per task:
  base   the sibling task of the same scene that the prior already solves (zero-shot >= 0.8, ledger/zeroshot90) and that
         shares the most slots with the task (ties: lowest index); the task's own instruction when there is none;
  object factor  plus = the base's instruction with the base's object replaced by the task's (factors.substitute; this is
         usually the task's own instruction), minus = the base's instruction: the minimal pair that differs in the object
         only, so v(plus) - v(minus) is the object factor read at this observation (the ICLR paper's d_j);
  phase  the object factor acts while the gripper is open (approach and grasp) and is released once the policy's own
         chunk closes the gripper (the prototype tree's top split is gripper state, >= 0.99 pure); "all" keeps it on.
Controls with the same machinery: cfg (the task's instruction against the empty prompt, the untrained null), placebo
(a random pair of training instructions at the same weight).  Bytes of a code: its prompt ids and weights (code_bytes).
"""
from __future__ import annotations

import json

import numpy as np

from . import factors as F
from .conditionals import Composition


def code_for(task, tasks, zeroshot, lam=2.0, phase="approach"):
    """The rule-built code of `task` (a factors.Task).  zeroshot: {libero_90 index: success} for the base rule."""
    same = [t for t in tasks if t.scene == task.scene and t.tid != task.tid and t.suite == "libero_90"
            and (zeroshot.get(t.index) or 0) >= 0.8 and len(set(t.moved)) == 1]
    def overlap(t):
        return sum(1 for a, b in zip(t.slots, task.slots) if a[:1] + a[2:] == b[:1] + b[2:])
    same = [t for t in same if overlap(t) == len(task.slots) == len(t.slots)]     # the object factor only: relations match
    base = max(same, key=lambda t: (overlap(t), -t.index)) if same else None
    if base is None or len(set(task.moved)) != 1:
        return dict(task=task.tid, base=task.language, base_task=None, factors=[],
                    note="no readable sibling with the same relation")
    ph = F.scene_phrases(base, tasks)
    plus = F.substitute(base, task.moved[0], ph)
    return dict(task=task.tid, base=base.language, base_task=base.tid, note="",
                factors=[dict(slot="object", value=task.moved[0], plus=plus, minus=base.language, lam=float(lam),
                              phase=phase)] if plus and plus != base.language else [])


def code_bytes(code):
    """Bytes the code keeps: prompt ids (2 B each, into the vocabulary of instructions) + one fp16 weight + a phase byte per
    factor, + the base id."""
    return 2 + sum(2 + 2 + 2 + 1 for _ in code["factors"])


class FactorHooks:
    """Per query the composition the server samples (GuidedPolicy calls compose(); observe() sees the chosen chunk)."""

    def __init__(self, code, mode="factor", w=None, rng_seed=0, train_prompts=None):
        self.code, self.mode = code, mode
        self.w = float(w) if w is not None else None
        self.rng = np.random.default_rng(rng_seed)
        self.train_prompts = list(train_prompts or [])
        self.closed = False
        self.log = []

    def reset(self, meta):
        self.closed = False
        self.log.append("reset")

    def compose(self, obs, meta=None):
        own = str(obs.get("prompt", ""))
        if self.mode == "cfg":                           # s v(task) + (1 - s) v(empty): guidance against the untrained null
            s = self.w if self.w is not None else 2.0
            return Composition([own, ""], np.array([s, 1.0 - s], np.float32)).check()
        if self.mode == "placebo":                       # a random pair of training instructions at the same weight
            a, b = self.rng.choice(len(self.train_prompts), 2, replace=False)
            lam = self.code["factors"][0]["lam"] if self.code["factors"] else 2.0
            return Composition.factor(own, [(self.train_prompts[a], self.train_prompts[b])], [lam])
        pairs, lams = [], []
        for f in self.code["factors"]:
            on = f["phase"] == "all" or (f["phase"] == "approach" and not self.closed) or (f["phase"] == "carry" and self.closed)
            pairs.append((f["plus"], f["minus"]))
            lams.append(f["lam"] if on else 0.0)
        return Composition.factor(self.code["base"], pairs, lams)

    def observe(self, actions):
        """The executed part of the chosen chunk: the policy's own gripper command sets the phase (close > 0)."""
        a = np.asarray(actions)
        if a.ndim == 2 and a.shape[1] >= 7:
            self.closed = bool(a[min(4, len(a) - 1), 6] > 0)

    def __call__(self, obs, pre, meta=None):
        return None, None


def load_code(path_or_json):
    s = str(path_or_json)
    return json.load(open(s)) if s.endswith(".json") else json.loads(s)
