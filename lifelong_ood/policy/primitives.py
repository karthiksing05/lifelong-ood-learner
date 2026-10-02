"""REUSABLE MOTION PRIMITIVES (docs/PLAN_PRIMITIVES.md, plan C): a step of an instruction, a keyed LoRA on the action expert fitted
on the matching segment of one demonstration, stored by its STEP KEY and reused by later tasks with the same step.

    steps_of(language)          -> the instruction's steps (bind_hooks.split_conj, else the whole instruction)
    step_key(clause)            -> "verb|preposition|target", the object left out (binding supplies it)
    segment(actions, n_steps)   -> frame ranges of the demo's steps, cut at its first sustained release
    PrimHooks(bk, steps, paths) -> serve hooks: step prompts and primitives, switched at the policy's own release
"""
from __future__ import annotations

import re

import numpy as np

from .bind_hooks import split_conj

PREP = r"(on top of|in|on|under|to the right of|to the left of|to the front of|into|onto)"
RELEASE_RUN = 10            # demo frames the gripper must stay closed before an opening counts as a release (demos at 20 Hz)


def steps_of(language):
    s = split_conj(language)
    return s if s else [language]


def step_key(clause):
    """(verb, preposition, target noun phrase) of a step; "pick up X and place it P T" is the step "put X P T"."""
    c = re.sub(r"^pick up (.+?) and (place|put) it", r"put \1", clause)
    verb = "turn on" if c.startswith("turn on") else "turn off" if c.startswith("turn off") else c.split()[0]
    rest = c[len(verb):]
    m = re.search(rf"\b{PREP} (the (?:[a-z]+ ){{0,3}}[a-z]+(?: of the [a-z]+)?)$", rest)
    if m:
        return f"{verb}|{m.group(1)}|{m.group(2)}"
    m = re.search(r"\b(the (?:[a-z]+ ){0,3}[a-z]+(?: of the [a-z]+)?)$", rest)
    return f"{verb}||{m.group(1) if m else c}"


def releases(gripper, run):
    """Frames where a closed run of at least `run` frames ends (the gripper opens)."""
    c = np.asarray(gripper) > 0
    out, n = [], 0
    for i, x in enumerate(c):
        if x:
            n += 1
        else:
            if n >= run:
                out.append(i)
            n = 0
    return out


def segment(actions, n_steps, run=RELEASE_RUN):
    """Frame ranges [(start, end)] of the demo's steps: one step = the whole demo; two steps = cut at the first sustained
    release (the event PrimHooks switches on at run time).  None if the demo has no such release."""
    T = len(actions)
    if n_steps == 1:
        return [(0, T)]
    r = releases(np.asarray(actions)[:, 6], run)
    if not r:
        return None
    return [(0, r[0]), (r[0], T)]


class PrimHooks:
    """Serve hooks for a sequence of steps: the step's clause as the prompt and its primitive (a keyed LoRA, or none = the policy
    itself) switched on; the next step begins when the policy's own commanded gripper releases a grasp held for 3 queries.
    `inner`: hooks for the first step (e.g. BindHooks binding the step's object); its bias applies only during that step."""

    def __init__(self, bk, steps, paths, inner=None):
        from . import adapt
        self.steps, self.inner = list(steps), inner
        self.state, self.keys = None, []
        for p in paths:
            if p is None:
                self.keys.append(None)
                continue
            self.state, key, _ = adapt.load(bk, p, state=self.state)
            self.keys.append(key)
        if self.state is None:
            self.state = {"on": frozenset()}
        self.step, self.run, self.closed = 0, 0, False
        self.log = []

    def _apply(self):
        k = self.keys[self.step] if self.step < len(self.keys) else None
        self.state["on"] = frozenset({k}) if k else frozenset()

    def reset(self, meta):
        self.step, self.run, self.closed = 0, 0, False
        self._apply()
        if self.inner is not None and hasattr(self.inner, "reset"):
            self.inner.reset(meta)

    def rewrite(self, obs, meta):
        obs = dict(obs)
        obs["prompt"] = self.steps[self.step]
        return obs

    def observe(self, actions):
        a = np.asarray(actions)
        self.closed = bool(a[min(4, len(a) - 1), 6] > 0)
        if self.step < len(self.steps) - 1:
            if self.closed:
                self.run += 1
            else:
                if self.run >= 3:
                    self.step += 1
                    self._apply()
                    self.log.append(("switch", self.step))
                self.run = 0
        if self.inner is not None and hasattr(self.inner, "observe"):
            self.inner.observe(actions)

    def __call__(self, obs, pre, meta):
        if self.inner is not None and self.step == 0:
            return self.inner(obs, pre, meta)
        return None, None
