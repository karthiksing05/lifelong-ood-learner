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
    m = re.search(rf"^(.*)\b{PREP} (the (?:[a-z]+ ){{0,3}}[a-z]+(?: of the [a-z]+)?)$", rest)   # the RIGHTMOST preposition: "the
    if m:                                                    # black bowl on the left ... in the tray" goes in the tray (fixed 2026-10-04;
        return f"{verb}|{m.group(2)}|{m.group(3)}"           # before, tasks 60 and 87 read "on | the left ...")
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


def goal_key(clause):
    """Plan D: the GOAL of a step, object and verb left out: (preposition, target), e.g. "in|the basket"."""
    k = step_key(clause).split("|")
    return f"{k[1]}|{k[2]}"


def transport(actions, lo, hi, run=RELEASE_RUN):
    """Plan D: the frames of a step that carry the object -- from the step's first sustained grasp (closed >= run frames) to the
    step's end.  None if the step never grasps (closing a drawer, turning a knob)."""
    c = np.asarray(actions)[lo:hi, 6] > 0
    starts = [i for i in range(len(c) - run + 1) if c[i:i + run].all() and (i == 0 or not c[i - 1])]
    return None if not starts else (lo + starts[0], hi)


QUALIFIERS = r"\b(at|on|in) the (front|back|left|right|middle)\b|\b(front|back|middle|left|right)\b"   # "in the middle": 2026-10-14


def object_key(clause):
    """Plan E: the OBJECT of a step, its instance qualifier left out: (verb, object), e.g. "put|the black bowl" for "put the black
    bowl at the back on the plate".  The qualifier stays in the step's prompt, so a primitive fitted on several instances sees it."""
    verb, prep, target = step_key(clause).split("|")
    c = re.sub(r"^pick up (.+?) and (place|put) it", r"put \1", clause)
    rest = " " + c[len(verb):].strip()
    if prep and rest.endswith(f" {prep} {target}"):
        rest = rest[: -len(f" {prep} {target}")]
    return f"{verb}|{re.sub(r' +', ' ', re.sub(QUALIFIERS, '', rest)).strip()}"


def instance_key(clause):
    """Plan E2: the INSTANCE child of a step's object concept -- its object key and the qualifier the instruction names for it
    ("put|the black bowl|middle"); None when the object phrase names no instance (the type concept serves it)."""
    verb, prep, target = step_key(clause).split("|")
    c = re.sub(r"^pick up (.+?) and (place|put) it", r"put \1", clause)
    rest = " " + c[len(verb):].strip()
    if prep and rest.endswith(f" {prep} {target}"):
        rest = rest[: -len(f" {prep} {target}")]
    q = [w for w in ("front", "back", "middle", "left", "right") if re.search(rf"\b{w}\b", rest)]
    return f"{object_key(clause)}|{q[0]}" if q else None


def approach(actions, lo, hi, run=RELEASE_RUN):
    """Plan E: the frames of a step before its first sustained grasp (reaching the object; the chunks of the last frames carry the
    closing).  None if the step never grasps, or grasps at its first frame."""
    tr = transport(actions, lo, hi, run)
    return None if tr is None or tr[0] <= lo else (lo, tr[0])


class PrimHooks:
    """Serve hooks for a sequence of steps: the step's clause as the prompt and its primitive (a keyed LoRA, or none = the policy
    itself) switched on; the next step begins when the policy's own commanded gripper releases a grasp held for 3 queries.
    `inner`: hooks for the first step (e.g. BindHooks binding the step's object); its bias applies only during that step.
    `approach_paths` (plan E, with goal_only): per step, the OBJECT primitive switched on before the grasp."""

    def __init__(self, bk, steps, paths, inner=None, goal_only=False, approach_paths=None):
        from . import adapt
        self.steps, self.inner = list(steps), inner
        self.goal_only = bool(goal_only)                 # plan D: a primitive acts only from the grasp to the release
        self.carrying = False
        self.state, self.keys, self.akeys = None, [], []
        for p in paths:
            if p is None:
                self.keys.append(None)
                continue
            self.state, key, _ = adapt.load(bk, p, state=self.state)
            self.keys.append(key)
        for p in approach_paths or []:
            if p is None:
                self.akeys.append(None)
                continue
            self.state, key, _ = adapt.load(bk, p, state=self.state)
            self.akeys.append(key)
        if self.state is None:
            self.state = {"on": frozenset()}
        self.step, self.run, self.closed = 0, 0, False
        self.log = []

    def _apply(self):
        k = self.keys[self.step] if self.step < len(self.keys) else None
        if self.goal_only and not self.carrying:         # the approach is the object primitive's, else the policy's (and the
            k = self.akeys[self.step] if self.step < len(self.akeys) else None   # object binding's)
        self.state["on"] = frozenset({k}) if k else frozenset()

    def reset(self, meta):
        self.step, self.run, self.closed, self.carrying = 0, 0, False, False
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
        if self.goal_only:                               # grasp (closed 3 queries) -> carry; release -> next step's approach
            if self.closed:
                self.run += 1
                if self.run >= 3 and not self.carrying:
                    self.carrying = True
                    self._apply()
                    self.log.append(("carry", self.step))
            else:
                if self.carrying:
                    self.carrying = False
                    self.step = min(self.step + 1, len(self.steps) - 1)
                    self._apply()
                    self.log.append(("release", self.step))
                self.run = 0
        elif self.step < len(self.steps) - 1:
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
        if self.inner is not None and (not self.carrying if self.goal_only else self.step == 0):
            return self.inner(obs, pre, meta)
        return None, None
