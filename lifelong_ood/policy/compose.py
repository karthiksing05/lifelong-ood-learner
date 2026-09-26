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


def rotmat(aa):
    """Axis-angle [..., 3] -> rotation matrices [..., 9] (continuous where the axis-angle wraps near pi)."""
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    n = aa / np.maximum(th, 1e-9)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    Kx = np.stack([np.zeros_like(x), -z, y, z, np.zeros_like(x), -x, -y, x, np.zeros_like(x)], -1).reshape(*aa.shape[:-1], 3, 3)
    I = np.broadcast_to(np.eye(3), Kx.shape)
    st, ct = np.sin(th)[..., None], np.cos(th)[..., None]
    R = I + st * Kx + (1 - ct) * (Kx @ Kx)
    return R.reshape(*aa.shape[:-1], 9).astype(np.float32)


def align_key(states, mode="pos", rot_weight=1.0):
    """The alignment key of raw 8-D states [eef pos, axis-angle, finger qpos].  "pos": normalised position + fingers
    (blind to rotation-only phases such as turning a knob); "pose": + the rotation matrix, weighted."""
    st = np.atleast_2d(np.asarray(states, np.float32))
    k = state_norm()(st)[:, list(ALIGN_DIMS)]
    if mode == "pose":
        k = np.concatenate([k, rot_weight * rotmat(st[:, 3:6])], 1)
    return k.astype(np.float32)


@dataclasses.dataclass
class DemoProgram:
    mu: np.ndarray                 # [T, dim] teacher means (x0 space, normalised chunk rows)
    var: np.ndarray                # [T, dim] teacher variances
    key: np.ndarray                # [T, len(ALIGN_DIMS)] normalised alignment states
    codes: list                    # per frame: list of (node id, weight) (empty for "raw")
    variant: str
    fold: np.ndarray = None        # [T] bool: frames the library does NOT explain (the fold expert's frames)
    prompt_id: np.ndarray = None   # [T] int: index into prompt_table of the library task that explains the frame
    prompt_table: list = None      # the library tasks' instructions (strings)
    states: np.ndarray = None      # [T, 8] the demo's raw states (for pose keys; 16 B per frame in fp16)
    steer: np.ndarray = None       # [T] bool: frames where the prior DISAGREES with the demo (guide only there; 1 bit each)

    @classmethod
    def from_demo(cls, tree, rows, states, K=3, levels=None, variant="code", raw_var=1e-2):
        norm = state_norm()
        key = norm(states)[:, list(ALIGN_DIMS)].astype(np.float16).astype(np.float32)
        st16 = np.asarray(states, np.float32).astype(np.float16).astype(np.float32)
        if variant == "raw":
            return cls(mu=np.asarray(rows, np.float32), var=np.full_like(np.asarray(rows, np.float32), raw_var), key=key,
                       codes=[], variant=variant, states=st16)
        mus, vars_, codes = [], [], []
        for r in np.asarray(rows, np.float32):
            c = decompose(tree, r, K=K, levels=levels)
            mus.append(c.mu); vars_.append(c.var); codes.append(c.entries())
        return cls(mu=np.stack(mus), var=np.stack(vars_), key=key, codes=codes, variant=variant, states=st16)

    def with_prompts(self, rows, library_rows, library_task, task_text, k=5, smooth=10):
        """PROMPT PROGRAM: per frame, the library task whose chunks explain this frame's chunk best (majority over the
        k nearest library chunks, rms in the normalised chunk space).  The policy is then conditioned on that task's
        instruction at the aligned frame: composition through pi0.5's own language conditioning."""
        L = np.asarray(library_rows, np.float32)
        tasks = np.asarray(library_task)
        table = sorted(set(tasks.tolist()))
        pid = np.zeros(len(rows), int)
        for f, r in enumerate(np.asarray(rows, np.float32)):
            d = ((L - r[None]) ** 2).mean(1)
            nn = tasks[np.argsort(d)[:k]]
            vals, cnt = np.unique(nn, return_counts=True)
            pid[f] = table.index(int(vals[cnt.argmax()]))
        if smooth:                                    # majority over a +-smooth window: switch only at sustained changes
            sm = pid.copy()
            for f in range(len(pid)):
                w = pid[max(0, f - smooth): f + smooth + 1]
                sm[f] = np.bincount(w).argmax()
            pid = sm
        self.prompt_id = pid
        self.prompt_table = [task_text[int(t)] for t in table]
        return self

    def subsample(self, k):
        """Keep every k-th frame (the policy is queried every 5 steps and neighbouring chunks share 9 of 10 actions):
        a k-times smaller program.  The alignment window is counted in kept frames by the hooks."""
        k = int(k)
        sl = slice(0, None, k)
        take = lambda a: None if a is None else np.asarray(a)[sl]          # noqa: E731
        return DemoProgram(mu=take(self.mu), var=take(self.var), key=take(self.key), codes=list(self.codes[sl]) if self.codes else [],
                           variant=self.variant, fold=take(self.fold), prompt_id=take(self.prompt_id), prompt_table=self.prompt_table,
                           states=take(self.states), steer=take(self.steer))

    def bytes(self):
        """What the program needs to EXECUTE what it executes.  A "code" program's teacher is the per-dimension PoE whose
        weights were fitted to the demo chunk (decompose's per-dim softmax), so it needs those weights or its 70-dim mean
        per frame -- as much as the raw chunk.  (Charging only the node ids + one scalar weight each, as before
        2026-09-25 late, under-counted: ids + scalar weights rebuild a teacher 0.25-0.39 rms from the demo chunk where
        the executed one is 0.04-0.08; RETRACTED.)  + the alignment keys, + one uint8 prompt index per frame."""
        store = 2 * int(self.mu.size)
        keys = 2 * int(self.states.size) if self.states is not None else 2 * int(self.key.size)
        return int(store + keys + (0 if self.prompt_id is None else len(self.prompt_id))
                   + (0 if self.steer is None else (len(self.steer) + 7) // 8))

    def save(self, path):
        extra = {} if self.prompt_id is None else dict(prompt_id=self.prompt_id, prompt_table=np.array(self.prompt_table, dtype=object))
        if self.states is not None:
            extra["states"] = self.states
        if self.steer is not None:
            extra["steer"] = self.steer
        np.savez_compressed(path, mu=self.mu, var=self.var, key=self.key, variant=self.variant,
                            fold=np.zeros(len(self.mu), bool) if self.fold is None else self.fold,
                            codes=np.array([np.asarray(c, np.float32) for c in self.codes], dtype=object), allow_pickle=True, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(mu=z["mu"], var=z["var"], key=z["key"], codes=[list(map(tuple, c)) for c in z["codes"]], variant=str(z["variant"]),
                   fold=z["fold"] if "fold" in z else None,
                   prompt_id=z["prompt_id"] if "prompt_id" in z else None,
                   prompt_table=list(z["prompt_table"]) if "prompt_table" in z else None,
                   states=z["states"] if "states" in z else None,
                   steer=z["steer"] if "steer" in z else None)


class ComposeHooks:
    """Per-episode state: a pointer into the program, advanced by nearest-state alignment in a forward window."""

    def __init__(self, program, w0=1.2, wmax=2.0, window=20, lookahead=0, schedule="variance", var=None, use_guide=True,
                 use_prompt=False, stall=0, release=2, key_mode="pos", agree=None, contact=0, arm_only=False):
        """schedule: "variance" (Zekun's Eq. 16: strong at high noise, fading as the policy takes over) or
        "const" (w = w0 at every step: w0 = 1 samples the teacher alone, i.e. executes the program open-loop in
        chunk space; pi0.5's conditional is nearly deterministic, so a fading guide may be overwritten late)."""
        self.prog, self.w0, self.wmax, self.window, self.lookahead = program, float(w0), float(wmax), int(window), int(lookahead)
        self.schedule = schedule
        # Mixing eps with weight w is (per noise level) a product of experts p_policy^(1-w) p_teacher^w, whose mean is
        # precision-weighted: pi0.5's conditional std is ~0.024 (r2b), so a teacher looser than that barely moves the
        # sample unless w ~ 1.  `var` overrides the teacher's variance to set its precision explicitly.
        self.var = None if var is None else float(var)
        self.use_guide, self.use_prompt = bool(use_guide), bool(use_prompt)
        # STALL fallback (0 = off): after `stall` calls whose aligned frame does not advance, release the guide for
        # `release` calls so the prior acts alone (it re-grounds on the scene), then follow the program again.
        # Seen in the pilot: replay stuck at one demo frame for 65 calls, re-issuing the same chunk.
        self.stall, self.release = int(stall), int(release)
        # AGREEMENT gate (None = off): the server first samples the prior's own chunk; if its mean squared distance to
        # the aligned frame's teacher, in units of the program's variance, is below `agree`, the prior's chunk is
        # executed and the program stays silent.  Composition only where the prior is off the demo (a capable prior
        # on a known task was degraded by forcing the program: white mug, zero-shot 9/10 vs codes 3/10).
        self.agree = None if agree is None else float(agree)
        # CONTACT release (0 = off): within +-`contact` frames of the demo's gripper events the program is silent and the
        # prior acts -- the program carries the transport, pi0.5's visually grounded grasp and release do the contact.
        # Gripper events come from the program's own gripper dimension (first step of each chunk, sign changes).
        self.contact = int(contact)
        g = np.sign(np.asarray(program.mu)[:, 6])
        ev = np.where(np.diff(g) != 0)[0] + 1
        self.near_contact = np.zeros(len(program.mu), bool)
        for e in ev:
            self.near_contact[max(0, e - self.contact): e + self.contact + 1] = self.contact > 0
        # ARM ONLY: guide the six arm dimensions of every step; the gripper dimension stays the prior's.
        self.arm_only = bool(arm_only)
        self.arm_mask = np.ones(program.mu.shape[1], np.float32)
        self.arm_mask[6::7] = 0.0
        self.z2 = []
        self._jg = None
        self._still, self._released = 0, 0
        self._j = None
        self.norm = state_norm()
        # alignment keys: "pose" adds the eef orientation (turning a knob barely moves the eef position)
        self.key_mode = key_mode if (key_mode == "pos" or program.states is not None) else "pos"
        self.keys = program.key if self.key_mode == "pos" else align_key(program.states, "pose")
        self.p = 0
        self.trace = []

    def reset(self, meta=None):
        if self.trace:
            import logging
            tr = np.asarray(self.trace)
            extra = ""
            if self.z2:
                z = np.asarray(self.z2)
                extra = (f", agreement z2 median {np.median(z):.2f} (q10 {np.quantile(z, 0.1):.2f}, q90 {np.quantile(z, 0.9):.2f}), "
                         f"guided {float((z >= self.agree).mean()):.2f} of calls")
            logging.info(f"[compose] episode done: {len(tr)} calls, reached demo frame {int(tr.max())}/{len(self.prog.mu) - 1}, "
                         f"stalls {int((np.diff(tr) == 0).sum())}, trace {tr[:: max(1, len(tr) // 12)].tolist()}{extra}")
        self.z2 = []
        self.p = 0
        self.trace = []
        self._j = None
        self._still, self._released = 0, 0

    def _released_now(self):
        """Advance the stall counters for the frame just aligned; True while the guide is released."""
        if not self.stall:
            return False
        if self._released > 0:
            self._released -= 1
            return True
        if len(self.trace) >= 2 and self.trace[-1] <= self.trace[-2]:
            self._still += 1
        else:
            self._still = 0
        if self._still >= self.stall:
            self._still, self._released = 0, self.release - 1
            return True
        return False

    def rewrite(self, obs, meta=None):
        """Called by GuidedPolicy BEFORE the prefix: align to the demo once per call and, with a prompt program,
        swap the instruction for the library task that explains the aligned frame."""
        self._j = self.align(obs["observation/state"])
        self.trace.append(self._j)
        if self.use_prompt and self.prog.prompt_id is not None:
            obs = dict(obs)
            obs["prompt"] = str(self.prog.prompt_table[int(self.prog.prompt_id[self._j])])
        return obs

    def align(self, state):
        s = self.norm(state)[list(ALIGN_DIMS)] if self.key_mode == "pos" else align_key(state, "pose")[0]
        lo, hi = self.p, min(len(self.keys), self.p + self.window + 1)
        d = ((self.keys[lo:hi] - s[None]) ** 2).sum(1)
        self.p = lo + int(d.argmin())
        return min(self.p + self.lookahead, len(self.keys) - 1)

    def agrees(self, own_row):
        """The prior's own chunk vs the teacher of the frame last guided: True if within `agree` (program variance units)."""
        if self.agree is None or self._jg is None:
            return False
        j = self._jg
        z2 = float(np.mean((np.asarray(own_row, np.float32) - self.prog.mu[j]) ** 2 / np.maximum(self.prog.var[j], 1e-4)))
        self.z2.append(z2)
        return z2 < self.agree

    def guide_at(self, j):
        self._jg = j
        v = self.prog.var[j] if self.var is None else np.full_like(self.prog.var[j], self.var)
        if self.schedule == "const":
            g = guide(self.prog.mu[j], v, wmax=self.w0, schedule="const", path="flow")
        else:
            g = guide(self.prog.mu[j], v, w0=self.w0, wmax=self.wmax, path="flow")
        if self.arm_only:
            eps_fn, w_fn = g
            mask = self.arm_mask
            g = (eps_fn, lambda x, u: np.asarray(w_fn(x, u), np.float32) * mask)
        return g

    def _frame(self, obs):
        if self._j is None:                          # called without rewrite(): align here
            self._j = self.align(obs["observation/state"])
            self.trace.append(self._j)
        j, self._j = self._j, None
        return j

    def steering(self, j):
        """Guide at frame j only where the prior disagrees with the demo (no mask: everywhere)."""
        return self.use_guide and (self.prog.steer is None or bool(self.prog.steer[j]))

    def __call__(self, obs, pre, meta=None):
        j = self._frame(obs)
        if self._released_now() or self.near_contact[j]:
            return None, None
        return (self.guide_at(j) if self.steering(j) else None), None


class HybridHooks(ComposeHooks):
    """COMPOSE where the library explains the demo, FOLD where it does not: at an aligned frame flagged in
    `program.fold` the gated expert `key` is switched on and no guide is used; elsewhere the prototype code guides
    the frozen policy with the expert off (so everything the library explains runs on the untouched prior)."""

    def __init__(self, program, state, key, **kw):
        super().__init__(program, **kw)
        from ..experts.lora import keys_on
        self.on = keys_on(state, {key})

    def __call__(self, obs, pre, meta=None):
        j = self._frame(obs)
        if self.prog.fold is not None and bool(self.prog.fold[j]):
            return None, (lambda x, t: self.on)
        if self._released_now():
            return None, None
        return (self.guide_at(j) if self.steering(j) else None), None

