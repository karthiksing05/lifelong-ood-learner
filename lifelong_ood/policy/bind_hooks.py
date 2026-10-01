"""BindHooks: the policy's own chunk, with the action expert's attention biased toward the instruction's referent of the current
phase (plan of 2026-10-01 evening; lifelong_ood/policy/attn_bias.py).

Phases come from the policy's OWN commanded gripper (its previous chunk's 5th action, the last one executed with replan 5):
open -> the moved object (or the fixture of a state goal such as Open / Close), closed -> the target receptacle.  Referents are
read from the task's BDDL goal (instances, never types).  Where-sources:
    oracle  the simulator's per-token coverage sent by the client (`libero_client.py --seg`): ORACLE arms only
    w1      the policy's own PaliGemma grounding of the referent's phrase (where.vlm_grounding), layer fixed in advance
    w3      OWLv2 on the agent-view image (where.OwlDetector)
Controls: ctrl=placebo binds a random OTHER movable object (fixed per episode); ctrl=shift binds the referent's token pattern
moved onto empty table; ctrl=none is the method.  Deployable sources refuse any privileged meta key (seg_*, obj_*).
"""
from __future__ import annotations

import numpy as np

from . import attn_bias as AB

MIN_COVER = 0.1
BANDS = {"all": None, "mid": tuple(range(6, 12)), "early": tuple(range(0, 6)), "late": tuple(range(12, 18))}
PRIVILEGED = ("seg_", "obj_")


def instance_of(arg, names):
    """A goal argument ('basket_1_contain_region', 'plate_1', 'wooden_cabinet_1_top_region') -> the longest object / fixture
    instance name it starts with (None for a table region)."""
    best = None
    for n in names:
        if arg == n or arg.startswith(n + "_"):
            if best is None or len(n) > len(best):
                best = n
    return best


def referents(task):
    """-> {"open": [instances], "closed": [instances]} from the goal atoms (In / On: object while open, target while closed;
    Open / Close / Turnon / Turnoff: the fixture while open)."""
    names = list(task.objects) + list(task.fixtures)
    out = {"open": [], "closed": []}
    for atom in task.goal:
        pred, args = atom[0], atom[1:]
        if pred in ("In", "On") and len(args) == 2:
            o, tgt = instance_of(args[0], names), instance_of(args[1], names)
            if o:
                out["open"].append(o)
            if tgt:
                out["closed"].append(tgt)
        elif args:
            f = instance_of(args[0], names)
            if f:
                out["open"].append(f)
    return out


def shifted(mask16, avoid16):
    """The token pattern moved to the empty table position (lower half, no object token) farthest from where it was."""
    ij = np.argwhere(mask16 > 0)
    if len(ij) == 0:
        return None
    best, far = None, -1.0
    for di in range(-15, 16):
        for dj in range(-15, 16):
            q = ij + [di, dj]
            if q.min() < 0 or q.max() > 15 or q[:, 0].min() < 7 or avoid16[q[:, 0], q[:, 1]].max() > 0:
                continue
            d = float(np.hypot(di, dj))
            if d > far:
                far, best = d, q
    if best is None:
        return None
    m = np.zeros((16, 16), np.float32)
    m[best[:, 0], best[:, 1]] = 1.0
    return m


class BindHooks:
    def __init__(self, bk, task, src="oracle", rho=0.5, beta=None, band="all", cams="agent", ctrl="none", phase=True, seed=0,
                 w1_layer=None, keep=0.5, phrases=None):
        self.bk, self.task = bk, task
        self.src, self.ctrl, self.phase = src, ctrl, bool(phase)
        self.cfg = (dict(mode="fixed", beta=float(beta)) if beta is not None else dict(mode="mass", rho=float(rho)))
        self.cfg["layers"] = BANDS[band]
        self.cams = cams.split("+")
        self.ref = referents(task)
        self.movable = list(task.objects)
        self.rng = np.random.default_rng(seed)
        self.w1_layer = w1_layer
        self.keep = float(keep)                          # deployable maps keep the tokens >= keep x the map's peak
        self.phrases = phrases or {}
        self.owl = None
        self.closed = False
        self.placebo = None
        self.started = False                            # the server's warm-up calls (before any episode) run unbiased
        self.log = []                                   # per query: (phase, referent, mass before, mass after)

    # ── episode ───────────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self, meta):
        self.started = True
        self.closed = False
        named = set(self.ref["open"]) | set(self.ref["closed"])
        others = [n for n in self.movable if n not in named]
        self.placebo = str(self.rng.choice(others)) if others else None

    def observe(self, actions):
        a = np.asarray(actions)
        self.closed = bool(a[min(4, len(a) - 1), 6] > 0)

    # ── where ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def _referent(self):
        if self.ctrl == "placebo":
            return [self.placebo] if self.placebo else []
        key = "closed" if (self.phase and self.closed and self.ref["closed"]) else "open"
        return self.ref[key]

    def _oracle(self, meta, names, cam):
        if "seg_names" not in meta:
            raise RuntimeError("oracle binding needs the client's --seg maps")
        sn = [str(x) for x in meta["seg_names"]]
        M = np.asarray(meta[f"seg_{cam}"], np.float32).reshape(len(sn), 16, 16)
        m = np.zeros((16, 16), np.float32)
        for n in names:
            if n in sn:
                m = np.maximum(m, M[sn.index(n)])
        allobj = (M.max(0) >= MIN_COVER).astype(np.float32)
        return (m >= MIN_COVER).astype(np.float32), allobj

    def maps(self, obs, meta):
        names = self._referent()
        if not names:
            return {}
        if self.src == "oracle":
            out = {}
            for cam in self.cams:
                m, allobj = self._oracle(meta, names, cam)
                if self.ctrl == "shift" and cam == "agent":
                    m = shifted(m, allobj)
                    if m is None:
                        continue
                if m.max() > 0:
                    out[cam] = m
            return out
        if any(k.startswith(PRIVILEGED) for k in meta):
            raise RuntimeError("a deployable where-source received privileged simulator state")
        out = {}
        for n in names[:1]:                              # one referent per phase for the deployable sources
            phrase = self.phrase_of(n)
            if self.src == "w1":
                from . import where as W
                g = W.vlm_grounding(self.bk, obs, phrase, layers=[self.w1_layer], tokenizer=self._tok())
                m = g[self.w1_layer]["agent"]
            elif self.src == "w3":
                if self.owl is None:
                    from .where import OwlDetector
                    self.owl = OwlDetector()
                m, _ = self.owl(obs["observation/image"], phrase)
                m = m / max(float(m.max()), 1e-9)
            else:
                raise ValueError(f"unknown where-source {self.src}")
            out["agent"] = (m >= self.keep * m.max()).astype(np.float32)
        return out

    def phrase_of(self, inst):
        """The words the instruction uses for an instance: the scene's phrase for its type, else the type's words."""
        ty = self.task.objects.get(inst) or self.task.fixtures.get(inst) or inst
        return self.phrases.get(ty, str(ty).replace("_", " "))

    def _tok(self):
        if getattr(self, "_tokenizer", None) is None:
            from openpi.models import tokenizer as _t
            self._tokenizer = _t.PaligemmaTokenizer(200)
        return self._tokenizer

    # ── the policy's own chunk, with the bias ──────────────────────────────────────────────────────────────────────
    def __call__(self, obs, pre, meta):
        if pre is None or not self.started:
            return None, None
        maps = self.maps(obs, meta)
        if not maps:
            self.log.append(("closed" if self.closed else "open", None, None, None))
            return None, None
        log = []
        gate = AB.bias_gate(self.bk, maps, pre, log=log, **self.cfg)
        self._last_log = log
        self.log.append(("closed" if self.closed else "open", ",".join(self._referent()), None, None))
        return None, gate
