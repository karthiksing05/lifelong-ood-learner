"""PHASE PROGRAMS WITH OBJECT-RELATIVE BINDINGS (docs/PLAN_TRANSFER.md P1): the one demo executed through the frozen policy.

The demo is cut into phases from object state (experiments/p3_compose_sim.py `phases`: the approach in the grasped object's frame,
the object-centric carry relative to the receptacle, the rest; a demo that lifts nothing is one world-frame phase).  At rollout
each phase is re-anchored on the objects' CURRENT poses (sent by the client, experiments/libero_client.py --objects; privileged
simulator state, an oracle to be replaced by a learned reader), reached by a straight bridge, and tracked by its controller goals
(p + 0.05 a).  Every query plans the next H actions -- predicting that each commanded goal is reached -- and the frozen policy is
sampled through them as a constant-weight guide (w = 1, variance 1e-3: the COMPOSE hooks' setting); on the demo's contact frames
the gated fold expert acts instead (HybridHooks' rule).  After the last phase the gripper is released.

    hooks = PhaseBindHooks(source, parts, norm, fold=mask, state=state, key=key)
"""
from __future__ import annotations

import math

import numpy as np

from ..operators.decompose import guide


def q2m(q):
    x, y, z, w = np.asarray(q, np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def aa2m(aa):
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa)
    if th < 1e-9:
        return np.eye(3)
    u = aa / th
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def m2aa(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-6:
        return th * np.sqrt(np.maximum((np.diag(Rm) + 1) / 2, 0))
    return th * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (2 * s)


def pose(p, R_):
    X = np.eye(4)
    X[:3, :3] = R_
    X[:3, 3] = p
    return X


def track(p, R_, p_t, R_t, g):
    a = np.zeros(7)
    a[:3] = np.clip((p_t - p) / 0.05, -1, 1)
    a[3:6] = np.clip(m2aa(R_t @ R_.T) / 0.5, -1, 1)
    a[6] = g
    return a


class PhaseBindHooks:
    """Per-episode state: a queue of (goal position, goal rotation, gripper, source frame) targets, refilled phase by phase."""

    agree = None
    needs_prior = False

    def __init__(self, source, parts, norm, fold=None, state=None, key=None, var=1e-3, H=10, replan=5):
        self.z, self.parts, self.norm = source, parts, norm
        self.names = [str(x) for x in source["names"]]
        self.fold = None if fold is None else np.asarray(fold, bool)
        self.var, self.H, self.replan = float(var), int(H), int(replan)
        self.on = None
        if state is not None and key is not None:
            from ..experts.lora import keys_on
            self.on = keys_on(state, {key})
        self.reset()

    def reset(self, meta=None):
        self.k = 0                  # next phase to start
        self.queue = []             # [(p*, R*, g, source frame or -1)]
        self.done_phases = False
        self._objs = {}
        self.log = []

    # ── objects from the client ─────────────────────────────────────────────
    def _read(self, meta):
        names = meta.get("obj_names")
        if names is None:
            return
        pos = np.asarray(meta["obj_pos"], np.float64).reshape(-1, 3)
        quat = np.asarray(meta["obj_quat"], np.float64).reshape(-1, 4)
        self._objs = {str(n): (pos[i], q2m(quat[i])) for i, n in enumerate(names)}

    def rewrite(self, obs, meta=None):
        self._read(meta or {})
        return obs

    # ── planning ────────────────────────────────────────────────────────────
    def _start(self, k, p_now, R_now, g_now):
        z, GP, GQ, s0, s1, role, ref, held = self.parts[k]
        if ref is not None and self.names[ref] in self._objs:
            pn, Rn = self._objs[self.names[ref]]
            ps, Rs = z["obj_pos"][s0, ref].astype(np.float64), q2m(z["obj_quat"][s0, ref])
            M = pose(pn - Rn @ Rs.T @ ps, Rn @ Rs.T)
        else:
            M = np.eye(4)
        if held is not None and self.names[held] in self._objs:
            ho, Ho = self._objs[self.names[held]]
            Gs = np.linalg.inv(pose(z["obj_pos"][s0, held], q2m(z["obj_quat"][s0, held]))) @ pose(z["eef_pos"][s0], q2m(z["eef_quat"][s0]))
            Gt = np.linalg.inv(pose(ho, Ho)) @ pose(p_now, R_now)
            C = np.linalg.inv(Gs) @ Gt
        else:
            C = np.eye(4)
        goals = [M @ pose(GP[i], GQ[i]) @ C for i in range(s0, s1)]
        # the source's ACHIEVED poses under the same transform: where the arm will be when the goal is commanded (the controller
        # lags its goal; predicting that goals are reached would close the gripper before the hand arrives)
        achieved = [M @ pose(z["eef_pos"][i], q2m(z["eef_quat"][i])) @ C for i in range(s0, s1)]
        n = max(5, int(math.ceil(np.linalg.norm(goals[0][:3, 3] - p_now) / 0.01)))
        rel = m2aa(goals[0][:3, :3] @ R_now.T)
        g_hold = float(z["action"][s0 - 1, 6]) if s0 > 0 else g_now
        prev = (p_now, R_now)
        for j in range(1, n + 1):
            pt, Rt = p_now + (goals[0][:3, 3] - p_now) * j / n, aa2m(rel * j / n) @ R_now
            self.queue.append((pt, Rt, g_hold, -1, prev[0], prev[1]))
            prev = (pt, Rt)
        for i, X, A in zip(range(s0, s1), goals, achieved):
            self.queue.append((X[:3, 3], X[:3, :3], float(z["action"][i, 6]), i, A[:3, 3], A[:3, :3]))

    def _fill(self, p_now, R_now, g_now):
        while len(self.queue) < self.H and self.k < len(self.parts):
            last = self.queue[-1] if self.queue else (p_now, R_now, g_now, -1, p_now, R_now)
            self._start(self.k, last[0], last[1], last[2])
            self.k += 1
        if self.k >= len(self.parts) and len(self.queue) < self.H:
            last = self.queue[-1] if self.queue else (p_now, R_now, g_now, -1, p_now, R_now)
            g = -1.0                                         # the release phase after the last phase
            while len(self.queue) < self.H:
                self.queue.append((last[0], last[1], g, -1, last[0], last[1]))

    def plan(self, state8):
        p = np.asarray(state8[:3], np.float64)
        R_ = aa2m(state8[3:6])
        g_now = 1.0 if (self.queue and self.queue[0][2] > 0) else -1.0
        self._fill(p, R_, g_now)
        acts, src = [], []
        for h, (pt, Rt, g, i, pa, Ra) in enumerate(self.queue[: self.H]):
            pc, Rc = (p, R_) if h == 0 else (pa, Ra)          # feedback on the first action, the source's lagged path after it
            acts.append(track(pc, Rc, pt, Rt, g))
            src.append(i)
        del self.queue[: self.replan]
        return np.asarray(acts, np.float32), src

    def __call__(self, obs, pre, meta=None):
        acts, src = self.plan(np.asarray(obs["observation/state"], np.float64))
        frame = next((i for i in src[: self.replan] if i >= 0), -1)
        self.log.append(frame)
        if self.on is not None and self.fold is not None and frame >= 0 and frame < len(self.fold) and bool(self.fold[frame]):
            return None, (lambda x, t: self.on)
        rows = self.norm.normalize(acts).reshape(-1)
        return guide(rows, np.full_like(rows, self.var), wmax=1.0, schedule="const", path="flow"), None
