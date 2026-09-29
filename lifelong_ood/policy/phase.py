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
import re

import numpy as np

from ..operators.decompose import guide


def q2m(q):
    x, y, z, w = np.asarray(q, np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(Rm):
    """Rotation matrix -> quaternion (x, y, z, w), the convention of q2m."""
    Rm = np.asarray(Rm, np.float64)
    w = math.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    x = math.sqrt(max(0.0, 1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - Rm[0, 0] + Rm[1, 1] - Rm[2, 2])) / 2
    z = math.sqrt(max(0.0, 1 - Rm[0, 0] - Rm[1, 1] + Rm[2, 2])) / 2
    x = math.copysign(x, Rm[2, 1] - Rm[1, 2])
    y = math.copysign(y, Rm[0, 2] - Rm[2, 0])
    z = math.copysign(z, Rm[1, 0] - Rm[0, 1])
    return np.array([x, y, z, w])


def mean_rotation(Rs):
    """The chordal mean of rotation matrices [n, 3, 3] (the SVD projection of their average)."""
    U, _, Vt = np.linalg.svd(np.mean(np.asarray(Rs, np.float64), 0))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


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


# ── the demo as a phase program, rules v2 (docs/PLAN_TRANSFER.md; v1 = experiments/p3_compose_sim.py `phases`) ─────────
MOVE, CLOSE_TAIL, OPEN_TAIL = 0.015, 8, 5


def load_demo(results_dir, t):
    """Task t's ONE demo with its objects' poses at every kept step (experiments/p0_demo_init.py)."""
    d = np.load(f"{results_dir}/p0_demo/{t}.npz", allow_pickle=True)
    m = np.load(f"{results_dir}/p0_demo/{t}_match.npz", allow_pickle=True)
    return dict(names=d["names"], movable=d["movable"], obj_pos=d["demo_obj_pos"], obj_quat=d["demo_obj_quat"],
                eef_pos=d["demo_eef_pos"], eef_quat=d["demo_eef_quat"], action=np.asarray(m["actions"], np.float32))


def at_rest(z, window=20, tol=1e-3):
    """The demo's objects read AT REST: an object still falling after the scene spawned (the first kept steps of the living-room
    demos catch it up to 12 cm above the table) takes its rest pose over those frames -- the convention of the evaluation
    placements, which the client reads after its settle steps.  Rest = the first frame within `window` whose next step moves it
    less than `tol`; nothing changes for an object already at rest."""
    pos, quat = np.array(z["obj_pos"], np.float64), np.array(z["obj_quat"], np.float64)
    for j in range(pos.shape[1]):
        step = np.linalg.norm(np.diff(pos[:window + 1, j], axis=0), axis=-1)
        r = next((i for i, s in enumerate(step) if s < tol), None)
        if r is not None and r > 0:
            pos[:r, j], quat[:r, j] = pos[r, j], quat[r, j]
    return dict(z, obj_pos=pos, obj_quat=quat)


def controller_goals(z):
    """The demo's CONTROLLER goals per step (OSC delta mode: p* = p + 0.05 a, R* = R(0.5 a_rot) R; as experiments/p0_retarget.py
    `goals`): tracking them reproduces the demo's actions on its own path.  -> (positions [T,3], rotations [T,3,3])."""
    P, Q = [], []
    for i, a in enumerate(np.clip(np.asarray(z["action"], np.float64), -1, 1)):
        P.append(np.asarray(z["eef_pos"][i], np.float64) + 0.05 * a[:3])
        Q.append(aa2m(0.5 * a[3:6]) @ q2m(z["eef_quat"][i]))
    return np.asarray(P), np.asarray(Q)


def cut(z):
    """Rules v2 -> phases [(z, GP, GQ, s0, s1, role, ref, held)]:
    - the manipulated object = the first MOVABLE object displaced > MOVE (lifted or pushed) while the gripper commands close;
    - approach [0, c + CLOSE_TAIL) in its frame (c = the last close before that displacement);
    - carry, to the first release after it (+ OPEN_TAIL), holding it, in the frame of the other object of interest, movable or
      FIXTURE (the world when there is none); the rest in the same frame;
    - no manipulated object (a fixture task: a knob, a door): the whole demo in the frame of the first object of interest."""
    GP, GQ = controller_goals(z)
    cmd = np.asarray(z["action"])[:, 6] > 0
    T, n = len(cmd), len(z["names"])
    moved = []
    for j in range(n):
        if bool(z["movable"][j]):
            disp = np.linalg.norm(np.asarray(z["obj_pos"])[:, j] - np.asarray(z["obj_pos"])[0, j], axis=-1)
            up = np.where((disp > MOVE) & cmd)[0]
            if len(up):
                moved.append((int(up[0]), j))
    if not moved:
        ref = 0 if n else None
        return [(z, GP, GQ, 0, T, "fix" if ref is not None else None, ref, None)]
    L, obj = min(moved)
    closes = [t for t in range(1, L + 1) if cmd[t] and not cmd[t - 1]]
    c = closes[-1] if closes else 0
    rel = [t for t in range(L + 1, T) if not cmd[t] and cmd[t - 1]]
    others = [j for j in range(n) if j != obj]
    rec = next((j for j in others if bool(z["movable"][j])), others[0] if others else None)
    a_end = min(T, c + CLOSE_TAIL)
    c_end = min(T, rel[0] + OPEN_TAIL) if rel else T
    role = "rec" if rec is not None else None
    out = [(z, GP, GQ, 0, a_end, "obj", obj, None), (z, GP, GQ, a_end, c_end, role, rec, obj)]
    if c_end < T:
        out.append((z, GP, GQ, c_end, T, role, rec, None))
    return [p for p in out if p[4] > p[3]]


def type_of(name):
    """An object's TYPE: its name without the instance suffix (ketchup_1 -> ketchup)."""
    return re.sub(r"_\d+$", "", str(name))


def compose(target_names, grasp_src, place_src, obj_type=None, obj_name=None, rec_name=None, transport=False):
    """Cross-task composition (rules v2): the approach of `grasp_src` (a demo that manipulates an object of the target's object type)
    followed by the carry and the rest of `place_src` (a demo with the target's goal relation), each relabelled onto the target
    scene's objects: the grasp source's manipulated object -> the target object of its type; the place source's manipulated
    object -> that target object, its receptacle -> the target object of the receptacle's type.  Other source objects are renamed
    so nothing anchors on them.  obj_type: the target's object type when the grasp source manipulates another type (the sibling
    control: the grasp source's object -> the target's object by ROLE).  obj_name / rec_name: the target's object and receptacle
    by NAME (from its goal), needed when the scene holds several objects of one type.  -> phases."""
    tn = [str(n) for n in target_names]
    g, p = cut(grasp_src), cut(place_src)
    if g[0][5] != "obj" or len(p) < 2 or p[1][5] != "rec":
        raise ValueError("a source has no approach or no carry relative to a receptacle")
    gn, pn = [str(n) for n in grasp_src["names"]], [str(n) for n in place_src["names"]]
    obj_t = obj_name or next(n for n in tn if type_of(n) == (obj_type or type_of(gn[g[0][6]])))
    rec_t = rec_name or next(n for n in tn if type_of(n) == type_of(pn[p[1][6]]) and n != obj_t)

    def relabel(z, names, mapping):
        new = [mapping.get(i, "src:" + n) for i, n in enumerate(names)]
        return dict(z, names=np.array(new))

    zg = relabel(grasp_src, gn, {g[0][6]: obj_t})
    zp = relabel(place_src, pn, {p[1][7]: obj_t, p[1][6]: rec_t})
    rest = [(zp,) + q[1:] for q in p[1:]]
    if transport:
        # keep only the PLACE binding of the source's carry: its descent onto the receptacle (from the held object's last frame at the
        # carry's height); the route there is the source layout's and is replaced by a generic lift-move transport (role "place")
        z_, GP, GQ, s0, s1, role, ref, held = rest[0]
        hz = np.asarray(z_["obj_pos"])[s0:s1, held, 2]
        d0 = s0 + int(np.where(hz >= hz.max() - 0.01)[0][-1])
        rest[0] = (z_, GP, GQ, d0, s1, "place", ref, held)
    return [(zg,) + g[0][1:]] + rest


class PhaseBindHooks:
    """Per-episode state: a queue of (goal position, goal rotation, gripper, source frame) targets, refilled phase by phase."""

    agree = None
    needs_prior = False

    def __init__(self, source, parts, norm, fold=None, state=None, key=None, var=1e-3, H=10, replan=5, reader=None, carry="live",
                 warp=False):
        """reader = dict(W, mx, my, names, enc): a LEARNED pose reader (experiments/p6_reader.py) in place of the client's
        simulator poses -- the objects' positions are predicted from the first frame's images, their orientations are the
        demo's (fixed yaw), and the carry assumes the planned grasp (no live held-object pose)."""
        self.z, self.parts, self.norm = source, parts, norm
        self.reader = reader
        # carry: "live" = the held object's current pose (simulator; identity with a reader); "predicted" = the in-hand pose the
        # approach's source had after its close (the grasp binding), no sensing: identity for a task's own demo, the correction a
        # composed program needs when the place source carried another object type
        self.carry = carry
        # warp: a carry starts where the hand holds the object and ends relative to the receptacle -- the start mismatch (a source
        # whose object-receptacle layout differs from the scene's) is blended out over the transport (until the held object's last
        # frame at the carry's height), so the descent onto the receptacle stays receptacle-relative
        self.warp = bool(warp)
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
        self._inhand = {}
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
        if self.reader is None:
            self._read(meta or {})
        elif not self._objs:
            rd = self.reader
            im, wr = np.asarray(obs["observation/image"])[None], np.asarray(obs["observation/wrist_image"])[None]
            f = rd["enc"]._embed(im, wr) if rd.get("features", "pooled") == "pooled" else rd["enc"]._embed_spatial(im, wr, pool=rd["pool"])
            f = f.float().cpu().numpy().astype(np.float64)
            pos = ((f - rd["mx"]) @ rd["W"] + rd["my"]).reshape(-1, 3)
            quat = {str(n): self.z["obj_quat"][0, j] for j, n in enumerate(self.names)}
            self._objs = {str(n): (pos[i], q2m(quat.get(str(n), np.array([0, 0, 0, 1.0])))) for i, n in enumerate(rd["names"])}
            yr = rd.get("yaw")
            if yr is not None and "W" in yr:                   # objects whose yaw varies (books): read it (experiments/p6_yaw_reader.py)
                P = ((f - yr["mx"]) @ yr["W"] + yr["my"]).reshape(-1)
                vn = [str(n) for n, v in zip(yr["names"], yr["vary"]) if v]
                k = len(vn)
                for j, n in enumerate(vn):
                    if n in self._objs:
                        p_, R_ref = self._objs[n]
                        th = math.atan2(P[k + j], P[j]) - math.atan2(R_ref[1, 0], R_ref[0, 0])
                        self._objs[n] = (p_, aa2m([0.0, 0.0, th]) @ R_ref)
            print(f"[phasebind] reader episode {(meta or {}).get('episode')}: {pos.round(5).tolist()}", flush=True)
        return obs

    # ── planning ────────────────────────────────────────────────────────────
    def _start(self, k, p_now, R_now, g_now):
        z, GP, GQ, s0, s1, role, ref, held = self.parts[k]
        names = [str(x) for x in z["names"]]                 # the part's own source (a composed program mixes sources)
        if ref is not None and names[ref] in self._objs:
            pn, Rn = self._objs[names[ref]]
            ps, Rs = z["obj_pos"][s0, ref].astype(np.float64), q2m(z["obj_quat"][s0, ref])
            M = pose(pn - Rn @ Rs.T @ ps, Rn @ Rs.T)
        else:
            M = np.eye(4)
        if role == "obj" and ref is not None:              # the grasp binding: the source's object -> hand after its close
            e = s1 - 1
            self._inhand[names[ref]] = np.linalg.inv(pose(z["obj_pos"][e, ref], q2m(z["obj_quat"][e, ref]))) @ \
                pose(z["eef_pos"][e], q2m(z["eef_quat"][e]))
        Gs = None
        if held is not None:
            Gs = np.linalg.inv(pose(z["obj_pos"][s0, held], q2m(z["obj_quat"][s0, held]))) @ pose(z["eef_pos"][s0], q2m(z["eef_quat"][s0]))
        if Gs is not None and self.carry == "predicted" and names[held] in self._inhand:
            C = np.linalg.inv(Gs) @ self._inhand[names[held]]
        elif Gs is not None and names[held] in self._objs and self.reader is None:
            ho, Ho = self._objs[names[held]]
            Gt = np.linalg.inv(pose(ho, Ho)) @ pose(p_now, R_now)
            C = np.linalg.inv(Gs) @ Gt
        else:
            C = np.eye(4)
        goals = [M @ pose(GP[i], GQ[i]) @ C for i in range(s0, s1)]
        # the source's ACHIEVED poses under the same transform: where the arm will be when the goal is commanded (the controller
        # lags its goal; predicting that goals are reached would close the gripper before the hand arrives)
        achieved = [M @ pose(z["eef_pos"][i], q2m(z["eef_quat"][i])) @ C for i in range(s0, s1)]
        if self.warp and held is not None and len(goals) > 1:
            hz = np.asarray(z["obj_pos"])[s0:s1, held, 2]
            apex = int(np.where(hz >= hz.max() - 0.01)[0][-1])       # the end of the transport: the last frame at the carry's height
            Dp, Da = p_now - goals[0][:3, 3], m2aa(R_now @ goals[0][:3, :3].T)
            for i in range(len(goals)):
                wgt = max(0.0, 1.0 - i / apex) if apex > 0 else 0.0
                Rc = aa2m(Da * wgt)
                goals[i] = pose(goals[i][:3, 3] + wgt * Dp, Rc @ goals[i][:3, :3])
                achieved[i] = pose(achieved[i][:3, 3] + wgt * Dp, Rc @ achieved[i][:3, :3])
        rel = m2aa(goals[0][:3, :3] @ R_now.T)
        g_hold = float(z["action"][s0 - 1, 6]) if s0 > 0 else g_now
        # the bridge: straight to the part's first goal; for a composed place (role "place") a generic transport -- lift to the
        # descent's height, then across -- in 1 cm steps
        way = [p_now]
        if role == "place" and goals[0][2, 3] > p_now[2]:
            way.append(np.array([p_now[0], p_now[1], goals[0][2, 3]]))
        way.append(goals[0][:3, 3])
        pts = []
        for a_, b_ in zip(way[:-1], way[1:]):
            k = max(1, int(math.ceil(np.linalg.norm(b_ - a_) / 0.01)))
            pts += [a_ + (b_ - a_) * j / k for j in range(1, k + 1)]
        n = max(5, len(pts))
        if len(pts) < n:
            pts = [p_now + (goals[0][:3, 3] - p_now) * j / n for j in range(1, n + 1)]
        prev = (p_now, R_now)
        for j, pt in enumerate(pts, 1):
            Rt = aa2m(rel * j / n) @ R_now
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
