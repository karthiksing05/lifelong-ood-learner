"""PhaseBindHooks (lifelong_ood/policy/phase.py; docs/PLAN_TRANSFER.md P1, P6): the retargeting geometry on a synthetic demo.

- rotation helpers round-trip;
- with the objects where the demo had them, the plan IS the demo's controller goals (retarget at identity = identity);
- translating the referenced object moves every goal of its phase by the same vector and leaves the rotations alone;
- rotating it about its own origin keeps each goal fixed in the object's frame;
- the first planned action is feedback from the arm's actual pose, later ones follow the demo's achieved (lagged) path;
- after the last phase the gripper is released;
- reader mode (P6) reads positions once from the first frame's features, takes the demo's orientations, and ignores the held
  object's live pose (C = I).
"""
import math

import numpy as np

from lifelong_ood.policy.phase import PhaseBindHooks, aa2m, m2aa, pose, q2m


def quat_z(th):
    return np.array([0.0, 0.0, math.sin(th / 2), math.cos(th / 2)])


def demo(T=12):
    """One movable object (index 0) and one receptacle (index 1); the hand moves toward the object."""
    names = np.array(["obj_1", "bin_1"])
    obj_pos = np.tile(np.array([[0.10, 0.20, 0.90], [0.30, -0.10, 0.90]]), (T, 1, 1))
    obj_quat = np.tile(np.stack([quat_z(0.3), quat_z(-0.2)]), (T, 1, 1))
    eef_pos = np.stack([np.array([0.0, 0.0, 1.1]) + (np.array([0.10, 0.20, 0.95]) - np.array([0.0, 0.0, 1.1])) * i / (T - 1)
                        for i in range(T)])
    eef_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    action = np.zeros((T, 7), np.float32)
    action[:, 6] = -1.0
    action[T // 2:, 6] = 1.0
    z = dict(names=names, movable=np.array([True, False]), obj_pos=obj_pos, obj_quat=obj_quat, eef_pos=eef_pos, eef_quat=eef_quat,
             action=action)
    GP = eef_pos + 0.01                                           # controller goals lead the achieved path
    GQ = np.stack([q2m(q) for q in eef_quat])
    return z, GP, GQ


def hooks(z, GP, GQ, ref=0, held=None, reader=None):
    parts = [(z, GP, GQ, 0, len(GP), "grasp", ref, held)]
    return PhaseBindHooks(z, parts, norm=None, reader=reader)


def meta_of(z, pos=None, quat=None):
    pos = z["obj_pos"][0] if pos is None else pos
    quat = z["obj_quat"][0] if quat is None else quat
    return {"obj_names": list(z["names"]), "obj_pos": np.asarray(pos).reshape(-1).tolist(), "obj_quat": np.asarray(quat).reshape(-1).tolist()}


def phase_goals(h, z):
    """The queue's goal entries (source frame >= 0) after starting phase 0 from the demo's first pose."""
    h._start(0, z["eef_pos"][0], q2m(z["eef_quat"][0]), -1.0)
    return [(q[0], q[1], q[3]) for q in h.queue if q[3] >= 0]


def test_rotation_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        aa = rng.normal(size=3)
        aa *= min(1.0, 3.0 / np.linalg.norm(aa))
        assert np.allclose(m2aa(aa2m(aa)), aa, atol=1e-8)
    q = quat_z(0.7)
    assert np.allclose(q2m(q), aa2m([0, 0, 0.7]), atol=1e-12)


def test_identity_retarget_is_the_demo():
    z, GP, GQ = demo()
    h = hooks(z, GP, GQ)
    h.rewrite({}, meta_of(z))
    for p, R_, i in phase_goals(h, z):
        assert np.allclose(p, GP[i], atol=1e-12)
        assert np.allclose(R_, GQ[i], atol=1e-12)


def test_translation_moves_goals_by_the_same_vector():
    z, GP, GQ = demo()
    d = np.array([0.03, -0.02, 0.0])
    pos = z["obj_pos"][0].copy()
    pos[0] += d
    h = hooks(z, GP, GQ)
    h.rewrite({}, meta_of(z, pos=pos))
    for p, R_, i in phase_goals(h, z):
        assert np.allclose(p, GP[i] + d, atol=1e-12)
        assert np.allclose(R_, GQ[i], atol=1e-12)


def test_rotation_keeps_goals_fixed_in_the_object_frame():
    z, GP, GQ = demo()
    quat = z["obj_quat"][0].copy()
    quat[0] = quat_z(0.3 + 0.5)                                   # the object yawed by 0.5 rad about its origin
    h = hooks(z, GP, GQ)
    h.rewrite({}, meta_of(z, quat=quat))
    O_src = pose(z["obj_pos"][0, 0], q2m(z["obj_quat"][0, 0]))
    O_new = pose(z["obj_pos"][0, 0], q2m(quat[0]))
    for p, R_, i in phase_goals(h, z):
        rel_src = np.linalg.inv(O_src) @ pose(GP[i], GQ[i])
        rel_new = np.linalg.inv(O_new) @ pose(p, R_)
        assert np.allclose(rel_src, rel_new, atol=1e-10)


def test_plan_feedback_then_lagged_path_and_release():
    z, GP, GQ = demo()
    h = hooks(z, GP, GQ)
    h.rewrite({}, meta_of(z))
    state8 = np.concatenate([z["eef_pos"][0] + np.array([0.0, 0.0, 0.004]), m2aa(q2m(z["eef_quat"][0])), [0.04, -0.04]])
    q0 = list(h.queue)
    acts, src = h.plan(state8)
    assert acts.shape == (h.H, 7)
    # action 0: from the ACTUAL pose to the first target; action 1: from the first target's achieved pose to the second
    h2 = hooks(z, GP, GQ)
    h2.rewrite({}, meta_of(z))
    h2._fill(state8[:3], aa2m(state8[3:6]), -1.0)
    t0, t1 = h2.queue[0], h2.queue[1]
    assert np.allclose(acts[0, :3], np.clip((t0[0] - state8[:3]) / 0.05, -1, 1), atol=1e-6)
    assert np.allclose(acts[1, :3], np.clip((t1[0] - t1[4]) / 0.05, -1, 1), atol=1e-6)
    assert q0 == []                                               # the queue fills lazily
    # run the queue out: past the last phase the gripper opens
    for _ in range(20):
        acts, src = h.plan(state8)
    assert np.all(acts[:, 6] == -1.0) and all(s == -1 for s in src)


class FakeEncoder:
    """Stands in for VinnPolicy: _embed returns fixed features; with W = I the reader predicts them."""

    def __init__(self, feats):
        self.feats = feats

    def _embed(self, images, wrist):
        import torch
        return torch.as_tensor(self.feats[None])


def test_reader_mode_reads_once_takes_demo_orientations_and_ignores_the_held_pose():
    z, GP, GQ = demo()
    target = np.array([0.12, 0.18, 0.91, 0.31, -0.08, 0.90])
    reader = dict(W=np.eye(6), mx=np.zeros(6), my=np.zeros(6), names=["obj_1", "bin_1"], enc=FakeEncoder(target))
    h = hooks(z, GP, GQ, ref=1, held=0, reader=reader)
    obs = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/wrist_image": np.zeros((224, 224, 3), np.uint8)}
    h.rewrite(obs, {"episode": 30})
    assert np.allclose(h._objs["obj_1"][0], target[:3]) and np.allclose(h._objs["bin_1"][0], target[3:])
    assert np.allclose(h._objs["bin_1"][1], q2m(z["obj_quat"][0, 1]))            # the demo's orientation
    reader["enc"] = FakeEncoder(np.zeros(6))
    h.rewrite(obs, {"episode": 30})                                                 # read once per episode
    assert np.allclose(h._objs["obj_1"][0], target[:3])
    # the held object's live pose is not used (C = I): goals are the receptacle-frame transform of the demo's goals alone
    d = target[3:] - z["obj_pos"][0, 1]
    Rn = q2m(z["obj_quat"][0, 1])
    for p, R_, i in phase_goals(h, z):
        assert np.allclose(p, GP[i] + d, atol=1e-12) and np.allclose(R_, Rn @ Rn.T @ GQ[i], atol=1e-12)
    h.reset()
    assert h._objs == {}


def test_warped_carry_starts_at_the_hand_and_ends_receptacle_relative():
    """A carry whose source layout differs (the receptacle far from where the object is held) starts where the hand is and still
    ends relative to the receptacle; after the transport the warped and unwarped paths agree."""
    T = 30
    names = np.array(["obj_1", "bin_1"])
    obj_pos = np.zeros((T, 2, 3))
    obj_pos[:, 1] = [0.3, 0.3, 0.9]                                   # the receptacle
    eef_pos = np.zeros((T, 3))
    for i in range(T):                                                # grasp at 0-9, lift 10-14, transport 15-24, descend 25-29
        if i < 10:
            p = [0.0, 0.0, 0.95]
        elif i < 15:
            p = [0.0, 0.0, 0.95 + 0.02 * (i - 9)]
        elif i < 25:
            p = [0.3 * (i - 14) / 10, 0.3 * (i - 14) / 10, 1.05]
        else:
            p = [0.3, 0.3, 1.05 - 0.02 * (i - 24)]
        eef_pos[i] = p
        obj_pos[i, 0] = np.array(p) - [0, 0, 0.05]
    z = dict(names=names, movable=np.array([True, False]), obj_pos=obj_pos, obj_quat=np.tile([0, 0, 0, 1.0], (T, 2, 1)),
             eef_pos=eef_pos, eef_quat=np.tile([1.0, 0, 0, 0], (T, 1)), action=np.zeros((T, 7), np.float32))
    GP = eef_pos.copy()
    GQ = np.stack([q2m(q) for q in z["eef_quat"]])
    parts = [(z, GP, GQ, 10, T, "rec", 1, 0)]
    moved = np.array([-0.3, -0.6, 0.9])                               # the scene's receptacle, far from the source's layout
    meta = {"obj_names": ["obj_1", "bin_1"], "obj_pos": [0.0, 0.0, 0.9] + moved.tolist(), "obj_quat": [0, 0, 0, 1.0] * 2}
    plans = {}
    for warp in (False, True):
        h = PhaseBindHooks(z, parts, norm=None, warp=warp, carry="predicted")
        h._inhand["obj_1"] = np.linalg.inv(pose(obj_pos[9, 0], np.eye(3))) @ pose(eef_pos[9], q2m(z["eef_quat"][9]))
        h.rewrite({}, meta)
        h._start(0, eef_pos[9], q2m(z["eef_quat"][9]), 1.0)
        plans[warp] = [q[0] for q in h.queue if q[3] >= 0]
    assert np.linalg.norm(plans[True][0] - eef_pos[9]) < 0.02          # starts at the hand
    assert np.linalg.norm(plans[False][0] - eef_pos[9]) > 0.3          # unwarped: starts where the source layout puts it
    assert np.allclose(plans[True][-1], plans[False][-1])               # the same end relative to the receptacle
    assert np.allclose(plans[True][15:], plans[False][15:])             # after the transport (source frame 25+) unchanged


def test_handoff_releases_the_guide_between_grasp_and_descent():
    """Handoff: after the approach the prior acts alone (the plan is held, not extended into the carry); the guide resumes with
    the place source's descent once the hand is within the radius of its start, or with the whole carry after the patience."""
    T = 30
    names = np.array(["obj_1", "bin_1"])
    obj_pos = np.zeros((T, 2, 3))
    obj_pos[:, 1] = [0.3, 0.3, 0.9]
    eef_pos = np.zeros((T, 3))
    for i in range(T):                                                # grasp at 0-9, lift 10-14, transport 15-24, descend 25-29
        p = ([0.0, 0.0, 0.95] if i < 10 else [0.0, 0.0, 0.95 + 0.02 * (i - 9)] if i < 15 else
             [0.3 * (i - 14) / 10, 0.3 * (i - 14) / 10, 1.05] if i < 25 else [0.3, 0.3, 1.05 - 0.02 * (i - 24)])
        eef_pos[i] = p
        obj_pos[i, 0] = np.array(p) - [0, 0, 0.05]
    action = np.zeros((T, 7), np.float32)
    action[5:, 6] = 1.0
    z = dict(names=names, movable=np.array([True, False]), obj_pos=obj_pos, obj_quat=np.tile([0, 0, 0, 1.0], (T, 2, 1)),
             eef_pos=eef_pos, eef_quat=np.tile([1.0, 0, 0, 0], (T, 1)), action=action)
    GP = eef_pos.copy()
    GQ = np.stack([q2m(q) for q in z["eef_quat"]])
    parts = [(z, GP, GQ, 0, 10, "obj", 0, None), (z, GP, GQ, 10, T, "rec", 1, 0)]
    meta = {"obj_names": ["obj_1", "bin_1"], "obj_pos": [0.0, 0.0, 0.9, 0.3, 0.3, 0.9], "obj_quat": [0, 0, 0, 1.0] * 2}
    for far_q, expect in ((3, "descent"), (100, "timeout")):
        h = PhaseBindHooks(z, parts, norm=None, carry="predicted", handoff=True, radius=0.05, patience=1.5)
        h.rewrite({}, meta)
        st = np.concatenate([eef_pos[0], m2aa(q2m(z["eef_quat"][0])), [0, 0]])
        n = 0
        while not h._prior_carries(st):                               # the approach, then the hold until the prior takes over
            _, src = h.plan(st)
            assert all(i < 10 for i in src)                           # never into the carry
            n += 1
            assert n < 20
        assert h.hand is not None and h.hand["d0"] == 24 and not h.queue
        assert np.allclose(h.hand["target"], eef_pos[24])
        far = np.concatenate([[0.0, 0.0, 1.05], st[3:]])
        for _ in range(far_q):
            if not h._prior_carries(far):
                break
        near = np.concatenate([eef_pos[24] + [0.02, 0.0, 0.0], st[3:]])
        if expect == "descent":
            assert not h._prior_carries(near)
            src = [q[3] for q in h.queue if q[3] >= 0]
            assert src[0] == 24 and src[-1] == T - 1               # from the descent's start, bridged from the hand
        else:
            assert h.hand["done"]                                     # patience 1.5 x 20 frames / 5 per query = 6 queries
            src = [q[3] for q in h.queue if q[3] >= 0]
            assert src[0] == 10                                       # the whole carry
        assert h.k == 2


def test_events_fire_on_arrival_not_on_the_clock():
    """events=True: a release in the next replan window is held (gripper unchanged, queue not consumed past it) while the hand is
    more than tol from the event's achieved pose; it fires once the hand arrives, or after max_wait queries."""
    z, GP, GQ = demo(T=12)
    for arrive_after, expect_fire_at in ((3, 3), (100, 4)):
        h = PhaseBindHooks(z, [(z, GP, GQ, 0, 12, "grasp", 0, None)], norm=None, events=True, tol=0.02, max_wait=4)
        h.rewrite({}, meta_of(z))
        h.g_cmd = 1.0                                                   # holding
        ev = np.array([0.3, 0.3, 1.0])
        h.queue = [(ev + [0.0, 0.0, 0.01 * j], np.eye(3), 1.0, j, ev, np.eye(3)) for j in range(2)] + \
                  [(ev, np.eye(3), -1.0, 2 + j, ev, np.eye(3)) for j in range(20)]
        h.k = 1
        far = np.concatenate([ev + [0.2, 0.0, 0.0], [0, 0, 0], [0, 0]])
        near = np.concatenate([ev + [0.01, 0.0, 0.0], [0, 0, 0], [0, 0]])
        fired = None
        for q in range(8):
            acts, src = h.plan(near if q >= arrive_after else far)
            if h.g_cmd < 0:
                fired = q
                break
            assert all(a_[6] > 0 for a_ in acts)                          # the gripper stays closed while held
        assert fired == expect_fire_at, (arrive_after, fired)


def test_events_bridge_corners_are_passed_on_arrival():
    """events=True: a lift-then-across bridge (role "place") marks its corner; the plan holds at the corner until the hand is there."""
    z, GP, GQ = demo(T=12)
    h = PhaseBindHooks(z, [(z, GP, GQ, 0, 12, "place", 1, None)], norm=None, events=True, tol=0.02, max_wait=50)
    h.rewrite({}, meta_of(z))
    p0 = np.array([0.0, 0.0, 0.8])                                  # below the part's first goal: lift first, then across
    h._start(0, p0, np.eye(3), -1.0)
    corners = [e for e in h.queue if e[3] == -4]
    assert len(corners) >= 1 and abs(corners[0][0][2] - h.queue[len(h.queue) - 12][0][2]) < 1e-9   # the first corner: at the goal height
    h.k = 1
    st = np.concatenate([p0, [0, 0, 0], [0, 0]])
    for _ in range(40):
        _, src = h.plan(st)
        if -4 in src[:1]:
            break
    assert h.queue[0][3] == -4                                      # held at the corner while the hand stays below
    n_before = len(h.queue)
    h.plan(st)
    assert len(h.queue) == n_before                                 # nothing consumed
    arrived = np.concatenate([h.queue[0][0], [0, 0, 0], [0, 0]])
    h.plan(arrived)
    assert len(h.queue) < n_before                                  # passed on arrival


def test_contact_start_reads_the_last_free_space_verb():
    """contact_start: the frames after the phase's last free-space prototype verb ("... move"), less the lead-in; None without one."""
    from lifelong_ood.policy.phase import contact_start
    verbs = ["open move"] * 10 + ["open reach down"] * 4 + ["open grip change"] * 2
    assert contact_start(verbs, 0, 16, lead=3) == 10 - 3 + 0
    assert contact_start(verbs, 0, 16, lead=0) == 10
    assert contact_start(["open reach down"] * 5, 0, 5) is None
    assert contact_start(verbs, 8, 16, lead=5) == 8                     # never before the phase's start


def test_cross_type_place_transfers_position_only():
    """cross_pos: with a place source of another object type the in-hand correction is a pure translation that puts the TARGET
    object's origin where the source object's was (the hand keeps the source hand's orientation); same type: the full correction."""
    T = 30
    names = np.array(["obj_1", "bin_1"])
    obj_pos = np.zeros((T, 2, 3))
    obj_pos[:, 1] = [0.3, 0.3, 0.9]
    eef_pos = np.array([[0.0, 0.0, 1.0 + 0.001 * i] for i in range(T)])
    obj_pos[:, 0] = eef_pos - [0.0, 0.0, 0.05]
    z = dict(names=names, movable=np.array([True, False]), obj_pos=obj_pos, obj_quat=np.tile([0, 0, 0, 1.0], (T, 2, 1)),
             eef_pos=eef_pos, eef_quat=np.tile([1.0, 0, 0, 0], (T, 1)), action=np.ones((T, 7), np.float32))
    GP = eef_pos.copy()
    GQ = np.stack([q2m(q) for q in z["eef_quat"]])
    meta = {"obj_names": ["obj_1", "bin_1"], "obj_pos": [0.0, 0.0, 0.95, 0.3, 0.3, 0.9], "obj_quat": [0, 0, 0, 1.0] * 2}
    tilt = aa2m([0.4, 0.0, 0.0])                                       # the target's grip: tilted, object 8 cm below the hand
    inhand = np.linalg.inv(pose(np.array([0.0, 0.0, 0.0]), np.eye(3))) @ pose(np.array([0.0, 0.0, 0.08]), tilt)
    out = {}
    for cross in (False, True):
        zc = dict(z, cross_type=cross)
        h = PhaseBindHooks(zc, [(zc, GP, GQ, 0, T, "rec", 1, 0)], norm=None, carry="predicted", cross_pos=True)
        h._inhand["obj_1"] = inhand
        h.rewrite({}, meta)
        goals, _, _, _ = h._goals(0, eef_pos[0], np.eye(3), 1.0)
        out[cross] = goals[0]
    src = pose(GP[0], GQ[0])
    assert np.allclose(out[True][:3, :3], src[:3, :3])                 # position only: the source hand's orientation
    obj_t = out[True] @ np.linalg.inv(inhand)                         # where the target object's origin ends up
    assert np.allclose(obj_t[:3, 3], obj_pos[0, 0], atol=1e-9)         # = the source object's origin
    assert not np.allclose(out[False][:3, :3], src[:3, :3])           # same type: the full correction rotates the hand
