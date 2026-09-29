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
