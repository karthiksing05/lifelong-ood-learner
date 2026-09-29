"""P0c (docs/PLAN_TRANSFER.md; ledger m3_prereg tag p0): the ceiling of object-relative retargeting, no policy.

Source = the first successful recorded practice episode of a task (trial order; m5_practice, then m6_round1, m6_round2), with its
per-step eef and object poses from the bit-exact replay (experiments/p0_replay.py).  Its eef path is cut at the gripper-command
switches; a segment that ends in a close which grasps a movable object is expressed in that object's frame at the segment start,
a segment that carries an object to its release in the frame of the other movable object of interest (the receptacle), anything
else in the world frame (fixtures move <= 1 cm); 8 steps after a close and 5 after a release stay with the segment before.  At a
target placement each segment is retargeted with the MimicGen transform T_new = T_ref,new T_ref,src^-1 T_ee,src (T_ref,new read
at the segment's start), reached by a straight interpolation (1 cm per step, >= 5 steps), and tracked with OSC delta actions
(robosuite: position goal = current + 0.05 a, orientation goal = R(0.5 a) R_current) and the source's gripper commands.

Arms per (task, init state 30-49), each run as the evaluation sessions run (one env per task and arm, states 30-49 in order):
    replay    the source's recorded actions, open loop (the pre-registered control)
    track     the source's eef poses tracked untransformed (the tracker alone)
    retarget  the transformed segments tracked

    PYTHONPATH=.../third_party/libero .venv/bin/python experiments/p0_retarget.py --tasks 42+32+1+35+40+36+52+62+41 --workers 9
"""
import argparse
import glob
import math
import os
import pathlib
import time
from multiprocessing import Pool

import numpy as np

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
R = os.path.join(REPO, "results", "robot")
DUMMY = [0.0] * 6 + [-1.0]
SEED, WAIT, MAX_STEPS = 7, 10, 400
CLOSE_TAIL, OPEN_TAIL, GRASP_WINDOW = 8, 5, 20
GOALS = False


def q2m(q):
    x, y, z, w = np.asarray(q, np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2aa(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(th)
    if s < 1e-6:                                     # near pi: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(Rm) + 1) / 2, 0))
        return th * ax
    return th * v / (2 * s)


def slerp_path(R0, R1, n):
    rel = m2aa(R1 @ R0.T)
    out = []
    for k in range(1, n + 1):
        a = rel * k / n
        th = np.linalg.norm(a)
        if th < 1e-9:
            Rk = np.eye(3)
        else:
            u = a / th
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            Rk = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        out.append(Rk @ R0)
    return out


def source_episode(task):
    for e in ("m5_practice", "m6_round1", "m6_round2", "p3_seed"):         # p3_seed: the prior's recorded successes (P3b)
        fs = sorted(glob.glob(os.path.join(R, "p0_replay", f"{e}_{task}_trial*.npz")), key=lambda f: int(f.rsplit("trial", 1)[1][:-4]))
        for f in fs:
            z = dict(np.load(f, allow_pickle=True))
            if bool(z["exact"]) and bool(z["rec_success"]):
                return f, z
    return None, None


def goals(z):
    """The source's CONTROLLER goals per step (OSC delta mode: p_goal = p + 0.05 a, R_goal = R(0.5 a_rot) R): tracking these with
    delta actions reproduces the source's own actions on its own path (tracking the ACHIEVED poses under-commands: the controller
    lags its goal).  -> (positions [T,3], rotations [T,3,3])."""
    P, Q = [], []
    for i, a in enumerate(z["action"]):
        a = np.clip(np.asarray(a, np.float64), -1, 1)
        P.append(z["eef_pos"][i].astype(np.float64) + 0.05 * a[:3])
        th = np.linalg.norm(0.5 * a[3:6])
        if th < 1e-9:
            Rd = np.eye(3)
        else:
            u = 0.5 * a[3:6] / th
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            Rd = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        Q.append(Rd @ q2m(z["eef_quat"][i]))
    return np.asarray(P), np.asarray(Q)


def segments(z):
    """[(start, end, reference index or None)] over the source's steps."""
    cmd = z["action"][:, 6] > 0
    T = len(cmd)
    names = list(z["names"]); movable = list(z["movable"])
    sw = [t for t in range(1, T) if cmd[t] != cmd[t - 1]]
    bounds, refs = [0], []
    held = None
    for t in sw:
        if cmd[t]:                                   # a close: the segment before approaches the object grasped after it
            g = [j for j in range(len(names)) if movable[j] and z["grasp"][t:t + GRASP_WINDOW + 1, j].any()]
            ref = g[0] if g else None
            held = ref
            end = min(T, t + CLOSE_TAIL)
        else:                                        # a release: the segment before carries `held` to the receptacle
            rec = [j for j in range(len(names)) if movable[j] and j != held]
            ref = rec[0] if (held is not None and rec) else None
            held = None
            end = min(T, t + OPEN_TAIL)
        if end > bounds[-1]:
            refs.append(ref)
            bounds.append(end)
    if bounds[-1] < T:
        refs.append(refs[-1] if refs else None)
        bounds.append(T)
    return [(bounds[k], bounds[k + 1], refs[k]) for k in range(len(refs))]


def ref_pose(obs, inner, name, movable):
    if movable:
        return np.asarray(obs[f"{name}_pos"], np.float64), q2m(obs[f"{name}_quat"])
    bid = inner.obj_body_id[name]
    w, x, y, zq = inner.sim.data.body_xquat[bid]
    return np.array(inner.sim.data.body_xpos[bid], np.float64), q2m([x, y, zq, w])


def track_action(obs, p_t, R_t, g):
    p = np.asarray(obs["robot0_eef_pos"], np.float64)
    Rc = q2m(obs["robot0_eef_quat"])
    a = np.zeros(7)
    a[:3] = np.clip((p_t - p) / 0.05, -1, 1)
    a[3:6] = np.clip(m2aa(R_t @ Rc.T) / 0.5, -1, 1)
    a[6] = g
    return a


def run_arm(job):
    task, arm = job[0], job[1]
    goal_mode = GOALS
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    src_path, z = source_episode(task)
    if z is None:
        return task, arm, None, "no source success"
    names = [str(x) for x in z["names"]]; movable = [bool(m) for m in z["movable"]]
    segs = segments(z)
    GP, GQ = goals(z)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    t_ = suite.get_task(task)
    init = suite.get_task_init_states(task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / t_.problem_folder / t_.bddl_file
    np.random.seed(SEED)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(SEED)
    inner = env.env
    out = {}
    placements = list(range(30, 50))
    if len(job) > 2:                                  # self-check: the source's own placement, its session's reset count
        src_trial = int(z["trial"])
        env.close()
        np.random.seed(SEED)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(SEED)
        inner = env.env
        for _ in range(src_trial):
            env.reset()
        placements = [src_trial]
    for k in placements:
        env.reset()
        obs = env.set_init_state(init[k])
        for _ in range(WAIT):
            obs, _, done, _ = env.step(DUMMY)
        done, steps = False, 0

        def step(a):
            nonlocal obs, done, steps
            obs, _, done, _ = env.step(np.asarray(a, np.float64).tolist())
            steps += 1
            return done or steps >= MAX_STEPS

        if arm == "replay":
            for a in z["action"]:
                if step(a):
                    break
        else:
            stop = False
            for (s0, s1, ref) in segs:
                if stop:
                    break
                if arm == "retarget" and ref is not None:
                    pn, Rn = ref_pose(obs, inner, names[ref], movable[ref])
                    ps, Rs = z["obj_pos"][s0, ref].astype(np.float64), q2m(z["obj_quat"][s0, ref])
                    M_R, M_p = Rn @ Rs.T, pn - Rn @ Rs.T @ ps           # T_new T_src^-1 as (M_R, M_p)
                else:
                    M_R, M_p = np.eye(3), np.zeros(3)
                if goal_mode:
                    P = [M_R @ GP[i] + M_p for i in range(s0, s1)]
                    Q = [M_R @ GQ[i] for i in range(s0, s1)]
                else:
                    P = [M_R @ z["eef_pos"][i].astype(np.float64) + M_p for i in range(s0, s1)]
                    Q = [M_R @ q2m(z["eef_quat"][i]) for i in range(s0, s1)]
                # reach the segment's start (1 cm per step, >= 5 steps), holding the current gripper command
                p_now = np.asarray(obs["robot0_eef_pos"], np.float64)
                n = max(5, int(math.ceil(np.linalg.norm(P[0] - p_now) / 0.01)))
                g_hold = float(z["action"][max(s0 - 1, 0), 6])
                for k2, Rk in enumerate(slerp_path(q2m(obs["robot0_eef_quat"]), Q[0], n)):
                    p_k = p_now + (P[0] - p_now) * (k2 + 1) / n
                    if step(track_action(obs, p_k, Rk, g_hold)):
                        stop = True
                        break
                if stop:
                    break
                for i in range(s0, s1):
                    j = (i - s0) if goal_mode else min(i - s0 + 1, len(P) - 1)
                    if step(track_action(obs, P[j], Q[j], float(z["action"][i, 6]))):
                        stop = True
                        break
            while not done and steps < MAX_STEPS and steps < len(z["action"]) + 60:   # let a final release settle
                if step([0.0] * 6 + [float(z["action"][-1, 6])]):
                    break
        out[k] = bool(done)
    env.close()
    return task, arm, out, os.path.basename(src_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="42+32+1+35+40+36+52+62+41")
    ap.add_argument("--arms", default="replay+track+retarget")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default=os.path.join(R, "p0_retarget.json"))
    ap.add_argument("--goals", action="store_true", help="track the source's controller goals (exploratory P0c'; default: achieved poses, P0c)")
    ap.add_argument("--selfcheck", action="store_true", help="run at the source's own placement only")
    a = ap.parse_args()
    global GOALS
    GOALS = a.goals
    jobs = [(int(t), arm) + (("self",) if a.selfcheck else ()) for t in a.tasks.replace("+", ",").split(",") for arm in a.arms.replace("+", ",").split(",")]
    res = {}
    t0 = time.time()
    with Pool(a.workers) as pool:
        for task, arm, out, src in pool.imap_unordered(run_arm, jobs):
            res.setdefault(str(task), {})[arm] = dict(success=out, source=src, k=None if out is None else int(sum(out.values())))
            print(f"[p0c] task {task} {arm}: {None if out is None else sum(out.values())}/{None if out is None else len(out)} (source {src}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    import json
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p0c] -> {a.out}")


if __name__ == "__main__":
    main()
