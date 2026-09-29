"""P0 (docs/PLAN_TRANSFER.md; ledger m3_prereg tag p0): replay recorded LIBERO episodes bit-exactly and log the object state.

A recorded episode (experiments/libero_client.py --record: 8-d state, executed actions, images, success) is replayed the way the
recorded session produced it: a fresh env as at process start (np.random.seed, env, env.seed), as many resets as preceded the
episode in its session (the trial index: every practice session ran its init states from 0 in order), reset + set_init_state,
the client's 10 settle steps, then the recorded actions.  Per step: the 8-d state (checked against the recording), the eef pose,
the finger qpos, the gripper command, and for every object of interest its pose (movable objects from the observation, fixtures
from the simulator's body pose) and, for movable objects, robosuite's _check_grasp.

    PYTHONPATH=.../third_party/libero .venv/bin/python experiments/p0_replay.py --experiments m5_practice --tasks 42,52 --workers 8
"""
import argparse
import glob
import math
import os
import pathlib
import re
import time
from multiprocessing import Pool

import numpy as np

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
R = os.path.join(REPO, "results", "robot")
OUT = os.path.join(R, "p0_replay")
DUMMY = [0.0] * 6 + [-1.0]
SEED, WAIT = 7, 10


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def st8(obs):
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])).astype(np.float32)


def replay(job):
    rec_path, task, trial, out_path = job
    if os.path.exists(out_path):
        return out_path, "exists"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    rec = np.load(rec_path)
    acts, ref = rec["action"], rec["state"]
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    t = suite.get_task(task)
    init = suite.get_task_init_states(task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    t0 = time.time()
    np.random.seed(SEED)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(SEED)
    for _ in range(trial):
        env.reset()
    env.reset()
    obs = env.set_init_state(init[trial % len(init)])
    for _ in range(WAIT):
        obs, _, _, _ = env.step(DUMMY)
    inner = env.env
    names = list(env.obj_of_interest)
    movable = [n for n in names if f"{n}_pos" in obs]
    log = dict(state=[], eef_pos=[], eef_quat=[], grip=[], obj_pos=[], obj_quat=[], grasp=[])
    done = False
    for a in acts:
        log["state"].append(st8(obs))
        log["eef_pos"].append(np.asarray(obs["robot0_eef_pos"], np.float32))
        log["eef_quat"].append(np.asarray(obs["robot0_eef_quat"], np.float32))           # xyzw
        log["grip"].append(np.asarray(obs["robot0_gripper_qpos"], np.float32))
        P, Q, G = [], [], []
        for n in names:
            if n in movable:
                P.append(obs[f"{n}_pos"]); Q.append(obs[f"{n}_quat"])                     # xyzw
                G.append(bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=inner.objects_dict[n])))
            else:
                bid = inner.obj_body_id[n]
                w, x, y, z = inner.sim.data.body_xquat[bid]
                P.append(np.array(inner.sim.data.body_xpos[bid])); Q.append(np.array([x, y, z, w]))
                G.append(False)
        log["obj_pos"].append(np.asarray(P, np.float32)); log["obj_quat"].append(np.asarray(Q, np.float32))
        log["grasp"].append(np.asarray(G, bool))
        obs, _, done, _ = env.step(np.asarray(a, np.float32).tolist())
        if done:
            break
    env.close()
    S = np.stack(log["state"])
    n = min(len(S), len(ref))
    d = np.abs(S[:n] - ref[:n]).max(1)
    diff = np.where(d > 0)[0]
    out = {k: np.stack(v) for k, v in log.items()}
    np.savez_compressed(out_path, **out, action=acts[:len(S)], names=np.array(names), movable=np.array([x in movable for x in names]),
                        exact=bool(len(diff) == 0 and len(S) == len(ref)), first_diff=int(diff[0]) if len(diff) else -1,
                        success=bool(done), rec_success=bool(rec["success"]), task=task, trial=trial, secs=time.time() - t0)
    return out_path, f"exact={len(diff) == 0 and len(S) == len(ref)} success={bool(done)}/{bool(rec['success'])} ({time.time() - t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="m5_practice,m6_round1,m6_round2")
    ap.add_argument("--tasks", default="35,36,39,42,1,6,34,37,40,52,53,62,3,4,5,32,41,43,48,58,63")
    ap.add_argument("--trials", default="", help="only these trials (comma list)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    only = {int(x) for x in a.trials.replace("+", ",").split(",") if x}
    jobs = []
    for e in a.experiments.replace("+", ",").split(","):
        for t in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
            for f in sorted(glob.glob(os.path.join(R, f"{e}_{t}_*_rec", f"task{t}_trial*.npz"))):
                k = int(re.search(r"trial(\d+)\.npz$", f).group(1))
                if only and k not in only:
                    continue
                jobs.append((f, t, k, os.path.join(OUT, f"{e}_{t}_trial{k}.npz")))
    jobs.sort(key=lambda j: -j[2])                                         # long reset chains first
    print(f"[p0] {len(jobs)} episodes, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for path, msg in pool.imap_unordered(replay, jobs):
            print(f"[p0] {os.path.basename(path)}: {msg}", flush=True)


if __name__ == "__main__":
    main()
