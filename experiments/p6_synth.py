"""P6 follow-up: training scenes for the pose reader rendered by the simulator.

The per-task reader (experiments/p6_reader.py) learns from 30 placements (the practice init states) and misses the thin bottles by
1.3-1.8 cm.  LIBERO samples every object's placement from the task's BDDL regions at each reset, so the simulator renders as many
labelled first frames as wanted: reset (a fresh placement), the client's 10 settle steps, the first frame exactly as the client
sends it (flipped, 224-px padded), and the objects' positions (movable: the observation; fixtures: the body pose).  Privileged
simulator state LABELS these scenes for training only; the reader reads images at test.  The scoring states are never rendered
here (placements are fresh draws).

    PYTHONPATH=... libero .venv/bin/python experiments/p6_synth.py --tasks 52+53 --n 400 --workers 8
"""
import argparse
import importlib.util
import os
import pathlib
import time
from multiprocessing import Pool

import numpy as np

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
R = os.path.join(REPO, "results", "robot")
TASKS = "35+36+39+42+1+6+34+37+40+52+53+62+3+4+5+32+41+43+48+58+63"
spec = importlib.util.spec_from_file_location("p0", os.path.join(REPO, "experiments", "p0_retarget.py"))
P0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P0)


def render(job):
    t, n, seed, rot_only, robot_init, outdir = job
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    names = [str(x) for x in np.load(os.path.join(R, "p0_demo", f"{t}.npz"), allow_pickle=True)["names"]]
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(t)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    np.random.seed(seed + t)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed + t)
    inner = env.env
    init = suite.get_task_init_states(t)
    robot = inner.robots[0]
    idx = list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)
    rng = np.random.default_rng(seed + t)
    img, wr, pos, rot = [], [], [], []
    for k in range(n):
        obs = env.reset()
        if robot_init:                     # the arm as in a random PRACTICE init state (0-29): reset() leaves it at one fixed pose
            src = init[int(rng.integers(0, 30))]
            inner.sim.data.qpos[idx] = src[1:][idx]
            inner.sim.forward()
        for _ in range(P0.WAIT):
            obs, _, _, _ = env.step(P0.DUMMY)
        a = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        w = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224)))
        wr.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(w, 224, 224)))
        poses = [P0.ref_pose(obs, inner, nm, f"{nm}_pos" in obs) for nm in names]
        pos.append(np.stack([q[0] for q in poses]))
        rot.append(np.stack([q[1] for q in poses]))
    env.close()
    pos = np.asarray(pos)
    out = os.path.join(R, outdir, f"{t}.npz") if not rot_only else os.path.join(R, outdir, f"{t}_rot.npz")
    extra = {} if rot_only else dict(image=np.stack(img), wrist=np.stack(wr))
    np.savez_compressed(out, obj_pos=pos, obj_rot=np.asarray(rot), names=np.array(names), **extra)
    spread = np.linalg.norm(pos - pos.mean(0), axis=-1).mean(0) * 100
    return t, n, {nm: round(float(s), 2) for nm, s in zip(names, spread)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rot-only", action="store_true", help="save poses only (positions and rotations) to <outdir>/<t>_rot.npz")
    ap.add_argument("--robot-init", action="store_true", help="the arm's joints from a random practice init state (0-29)")
    ap.add_argument("--outdir", default="p6_synth")
    a = ap.parse_args()
    os.makedirs(os.path.join(R, a.outdir), exist_ok=True)
    jobs = [(int(t), a.n, a.seed, a.rot_only, a.robot_init, a.outdir) for t in a.tasks.replace("+", ",").split(",")]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for t, n, spread in pool.imap_unordered(render, jobs):
            print(f"[p6-synth] task {t}: {n} scenes; mean distance from the mean placement (cm) {spread} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
