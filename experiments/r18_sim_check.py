"""Is a LIBERO episode reproducible from its init state alone?  (Experiment 2's prefix gate; libero env, python 3.8.)

Replays one recorded episode's actions open-loop as the 1st, 2nd, ... episode of one env session (client order: np.random.seed,
env, env.seed, then reset + set_init_state per episode), compares the 8-d states with the recording, and lists every MuJoCo
model / data array and controller attribute that differs between sessions right after set_init_state.

    sbatch --partition=tail-lab --account=tail-lab --export=ALL,ENV=libero,SCRIPT="experiments/r18_sim_check.py --task 58 --trial 3" \
        slurm/jobs/robot_gpu.sbatch
"""
import argparse
import math
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def snapshot(env):
    """Every numeric array of sim.model / sim.data and of the robot's controllers."""
    sim = env.env.sim
    out = {}
    for src, obj in (("model", sim.model), ("data", sim.data)):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:      # noqa: BLE001
                continue
            if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.size:
                out[f"{src}.{name}"] = np.array(v, copy=True)
    for r in env.env.robots:
        for cname in ("controller", "gripper_controller"):
            c = getattr(r, cname, None)
            if c is None:
                continue
            cs = c.values() if isinstance(c, dict) else [c]
            for k, cc in enumerate(cs):
                for name, v in vars(cc).items():
                    if isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
                        out[f"{cname}{k}.{name}"] = np.array(v, copy=True)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        out[f"{cname}{k}.{name}"] = np.array(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=58)
    ap.add_argument("--trial", type=int, default=3)
    ap.add_argument("--rec", default="")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", default="plain", help="plain | reseed (np.random.seed + env.seed before each episode) | "
                    "count (reseed, then `trial` extra resets: the recorded episode's reset count)")
    a = ap.parse_args()
    R = pathlib.Path("/coc/flash10/ksingara3/lifelong-ood-learner/results/robot")
    rec = np.load(a.rec or next(R.glob(f"m5_practice_{a.task}_*_rec/task{a.task}_trial{a.trial}.npz")))
    acts, ref = rec["action"], rec["state"]
    np.random.seed(a.seed)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(a.task)
    init = suite.get_task_init_states(a.task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(a.seed)
    snaps = []
    for e in range(a.episodes):
        if a.mode in ("reseed", "count"):
            np.random.seed(a.seed)
            env.seed(a.seed)
            if a.mode == "count":
                for _ in range(a.trial):
                    env.reset()
        env.reset()
        obs = env.set_init_state(init[a.trial % len(init)])
        snaps.append(snapshot(env))
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
        sts = []
        for t, act in enumerate(acts):
            sts.append(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])))
            obs, _, done, _ = env.step(np.asarray(act, np.float32).tolist())
            if done:
                break
        sts = np.asarray(sts, np.float32)
        n = min(len(sts), len(ref))
        d = np.abs(sts[:n] - ref[:n]).max(1)
        first = np.where(d > 0)[0]
        print(f"[sim] mode {a.mode}, episode {e + 1} of the session: {len(sts)} steps (recorded {len(ref)}), success {bool(done)} "
              f"(recorded {bool(rec['success'])}), first state difference at step {first[0] if len(first) else None}, "
              f"max {d.max():.2e}", flush=True)
    keys = sorted(set.intersection(*[set(s) for s in snaps]))
    for k in keys:
        vals = [s[k] for s in snaps]
        if any(v.shape != vals[0].shape for v in vals):
            print(f"[diff] {k}: shapes differ {[v.shape for v in vals]}")
            continue
        dif = [float(np.abs(v.astype(np.float64) - vals[0].astype(np.float64)).max()) for v in vals]
        if max(dif) > 0:
            print(f"[diff] {k} {vals[0].shape}: max |x_e - x_1| per episode {['%.2e' % x for x in dif]}")
    env.close()


if __name__ == "__main__":
    main()
