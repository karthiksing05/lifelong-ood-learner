"""FRESH init states: LIBERO's own placement sampler (the BDDL regions) run under fixed seeds, saved once per task so every arm is
scored from the same never-before-used start states (all 50 stored init states of LIBERO-90 have been used by earlier reads).
Run in the LIBERO client's venv (experiments/fresh_states.sbatch).  Output: results/robot/fresh_states/libero_90_<task>.npy
[n, state_dim], in the stored init states' format (MjSimState flattened), checked against it.

    sbatch --export=ALL,TASKS=10+15,N=20,BASE=100000 experiments/fresh_states.sbatch
"""
import os
import pathlib
import sys

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OUT = "/coc/flash10/ksingara3/lifelong-ood-learner/results/robot/fresh_states"


def main():
    tasks = [int(x) for x in os.environ["TASKS"].split("+")]
    n, base = int(os.environ.get("N", 20)), int(os.environ.get("BASE", 100000))
    os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    for tid in tasks:
        task = suite.get_task(tid)
        stored = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
        states = []
        for k in range(n):
            np.random.seed(base + k)
            env.seed(base + k)
            env.reset()                                         # a new placement from the task's regions
            s = env.sim.get_state().flatten()
            assert s.shape == stored[0].shape, (s.shape, stored[0].shape)
            states.append(s)
        states = np.stack(states)
        d = np.min(np.linalg.norm(states[:, None, :] - stored[None, :, :], axis=-1))
        np.save(os.path.join(OUT, f"libero_90_{tid}.npy"), states)
        print(f"[fresh] task {tid}: {n} states (seed base {base}), min distance to a stored init state {d:.3f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
