"""M0 smoke (runs in the LIBERO client venv): create every LIBERO-Goal task headless, reset to the
benchmark's first initial state, step a few dummy actions, save one rendered frame per task, and print
the observation keys and shapes.  Writes results/robot/r0_env_smoke.json.

    sbatch --export=ALL,ENV=libero,SCRIPT="experiments/r0_env_smoke.py" slurm/jobs/robot_gpu.sbatch
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "robot")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures", "robot"), exist_ok=True)

from libero.libero import benchmark, get_libero_path          # noqa: E402
from libero.libero.envs import OffScreenRenderEnv              # noqa: E402

suite_name = sys.argv[1] if len(sys.argv) > 1 else "libero_goal"
t0 = time.time()
suite = benchmark.get_benchmark_dict()[suite_name]()
print(f"[r0] suite {suite_name}: {suite.n_tasks} tasks; libero path {get_libero_path('bddl_files')}", flush=True)
rows = []
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    init_states = suite.get_task_init_states(tid)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init_states[0])
    t1 = time.time()
    for _ in range(10):
        obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
    dt = (time.time() - t1) / 10
    keys = {k: (tuple(np.asarray(v).shape) if hasattr(v, "shape") else type(v).__name__) for k, v in obs.items()}
    img = np.asarray(obs["agentview_image"])[::-1]        # LIBERO renders upside down
    try:
        import imageio
        imageio.imwrite(os.path.join(ROOT, "figures", "robot", f"r0_{suite_name}_task{tid}.png"), img)
    except Exception as e:
        print("imageio failed:", e)
    rows.append(dict(task=tid, name=task.name, language=task.language, n_init_states=len(init_states),
                     step_secs=dt, obs_keys=keys, done=bool(done)))
    print(f"[r0] task {tid}: {task.language!r}; {len(init_states)} init states; {dt*1000:.1f} ms/step; keys {list(keys)[:6]}...", flush=True)
    env.close()
json.dump(dict(suite=suite_name, secs=time.time() - t0, tasks=rows), open(os.path.join(OUT, f"r0_env_smoke_{suite_name}.json"), "w"), indent=1)
print(f"[r0] PASS: {len(rows)} tasks rendered in {time.time() - t0:.0f}s", flush=True)
