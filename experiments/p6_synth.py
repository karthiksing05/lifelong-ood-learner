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
    t, n, seed = job
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
    img, wr, pos = [], [], []
    for k in range(n):
        obs = env.reset()
        for _ in range(P0.WAIT):
            obs, _, _, _ = env.step(P0.DUMMY)
        a = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        w = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(a, 224, 224)))
        wr.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(w, 224, 224)))
        pos.append(np.stack([P0.ref_pose(obs, inner, nm, f"{nm}_pos" in obs)[0] for nm in names]))
    env.close()
    pos = np.asarray(pos)
    np.savez_compressed(os.path.join(R, "p6_synth", f"{t}.npz"), image=np.stack(img), wrist=np.stack(wr), obj_pos=pos, names=np.array(names))
    spread = np.linalg.norm(pos - pos.mean(0), axis=-1).mean(0) * 100
    return t, n, {nm: round(float(s), 2) for nm, s in zip(names, spread)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(R, "p6_synth"), exist_ok=True)
    jobs = [(int(t), a.n, a.seed) for t in a.tasks.replace("+", ",").split(",")]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for t, n, spread in pool.imap_unordered(render, jobs):
            print(f"[p6-synth] task {t}: {n} scenes; mean distance from the mean placement (cm) {spread} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
