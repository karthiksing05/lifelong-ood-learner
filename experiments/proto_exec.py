"""Each verb's prototype, EXECUTED: the mode of its most-used t = 0.4 prototype (results/robot/proto_exec_chunks.json: ten raw OSC
actions, decoded from the 40-task hierarchy) replayed open-loop four times in a row from one fixed start state of one LIBERO-90 scene,
no policy -- what the prototype commands the robot to do.  Frames (agent view, upright, 224 x 224) -> results/robot/proto_exec.npz.

    sbatch ... examples/libero/.venv/bin/python experiments/proto_exec.py [--task 6 --reps 4]   (LIBERO venv, EGL on a GPU node)
"""
import argparse
import json
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
DUMMY = [0.0] * 6 + [-1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=6)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--state", type=int, default=0)
    a = ap.parse_args()
    chunks = json.load(open(os.path.join(R, "proto_exec_chunks.json")))
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(a.task)
    init = suite.get_task_init_states(a.task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    out = {}
    for verb, c in chunks.items():
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        obs = env.set_init_state(init[a.state])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
        frames, eef = [], []
        acts = np.asarray(c["actions"], np.float32)
        for r in range(a.reps):
            for act in acts:
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                frames.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
                eef.append(np.asarray(obs["robot0_eef_pos"], np.float32))
                obs, _, _, _ = env.step(np.clip(act, -1, 1).tolist())
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        frames.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
        eef.append(np.asarray(obs["robot0_eef_pos"], np.float32))
        env.close()
        k = verb.replace(" ", "_")
        out[f"frames_{k}"] = np.stack(frames)
        out[f"eef_{k}"] = np.stack(eef)
        print(f"[proto_exec] {verb}: prototype {c['prototype']}, hand moved {np.round((eef[-1] - eef[0]) * 100, 1).tolist()} cm", flush=True)
    out["task"] = np.array(a.task)
    out["language"] = np.array(str(task.language))
    np.savez_compressed(os.path.join(R, "proto_exec.npz"), **out)


if __name__ == "__main__":
    main()
