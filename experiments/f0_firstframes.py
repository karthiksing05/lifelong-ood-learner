"""First observations and EVERY object's position, for the offline readability probe (overhaul P0b; experiments/f1_factor_probe.py).

For each task and init state: the client's observation after its 10 settle steps (agent view and wrist, flipped 180 degrees,
resized with padding to 224, the 8-D state), and the position of every movable object and fixture in the scene (not only
the objects of interest).  Practice states only (0-9 by default): the scoring states 30-49 are never rendered here.

    sbatch --export=ALL,ENV=libero,SCRIPT="experiments/f0_firstframes.py --suites libero_90,libero_goal,libero_object,libero_10" \
        slurm/jobs/robot_gpu.sbatch           (LIBERO venv, EGL)
-> results/robot/f0_firstframes/<suite>_<task>.npz: img, wimg [S,224,224,3] uint8, state8 [S,8], names [N], kinds [N]
   (object / fixture), pos [S,N,3], states [S], language.
"""
import argparse
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "robot", "f0_firstframes")
DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    q = np.asarray(quat, np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="libero_90")
    ap.add_argument("--tasks", default="all", help="task ids (comma) within each suite, or all")
    ap.add_argument("--states", default="0-9")
    ap.add_argument("--settle", type=int, default=10)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.states.split("-"))
    assert hi < 30, "the scoring states 30-49 are never rendered by this probe"
    os.makedirs(OUT, exist_ok=True)
    for sname in a.suites.split(","):
        suite = benchmark.get_benchmark_dict()[sname]()
        tids = range(suite.n_tasks) if a.tasks == "all" else [int(x) for x in a.tasks.split(",")]
        for tid in tids:
            out = os.path.join(OUT, f"{sname}_{tid}.npz")
            if os.path.exists(out):
                continue
            task = suite.get_task(tid)
            init = suite.get_task_init_states(tid)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(0)
            inner = env.env
            names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
            kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
            rec = dict(img=[], wimg=[], state8=[], pos=[], states=[])
            for k in range(lo, hi + 1):
                env.reset()
                obs = env.set_init_state(init[k])
                for _ in range(a.settle):
                    obs, _, _, _ = env.step(DUMMY)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                rec["img"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
                rec["wimg"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)))
                rec["state8"].append(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                     obs["robot0_gripper_qpos"])).astype(np.float32))
                P = []
                for n in names:
                    if f"{n}_pos" in obs:
                        P.append(np.asarray(obs[f"{n}_pos"], np.float64))
                    else:
                        P.append(np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]]))
                rec["pos"].append(np.asarray(P, np.float32))
                rec["states"].append(k)
            env.close()
            np.savez_compressed(out, img=np.stack(rec["img"]), wimg=np.stack(rec["wimg"]), state8=np.stack(rec["state8"]),
                                pos=np.stack(rec["pos"]), names=np.array(names), kinds=np.array(kinds),
                                states=np.array(rec["states"]), language=str(task.language))
            print(f"[f0] {sname} {tid}: {len(rec['states'])} states, {len(names)} bodies -> {out}", flush=True)


if __name__ == "__main__":
    main()
