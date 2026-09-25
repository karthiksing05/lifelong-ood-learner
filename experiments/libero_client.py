"""LIBERO rollout client (python 3.8, openpi's examples/libero/.venv): openpi's examples/libero/main.py logic, plus
structured results and an episode-start signal for stateful servers.

    PYTHONPATH=third_party/libero examples/libero/.venv/bin/python experiments/libero_client.py \
        --port 8123 --suite libero_goal --tasks 0,1 --trials 10 --out results/robot/roll_x.json

Identical to main.py in what reaches the policy (180-degree flip, resize_with_pad to 224, 8-D state = eef pos +
axis-angle + gripper qpos, the task language as prompt, 10 settle steps, replan every 5 steps, the suite's max
steps, env.seed(seed), the benchmark's init states in order).  Extra keys under "meta/" (episode_start, task_id,
episode) are stripped by our servers before the policy's transforms.
"""
import argparse
import collections
import json
import logging
import math
import os
import pathlib
import time

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--first-trial", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--wait", type=int, default=10)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--videos", type=int, default=2, help="save a video for the first N episodes of each task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-meta", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    np.random.seed(a.seed)
    suite = benchmark.get_benchmark_dict()[a.suite]()
    tasks = list(range(suite.n_tasks)) if a.tasks == "all" else [int(x) for x in a.tasks.split(",")]
    max_steps = a.max_steps or MAX_STEPS[a.suite]
    client = _wcp.WebsocketClientPolicy(a.host, a.port)
    vid_dir = pathlib.Path(a.out).with_suffix("")
    vid_dir.mkdir(parents=True, exist_ok=True)
    res = dict(suite=a.suite, tasks={}, args=vars(a), started=time.time())
    for tid in tasks:
        task = suite.get_task(tid)
        init_states = suite.get_task_init_states(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(a.seed)
        eps = []
        for k in range(a.first_trial, a.first_trial + a.trials):
            env.reset()
            obs = env.set_init_state(init_states[k % len(init_states)])
            plan = collections.deque()
            t, calls, done, frames, t0 = 0, 0, False, [], time.time()
            err = None
            while t < max_steps + a.wait:
                try:
                    if t < a.wait:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, a.resize, a.resize))
                    wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, a.resize, a.resize))
                    if len(eps) < a.videos:
                        frames.append(img)
                    if not plan:
                        el = {"observation/image": img, "observation/wrist_image": wimg,
                              "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                              "prompt": str(task.language)}
                        if not a.no_meta:
                            el.update({"meta/episode_start": calls == 0, "meta/task_id": tid, "meta/episode": k, "meta/t": t - a.wait})
                        chunk = client.infer(el)["actions"]
                        calls += 1
                        plan.extend(chunk[: a.replan])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                    if done:
                        break
                    t += 1
                except Exception as ex:   # noqa: BLE001
                    err = repr(ex)
                    logging.error(f"episode error: {ex}")
                    break
            eps.append(dict(trial=k, success=bool(done), steps=t - a.wait, calls=calls, secs=time.time() - t0, error=err))
            if frames:
                import imageio
                imageio.mimwrite(str(vid_dir / f"task{tid}_trial{k}_{'success' if done else 'failure'}.mp4"), frames, fps=10)
            logging.info(f"[client] task {tid} '{task.language}' trial {k}: success={done} steps={t - a.wait} ({time.time() - t0:.0f}s)")
        sr = float(np.mean([e["success"] for e in eps]))
        res["tasks"][str(tid)] = dict(language=task.language, success_rate=sr, episodes=eps)
        logging.info(f"[client] task {tid} success rate {sr:.2f} over {len(eps)}")
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        env.close()
    allsr = [e["success"] for t in res["tasks"].values() for e in t["episodes"]]
    res["success_rate"] = float(np.mean(allsr))
    res["n_episodes"] = len(allsr)
    res["secs"] = time.time() - res["started"]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    logging.info(f"[client] TOTAL success rate {res['success_rate']:.3f} over {len(allsr)} episodes")


if __name__ == "__main__":
    main()
