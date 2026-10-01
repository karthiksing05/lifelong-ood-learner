"""Preview of the ORACLE role swap (experiments/libero_client.py --swap): what the policy would see at a task's first query.

    sbatch --export=ALL,ENV=libero,SCRIPT="experiments/b3_swap_preview.py --suite libero_90 --task 48 --swap ketchup_1:alphabet_soup_1" \
        slurm/jobs/robot_gpu.sbatch
-> results/robot/b3_swap_preview_<suite>_<task>.png: rows = practice states 0-2; columns = real agent view | swapped | real wrist | swapped
"""
import argparse
import importlib.util
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("lc", os.path.join(ROOT, "experiments", "libero_client.py"))
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--task", type=int, default=48)
    ap.add_argument("--swap", default="ketchup_1:alphabet_soup_1")
    ap.add_argument("--states", default="0-2")
    ap.add_argument("--mode", default="render", choices=("render", "pixel"))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.states.split("-"))
    assert hi < 30
    suite = benchmark.get_benchmark_dict()[a.suite]()
    task = suite.get_task(a.task)
    init = suite.get_task_init_states(a.task)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, camera_segmentations="element")
    env.seed(0)
    pairs = [tuple(p.split(":")) for p in a.swap.replace(",", "+").split("+") if p]
    rows = []
    for k in range(lo, hi + 1):
        env.reset()
        obs = env.set_init_state(init[k])
        for _ in range(10):
            obs, _, _, _ = env.step(LC.DUMMY)
        tables = LC.seg_tables(env)
        cache = {}
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        g0 = env.env.sim.model.geom_group.copy()
        if a.mode == "pixel":
            sw = LC.role_swap(env, tables, "agentview", img, pairs, cache)
            wsw = LC.role_swap(env, tables, "robot0_eye_in_hand", wimg, pairs, cache)
        else:
            sim = env.env.sim
            r = lambda cam: np.ascontiguousarray(sim.render(camera_name=cam, width=256, height=256, depth=False)[::-1, ::-1])  # noqa: E731
            img, wimg = r("agentview"), r("robot0_eye_in_hand")     # observations are sampled within the control step: render now
            LC.swap_visuals(env, pairs)
            sw, wsw = r("agentview"), r("robot0_eye_in_hand")
            p0 = np.array(sim.data.qpos).copy()
            for _ in range(20):
                env.step(LC.DUMMY)
            moved = float(np.abs(np.array(sim.data.qpos) - p0)[:].max())
            print(f"[b3] state {k}: max qpos change over 20 idle steps after the swap {moved:.4f} (physics untouched)", flush=True)
        assert np.array_equal(g0, env.env.sim.model.geom_group), "geom groups not restored"
        print(f"[b3] state {k}: agent view changed px {int((sw != img).any(-1).sum())}, wrist {int((wsw != wimg).any(-1).sum())}",
              flush=True)
        rows.append(np.concatenate([img, sw, wimg, wsw], axis=1))
    env.close()
    from PIL import Image
    out = os.path.join(ROOT, "results", "robot", f"b3_swap_preview_{a.suite}_{a.task}_{a.mode}.png")
    Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(out)
    print(f"[b3] -> {out}", flush=True)


if __name__ == "__main__":
    main()
