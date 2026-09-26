"""M2 validation (iv), run retroactively: FORCED EXECUTION of prototypes in the simulator (no policy, no rendering).

Each prototype's medoid chunk (results/robot/atlas.json, 10 raw actions) is executed open-loop from several LIBERO-90
scenes and init states, after 10 settle steps.  Measured: the realised hand displacement against the commanded one
(sum of the chunk's position actions x the demo-measured scale POS), as the realised fraction
(realised . commanded / |commanded|^2) and the cosine between them, for prototypes that command at least 1.5 cm; the
hand's drift for "hold" prototypes; the finger opening change for gripper events.  The plan's kill: < 0.5 realised.
Runs in the LIBERO venv (python 3.8) on a CPU:

    python experiments/r12_forced.py --level 4 --top 20 --tasks 1,19,36,53,58 --states 2 --out results/robot/r12_forced.json
"""
import argparse
import json
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

POS = 0.011                      # m of hand motion per unit position action (measured on the demos; experiments/r2c_atlas.py)
DUMMY = [0.0] * 6 + [-1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="results/robot/atlas.json")
    ap.add_argument("--level", default="4")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--tasks", default="1,19,36,53,58")
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--out", default="results/robot/r12_forced.json")
    a = ap.parse_args()
    atlas = json.load(open(a.atlas))[a.level]
    protos = sorted(atlas, key=lambda r: (-len(r["tasks"]), -r["frames"]))[:a.top]
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    runs = []
    for tid in [int(x) for x in a.tasks.split(",")]:
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        inits = suite.get_task_init_states(tid)
        for s in range(a.states):
            for p in protos:
                env.reset()
                obs = env.set_init_state(inits[s])
                for _ in range(10):
                    obs, _, _, _ = env.step(DUMMY)
                p0, g0 = np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_gripper_qpos"])
                raw = np.array(p["raw"])
                for act in raw:
                    obs, _, _, _ = env.step(act.tolist())
                p1, g1 = np.array(obs["robot0_eef_pos"]), np.array(obs["robot0_gripper_qpos"])
                cmd = raw[:, :3].sum(0) * POS
                real = p1 - p0
                runs.append(dict(task=tid, state=s, node=p["node"], name=p["name"], cmd=cmd.tolist(), real=real.tolist(),
                                 grip_cmd=[float(raw[:3, 6].mean()), float(raw[-3:, 6].mean())],
                                 finger_change=float(np.abs(g1).sum() - np.abs(g0).sum())))
        env.close()
        print(f"[r12] task {tid}: {a.states} states x {len(protos)} prototypes done", flush=True)
    per = {}
    for r in runs:
        per.setdefault(r["node"], []).append(r)
    summ = []
    for nid, rs in per.items():
        cmd = np.array([r["cmd"] for r in rs]); real = np.array([r["real"] for r in rs])
        c = np.linalg.norm(cmd[0])
        frac = (real * cmd).sum(1) / np.maximum((cmd * cmd).sum(1), 1e-12)
        cos = (real * cmd).sum(1) / np.maximum(np.linalg.norm(real, axis=1) * np.linalg.norm(cmd, axis=1), 1e-12)
        summ.append(dict(node=nid, name=rs[0]["name"], commanded_cm=float(c * 100), realised_cm=float(np.linalg.norm(real, axis=1).mean() * 100),
                         frac_mean=float(frac.mean()), frac_sd=float(frac.std()), cos_mean=float(cos.mean()),
                         finger_change_mean=float(np.mean([r["finger_change"] for r in rs])), grip_cmd=rs[0]["grip_cmd"], runs=len(rs)))
        s = summ[-1]
        print(f"[r12] {nid} '{s['name']}': commanded {s['commanded_cm']:.1f} cm, realised {s['realised_cm']:.1f} cm, fraction "
              f"{s['frac_mean']:.2f} +- {s['frac_sd']:.2f}, cosine {s['cos_mean']:.2f}, finger change {s['finger_change_mean']:+.4f}", flush=True)
    moving = [s for s in summ if s["commanded_cm"] >= 1.5]
    out = dict(level=a.level, tasks=a.tasks, states=a.states, prototypes=summ, runs=runs,
               moving=len(moving), moving_frac_median=float(np.median([s["frac_mean"] for s in moving])) if moving else None,
               moving_cos_median=float(np.median([s["cos_mean"] for s in moving])) if moving else None,
               killed=bool(moving and np.median([s["frac_mean"] for s in moving]) < 0.5))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[r12] {len(moving)} prototypes command >= 1.5 cm: median realised fraction {out['moving_frac_median']}, median cosine "
          f"{out['moving_cos_median']}; plan kill (< 0.5 realised): {out['killed']}")


if __name__ == "__main__":
    main()
