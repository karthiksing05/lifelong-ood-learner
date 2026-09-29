"""P0a (docs/PLAN_TRANSFER.md; ledger m3_prereg tag p0): the ONE demo's object poses, from LIBERO's raw demonstrations.

The IPEC LeRobot port (our demos) re-rendered LIBERO's raw hdf5 demos with the no-op steps removed (|arm action| < 1e-4 and an
unchanged gripper command) and failed replays dropped, so it carries no object state.  The raw files carry the full simulator
state at every step.  Two steps:

    match  (any env with h5py): IPEC episode 0 of each task's text -> the raw demo whose no-op-filtered arm actions equal it; writes
           results/robot/p0_demo/<task>_match.npz (raw index, kept raw steps, the raw states of those steps, the match error)
    poses  (LIBERO venv): set every kept state and read the objects of interest, the eef and the fingers; the same for the benchmark
           init states 30-49 (the evaluation placements); writes results/robot/p0_demo/<task>.npz

    python experiments/p0_demo_init.py match --tasks 42+52
    PYTHONPATH=.../third_party/libero .venv/bin/python experiments/p0_demo_init.py poses --tasks 42+52
"""
import argparse
import importlib.util
import os
import pathlib
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(R, "p0_demo")
RAW = "/coc/flash10/ksingara3/robot/data/libero_raw/libero_90"
TASKMAP = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
TASKS = "35+36+39+42+1+6+34+37+40+52+53+62+3+4+5+32+41+43+48+58+63+49+59+46+47+50+51+55+56+57+61"


def names90():
    src = open(TASKMAP).read()
    block = src[src.index('"libero_90"'):]
    block = block[block.index("[") + 1: block.index("]")]
    return re.findall(r'"([^"]+)"', block)


def noop_mask(actions):
    keep, prev = [], None
    for a in actions:
        is_noop = np.linalg.norm(a[:-1]) < 1e-4 and prev is not None and a[-1] == prev[-1]
        keep.append(not is_noop)
        prev = a
    return np.asarray(keep)


def match(tasks):
    import h5py
    sys.path.insert(0, ROOT)
    from lifelong_ood.policy import data as D
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90 = D.Demos90()
    names = names90()
    os.makedirs(OUT, exist_ok=True)
    for t in tasks:
        # IPEC merges tasks by language: the demo is the FIRST episode of the text that comes from this task's scene
        eps = d90.episodes_of_text(texts[t])
        if not eps:
            print(f"[p0a] task {t}: no IPEC episode for '{texts[t]}'", flush=True)
            continue
        with h5py.File(os.path.join(RAW, f"{names[t]}_demo.hdf5"), "r") as f:
            raw = {key: np.asarray(f["data"][key]["actions"], np.float64) for key in f["data"].keys()}
            best, used = None, None
            for i, e in enumerate(eps):
                ep = d90.load(e)
                ipec = np.asarray(ep["actions"], np.float64)[:, :6]
                for key, acts in raw.items():
                    keep = noop_mask(acts)
                    A = acts[keep][:, :6]
                    if len(A) != len(ipec):
                        continue
                    err = float(np.abs(A - ipec).max())
                    if err < 1e-5 and (best is None or err < best[0]):
                        best = (err, key, np.where(keep)[0], np.asarray(f["data"][key]["states"]))
                if best is not None:
                    used = (i, ep)
                    break
        if best is None:
            print(f"[p0a] task {t}: no raw demo matches the text's IPEC episodes", flush=True)
            continue
        err, key, kept, states = best
        i, ep = used
        np.savez_compressed(os.path.join(OUT, f"{t}_match.npz"), raw_key=key, kept=kept, states=states[kept], max_err=err,
                            ipec_state=np.asarray(ep["state"], np.float32), ipec_rank=i)
        print(f"[p0a] task {t}: IPEC episode {i} of the text = raw {key} ({len(kept)} kept steps), max |action diff| {err:.2e}", flush=True)


def quat2axisangle(q):
    import math
    q = np.array(q, dtype=np.float64)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


def read(env, obs, names, movable):
    inner = env.env
    P, Q = [], []
    for n in names:
        if n in movable:
            P.append(obs[f"{n}_pos"]); Q.append(obs[f"{n}_quat"])
        else:
            bid = inner.obj_body_id[n]
            w, x, y, z = inner.sim.data.body_xquat[bid]
            P.append(np.array(inner.sim.data.body_xpos[bid])); Q.append(np.array([x, y, z, w]))
    s8 = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return np.asarray(P, np.float32), np.asarray(Q, np.float32), np.asarray(obs["robot0_eef_pos"], np.float32), \
        np.asarray(obs["robot0_eef_quat"], np.float32), s8.astype(np.float32)


def poses(tasks):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    for t in tasks:
        mpath = os.path.join(OUT, f"{t}_match.npz")
        if not os.path.exists(mpath):
            continue
        m = np.load(mpath)
        task = suite.get_task(t)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(7)
        obs = env.reset()
        names = list(env.obj_of_interest)
        movable = [n for n in names if f"{n}_pos" in obs]
        demo = dict(obj_pos=[], obj_quat=[], eef_pos=[], eef_quat=[], state8=[])
        for s in m["states"]:
            obs = env.set_init_state(s)
            for k, v in zip(demo, read(env, obs, names, movable)):
                demo[k].append(v)
        init = suite.get_task_init_states(t)
        ev = dict(obj_pos=[], obj_quat=[])
        for k in range(30, 50):
            env.reset()
            obs = env.set_init_state(init[k])
            for _ in range(10):                      # the client's settle steps: objects spawn above the table and drop
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            P, Q, _, _, _ = read(env, obs, names, movable)
            ev["obj_pos"].append(P); ev["obj_quat"].append(Q)
        env.close()
        D = {k: np.stack(v) for k, v in demo.items()}
        E = {k: np.stack(v) for k, v in ev.items()}
        s_err = float(np.abs(D["state8"] - m["ipec_state"][: len(D["state8"])]).max())
        disp = np.linalg.norm(E["obj_pos"] - D["obj_pos"][0][None], axis=-1)            # [20, n_obj] m (both settled)
        np.savez_compressed(os.path.join(OUT, f"{t}.npz"), names=np.array(names), movable=np.array([n in movable for n in names]),
                            demo_obj_pos=D["obj_pos"], demo_obj_quat=D["obj_quat"], demo_eef_pos=D["eef_pos"], demo_eef_quat=D["eef_quat"],
                            demo_state8=D["state8"], eval_obj_pos=E["obj_pos"], eval_obj_quat=E["obj_quat"], state8_err=s_err,
                            raw_key=m["raw_key"])
        print(f"[p0a] task {t}: {len(D['state8'])} demo frames, 8-d state vs IPEC max err {s_err:.2e}; eval placements vs the demo's: " +
              ", ".join(f"{n} median {100 * np.median(disp[:, j]):.1f} cm (max {100 * disp[:, j].max():.1f})" for j, n in enumerate(names)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["match", "poses"])
    ap.add_argument("--tasks", default=TASKS)
    a = ap.parse_args()
    tasks = [int(x) for x in a.tasks.replace("+", ",").split(",")]
    (match if a.cmd == "match" else poses)(tasks)


if __name__ == "__main__":
    main()
