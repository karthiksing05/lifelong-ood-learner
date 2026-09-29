"""3.8 (docs/PLAN_TRANSFER.md; ledger m3_prereg tag g38): the ONE demo as a data generator.

The task's demo (IPEC episode 0 = a raw LIBERO demo; experiments/p0_demo_init.py: its object poses and actions at every kept step)
is executed at init states 0-19 by P3's executor (experiments/p3_compose_sim.py, object_carry mode: object-state phases, the
approach in the object's frame, the object-centric carry relative to the receptacle or the world, a release phase; controller-goal
tracking).  A demo that lifts no movable object (fixture tasks) is tracked in the world frame.  Every executed step is recorded in
the practice format of experiments/libero_client.py --record (224-px flipped images, 8-d state, executed action, success), so
experiments/r15_consolidate.py --lora --practice g25 learns from the verified successes.  Privileged simulator state is used only
here, to generate; the learned policy acts from images.

    PYTHONPATH=.../third_party/libero .venv/bin/python experiments/p38_generate.py --tasks 42+52 --workers 2

--rules v2 cuts the demo by lifelong_ood/policy/phase.py (`at_rest`, `cut`: the demo's objects read at rest, the manipulated
object by displacement, receptacles and fixture tasks anchored on fixtures too); --tag names the record directory
(results/robot/<tag>_<t>_gen_rec) and the output json.
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import time
from multiprocessing import Pool

import numpy as np

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
R = os.path.join(REPO, "results", "robot")
TASKS = "35+36+39+42+1+6+34+37+40+52+53+62+3+4+5+32+41+43+48+58+63"
spec = importlib.util.spec_from_file_location("p3", os.path.join(REPO, "experiments", "p3_compose_sim.py"))
P3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P3)
P0 = P3.P0


def demo_source(t):
    """The demo as a replay-like record: names, movable, per-step object / eef poses, actions."""
    d = np.load(os.path.join(R, "p0_demo", f"{t}.npz"), allow_pickle=True)
    m = np.load(os.path.join(R, "p0_demo", f"{t}_match.npz"), allow_pickle=True)
    return dict(names=d["names"], movable=d["movable"], obj_pos=d["demo_obj_pos"], obj_quat=d["demo_obj_quat"],
                eef_pos=d["demo_eef_pos"], eef_quat=d["demo_eef_quat"], action=np.asarray(m["actions"], np.float32))


def pose(p, R_):
    X = np.eye(4)
    X[:3, :3] = R_
    X[:3, 3] = p
    return X


def program(t, rules="v1", g=None, p=None):
    """-> (the demo, its phases, a label).  g, p: a program COMPOSED from other tasks' demos (lifelong_ood/policy/phase.py compose;
    the target's object and receptacle by name from results/robot/p5_pool.json)."""
    if g is not None:
        import json
        import sys
        sys.path.insert(0, REPO)
        from lifelong_ood.policy import phase as PH
        info = json.load(open(os.path.join(R, "p5_pool.json")))[str(t)]
        names = np.load(os.path.join(R, "p0_demo", f"{t}.npz"), allow_pickle=True)["names"]
        parts = PH.compose(names, PH.at_rest(PH.load_demo(R, g)), PH.at_rest(PH.load_demo(R, p)), obj_name=info["object"],
                           rec_name=info["receptacle"])
        return parts[0][0], parts, f"composed {g}+{p}"
    if rules == "v2":
        import sys
        sys.path.insert(0, REPO)
        from lifelong_ood.policy import phase as PH
        z = PH.at_rest(PH.load_demo(R, t))
        parts = PH.cut(z)
        return z, parts, "v2:" + "+".join(str(q[5]) for q in parts)
    z = demo_source(t)
    g, p = P3.phases(z, "grasp"), P3.phases(z, "place")
    if g and p:
        return z, g + p, "phases"
    GP, GQ = P0.goals(z)                                   # no lifted object: track the whole demo in the world frame
    return z, [(z, GP, GQ, 0, len(z["action"]), None, None, None)], "world"


def generate(job):
    t, first, n, rules, tag = job[:5]
    g_src, p_src = (job[5], job[6]) if len(job) > 5 else (None, None)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    z, parts, how = program(t, rules, g_src, p_src)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(t)
    init = suite.get_task_init_states(t)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    np.random.seed(P0.SEED)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(P0.SEED)
    inner = env.env
    rec_dir = os.path.join(R, f"{tag}_{t}_gen_rec" if g_src is None else f"{tag}_{t}_{g_src}_{p_src}_gen_rec")
    os.makedirs(rec_dir, exist_ok=True)
    out = {}
    for k in range(first, first + n):
        env.reset()
        obs = env.set_init_state(init[k])
        for _ in range(P0.WAIT):
            obs, _, done, _ = env.step(P0.DUMMY)
        movable = [x for x in env.obj_of_interest if f"{x}_pos" in obs]
        rec = dict(state=[], action=[], image=[], wrist=[])
        done, steps = False, 0

        def step(a):
            nonlocal obs, done, steps
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            rec["image"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)))
            rec["wrist"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, 224, 224)))
            s8 = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
            rec["state"].append(s8.astype(np.float32))
            a = np.clip(np.asarray(a, np.float64), -1, 1)
            rec["action"].append(a.astype(np.float32))
            obs, _, done, _ = env.step(a.tolist())
            steps += 1
            return done or steps >= P0.MAX_STEPS

        stop = False
        for (zz, GP, GQ, s0, s1, role, ref, held) in parts:
            if stop:
                break
            pnames = [str(x) for x in zz["names"]]            # the part's own source (a composed program mixes sources)
            tgt = pnames[ref] if (role is not None and ref is not None) else None
            if tgt is not None:
                pn, Rn = P0.ref_pose(obs, inner, tgt, tgt in movable)
                ps, Rs = zz["obj_pos"][s0, ref].astype(np.float64), P0.q2m(zz["obj_quat"][s0, ref])
                M_R, M_p = Rn @ Rs.T, pn - Rn @ Rs.T @ ps
            else:
                M_R, M_p = np.eye(3), np.zeros(3)
            if held is not None:
                hname = pnames[held]
                Gs = np.linalg.inv(pose(zz["obj_pos"][s0, held], P0.q2m(zz["obj_quat"][s0, held]))) @ \
                    pose(zz["eef_pos"][s0], P0.q2m(zz["eef_quat"][s0]))
                Gt = np.linalg.inv(pose(obs[f"{hname}_pos"], P0.q2m(obs[f"{hname}_quat"]))) @ \
                    pose(obs["robot0_eef_pos"], P0.q2m(obs["robot0_eef_quat"]))
                goals_ = [pose(M_p, M_R) @ pose(GP[i], GQ[i]) @ np.linalg.inv(Gs) @ Gt for i in range(s0, s1)]
                Pp, Qq = [X[:3, 3] for X in goals_], [X[:3, :3] for X in goals_]
            else:
                Pp = [M_R @ GP[i] + M_p for i in range(s0, s1)]
                Qq = [M_R @ GQ[i] for i in range(s0, s1)]
            p_now = np.asarray(obs["robot0_eef_pos"], np.float64)
            nb = max(5, int(math.ceil(np.linalg.norm(Pp[0] - p_now) / 0.01)))
            g_hold = float(zz["action"][s0 - 1, 6]) if s0 > 0 else -1.0
            for k2, Rk in enumerate(P0.slerp_path(P0.q2m(obs["robot0_eef_quat"]), Qq[0], nb)):
                if step(P0.track_action(obs, p_now + (Pp[0] - p_now) * (k2 + 1) / nb, Rk, g_hold)):
                    stop = True
                    break
            if stop:
                break
            for i in range(s0, s1):
                if step(P0.track_action(obs, Pp[i - s0], Qq[i - s0], float(zz["action"][i, 6]))):
                    stop = True
                    break
        last = float(parts[-1][0]["action"][-1, 6])
        if not done and not stop and last > 0:
            last = -1.0                                      # the release phase
        budget = sum(q[4] - q[3] for q in parts) + 120
        while not done and not stop and steps < min(P0.MAX_STEPS, budget):
            if step([0.0] * 6 + [last]):
                break
        out[k] = bool(done)
        np.savez_compressed(os.path.join(rec_dir, f"task{t}_trial{k}.npz"), state=np.stack(rec["state"]), action=np.stack(rec["action"]),
                            image=np.stack(rec["image"]), wrist=np.stack(rec["wrist"]), success=bool(done), trial=k, task=t)
    env.close()
    return t, out, how


def quat2axisangle(q):
    q = np.array(q, dtype=np.float64)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rules", default="v1", choices=("v1", "v2"))
    ap.add_argument("--combos", default=None, help="a JSON list of [task, grasp source, place source] (results/robot/<file>)")
    ap.add_argument("--composed-plans", default=None, help="+-separated stream plans (results/robot/*.json): run every distinct "
                    "composed (task, grasp source, place source) instead of --tasks")
    ap.add_argument("--tag", default="g25")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(R, f"{a.tag}_generate.json")
    jobs = [(int(t), a.first, a.n, a.rules, a.tag) for t in a.tasks.replace("+", ",").split(",")]
    if a.composed_plans:
        seen = set()
        for f in a.composed_plans.split("+"):
            for o in json.load(open(os.path.join(R, f))).values():
                seen |= {(s["task"], s["g"], s["p"]) for s in o["steps"] if s["action"] == "compose"}
        jobs = [(t, a.first, a.n, "v2", a.tag, g, p) for t, g, p in sorted(seen)]
    if a.combos:
        jobs = [(int(t), a.first, a.n, "v2", a.tag, int(g), int(p)) for t, g, p in json.load(open(os.path.join(R, a.combos)))]
    res, t0 = {}, time.time()
    with Pool(a.workers) as pool:
        for t, out, how in pool.imap_unordered(generate, jobs):
            key = str(t) if not how.startswith("composed") else f"{t}_{how.split()[1].replace('+', '_')}"
            res[key] = dict(success=out, k=int(sum(out.values())), n=len(out), how=how)
            print(f"[g25] task {t}: {sum(out.values())}/{len(out)} generated successes ({how}) [{time.time() - t0:.0f}s]", flush=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev.update(res)
    json.dump(prev, open(a.out, "w"), indent=1)
    print(f"[g25] -> {a.out}")


if __name__ == "__main__":
    main()
