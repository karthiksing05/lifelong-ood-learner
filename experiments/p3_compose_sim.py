"""P3a (archive/2026-10-program/PLAN_TRANSFER.md; ledger m3_prereg tag p3a): do object-relative bindings COMPOSE across tasks?  No policy.

A target task is executed from PARTS of other tasks' recorded successes: the GRASP part (every segment up to and including the one
whose close grasps the object) of a task with the same object, and the PLACE part (every segment after that source's grasp segment)
of a task with the same receptacle.  Segments, frames and the tracker are P0c's (experiments/p0_retarget.py: cut at gripper-command
switches; approach segments in the grasped object's frame, carry segments in the receptacle's, the rest in the world frame; the
MimicGen transform T_new = T_ref,new T_ref,src^-1 T_goal,src on the source's controller goals; straight bridges between segments).
Roles map source frames to the target: the grasp part's object -> the target's object, the place part's receptacle -> the target's
receptacle (fixtures stay in the world frame).  Arms: own (the target's own success), cross, mismatch (the grasp part of a task with
a DIFFERENT object, applied to the target's object).

    PYTHONPATH=.../third_party/libero .venv/bin/python experiments/p3_compose_sim.py --workers 15
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import re
import time
from multiprocessing import Pool

import numpy as np

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
R = os.path.join(REPO, "results", "robot")
spec = importlib.util.spec_from_file_location("p0r", os.path.join(REPO, "experiments", "p0_retarget.py"))
P0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P0)

# target -> roles and the pre-registered sources (ledger m3_prereg tag p3a)
TARGETS = {
    58: dict(obj="ketchup", rec="wooden_tray", own=58, grasp=48, place=62, mismatch=52),
    48: dict(obj="ketchup", rec="basket", own=48, grasp=58, place=52, mismatch=62),
    43: dict(obj="white_bowl", rec=None, own=43, grasp=36, place=41, mismatch=41),
    3: dict(obj="butter", rec=None, own=3, grasp=4, place=4, mismatch=32),
    4: dict(obj="butter", rec=None, own=4, grasp=3, place=3, mismatch=32),
}
# P3b (ledger m3_prereg tag p3b): confirmatory targets, none used in P3a / P3a' / P3a''
TARGETS_B = {
    49: dict(obj="tomato_sauce", rec="basket", own=None, grasp=54, place=47, mismatch=4),
    59: dict(obj="tomato_sauce", rec="wooden_tray", own=None, grasp=54, place=57, mismatch=4),
    5: dict(obj="chocolate_pudding", rec=None, own=None, grasp=61, place=4, mismatch=32),
    36: dict(obj="white_bowl", rec="plate", own=None, grasp=43, place=67, mismatch=48),
}


def category(name):
    return re.sub(r"_\d+$", "", str(name))


def grasped(z, segs):
    """(index of the first segment whose close grasps a movable object, that object's index in the source)."""
    cmd = z["action"][:, 6] > 0
    for k, (s0, s1, ref) in enumerate(segs):
        if ref is not None and bool(z["movable"][ref]) and cmd[min(s1 - 1, len(cmd) - 1)]:
            if z["grasp"][s0:min(len(cmd), s1 + P0.GRASP_WINDOW), ref].any():
                return k, ref
    return None, None


def pieces(z, part):
    """[(z, GP, GQ, s0, s1, role, ref)] for the 'grasp' or 'place' part of source z; role in {'obj', 'rec', None}."""
    segs = P0.segments(z)
    GP, GQ = P0.goals(z)
    g, obj = grasped(z, segs)
    if g is None:
        return None
    movable = list(z["movable"])
    out = []
    chosen = segs[: g + 1] if part == "grasp" else segs[g + 1:]
    for s0, s1, ref in chosen:
        if ref is None:
            role = None
        elif ref == obj:
            role = "obj"
        elif movable[ref]:
            role = "rec"
        else:
            role = None
        out.append((z, GP, GQ, s0, s1, role, ref, None))
    return out


LIFT = 0.015


def phases(z, part):
    """P3a' (exploratory): phases from OBJECT state.  The manipulated object = the object of interest that first rises > 1.5 cm
    above its initial height while the gripper commands close (robosuite's _check_grasp misses some objects: the dressing, the
    pan); approach = up to the close that produced that lift (+ CLOSE_TAIL), in the object's frame; carry = from there to the
    first release after the lift (+ OPEN_TAIL) or the episode's end, in the receptacle's frame (the other movable object of
    interest; else the world); after = the rest, in the same frame as the carry.  -> the part's pieces."""
    cmd = z["action"][:, 6] > 0
    T = len(cmd)
    movable = list(z["movable"])
    lifts = []
    for j in range(len(movable)):
        if not movable[j]:
            continue
        up = np.where((z["obj_pos"][:, j, 2] - z["obj_pos"][0, j, 2] > LIFT) & cmd)[0]
        if len(up):
            lifts.append((int(up[0]), j))
    if not lifts:
        return None
    L, obj = min(lifts)
    closes = [t for t in range(1, L + 1) if cmd[t] and not cmd[t - 1]]
    c = closes[-1] if closes else 0
    rel = [t for t in range(L + 1, T) if not cmd[t] and cmd[t - 1]]
    recs = [j for j in range(len(movable)) if movable[j] and j != obj]
    rec = recs[0] if recs else None
    GP, GQ = P0.goals(z)
    a_end = min(T, c + P0.CLOSE_TAIL)
    c_end = min(T, rel[0] + P0.OPEN_TAIL) if rel else T
    if part == "grasp":
        return [(z, GP, GQ, 0, a_end, "obj", obj, None)]
    out = [(z, GP, GQ, a_end, c_end, "rec" if rec is not None else None, rec, obj)]            # carry: holds `obj`
    if c_end < T:
        out.append((z, GP, GQ, c_end, T, "rec" if rec is not None else None, rec, None))
    return [p for p in out if p[4] > p[3]]


def run(job):
    target, arm = job[0], job[1]
    mode = job[2] if len(job) > 2 else "switch"
    table = TARGETS_B if (len(job) > 3 and job[3] == "p3b") else TARGETS
    carry_mode = mode == "object_carry"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    T = table[target]
    src_g = T["own"] if arm == "own" else (T["grasp"] if arm == "cross" else T["mismatch"])
    src_p = T["own"] if arm == "own" else T["place"]
    fg, zg = P0.source_episode(src_g)
    fp, zp = P0.source_episode(src_p)
    if zg is None or zp is None:
        return target, arm, None, "no source success"
    split = phases if mode in ("object", "object_carry") else pieces
    pg, pp = split(zg, "grasp"), split(zp, "place")
    if not pg or not pp:                                  # never execute half a composition
        return target, arm, None, f"no {'grasp' if not pg else 'place'} part found in a source"
    parts = pg + pp
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    t_ = suite.get_task(target)
    init = suite.get_task_init_states(target)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / t_.problem_folder / t_.bddl_file
    np.random.seed(P0.SEED)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(P0.SEED)
    inner = env.env
    out = {}
    for k in range(30, 50):
        env.reset()
        obs = env.set_init_state(init[k])
        for _ in range(P0.WAIT):
            obs, _, done, _ = env.step(P0.DUMMY)
        names = list(env.obj_of_interest)
        movable = [n for n in names if f"{n}_pos" in obs]
        tobj = next((n for n in names if category(n) == T["obj"]), None)
        trec = next((n for n in names if T["rec"] and category(n) == T["rec"]), None)
        done, steps = False, 0

        def step(a):
            nonlocal obs, done, steps
            obs, _, done, _ = env.step(np.asarray(a, np.float64).tolist())
            steps += 1
            return done or steps >= P0.MAX_STEPS

        stop = False
        for (z, GP, GQ, s0, s1, role, ref, held) in parts:
            if stop:
                break
            tgt = tobj if role == "obj" else (trec if role == "rec" else None)
            if tgt is not None:
                pn, Rn = P0.ref_pose(obs, inner, tgt, tgt in movable)
                ps, Rs = z["obj_pos"][s0, ref].astype(np.float64), P0.q2m(z["obj_quat"][s0, ref])
                M_R, M_p = Rn @ Rs.T, pn - Rn @ Rs.T @ ps
            else:
                M_R, M_p = np.eye(3), np.zeros(3)
            if held is not None and carry_mode:
                # OBJECT-CENTRIC CARRY: move the held object along the source object's path (relative to the receptacle) and put
                # the gripper where the TARGET's grasp holds it: T_goal,new = M T_goal,src G_src^-1 G_tgt, G = T_obj^-1 T_ee
                def pose(p, R_):
                    X = np.eye(4); X[:3, :3] = R_; X[:3, 3] = p
                    return X
                Gs = np.linalg.inv(pose(z["obj_pos"][s0, held], P0.q2m(z["obj_quat"][s0, held]))) @ \
                    pose(z["eef_pos"][s0], P0.q2m(z["eef_quat"][s0]))
                Gt = np.linalg.inv(pose(obs[f"{tobj}_pos"], P0.q2m(obs[f"{tobj}_quat"]))) @ \
                    pose(obs["robot0_eef_pos"], P0.q2m(obs["robot0_eef_quat"]))
                Mx = pose(M_p, M_R)
                goals_ = [Mx @ pose(GP[i], GQ[i]) @ np.linalg.inv(Gs) @ Gt for i in range(s0, s1)]
                Pp = [X[:3, 3] for X in goals_]
                Qq = [X[:3, :3] for X in goals_]
            else:
                Pp = [M_R @ GP[i] + M_p for i in range(s0, s1)]
                Qq = [M_R @ GQ[i] for i in range(s0, s1)]
            p_now = np.asarray(obs["robot0_eef_pos"], np.float64)
            n = max(5, int(math.ceil(np.linalg.norm(Pp[0] - p_now) / 0.01)))
            g_hold = float(z["action"][s0 - 1, 6]) if s0 > 0 else -1.0
            for k2, Rk in enumerate(P0.slerp_path(P0.q2m(obs["robot0_eef_quat"]), Qq[0], n)):
                if step(P0.track_action(obs, p_now + (Pp[0] - p_now) * (k2 + 1) / n, Rk, g_hold)):
                    stop = True
                    break
            if stop:
                break
            for i in range(s0, s1):
                if step(P0.track_action(obs, Pp[i - s0], Qq[i - s0], float(z["action"][i, 6]))):
                    stop = True
                    break
        last = float(parts[-1][0]["action"][-1, 6])
        if carry_mode and not done and not stop and last > 0:
            # RELEASE phase: a source that succeeded while still holding never released; the grammar ends a place with a release
            last = -1.0
        budget = sum(p[4] - p[3] for p in parts) + 120
        while not done and not stop and steps < min(P0.MAX_STEPS, budget):
            if step([0.0] * 6 + [last]):
                break
        out[k] = bool(done)
    env.close()
    return target, arm, out, f"grasp {os.path.basename(fg)}, place {os.path.basename(fp)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="58+48+43+3+4")
    ap.add_argument("--set", default="p3a", choices=["p3a", "p3b"])
    ap.add_argument("--arms", default="own+cross+mismatch")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(R, "p3_compose_sim.json"))
    ap.add_argument("--phases", default="switch", choices=["switch", "object", "object_carry"],
                    help="switch: the pre-registered P3a segmentation; object: P3a' phases from object state; object_carry: + the "
                         "object-centric carry and the release phase")
    a = ap.parse_args()
    jobs = [(int(t), arm, a.phases, a.set) for t in a.targets.replace("+", ",").split(",") for arm in a.arms.replace("+", ",").split(",")]
    res, t0 = {}, time.time()
    with Pool(a.workers) as pool:
        for target, arm, out, src in pool.imap_unordered(run, jobs):
            res.setdefault(str(target), {})[arm] = dict(success=out, sources=src, k=None if out is None else int(sum(out.values())))
            print(f"[p3a] target {target} {arm}: {None if out is None else sum(out.values())}/20 ({src}) [{time.time() - t0:.0f}s]", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[p3a] -> {a.out}")


if __name__ == "__main__":
    main()
