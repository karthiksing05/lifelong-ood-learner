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
GEOM = 5                                    # mujoco mjOBJ_GEOM
TOKEN_OF_PIXEL = (np.arange(256)[:, None] // 16) * 16 + (np.arange(256)[None, :] // 16)    # 16-px block -> its 14-px patch


def seg_tables(env):
    """-> (names, geom_root [ngeom] index into names or -1): every movable object and fixture and the geoms under it.  Rebuilt
    after every reset (a hard reset rebuilds the simulator)."""
    inner = env.env
    m = inner.sim.model
    names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
    root_of_body = {int(inner.obj_body_id[n]): i for i, n in enumerate(names) if n in inner.obj_body_id}
    body_root = np.full(m.nbody, -1, np.int64)
    for b in range(m.nbody):
        x = b
        while True:
            if x in root_of_body:
                body_root[b] = root_of_body[x]
                break
            p = int(m.body_parentid[x])
            if p == x or x == 0:
                break
            x = p
    return names, body_root[np.asarray(m.geom_bodyid)]


def seg_maps(env, tables, cam):
    """ORACLE where-maps: [N, 256] coverage of each object / fixture over the 16 x 16 tokens of the client's (flipped) 224 image."""
    names, geom_root = tables
    s = env.env.sim.render(camera_name=cam, width=256, height=256, depth=False, segmentation=True)
    ids = np.where(s[..., 0] == GEOM, s[..., 1], -1)[::-1, ::-1]
    root = np.where(ids >= 0, geom_root[np.clip(ids, 0, len(geom_root) - 1)], -1)
    ok = root >= 0
    cnt = np.bincount(root[ok] * 256 + TOKEN_OF_PIXEL[ok], minlength=len(names) * 256)
    return (cnt.reshape(len(names), 256) / 256.0).astype(np.float16)


def _flip(x):
    return np.ascontiguousarray(x[::-1, ::-1])


def seg_depth(sim, cam):
    """Per-geom ids and metric depth of one camera at 256, in the CLIENT orientation (flipped like the policy's images)."""
    s = sim.render(camera_name=cam, width=256, height=256, depth=False, segmentation=True)
    ids = np.where(s[..., 0] == GEOM, s[..., 1], -1)
    _, d = sim.render(camera_name=cam, width=256, height=256, depth=True)
    ext = sim.model.stat.extent
    far, near = sim.model.vis.map.zfar * ext, sim.model.vis.map.znear * ext
    return _flip(ids), _flip(near / (1.0 - d * (1.0 - near / far)))


def background(sim, cam, geoms):
    """The camera's render with the given geoms hidden (moved to an unrendered geom group for one render)."""
    g = sim.model.geom_group
    old = g[geoms].copy()
    g[geoms] = 5
    try:
        return _flip(sim.render(camera_name=cam, width=256, height=256, depth=False))
    finally:
        g[geoms] = old


def _bottom_center(m):
    r, c = np.where(m)
    rb = r.max()
    return int(rb), float(c[r >= rb - 2].mean())


def _paste(dst, crop, mask, anchor, scale):
    import cv2
    h, w = mask.shape
    H2, W2 = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    rgb2 = cv2.resize(crop, (W2, H2), interpolation=cv2.INTER_LINEAR)
    m2 = cv2.resize(mask.astype(np.uint8), (W2, H2), interpolation=cv2.INTER_NEAREST).astype(bool)
    r1 = anchor[0] + 1
    r0 = r1 - H2
    c0 = int(round(anchor[1] - W2 / 2.0))
    c1 = c0 + W2
    R0, R1, C0, C1 = max(r0, 0), min(r1, dst.shape[0]), max(c0, 0), min(c1, dst.shape[1])
    if R1 <= R0 or C1 <= C0:
        return
    sub = m2[R0 - r0:R1 - r0, C0 - c0:C1 - c0]
    dst[R0:R1, C0:C1][sub] = rgb2[R0 - r0:R1 - r0, C0 - c0:C1 - c0][sub]


def role_swap(env, tables, cam, rgb, pairs, cache, min_px=20):
    """ORACLE role rebinding of one camera's image (256, client orientation): for each pair (a, b) of instances, the policy
    sees b's appearance where a is and a's appearance where b is (bottom-aligned, scaled by depth); hidden pixels come from a
    render without the swapped objects.  An object visible without its partner shows the partner's last sprite in this camera
    (`cache`).  The scene's physics is untouched."""
    names, geom_root = tables
    sim = env.env.sim
    ids, depth = seg_depth(sim, cam)
    root = np.where(ids >= 0, geom_root[np.clip(ids, 0, len(geom_root) - 1)], -1)
    inst = sorted({n for p in pairs for n in p})
    masks = {n: root == names.index(n) for n in inst}
    vis = {n: int(masks[n].sum()) >= min_px for n in inst}
    sprites = {}
    for n in inst:
        if vis[n]:
            r, c = np.where(masks[n])
            r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
            sprites[n] = (rgb[r0:r1, c0:c1].copy(), masks[n][r0:r1, c0:c1].copy(), float(np.median(depth[masks[n]])))
            cache[(cam, n)] = sprites[n]
    todo = []
    for a_, b_ in pairs:
        for src, dst in ((b_, a_), (a_, b_)):
            if vis[dst] and (src in sprites or (cam, src) in cache):
                todo.append((src, dst))
    if not todo:
        return rgb
    hide = sorted({d for _, d in todo})
    geoms = np.where(np.isin(geom_root, [names.index(n) for n in hide]))[0]
    bg = background(sim, cam, geoms)
    out = rgb.copy()
    holes = np.zeros(rgb.shape[:2], bool)
    for n in hide:
        holes |= masks[n]
    out[holes] = bg[holes]
    todo.sort(key=lambda sd: -float(np.median(depth[masks[sd[1]]])))     # farther first
    for src, dst in todo:
        crop, m, d_src = sprites.get(src) or cache[(cam, src)]
        _paste(out, crop, m, _bottom_center(masks[dst]), d_src / max(float(np.median(depth[masks[dst]])), 1e-6))
    return out


def _quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_quat(R):
    """3x3 rotation -> (w, x, y, z)."""
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


class RenderSwap:
    """ORACLE role rebinding in the RENDERER: each pair's visual geoms exchange root bodies (each object's visual set expressed
    in its own root frame at the moment of construction, bottoms aligned), so every camera shows b's appearance on a's body and
    a's on b's, with true perspective, occlusion and carrying.  Collision geometry, masses and physics are untouched (visual
    geoms have no contacts; inertia is fixed at compile time).  Build it right after an episode's reset (objects at rest)."""

    def __init__(self, env, pairs):
        self.sim = sim = env.env.sim
        m, d = sim.model, sim.data
        names, geom_root = seg_tables(env)
        self.orig, self.new = {}, {}

        def visual(n):
            gs = [int(g) for g in np.where(geom_root == names.index(n))[0]
                  if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] == 1]
            if not gs:
                raise RuntimeError(f"{n}: no visual geoms")
            return gs

        def rel(g, r):
            Rr, pr = np.asarray(d.xmat[r]).reshape(3, 3), np.asarray(d.xpos[r])
            Rg, pg = np.asarray(d.geom_xmat[g]).reshape(3, 3), np.asarray(d.geom_xpos[g])
            return Rr.T @ Rg, Rr.T @ (pg - pr)

        def bottom(g, R, p):
            c, h = np.asarray(m.geom_aabb[g][:3], np.float64), np.asarray(m.geom_aabb[g][3:], np.float64)
            corners = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], np.float64)
            return float((corners @ R.T + p)[:, 2].min())

        for a_, b_ in pairs:
            ra, rb = int(env.env.obj_body_id[a_]), int(env.env.obj_body_id[b_])
            ga, gb = visual(a_), visual(b_)
            relA = {g: rel(g, ra) for g in ga}
            relB = {g: rel(g, rb) for g in gb}
            za = min(bottom(g, *relA[g]) for g in ga)
            zb = min(bottom(g, *relB[g]) for g in gb)
            for g in ga + gb:
                self.orig[g] = (int(m.geom_bodyid[g]), np.array(m.geom_pos[g]), np.array(m.geom_quat[g]))
            for g, (R, p) in relA.items():
                self.new[g] = (rb, p + np.array([0.0, 0.0, zb - za]), _mat_quat(R))
            for g, (R, p) in relB.items():
                self.new[g] = (ra, p + np.array([0.0, 0.0, za - zb]), _mat_quat(R))
        self.on = False

    def set(self, on):
        m = self.sim.model
        for g, (b, p, q) in (self.new if on else self.orig).items():
            m.geom_bodyid[g] = b
            m.geom_pos[g] = p
            m.geom_quat[g] = q
        self.sim.forward()
        self.on = bool(on)


def swap_visuals(env, pairs):
    """(kept for the preview script) build and switch on a RenderSwap."""
    rs = RenderSwap(env, pairs)
    rs.set(True)
    return rs


def movable_positions(env, obs):
    names = list(env.env.objects_dict.keys())
    P = []
    for n in names:
        P.append(np.asarray(obs[f"{n}_pos"], np.float64) if f"{n}_pos" in obs
                 else np.array(env.env.sim.data.body_xpos[env.env.obj_body_id[n]]))
    return names, np.asarray(P)
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
    ap.add_argument("--state-mod", type=int, default=0, help="trial k starts from init state k %% state_mod (0: k %% 50); online "
                    "learners practise on states 0..state_mod-1 only")
    ap.add_argument("--first-trial", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fresh", default="", help="a directory of FRESH init states (experiments/fresh_states.py): trial k starts from "
                    "<dir>/<suite>_<task>.npy[k] instead of the stored init state k")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--wait", type=int, default=10)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--videos", type=int, default=2, help="save a video for the first N episodes of each task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--record", action="store_true", help="save every episode's policy-input frames, states, executed actions "
                    "and outcome to <out without .json>_rec/task<t>_trial<k>.npz (learning from practice)")
    ap.add_argument("--light", action="store_true", help="with --record: no images (states, actions, object poses: phase-level "
                    "outcomes of every episode, experiments/phase_outcomes.py)")
    ap.add_argument("--all-objects", action="store_true", help="with --objects: every movable object of the scene, not only the "
                    "objects of interest (which object the policy moves: the overhaul's binding read)")
    ap.add_argument("--objects", action="store_true", help="send (and record) the poses of the objects of interest: meta/obj_names, "
                    "meta/obj_pos, meta/obj_quat (xyzw); privileged simulator state for the phase-binding server (archive/2026-10-program/PLAN_TRANSFER.md P1)")
    ap.add_argument("--branches", default="", help="a JSON list of stall-point branches of ONE task (experiments/r18_branch.py): "
                    "{id, trial, seed_base, at, variant}; each replays a recorded episode's noise up to query `at`, then the "
                    "server changes one knob; replaces --first-trial / --trials.  The prefix (queries < at) executes the recorded "
                    "actions of `rec` (the simulator is bit-deterministic given the actions and the session's reset count, "
                    "`resets`); the server only follows it (meta/ff)")
    ap.add_argument("--seg", action="store_true", help="ORACLE arms only: send per-token coverage maps of every object and fixture "
                    "from the simulator's segmentation (meta/seg_names, meta/seg_agent, meta/seg_wrist)")
    ap.add_argument("--swap", default="", help="ORACLE role rebinding: 'a:b[+c:d]' instance pairs whose appearances the policy sees "
                    "exchanged in both cameras (the objects stay where they are); e.g. ketchup_1:alphabet_soup_1")
    ap.add_argument("--swap-mode", default="render", choices=("render", "pixel"), help="render: the meshes exchange bodies in the "
                    "simulator's renderer (every camera consistent); pixel: sprites exchanged in the two images (the operation a "
                    "deployable editor would perform)")
    ap.add_argument("--swap-cams", default="both", choices=("both", "agent"), help="render swap: 'agent' re-renders the WRIST view "
                    "real at each query (the closed-loop grasp sees the true object)")
    ap.add_argument("--swap-phase", default="all", choices=("all", "open", "closed"), help="render swap: 'open' keeps the swap only while the "
                    "last commanded gripper is open (approach), the real scene once it closes")
    ap.add_argument("--bind", default="", choices=("", "track", "stop"), help="track the first movable object displaced > 3 cm "
                    "while the commanded gripper is closed (episode field 'bound'); 'stop' ends the episode there")
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
        init_states = (np.load(os.path.join(a.fresh, f"{a.suite}_{tid}.npy")) if a.fresh
                       else suite.get_task_init_states(tid))
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     **({"camera_segmentations": "element"} if (a.seg or a.swap) else {}))
        env.seed(a.seed)
        eps = []
        todo = ([(int(b["trial"]), b) for b in json.load(open(a.branches))] if a.branches
                else [(k, None) for k in range(a.first_trial, a.first_trial + a.trials)])
        for k, br in todo:
            prefix = np.load(br["rec"])["action"] if br is not None and br.get("rec") else None
            if br is not None:
                # the recorded episode's simulator.  MuJoCo's solver warm-start is not in the init state and depends on the random
                # placement of the reset that precedes set_init_state, i.e. on the resets since env.seed (r18_sim_check.py); and on
                # some tasks (articulated fixtures) earlier episodes of the process leave state that re-seeding does not clear
                # (r18 branches: only a shard's first branch replayed).  So: a fresh env as at process start, re-seeded, and the
                # recorded session's resets
                env.close()
                np.random.seed(a.seed)
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     **({"camera_segmentations": "element"} if (a.seg or a.swap) else {}))
                env.seed(a.seed)
                for _ in range(int(br.get("resets", 0))):
                    env.reset()
            env.reset()
            obs = env.set_init_state(init_states[k % (a.state_mod or len(init_states))])
            swapped, rswap = False, None
            if a.swap and a.swap_mode == "render":
                rswap = RenderSwap(env, [tuple(p.split(":")) for p in a.swap.split("+") if p])
                if a.swap_phase != "closed":                 # the episode starts with the gripper open
                    rswap.set(True)
                    swapped = True
            plan = collections.deque()
            t, calls, done, frames, t0 = 0, 0, False, [], time.time()
            tables = seg_tables(env) if (a.seg or a.swap) else None
            pairs = [tuple(p.split(":")) for p in a.swap.split("+") if p] if a.swap else []
            swap_cache, last_q, rec_act = {}, None, []
            bound, bound_step, start_pos = None, None, None
            rec = dict(state=[], action=[], image=[], wrist=[], obj_pos=[], obj_quat=[]) if a.record else None

            def objects(obs):
                names = list(env.env.objects_dict.keys()) if a.all_objects else list(env.obj_of_interest)
                P, Q = [], []
                for n in names:
                    if f"{n}_pos" in obs:
                        P.append(np.asarray(obs[f"{n}_pos"], np.float64)); Q.append(np.asarray(obs[f"{n}_quat"], np.float64))
                    else:
                        bid = env.env.obj_body_id[n]
                        w, x, y, zq = env.env.sim.data.body_xquat[bid]
                        P.append(np.array(env.env.sim.data.body_xpos[bid])); Q.append(np.array([x, y, zq, w]))
                return names, np.asarray(P), np.asarray(Q)
            err = None
            while t < max_steps + a.wait:
                try:
                    if t < a.wait:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wimg = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if pairs and not plan and a.swap_mode == "render":
                        closed_now = bool(len(rec_act) and rec_act[-1][6] > 0)
                        want = {"all": True, "open": not closed_now, "closed": closed_now}[a.swap_phase]
                        if want != swapped:                  # phase-gated swap: the meshes follow the commanded gripper
                            rswap.set(want)
                            swapped = want
                            sim_ = env.env.sim
                            img = np.ascontiguousarray(sim_.render(camera_name="agentview", width=256, height=256)[::-1, ::-1])
                            wimg = np.ascontiguousarray(sim_.render(camera_name="robot0_eye_in_hand", width=256, height=256)[::-1, ::-1])
                        if a.swap_cams == "agent" and swapped:   # the wrist view real: un-swap, render, re-swap
                            rswap.set(False)
                            wimg = np.ascontiguousarray(env.env.sim.render(camera_name="robot0_eye_in_hand", width=256, height=256)[::-1, ::-1])
                            rswap.set(True)
                    if pairs and not plan and a.swap_mode == "pixel":    # what the POLICY sees at its queries (recordings keep the real scene)
                        img_q = role_swap(env, tables, "agentview", img, pairs, swap_cache)
                        wimg_q = role_swap(env, tables, "robot0_eye_in_hand", wimg, pairs, swap_cache)
                        img_q = image_tools.convert_to_uint8(image_tools.resize_with_pad(img_q, a.resize, a.resize))
                        wimg_q = image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg_q, a.resize, a.resize))
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, a.resize, a.resize))
                    wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(wimg, a.resize, a.resize))
                    if not (pairs and not plan and a.swap_mode == "pixel"):
                        img_q, wimg_q = (last_q if (pairs and a.swap_mode == "pixel" and last_q is not None) else (img, wimg))
                    last_q = (img_q, wimg_q)
                    if len(eps) < a.videos:
                        frames.append(np.concatenate([img, img_q], axis=1) if (pairs and a.swap_mode == "pixel") else img)
                    st8 = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                    if not plan:
                        el = {"observation/image": img_q, "observation/wrist_image": wimg_q,
                              "observation/state": st8,
                              "prompt": str(task.language)}
                        if not a.no_meta:
                            el.update({"meta/episode_start": calls == 0, "meta/task_id": tid, "meta/episode": k, "meta/t": t - a.wait})
                            if calls == 0 and eps:                  # the previous episode's outcome (online learners: lifelong_ood/policy/dsrl.py)
                                el["meta/prev_success"] = bool(eps[-1]["success"])
                            if br is not None:
                                el.update({"meta/seed_base": int(br["seed_base"]), "meta/branch_at": int(br["at"]),
                                           "meta/variant": str(br["variant"])})
                        if a.seg:
                            el.update({"meta/seg_names": tables[0], "meta/seg_agent": seg_maps(env, tables, "agentview"),
                                       "meta/seg_wrist": seg_maps(env, tables, "robot0_eye_in_hand")})
                        if a.objects:
                            on_, op_, oq_ = objects(obs)
                            el.update({"meta/obj_names": on_, "meta/obj_pos": op_.reshape(-1).tolist(), "meta/obj_quat": oq_.reshape(-1).tolist()})
                        ff = prefix is not None and calls < int(br["at"]) and a.replan * (calls + 1) <= len(prefix)
                        if ff:
                            el["meta/ff"] = True
                        chunk = client.infer(el)["actions"]
                        if ff:                          # the branch's prefix: the recorded episode's own actions
                            chunk = prefix[a.replan * calls: a.replan * (calls + 1)]
                        calls += 1
                        plan.extend(chunk[: a.replan])
                    act = np.asarray(plan.popleft(), np.float32)
                    rec_act.append(act)
                    if rec is not None:
                        rec["state"].append(np.asarray(st8, np.float32)); rec["action"].append(act)
                        if not a.light:
                            if a.swap and a.swap_mode == "render" and swapped:
                                # the REAL scene for the recording (a fold learns real image + own instruction -> these actions):
                                # the swap is an involution, so applying it twice restores the meshes
                                rswap.set(False)
                                sim_ = env.env.sim
                                ri = np.ascontiguousarray(sim_.render(camera_name="agentview", width=256, height=256)[::-1, ::-1])
                                rw = np.ascontiguousarray(sim_.render(camera_name="robot0_eye_in_hand", width=256, height=256)[::-1, ::-1])
                                rswap.set(True)
                                rec["image"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(ri, a.resize, a.resize)))
                                rec["wrist"].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(rw, a.resize, a.resize)))
                            else:
                                rec["image"].append(img); rec["wrist"].append(wimg)
                        if a.objects:
                            _, op_, oq_ = objects(obs)
                            rec["obj_pos"].append(op_.astype(np.float32)); rec["obj_quat"].append(oq_.astype(np.float32))
                    obs, _, done, _ = env.step(act.tolist())
                    if a.bind and bound is None:
                        mn, mp = movable_positions(env, obs)
                        if start_pos is None:
                            start_pos = mp
                        elif act[6] > 0:
                            disp = np.linalg.norm(mp - start_pos, axis=-1)
                            if disp.max() > 0.03:
                                bound, bound_step = mn[int(np.argmax(disp))], t - a.wait
                                if a.bind == "stop":
                                    break
                    if done:
                        break
                    t += 1
                except Exception as ex:   # noqa: BLE001
                    err = repr(ex)
                    logging.error(f"episode error: {ex}")
                    break
            eps.append(dict(trial=k, success=bool(done), steps=t - a.wait, calls=calls, secs=time.time() - t0, error=err,
                            **(dict(bound=bound, bound_step=bound_step) if a.bind else {}),
                            **({} if br is None else dict(branch=br["id"], variant=br["variant"], at=int(br["at"])))))
            if rec is not None and rec["action"]:
                rec_dir = pathlib.Path(a.out).with_suffix("").parent / (pathlib.Path(a.out).stem + "_rec")
                rec_dir.mkdir(parents=True, exist_ok=True)
                extra = dict(obj_pos=np.stack(rec["obj_pos"]), obj_quat=np.stack(rec["obj_quat"]), obj_names=np.array(list(env.env.objects_dict.keys()) if a.all_objects else list(env.obj_of_interest))) \
                    if (a.objects and rec["obj_pos"]) else {}
                np.savez_compressed(str(rec_dir / (f"task{tid}_trial{k}" + ("" if br is None else f"_{br['id']}") + ".npz")), state=np.stack(rec["state"]), action=np.stack(rec["action"]),
                                    **({} if a.light else dict(image=np.stack(rec["image"]), wrist=np.stack(rec["wrist"]))),
                                    success=bool(done), trial=k, task=tid, **extra)
            if frames:
                import imageio
                imageio.mimwrite(str(vid_dir / f"task{tid}_trial{k}_{'success' if done else 'failure'}.mp4"), frames, fps=10)
            logging.info(f"[client] task {tid} '{task.language}' trial {k}{'' if br is None else ' branch ' + br['id']}: success={done} "
                         f"steps={t - a.wait} ({time.time() - t0:.0f}s)")
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
