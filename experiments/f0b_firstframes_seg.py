"""First observations with ORACLE segmentation, depth and camera matrices (plan of 2026-10-01 evening, Q0).

Like experiments/f0_firstframes.py (same env, seed, settle steps and image path), plus, per init state:
  seg  / wseg   [S,256,256] int16  per-geom ids of the agent-view / wrist render (-1 = not a geom), flipped [::-1,::-1] like
                the client's images (so the 224 image's 14-px patch (i, j) is the 16-px block (i, j) of this map)
  depth         [S,256,256] float16 metres, agent view, same flip
  K, wK         [3,3] intrinsics at 256;  E, wE [S,4,4] camera pose in the world (agent view fixed, wrist moves)
  geom_body     [n_geom] body name of every geom;  body_root [n_body] the object / fixture each body belongs to ('' if none)
  sites         names, xpos [S,n,3], xmat [S,n,9], size [n,3] of every site whose name contains 'region'
The simulator's segmentation is the ORACLE: it scores where-sources and drives the oracle arm, never a deployable one.
Practice states only (0-9 by default); the scoring states 30-49 are never rendered here.

    sbatch --export=ALL,ENV=libero,SCRIPT="experiments/f0b_firstframes_seg.py --suites libero_90,libero_goal,libero_object,libero_10" \
        slurm/jobs/robot_gpu.sbatch
-> results/robot/f0b_firstframes/<suite>_<task>.npz
"""
import argparse
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "robot", "f0b_firstframes")
DUMMY = [0.0] * 6 + [-1.0]
CAMS = ("agentview", "robot0_eye_in_hand")
GEOM = 5                                   # mujoco mjOBJ_GEOM


class CU:
    """robosuite.utils.camera_utils' three functions, inlined (that module imports h5py, absent from the LIBERO venv)."""

    @staticmethod
    def get_camera_intrinsic_matrix(sim, cam, H, W):
        fovy = sim.model.cam_fovy[sim.model.camera_name2id(cam)]
        f = 0.5 * H / np.tan(fovy * np.pi / 360)
        return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

    @staticmethod
    def get_camera_extrinsic_matrix(sim, cam):
        i = sim.model.camera_name2id(cam)
        R = np.eye(4)
        R[:3, :3] = np.asarray(sim.data.cam_xmat[i]).reshape(3, 3)
        R[:3, 3] = np.asarray(sim.data.cam_xpos[i])
        return R @ np.diag([1.0, -1.0, -1.0, 1.0])

    @staticmethod
    def get_real_depth_map(sim, d):
        extent = sim.model.stat.extent
        far, near = sim.model.vis.map.zfar * extent, sim.model.vis.map.znear * extent
        return near / (1.0 - d * (1.0 - near / far))


def quat2axisangle(quat):
    q = np.asarray(quat, np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


def seg_ids(sim, cam, H=256, W=256):
    """Per-geom ids of a render in the observation's orientation (robosuite's 'opengl' convention), -1 for non-geoms."""
    s = sim.render(camera_name=cam, width=W, height=H, depth=False, segmentation=True)
    ids = np.where(s[..., 0] == GEOM, s[..., 1], -1).astype(np.int16)
    return ids


def body_roots(inner):
    """body index -> the object / fixture it belongs to (walk up the body tree to a registered root), '' otherwise."""
    m = inner.sim.model
    roots = {}
    for n in list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys()):
        if n in inner.obj_body_id:
            roots[int(inner.obj_body_id[n])] = n
    out = []
    for b in range(m.nbody):
        x, name = b, ""
        while True:
            if x in roots:
                name = roots[x]
                break
            p = int(m.body_parentid[x])
            if p == x or x == 0:
                break
            x = p
        out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="libero_90")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--states", default="0-9")
    ap.add_argument("--settle", type=int, default=10)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.states.split("-"))
    assert hi < 30, "the scoring states 30-49 are never rendered by this probe"
    os.makedirs(OUT, exist_ok=True)
    for sname in a.suites.replace("+", ",").split(","):   # "+" too: sbatch --export splits on commas
        suite = benchmark.get_benchmark_dict()[sname]()
        tids = range(suite.n_tasks) if a.tasks == "all" else [int(x) for x in a.tasks.replace("+", ",").split(",")]
        for tid in tids:
            out = os.path.join(OUT, f"{sname}_{tid}.npz")
            if os.path.exists(out):
                continue
            task = suite.get_task(tid)
            init = suite.get_task_init_states(tid)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            # element segmentation on, so robosuite shrinks the (invisible) region sites out of the masks
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     camera_segmentations="element")
            env.seed(0)
            inner = env.env
            names = list(inner.objects_dict.keys()) + list(inner.fixtures_dict.keys())
            kinds = ["object"] * len(inner.objects_dict) + ["fixture"] * len(inner.fixtures_dict)
            rec = {k: [] for k in ("img", "wimg", "state8", "pos", "states", "seg", "wseg", "depth", "E", "wE", "sxpos", "sxmat")}
            for k in range(lo, hi + 1):
                env.reset()
                obs = env.set_init_state(init[k])
                sim = inner.sim                     # a hard reset rebuilds the simulator: never keep an old handle
                m = sim.model
                if k == lo:
                    geom_body = [m.body_id2name(int(m.geom_bodyid[g])) or "" for g in range(m.ngeom)]
                    roots = body_roots(inner)
                    body_names = [m.body_id2name(b) or "" for b in range(m.nbody)]
                    site_ids = [i for i in range(m.nsite) if "region" in (m.site_id2name(i) or "")]
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
                    P.append(np.asarray(obs[f"{n}_pos"], np.float64) if f"{n}_pos" in obs
                             else np.array(sim.data.body_xpos[inner.obj_body_id[n]]))
                rec["pos"].append(np.asarray(P, np.float32))
                rec["seg"].append(np.ascontiguousarray(seg_ids(sim, CAMS[0])[::-1, ::-1]))
                rec["wseg"].append(np.ascontiguousarray(seg_ids(sim, CAMS[1])[::-1, ::-1]))
                _, d = sim.render(camera_name=CAMS[0], width=256, height=256, depth=True)
                rec["depth"].append(np.ascontiguousarray(CU.get_real_depth_map(sim, d)[::-1, ::-1]).astype(np.float16))
                rec["E"].append(CU.get_camera_extrinsic_matrix(sim, CAMS[0]).astype(np.float32))
                rec["wE"].append(CU.get_camera_extrinsic_matrix(sim, CAMS[1]).astype(np.float32))
                rec["sxpos"].append(np.array([sim.data.site_xpos[i] for i in site_ids], np.float32).reshape(-1, 3))
                rec["sxmat"].append(np.array([sim.data.site_xmat[i] for i in site_ids], np.float32).reshape(-1, 9))
                rec["states"].append(k)
            K = CU.get_camera_intrinsic_matrix(sim, CAMS[0], 256, 256).astype(np.float32)
            wK = CU.get_camera_intrinsic_matrix(sim, CAMS[1], 256, 256).astype(np.float32)
            env.close()
            np.savez_compressed(
                out, img=np.stack(rec["img"]), wimg=np.stack(rec["wimg"]), state8=np.stack(rec["state8"]),
                pos=np.stack(rec["pos"]), names=np.array(names), kinds=np.array(kinds), states=np.array(rec["states"]),
                language=str(task.language), seg=np.stack(rec["seg"]), wseg=np.stack(rec["wseg"]), depth=np.stack(rec["depth"]),
                K=K, wK=wK, E=np.stack(rec["E"]), wE=np.stack(rec["wE"]), geom_body=np.array(geom_body),
                body_names=np.array(body_names), body_root=np.array(roots),
                site_names=np.array([m.site_id2name(i) for i in site_ids]),
                site_size=np.array([m.site_size[i] for i in site_ids], np.float32).reshape(-1, 3),
                site_xpos=np.stack(rec["sxpos"]), site_xmat=np.stack(rec["sxmat"]))
            print(f"[f0b] {sname} {tid}: {len(rec['states'])} states, {len(names)} bodies, {len(site_ids)} region sites -> {out}",
                  flush=True)


if __name__ == "__main__":
    main()
