"""Inspect the geoms of a scene's objects (visual vs collision, body, mesh, offsets): for the render-level role swap.
    sbatch --export=ALL,ENV=libero,SCRIPT="experiments/b3_geom_inspect.py --task 48" slurm/jobs/robot_gpu.sbatch"""
import argparse, os, pathlib, importlib.util
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("lc", os.path.join(ROOT, "experiments", "libero_client.py"))
LC = importlib.util.module_from_spec(spec); spec.loader.exec_module(LC)
ap = argparse.ArgumentParser(); ap.add_argument("--suite", default="libero_90"); ap.add_argument("--task", type=int, default=48)
a = ap.parse_args()
suite = benchmark.get_benchmark_dict()[a.suite](); task = suite.get_task(a.task)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
env.seed(0); env.reset(); env.set_init_state(suite.get_task_init_states(a.task)[0])
sim = env.env.sim; m = sim.model
names, geom_root = LC.seg_tables(env)
print("vopt geomgroup", list(sim._render_context_offscreen.vopt.geomgroup))
for i, n in enumerate(names):
    gs = np.where(geom_root == i)[0]
    print(f"== {n}: root body {m.body_id2name(int(env.env.obj_body_id[n])) if n in env.env.obj_body_id else None}, {len(gs)} geoms")
    for g in gs:
        print(f"   geom {g} body {m.body_id2name(int(m.geom_bodyid[g]))} type {m.geom_type[g]} group {m.geom_group[g]} "
              f"contype {m.geom_contype[g]} conaff {m.geom_conaffinity[g]} dataid {m.geom_dataid[g]} matid {m.geom_matid[g]} "
              f"pos {np.round(m.geom_pos[g],3)} size {np.round(m.geom_size[g],3)}")
    bid = int(env.env.obj_body_id[n]) if n in env.env.obj_body_id else None
    if bid is not None:
        print(f"   body xpos {np.round(sim.data.body_xpos[bid],3)} parent {m.body_id2name(int(m.body_parentid[bid]))}")
