"""Where the moved object ends up relative to the goal region (LIBERO SiteObject.under), from recorded object poses (launch_pack diag_cabpose --record --light --all-objects); run in the LIBERO venv (results/robot/.scratch/diag_site.sbatch)."""
import json, pathlib, sys, glob
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
R = "/coc/flash10/ksingara3/lifelong-ood-learner/results/robot"
suite = benchmark.get_benchmark_dict()["libero_90"]()
for tid, site, obj in ((25, None, None), (43, None, None), (31, None, None)):
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    env.seed(0)
    inner = env.env
    goal = inner.parsed_problem["goal_state"]
    print(tid, task.language, "goal", goal)
    if site is None:
        g = goal[0]; obj, site = g[1], g[2]
    so = inner.object_sites_dict[site]
    init = suite.get_task_init_states(tid)
    for f in sorted(glob.glob(f"{R}/diag_cabpose_{tid}_zs_rec/*.npz")):
        z = np.load(f)
        k = int(z["trial"]); env.reset(); env.set_init_state(init[k])
        sp, sm = inner.sim.data.get_site_xpos(site).copy(), inner.sim.data.get_site_xmat(site).copy()
        names = list(z["obj_names"]); P = z["obj_pos"].reshape(len(z["obj_pos"]), len(names), 3)[:, names.index(obj)]
        d = (sm @ (P - sp).T).T
        size = np.asarray(so.size, float)
        inxy = np.all(np.abs(d[:, :2]) < size[:2], 1); inz = (size[2] - 0.005 < d[:, 2]) & (d[:, 2] < size[2] + 0.10)
        best = np.argmin(np.abs(d[:, :2]).max(1) - size[:2].min() + np.where(inz, 0, 1))
        print(f"  trial {k} success {bool(z['success'])} size {np.round(size,3)} | ever in xy {inxy.any()} ever in z {inz.any()} both {(inxy&inz).any()}"
              f" | final d {np.round(d[-1],3)} | closest step {best} d {np.round(d[best],3)}")
    env.close()
