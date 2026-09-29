import pathlib, numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
t = 39
suite = benchmark.get_benchmark_dict()["libero_90"]()
task = suite.get_task(t); init = suite.get_task_init_states(t)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
np.random.seed(7)
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128); env.seed(7)
inner = env.env
obj = inner.get_object("flat_stove_1")
joints = list(obj.joints)
print("joints", joints)
def knob():
    return [float(inner.sim.data.qpos[inner.sim.model.get_joint_qpos_addr(j)]) for j in joints]
def site():
    out = {}
    for n in inner.sim.model.body_names:
        if "stove" in n: out[n] = np.round(inner.sim.data.body_xpos[inner.sim.model.body_name2id(n)], 3).tolist()
    return out
vals = []
for k in range(50):
    env.reset(); env.set_init_state(init[k])
    for _ in range(10): env.step([0.0] * 6 + [-1.0])
    vals.append(knob())
    if k < 3: print("init", k, "knob", knob(), "bodies", site())
vals = np.array(vals); print("knob over 50 init states: min", vals.min(0), "max", vals.max(0), "std", vals.std(0))
m = np.load("results/robot/p0_demo/39_match.npz", allow_pickle=True)
env.reset(); env.set_init_state(m["states"][0]); print("demo kept 0 knob", knob(), "bodies", site())
env.set_init_state(m["states"][-1]); print("demo last knob", knob(), "turn_off", [bool(obj.turn_off(q)) for q in knob()])
for i in (60, 70, 80, 90, 100):
    env.set_init_state(m["states"][i]); print("demo kept", i, "knob", np.round(knob(), 3).tolist(), "eef", np.round(inner.sim.data.site_xpos[inner.robots[0].eef_site_id], 3).tolist())
import inspect; print(inspect.getsource(type(obj).turn_off))
env.close()
