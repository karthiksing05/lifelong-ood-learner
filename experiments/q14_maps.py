"""What a 4 KB object prototype points at: its cosine map over the policy's own SigLIP patch features in a task's first frames,
for the task's own prototype and for one borrowed from another task's demo (ledger q14_borrow).  Writes the maps and a figure.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q14_maps.py" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")

R = os.path.join(ROOT, "results", "robot")
FR = os.path.join(R, "f0b_firstframes")
PAIRS = [(43, 43), (43, 36), (70, 70), (70, 5), (13, 13), (13, 12), (12, 12), (12, 13)]   # (scene task, prototype's task)


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("q4", os.path.join(ROOT, "experiments", "q4_template.py"))
    q4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q4)
    bk = PolicyBackbone("pi05_libero", device="cuda")
    P = np.load(os.path.join(R, "q4_object_protos.npz"))
    out = {}
    for t, src in PAIRS:
        z = np.load(os.path.join(FR, f"libero_90_{t}.npz"), allow_pickle=True)
        f = q4.feats(bk, z["img"][0], z["wimg"][0], z["state8"][0])
        out[f"{t}_{src}"] = (f @ P[str(src)]).reshape(16, 16).astype(np.float32)
        out[f"img_{t}"] = z["img"][0]
        print(f"[q14m] scene {t} prototype {src}: top {np.argsort(out[f'{t}_{src}'].reshape(-1))[::-1][:4].tolist()}", flush=True)
    np.savez(os.path.join(R, "q14_maps.npz"), **out)
    try:
        figure(out)
    except ImportError as e:                         # the openpi venv may lack matplotlib: rerun with --figure elsewhere
        print(f"[q14m] no figure here ({e}); run experiments/q14_maps.py --figure", flush=True)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
    for a, (t, src) in zip(ax.T.reshape(-1), PAIRS):
        m = out[f"{t}_{src}"]
        a.imshow(out[f"img_{t}"])
        a.imshow(np.kron(m, np.ones((14, 14))), cmap="magma", alpha=0.55, extent=(0, 224, 224, 0))
        for p in np.argsort(m.reshape(-1))[::-1][:4]:
            i, j = divmod(int(p), 16)
            a.add_patch(plt.Rectangle((j * 14, i * 14), 14, 14, fill=False, ec="#39d353", lw=1.6))
        a.set_title(f"task {t}, {'own' if t == src else f'from task {src}'} prototype", fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "robot", "q14_maps.png"), dpi=110)
    print("[q14m] figure figures/robot/q14_maps.png", flush=True)


if __name__ == "__main__":
    if "--figure" in sys.argv:
        figure(dict(np.load(os.path.join(R, "q14_maps.npz"))))
    else:
        main()
