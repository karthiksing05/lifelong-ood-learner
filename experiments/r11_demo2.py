"""Demonstration robustness: the programs of a held-out task built from ANOTHER demo (IPEC episode --demo of its text).

Writes, with the suffix _d<demo>: results/robot/r3_prog_<t>_raw_lib40_d<k>.npz (replay), r7_dim_<t>_s1_d<k>.npz
(dimension-wise borrowing from the tree's band, K = 8) and r10_seg_<t>_s5_d<k>.npz (segment borrowing from the leaves).
The fold and the LoRA of that demo come from r4_fold.py / r3_fewshot.py with --demo k (experiments/prep_demo.sbatch).

    python experiments/r11_demo2.py --tasks 35,36,39,42 --demo 1
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy import dimcode                                 # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                     # noqa: E402
from lifelong_ood.tree.nodes import Tree                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--demo", type=int, default=1)
    a = ap.parse_args()
    os.chdir(ROOT)
    lib = dimcode.Library(Tree.load("results/robot/r2b_libero_40_tree.npz"))
    leaves = np.load("results/robot/r2b_libero_40.npz")["mu"].astype(np.float32).reshape(-1, 10, 7)
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    k = a.demo
    for t in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
        ep_idx = d90.episodes_of_task(t)[k]
        ep = d90.load(ep_idx)
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        raw = DemoProgram.from_demo(None, rows, ep["state"], variant="raw")
        raw.save(f"results/robot/r3_prog_{t}_raw_lib40_d{k}.npz")
        dim = DemoProgram.from_dimcode(lib, rows, ep["state"], K=8)
        dim.save(f"results/robot/r7_dim_{t}_s1_d{k}.npz")
        seg = DemoProgram.from_segcode(leaves, rows, ep["state"]).subsample(5)
        seg.save(f"results/robot/r10_seg_{t}_s5_d{k}.npz")
        rms = lambda p, s=1: float(np.sqrt(((p.mu - rows[::s]) ** 2).mean()))   # noqa: E731
        print(f"[r11] task {t} demo {k} (IPEC episode {ep_idx}, {T} frames, gripper events {D.gripper_events(ep['actions']).tolist()}): "
              f"replay {raw.bytes()} B; borrowing rms {rms(dim):.3f} {dim.bytes()} B; segments rms {rms(seg, 5):.3f} {seg.bytes()} B", flush=True)


if __name__ == "__main__":
    main()
