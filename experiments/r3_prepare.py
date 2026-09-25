"""M3 (prepare): turn ONE LIBERO-90 demonstration per task into programs over the prototype library.

    python experiments/r3_prepare.py --tree results/robot/r2b_all_tree.npz --tasks 19,33,24,67,35,39 --K 3

For each LIBERO-90 benchmark task (unique language in the IPEC port), THE demo = the first IPEC episode of that text;
the next `--held` episodes are held out for the novelty read.  Writes results/robot/r3_prog_<tid>_{code,raw}.npz and a
ledger row per task: program bytes, how well the library explains the demo's chunks (PoE mean rms to the chunk,
per-dim NLL of the chunk under the PoE Gaussian, best-node NLL per level vs the library's own frames), gripper
events.  Runs on CPU (chunks env): numpy and parquet only.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.operators.address import best_nll                    # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                    # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402

L90_TEXT = None


def libero90_texts():
    """benchmark id -> language, from the LIBERO package's task list (bddl names; no simulator import)."""
    import re
    d = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
    src = open(d).read()
    block = src[src.index('"libero_90"'):]
    block = block[block.index("[") + 1: block.index("]")]
    names = re.findall(r'"([^"]+)"', block)
    out = {}
    for i, n in enumerate(names):
        m = re.match(r"^(?:[A-Z_]+SCENE\d+)_(.*)$", n)
        out[i] = (m.group(1) if m else n).replace("_", " ")
    return out, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tasks", required=True, help="LIBERO-90 benchmark ids")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--held", type=int, default=5)
    ap.add_argument("--library-npz", default="", help="r2b npz: the library's own frames for the NLL calibration")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tree = Tree.load(a.tree)
    levels = [int(x) for x in a.levels.split(",")]
    texts, names = libero90_texts()
    d90 = D.Demos90()
    norm = D.Norm.load()
    cal = None
    if a.library_npz:
        z = np.load(a.library_npz)
        lib = z["mu"] if "mu" in z else z["rows"]
        sub = np.random.default_rng(0).choice(len(lib), size=min(len(lib), 1000), replace=False)
        cal = {l: np.quantile(best_nll(tree, lib[sub], l, seed=1)[0], [0.5, 0.95]).tolist() for l in range(len(tree.u_grid))}
        print(f"[r3p] library NLL quantiles (median, q95) per level: {cal}", flush=True)
    os.makedirs("results/robot", exist_ok=True)
    for tid in [int(x) for x in a.tasks.split(",")]:
        text = texts[tid]
        eps = d90.episodes_of_text(text)
        if not eps:
            print(f"[r3p] task {tid} '{text}': no IPEC episodes (merged or dropped)", flush=True)
            continue
        demo = d90.load(eps[0])
        T = len(demo["actions"])
        idx, _ = D.chunk_indices(T, 10)
        rows = D.to_rows(demo["actions"][idx], norm)
        m = dict(task_id=tid, text=text, bddl=names[tid], demo_episode=int(eps[0]), n_frames=int(T),
                 gripper_events=D.gripper_events(demo["actions"]).tolist(), n_ipec_episodes=len(eps),
                 out_of_range_frac=float((np.abs(rows) > 1.0).mean()))
        for variant in ("code", "raw"):
            prog = DemoProgram.from_demo(tree, rows, demo["state"], K=a.K, levels=levels, variant=variant)
            path = f"results/robot/r3_prog_{tid}_{variant}{a.tag}.npz"
            prog.save(path)
            resid = float(np.sqrt(((prog.mu - rows) ** 2).mean()))
            nll = float((0.5 * ((rows - prog.mu) ** 2 / prog.var + np.log(2 * np.pi * prog.var))).mean())
            m[variant] = dict(bytes=prog.bytes(), mean_rms_to_demo=resid, nll_per_dim=nll, path=path)
        # novelty read: best-node NLL of the demo's chunks and of held-out demos per level
        held = [d90.load(e) for e in eps[1:1 + a.held]]
        hrows = np.concatenate([D.to_rows(h["actions"][D.chunk_indices(len(h["actions"]), 10)[0]], norm) for h in held]) if held else rows
        m["nll_by_level"] = {str(tree.u_grid[l]): dict(demo=float(np.median(best_nll(tree, rows, l, seed=2)[0])),
                                                        held=float(np.median(best_nll(tree, hrows, l, seed=3)[0])),
                                                        library=None if cal is None else cal[l]) for l in range(len(tree.u_grid))}
        print(f"[r3p] task {tid} '{text}' ({T} frames, {len(m['gripper_events'])} gripper events, {m['out_of_range_frac']:.2f} of dims outside the prior's q01-q99): "
              f"code {m['code']['bytes']} B rms {m['code']['mean_rms_to_demo']:.3f} nll {m['code']['nll_per_dim']:.2f} | "
              f"raw {m['raw']['bytes']} B | median best-node NLL by level {[round(v['demo'], 2) for v in m['nll_by_level'].values()]}", flush=True)
        ledger.record("r3_prepare", bench="libero_90", tag=a.tag, seed=0, keys=dict(task=tid), metrics=m,
                      bytes=dict(experts=0, tree=0, codes=m["code"]["bytes"], samples=0, teacher_copies=0, total=m["code"]["bytes"]),
                      config=vars(a), script="experiments/r3_prepare.py", command=" ".join(sys.argv),
                      artifacts=[m["code"]["path"], m["raw"]["path"]])


if __name__ == "__main__":
    main()
