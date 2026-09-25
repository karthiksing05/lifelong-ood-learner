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
from lifelong_ood.policy import novelty                              # noqa: E402
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
    all_levels = list(range(len(tree.u_grid)))
    thr = None
    if a.library_npz:
        lib = np.load(a.library_npz)
        lib_text = D.Demos().tasks
        cal_rows = novelty.heldout_rows(D.Demos(), lib)
        thr = novelty.calibrate(tree, cal_rows, all_levels)
        print(f"[r3p] q95 NLL of {len(cal_rows)} held-out library frames per level: {[round(v, 2) for v in thr.values()]}", flush=True)
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
            if a.library_npz:
                prog.with_prompts(rows, lib["demo"], lib["task"], lib_text, k=5)
            path = f"results/robot/r3_prog_{tid}_{variant}{a.tag}.npz"
            prog.save(path)
            resid = float(np.sqrt(((prog.mu - rows) ** 2).mean()))
            nll = float((0.5 * ((rows - prog.mu) ** 2 / prog.var + np.log(2 * np.pi * prog.var))).mean())
            m[variant] = dict(bytes=prog.bytes(), mean_rms_to_demo=resid, nll_per_dim=nll, path=path)
            if prog.prompt_id is not None and variant == "code":
                ids, cnt = np.unique(prog.prompt_id, return_counts=True)
                m["prompt_program"] = {prog.prompt_table[int(i)]: int(c) for i, c in zip(ids, cnt)}
        # novelty read (ADDRESS): the demo's chunks vs held-out library frames, per level
        if thr is not None:
            nll, unexp, depth = novelty.read(tree, rows, all_levels, thr)
            m["unexplained_frac"] = float(unexp.mean())
            m["unexplained_by_level"] = {str(tree.u_grid[l]): float((nll[l] > thr[l]).mean()) for l in all_levels}
            m["depth_hist"] = {str(tree.u_grid[l]) if l >= 0 else "explained": int((depth == l).sum()) for l in [-1] + all_levels}
            m["nll_median"] = {str(tree.u_grid[l]): float(np.median(nll[l])) for l in all_levels}
        print(f"[r3p] task {tid} '{text}' ({T} frames, {len(m['gripper_events'])} gripper events, {m['out_of_range_frac']:.2f} of dims outside q01-q99): "
              f"code {m['code']['bytes']} B rms {m['code']['mean_rms_to_demo']:.3f} | raw {m['raw']['bytes']} B"
              + ("" if thr is None else f" | unexplained {m['unexplained_frac']:.2f}; by level {[round(v, 2) for v in m['unexplained_by_level'].values()]}"), flush=True)
        ledger.record("r3_prepare", bench="libero_90", tag=a.tag, seed=0, keys=dict(task=tid), metrics=m,
                      bytes=dict(experts=0, tree=0, codes=m["code"]["bytes"], samples=0, teacher_copies=0, total=m["code"]["bytes"]),
                      config=vars(a), script="experiments/r3_prepare.py", command=" ".join(sys.argv),
                      artifacts=[m["code"]["path"], m["raw"]["path"]])


if __name__ == "__main__":
    main()
