"""Dimension-wise borrowing programs for the LIBERO-90 tasks (CPU).

    python experiments/r7_dimcode.py --tasks 39,35,36,42,67,19 --K 8
Per task and stride (1: every frame; 5: every 5th, the policy is queried every 5 steps): the program, a check that the
stored code alone (ids + choices + the shared library) rebuilds exactly the executed chunk, its distance from the demo
chunk, its jerk vs the demo's, and its honest size.  Writes results/robot/r7_dim_<task>_s<stride>.npz, ledger r7_dimcode.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy import dimcode                                # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                    # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402


def jerk(x):
    return float(np.abs(np.diff(np.asarray(x).reshape(-1, 10, 7)[:, :, :6], 2, axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="39,35,36,42,67,19")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--strides", default="1,5")
    ap.add_argument("--tree", default="results/robot/r2b_libero_40_tree.npz")
    a = ap.parse_args()
    lib = dimcode.Library(Tree.load(a.tree))
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    for tid in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
        ep = d90.load(d90.episodes_of_text(texts[tid])[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        full = DemoProgram.from_dimcode(lib, rows, ep["state"], K=a.K)
        for s in [int(x) for x in a.strides.split(",")]:
            prog = full.subsample(s) if s > 1 else full
            path = f"results/robot/r7_dim_{tid}_s{s}.npz"
            prog.save(path)
            back = DemoProgram.load(path)
            rebuilt = np.stack([dimcode.decode(lib, i, c)[0] for i, c in zip(back.dim_ids, back.dim_choice)])
            exact = float(np.abs(rebuilt - back.mu).max())
            kept = rows[::s]
            m = dict(task=tid, text=texts[tid], stride=s, K=a.K, frames=int(T), kept=int(len(back.mu)),
                     rms=float(np.sqrt(((back.mu - kept) ** 2).mean())), jerk=jerk(back.mu), jerk_demo=jerk(kept),
                     rebuild_max_err=exact, bytes=back.bytes(), raw_bytes=int(2 * rows.size + 16 * T))
            print(f"[r7] task {tid} '{texts[tid][:36]}' stride {s}: rms {m['rms']:.3f}, jerk {m['jerk']:.4f} vs demo {m['jerk_demo']:.4f}, "
                  f"rebuilt from the code alone to {exact:.1e}; stores {m['bytes']} B (raw demo {m['raw_bytes']} B)", flush=True)
            ledger.record("r7_dimcode", bench="libero_90", tag=f"_s{s}", seed=0, keys=dict(task=tid), metrics=m,
                          bytes=dict(experts=0, tree=0, codes=m["bytes"], samples=0, teacher_copies=0, total=m["bytes"]),
                          config=vars(a), script="experiments/r7_dimcode.py", command=" ".join(sys.argv), artifacts=[path])


if __name__ == "__main__":
    main()
