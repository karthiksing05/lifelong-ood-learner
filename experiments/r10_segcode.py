"""Segment-wise borrowing programs (policy/segcode.py) for held-out LIBERO-90 tasks, from ONE demo each.

Leaves = the policy's chunk means at the 10,249 library frames (results/robot/r2b_libero_40.npz "mu"), the components of
its action marginal.  Writes results/robot/r10_seg_<task>_s{1,5}.npz (s = alignment-key stride), checks that the executed
chunks rebuild exactly from the stored ids, and records ledger rows r10_segcode (rms to the demo, jerk, bytes).

    python experiments/r10_segcode.py --tasks 39,35,36,42
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy import segcode                                 # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                     # noqa: E402


def jerk(rows):
    return float(np.abs(np.diff(np.asarray(rows).reshape(len(rows), 10, 7), 3, axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--npz", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--L", type=int, default=10)
    a = ap.parse_args()
    os.chdir(ROOT)
    leaves = np.load(a.npz)["mu"].astype(np.float32).reshape(-1, 10, 7)
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    for t in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
        ep = d90.load(d90.episodes_of_text(texts[t])[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        full = DemoProgram.from_segcode(leaves, rows, ep["state"], L=a.L)
        for s in (1, 5):
            prog = full.subsample(s) if s > 1 else full
            path = f"results/robot/r10_seg_{t}_s{s}.npz"
            prog.save(path)
            back = DemoProgram.load(path)
            hat = segcode.decode(leaves, back.seg_ids, back.seg_offs, T + 9, a.L)
            rebuilt = segcode.windows(hat, T)[::s]
            exact = float(np.abs(rebuilt - back.mu).max())
            m = dict(task=t, text=texts[t], stride=s, L=a.L, frames=T, kept=int(len(back.mu)), segments=int(len(back.seg_ids)),
                     rms=float(np.sqrt(((back.mu - rows[::s]) ** 2).mean())), jerk=jerk(back.mu), jerk_demo=jerk(rows[::s]),
                     rebuild_max_err=exact, bytes=back.bytes(), replay_bytes=DemoProgram.from_demo(None, rows, ep["state"], variant="raw").bytes())
            print(f"[r10] task {t} '{texts[t][:34]}' key stride {s}: rms {m['rms']:.3f}, jerk {m['jerk']:.4f} (demo {m['jerk_demo']:.4f}), "
                  f"rebuilt from the ids to {exact:.1e}; {m['bytes']} B (replayed demo {m['replay_bytes']} B)", flush=True)
            ledger.record("r10_segcode", bench="libero_90", tag=f"_s{s}", seed=0, keys=dict(task=t), metrics=m,
                          bytes=dict(experts=0, tree=0, codes=m["bytes"], samples=0, teacher_copies=0, total=m["bytes"]),
                          config=vars(a), script="experiments/r10_segcode.py", command=" ".join(sys.argv), artifacts=[path])


if __name__ == "__main__":
    main()
