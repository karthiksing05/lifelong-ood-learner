"""Fit a COMPOSER per LIBERO-90 task from its ONE demo and measure it offline (CPU); writes the programs.

    python experiments/r6_composer.py --tasks 39,35,36,42,67,19 [--M 32 --hidden 32]

Per task: vocabulary (M library prototypes from the band levels), fit, then: reconstruction rms of the composed chunk vs
the demo chunk; the effective number of prototypes borrowed per (step, dim) (exp of the entropy); jerk within the chunk
(mean |second difference over steps| of the arm dims) vs the demo's own; what the program stores.  Writes
results/robot/r6_comp_<task>{tag}.npz and a ledger row r6_composer per task.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import composer as C                          # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.tree.nodes import Tree                               # noqa: E402


def jerk(rows):
    x = np.asarray(rows, np.float32).reshape(-1, 10, 7)[:, :, :6]
    return float(np.abs(np.diff(x, 2, axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="39,35,36,42,67,19")
    ap.add_argument("--tree", default="results/robot/r2b_libero_40_tree.npz")
    ap.add_argument("--levels", default="2,3,4")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam-ent", type=float, default=1e-3)
    ap.add_argument("--lam-smooth", type=float, default=1e-2)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    tree = Tree.load(a.tree)
    levels = [int(x) for x in a.levels.split(",")]
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    for tid in [int(x) for x in a.tasks.split(",")]:
        t0 = time.time()
        ep = d90.load(d90.episodes_of_text(texts[tid])[0])
        T = len(ep["actions"])
        rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
        ids, cover = C.vocabulary(tree, rows, levels, M=a.M)
        Mk, Vk = C.prototypes(tree, ids)
        Z = C.features(ep["state"], np.arange(T) / max(T - 1, 1))
        comp = C.Composer(len(ids), Z.shape[1], a.hidden)
        hist = C.fit(comp, Z, rows, Mk, Vk, steps=a.steps, lam_ent=a.lam_ent, lam_smooth=a.lam_smooth)
        with torch.no_grad():
            mu, var, w = comp(torch.as_tensor(Z), Mk, Vk)
        mu = mu.numpy()
        wn = w.numpy()
        eff = float(np.exp(-(wn * np.log(wn + 1e-9)).sum(1)).mean())
        top1 = float(wn.max(1).mean())
        used = int((wn.max(axis=(0, 2, 3)) > 0.2).sum())
        best1 = float(np.sqrt(((Mk.numpy().reshape(len(ids), -1)[None] - rows[:, None]) ** 2).mean(2)).min(1).mean())
        prog = C.ComposerProgram(ids, {k: v.detach().numpy() for k, v in comp.state_dict().items()}, ep["state"][:: a.stride], T,
                                 a.stride, Z.shape[1], a.hidden, a.tree)
        path = f"results/robot/r6_comp_{tid}{a.tag}.npz"
        prog.save(path)
        m = dict(task=tid, text=texts[tid], n_frames=int(T), M=len(ids), vocab_cover=cover, n_params=comp.n_params(),
                 rms=float(np.sqrt(((mu - rows) ** 2).mean())), best_single_rms=best1, effective_prototypes=eff, top_weight=top1,
                 prototypes_used=used, jerk_composed=jerk(mu), jerk_demo=jerk(rows), program_bytes=prog.bytes(),
                 raw_bytes=int(2 * rows.size + 16 * T), curve=hist, secs=time.time() - t0)
        print(f"[r6] task {tid} '{texts[tid][:40]}': rms {m['rms']:.3f} (best single prototype {best1:.3f}); borrows "
              f"{eff:.1f} prototypes per (step, dim), {used}/{len(ids)} used; jerk {m['jerk_composed']:.4f} vs demo {m['jerk_demo']:.4f}; "
              f"stores {m['program_bytes']} B vs raw demo {m['raw_bytes']} B; {m['secs']:.0f}s", flush=True)
        ledger.record("r6_composer", bench="libero_90", tag=a.tag, seed=0, keys=dict(task=tid), metrics=m,
                      bytes=dict(experts=0, tree=0, codes=m["program_bytes"], samples=0, teacher_copies=0, total=m["program_bytes"]),
                      config=vars(a), script="experiments/r6_composer.py", command=" ".join(sys.argv), artifacts=[path])


if __name__ == "__main__":
    main()
