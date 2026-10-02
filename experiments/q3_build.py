"""Build a task's PHASE CODES from one demonstration (the prototype-PoE plan; selection only: the demo is never executed).

Per phase of the demo (gripper open / closed runs), Wang's coverage (poe.select, the task's own policy row forced in, K = 3,
tau 0.5) picks experts from the policy family at the demo's own observations (the task's instruction, "pick up the {noun}" for the
scene's object types, the 16 attention-focus cells) and the motion prototypes (tree nodes at t in {0.3, 0.4, 0.55}).  The code is
saved as observation-free ids + per-dim weights: results/robot/poe_codes/<task>_d<demo>.json, with its bytes and build seconds.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q3_build.py --tasks 52+53 --demo 0" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import attn_bias as AB       # noqa: E402
from lifelong_ood.policy import data as D             # noqa: E402
from lifelong_ood.policy import factors as F          # noqa: E402
from lifelong_ood.policy import poe                   # noqa: E402

R = os.path.join(ROOT, "results", "robot")
OUT = os.path.join(R, "poe_codes")
_q2 = importlib.util.spec_from_file_location("q2", os.path.join(ROOT, "experiments", "q2_explain.py"))
Q2 = importlib.util.module_from_spec(_q2)
_q2.loader.exec_module(Q2)


def rows_for(t, text, tasks):
    ph = F.scene_phrases(t, tasks)
    nouns = sorted({ph.get(ty) for ty in t.objects.values() if ph.get(ty)})
    rows = [poe.PolicyRow(text, name="task")] + [poe.PolicyRow(f"pick up the {n}", name=f"prompt:{n}") for n in nouns]
    for bi in range(4):
        for bj in range(4):
            m = np.zeros((16, 16), np.float32)
            m[4 * bi:4 * bi + 4, 4 * bj:4 * bj + 4] = 1.0
            rows.append(poe.PolicyRow(text, focus={"agent": m}, name=f"cell:{bi}{bj}"))
    return rows


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    tids = [int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].split("+")]
    demo = int(sys.argv[sys.argv.index("--demo") + 1]) if "--demo" in sys.argv else 0
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    vpol = np.load(os.path.join(R, "poe_policy_var.npy"))
    from lifelong_ood.tree.nodes import Tree
    tree = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    os.makedirs(OUT, exist_ok=True)
    for tid in tids:
        t0 = time.time()
        rows = rows_for(by[("libero_90", tid)], texts[tid], tasks)
        P = len(rows)
        ep = d90.load(d90.episodes_of_text(texts[tid])[demo])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        Q = D.to_rows(ep["actions"][idx], norm)
        fr = list(range(0, T, Q2.STRIDE))
        Mp = np.zeros((len(fr), P, 70), np.float32)
        for n, i in enumerate(fr):
            obs = d90.observation(ep, i)
            pre = bk.prefix(bk.observations([dict(obs, prompt=r.prompt) for r in rows]))
            noise = np.tile(np.random.default_rng(1000 * tid + i).standard_normal(bk.dim_full).astype(np.float32), (P, 1))
            maps = [r.focus or {} for r in rows]
            gate = (lambda maps_, pre_: (lambda x, tt: (lambda: AB.biased(bk, maps_, mode="mass", rho=0.7,
                                                                              width=int(pre_.pad_masks.shape[1])))))(maps, pre)
            Mp[n] = bk.sample(P, pre, noise=noise, step_gate=gate)
        codes = []
        for name, s, e in Q2.phases(ep["actions"]):
            sel = [n for n, f in enumerate(fr) if s <= f < e]
            if not sel:
                continue
            qn = poe.query_nodes(tree, Q[fr][sel], Q2.LEVELS)                       # prototypes reached FROM the demo phase
            Mall = np.concatenate([np.transpose(Mp[sel], (1, 0, 2)), np.repeat(np.stack([e.m for e, _, _ in qn])[:, None, :], len(sel), 1)])
            Vall = np.concatenate([np.tile(vpol, (P, 1)), np.stack([e.V for e, _, _ in qn])])
            S, Wt, _ = poe.select(Q[fr][sel], Mall, Vall, K=3, forced=(0,))
            ids = [rows[k].name if k < P else qn[k - P][1] for k in S]
            codes.append(dict(phase=name, start=int(s), end=int(e), experts=ids, weights=Wt.astype(np.float16).tolist()))
        nbytes = int(sum(len(x.encode()) for c in codes for x in c["experts"]) + sum(np.asarray(c["weights"], np.float16).nbytes for c in codes))
        out = dict(task=tid, demo=demo, episode=int(ep["episode_index"]), language=texts[tid], codes=codes, bytes=nbytes,
                   build_secs=float(time.time() - t0))
        json.dump(out, open(os.path.join(OUT, f"{tid}_d{demo}.json"), "w"), indent=1)
        print(f"[q3] {tid}: {len(codes)} phases, {nbytes} B, {out['build_secs']:.0f} s | " +
              " ; ".join(f"{c['phase']}: {', '.join(c['experts'])}" for c in codes), flush=True)


if __name__ == "__main__":
    main()
