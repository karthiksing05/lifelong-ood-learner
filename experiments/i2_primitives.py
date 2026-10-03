"""INTERPRETATION 2: what the motion primitives are, in weight space (exploratory, offline).  A primitive is a low-rank update
dW = B A on each Linear of the action expert; inner products of two updates are exact in rank space,
<B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T)), so every comparison is cheap.  Reports
  similarity -- cosine between primitives (all banks together): same step key against different keys, and consolidated against
                single-demonstration primitives and per-task LoRAs;
  convergence -- cosine between a stream key's successive refits as demonstrations accumulate;
  where      -- each primitive's update norm by layer (0-17) and by module (attention q/k/v/o, MLP gate/up/down).
Figures: figures/robot/i2_similarity.png, figures/robot/i2_layers.png.

    sbatch --export=ALL,SCRIPT="experiments/i2_primitives.py --record" slurm/jobs/cpu.sbatch
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
KEYCOL = {"put|on top of|the cabinet": "#2a78d6", "put|on|the plate": "#eb6834", "put|in|the basket": "#1baf7a",
          "put|in|the tray": "#eda100"}                      # the dataviz reference palette's slots 1-4 (validated order)
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load(path):
    z = torch.load(path, map_location="cpu")
    banks = {n: (w["A"].float().numpy(), w["B"].float().numpy()) for n, w in z["banks"].items()}
    return banks, z.get("meta", {})


def inner(p, q):
    s = 0.0
    for n, (A1, B1) in p.items():
        if n in q:
            A2, B2 = q[n]
            s += float(np.sum((B1.T @ B2) * (A1 @ A2.T)))
    return s


def norms(p):
    out = {}
    for n, (A, B) in p.items():
        out[n] = float(np.sqrt(max(np.sum((B.T @ B) * (A @ A.T)), 0.0)))
    return out


def main():
    st = json.load(open(os.path.join(R, "c2_state.json")))
    prims = {}                                                   # name -> (path, key, kind)
    for k, s in st.items():
        if k.startswith("_"):
            continue
        for i in range(1, s["n_fit"] + 1):
            prims[f"C2{k}{i}"] = (os.path.join(R, f"prim_C2{k}{i}_0.pt"), s["key"], "stream refit")
    for n in ("Jcab15", "Jcab25", "Jcab31"):
        prims[n] = (os.path.join(R, f"prim_{n}_0.pt"), "put|on top of|the cabinet", "leave-one-out joint")
    for n in ("Jplate12", "Jplate13", "Jplate14", "Jplate30"):
        prims[n] = (os.path.join(R, f"prim_{n}_0.pt"), "put|on|the plate", "leave-one-out joint")
    for f in sorted(glob.glob(os.path.join(R, "prim_Q27t*_0.pt"))):
        n = os.path.basename(f)[5:-5]
        prims[n] = (f, None, "leave-one-out joint")
    for t in (36, 37, 41, 43, 40, 42, 34):                     # own single-demonstration primitives (c0_fit) of one-step tasks
        f = os.path.join(R, f"prim_{t}_0.pt")
        if os.path.exists(f):
            prims[f"own{t}"] = (f, None, "single demonstration")
    loaded = {}
    for n, (f, key, kind) in prims.items():
        if not os.path.exists(f):
            continue
        b, meta = load(f)
        loaded[n] = dict(banks=b, key=key or meta.get("step_key"), kind=kind)
        print(f"[i2] {n}: {kind}, key {loaded[n]['key']}, {len(b)} banks", flush=True)
    names = list(loaded)
    sq = {n: inner(loaded[n]["banks"], loaded[n]["banks"]) for n in names}
    S = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            S[i, j] = S[j, i] = inner(loaded[a]["banks"], loaded[b]["banks"]) / np.sqrt(sq[a] * sq[b])
    # similarity: final-or-joint primitives, same key against different keys
    joint = [n for n in names if loaded[n]["kind"] != "single demonstration"]
    same, diff = [], []
    for i, a in enumerate(joint):
        for b in joint[i + 1:]:
            if a.rstrip("0123456789") == b.rstrip("0123456789") and loaded[a]["kind"] == "stream refit":
                continue                                         # successive refits of one key: reported as convergence
            (same if loaded[a]["key"] == loaded[b]["key"] else diff).append(S[names.index(a), names.index(b)])
    single = [n for n in names if loaded[n]["kind"] == "single demonstration"]
    s_single = [S[names.index(a), names.index(b)] for i, a in enumerate(single) for b in single[i + 1:]]
    conv = {}
    for k, s in st.items():
        if k.startswith("_"):
            continue
        seq = [f"C2{k}{i}" for i in range(1, s["n_fit"] + 1) if f"C2{k}{i}" in loaded]
        conv[k] = [float(S[names.index(a), names.index(b)]) for a, b in zip(seq, seq[1:])]
    # where: share of the squared update norm by layer and by module
    where = {}
    for n in names:
        nb = norms(loaded[n]["banks"])
        tot = sum(v ** 2 for v in nb.values())
        lay, mod = np.zeros(18), {m: 0.0 for m in MODULES}
        for bank, v in nb.items():
            m = re.search(r"layers\.(\d+)\.", bank)
            if m:
                lay[int(m.group(1))] += v ** 2 / tot
            for mm in MODULES:
                if bank.endswith(mm):
                    mod[mm] += v ** 2 / tot
        where[n] = dict(layers=lay.tolist(), modules=mod, total_norm=float(np.sqrt(tot)))
    res = dict(names=names, kinds={n: loaded[n]["kind"] for n in names}, keys={n: loaded[n]["key"] for n in names},
               same_key_cos=float(np.mean(same)) if same else None, diff_key_cos=float(np.mean(diff)) if diff else None,
               n_same=len(same), n_diff=len(diff), single_demo_cos=float(np.mean(s_single)) if s_single else None,
               convergence=conv, where=where, similarity=S.tolist())
    print(json.dumps({k: res[k] for k in ("same_key_cos", "diff_key_cos", "n_same", "n_diff", "single_demo_cos", "convergence")}, indent=1),
          flush=True)
    figures(res, names, S)
    json.dump(res, open(os.path.join(R, "i2_primitives.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("i2_primitives", bench="libero_90", tag="read", seed=0,
                            metrics={k: v for k, v in res.items() if k not in ("similarity", "where")} | {
                                "module_share_mean": {m: float(np.mean([w["modules"][m] for w in where.values()])) for m in MODULES},
                                "layer_share_mean": np.mean([w["layers"] for w in where.values()], 0).tolist()},
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i2_primitives.py", command=" ".join(sys.argv)))


def figures(res, names, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(range(len(names)), key=lambda i: (str(res["keys"][names[i]]), res["kinds"][names[i]], names[i]))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(S[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), [f"{names[i]}  ({res['keys'][names[i]]})" for i in order], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03, label="cosine of the weight updates")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "robot", "i2_similarity.png"), dpi=130)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, c in KEYCOL.items():
        fin = [n for n in names if res["keys"][n] == k and res["kinds"][n] != "single demonstration"]
        if fin:
            ax.plot(range(18), np.mean([res["where"][n]["layers"] for n in fin], 0), color=c, lw=2, label=k.replace("|", " "))
    ax.set_xlabel("action-expert layer")
    ax.set_ylabel("share of the update's squared norm")
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "robot", "i2_layers.png"), dpi=130)


if __name__ == "__main__":
    main()
