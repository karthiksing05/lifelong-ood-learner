"""P2 of the prototype-PoE plan: does the expert pool explain held-out demonstrations, and does a code built on ONE demo
transfer to another demo of the task when its scene options are recomputed at that demo's observations?

For each of the 21 held-out LIBERO-90 tasks, demos 0 and 1 (IPEC), phases from the demo's gripper command (open/closed runs),
query frames every 5 steps.  Expert pool at each frame's observation:
    policy family (live): row 0 = the task's instruction; "pick up the {noun}" for each movable object type of the scene;
                          the task's instruction with the action expert's attention raised to mass 0.7 on one of 4 x 4 token
                          blocks (no oracle).  Each row's mean = its sampled chunk (shared noise); variance = the policy's calibrated
                          error (results/robot/poe_policy_var.npy)
    motion prototypes:    nodes of the action-marginal tree at t in {0.3, 0.4, 0.55} (mode, extent); leaves (t = 0.05) only as
                          the replay ceiling
Selection on demo 0 per phase (poe.select: Wang's coverage, row 0 forced, K = 3, tau 0.5); the code's composed mean per frame
= sum_k beta_k m_k with beta from poe.make_beta (policy-family share lambda = 0.5, gripper from the policy family).
Reported per pool (policy only / + motion / + options / + both / + leaves): NLL and rms of the demo chunks under the composed
Gaussian, on demo 0 (fit) and on demo 1 with the options RECOMPUTED at demo 1's observations, against FROZEN options (demo 0's
values at the same phase position).  Pre-registered kill (m3_prereg q_probes): options cut the NLL by < 10%, or the recomputed
code beats the policy alone on < 12 of 21 tasks.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q2_explain.py --record" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import attn_bias as AB       # noqa: E402
from lifelong_ood.policy import data as D             # noqa: E402
from lifelong_ood.policy import factors as F          # noqa: E402
from lifelong_ood.policy import poe                   # noqa: E402

R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
if "--tasks" in sys.argv:
    HELD = [int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].split("+")]
LEVELS = (0.3, 0.4, 0.55)
LAM = 0.5
STRIDE = 5


def phases(actions):
    closed = actions[:, 6] > 0
    out, s = [], 0
    for i in range(1, len(closed) + 1):
        if i == len(closed) or closed[i] != closed[s]:
            out.append(("closed" if closed[s] else "open", s, i))
            s = i
    return [p for p in out if p[2] - p[1] >= STRIDE]


def tree_pool(levels):
    from lifelong_ood.tree.nodes import Tree
    tr = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
    M, V, names = [], [], []
    for u in levels:
        li = int(np.argmin([abs(x - u) for x in tr.u_grid]))
        for i in tr.levels[li]:
            nd = tr.nodes[i]
            M.append(np.asarray(nd.mode, np.float32))
            V.append(np.maximum(np.asarray(nd.extent_dim if hasattr(nd, "extent_dim") else nd.extent, np.float32), 1e-4))
            names.append(f"node:{u}:{i}")
    return np.stack(M), np.stack(V), names


def nll(Q, m, v):
    return float((0.5 * ((Q - m) ** 2 / v + np.log(2 * np.pi * v))).mean())


def main():
    import torch
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    if not os.path.exists(os.path.join(R, "poe_policy_var.npy")):          # the policy's calibrated error (also written by q0_share.py)
        z = np.load(os.path.join(R, "r2b_libero_40.npz"))
        np.save(os.path.join(R, "poe_policy_var.npy"),
                np.tile(((z["mu"] - z["demo"]) ** 2).reshape(-1, 10, 7).mean((0, 1)), 10).astype(np.float32))
    vpol = np.load(os.path.join(R, "poe_policy_var.npy"))
    global TREE
    from lifelong_ood.tree.nodes import Tree
    TREE = Tree.load(os.path.join(R, "r2b_libero_40_tree.npz"))
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    per_task = {}
    for tid in HELD:
        t = by[("libero_90", tid)]
        ph = F.scene_phrases(t, tasks)
        nouns = sorted({ph.get(ty) for ty in t.objects.values() if ph.get(ty)})
        rows = [poe.PolicyRow(texts[tid], name="task")]
        rows += [poe.PolicyRow(f"pick up the {n}", name=f"prompt:{n}") for n in nouns]
        for bi in range(4):
            for bj in range(4):
                m = np.zeros((16, 16), np.float32)
                m[4 * bi:4 * bi + 4, 4 * bj:4 * bj + 4] = 1.0
                rows.append(poe.PolicyRow(texts[tid], focus={"agent": m}, name=f"cell:{bi}{bj}"))
        P = len(rows)
        eps_ = d90.episodes_of_task(tid)
        demos = {}
        for d in (0, 1):
            if d >= len(eps_):
                continue
            ep = d90.load(eps_[d])
            T = len(ep["actions"])
            idx, _ = D.chunk_indices(T, bk.H)
            Q = D.to_rows(ep["actions"][idx], norm)
            fr = list(range(0, T, STRIDE))
            Mp = np.zeros((len(fr), P, 70), np.float32)
            for n, i in enumerate(fr):
                obs = d90.observation(ep, i)
                pre = bk.prefix(bk.observations([dict(obs, prompt=r.prompt) for r in rows]))
                noise = np.tile(np.random.default_rng(1000 * tid + i).standard_normal(bk.dim_full).astype(np.float32), (P, 1))
                maps = [r.focus or {} for r in rows]
                gate = (lambda maps_, pre_: (lambda x, tt: (lambda: AB.biased(bk, maps_, mode="mass", rho=0.7,
                                                                                  width=int(pre_.pad_masks.shape[1])))))(maps, pre)
                Mp[n] = bk.sample(P, pre, noise=noise, step_gate=gate)
            demos[d] = dict(Q=Q[fr], frames=fr, phases=phases(ep["actions"]), Mp=Mp, T=T)
            if "--dump" in sys.argv:
                np.savez(os.path.join(R, ".scratch", f"q2_dump_{tid}_d{d}.npz"), Q=Q[fr], Mp=Mp, frames=np.array(fr),
                         actions=ep["actions"])
        if 0 not in demos:
            continue
        res = {}
        d0 = demos[0]
        codes = []
        for name, (s, e) in [(p[0], (p[1], p[2])) for p in d0["phases"]]:
            sel = [n for n, f in enumerate(d0["frames"]) if s <= f < e]
            if not sel:
                continue
            Qp = d0["Q"][sel]
            pool = {}
            Mpol = np.transpose(d0["Mp"][sel], (1, 0, 2))                                   # [P, n, 70]
            Vpol = np.tile(vpol, (P, 1))
            qn = poe.query_nodes(TREE, Qp, LEVELS)                                        # Wang: prototypes reached FROM the query
            PMq = np.stack([e.m for e, _, _ in qn])
            PVq = np.stack([e.V for e, _, _ in qn])
            qnames = [nm for _, nm, _ in qn]
            ln = poe.query_nodes(TREE, Qp, (0.05,))
            LMq, LVq = np.stack([e.m for e, _, _ in ln]), np.stack([e.V for e, _, _ in ln])
            Mpro = np.repeat(PMq[:, None, :], len(sel), 1)
            pool["policy"] = (Mpol[:1], Vpol[:1], 1)
            pool["motion"] = (np.concatenate([Mpol[:1], Mpro]), np.concatenate([Vpol[:1], PVq]), 1)
            pool["options"] = (Mpol, Vpol, P)
            pool["both"] = (np.concatenate([Mpol, Mpro]), np.concatenate([Vpol, PVq]), P)
            pool["leaves"] = (np.concatenate([Mpol, Mpro, np.repeat(LMq[:, None, :], len(sel), 1)]),
                              np.concatenate([Vpol, PVq, LVq]), P)
            for pk, (Mall, Vall, npol) in pool.items():
                S, Wt, _ = poe.select(Qp, Mall, Vall, K=3, forced=(0,))
                fam = [k for k in S if k < npol]
                pro = [k for k in S if k >= npol]
                order = fam + pro
                Wo = np.stack([Wt[S.index(k)] for k in order])
                protos = [poe.ProtoExpert(Mall[k][0], Vall[k], str(k)) for k in pro]
                ex, betaf = poe.compose(Wo, len(fam), protos, policy_share=LAM)        # Wang's product of the prototypes
                beta = betaf(0.1)

                def comp(Mframes, fam=fam, beta=beta, ex=ex):                      # Mframes [J, n, 70]
                    m = sum(beta[i][None] * Mframes[k] for i, k in enumerate(fam))
                    v = sum(beta[i][None] * Vall[k][None] for i, k in enumerate(fam))
                    if ex:
                        m = m + beta[-1][None] * ex[0].m[None]
                        v = v + beta[-1][None] * ex[0].V[None]
                    return m, v
                m0, v0 = comp(Mall)
                names_all = [r_.name for r_ in rows][:npol] + qnames + ["leaf"] * len(LMq)
                r = dict(experts=[names_all[k] if k < len(names_all) else "leaf" for k in order],
                         nll_fit=nll(Qp, m0, v0), rms_fit=float(np.sqrt(((Qp - m0) ** 2).mean())))
                if 1 in demos and pk != "leaves":
                    d1 = demos[1]
                    ph1 = [p for p in d1["phases"] if p[0] == name]
                    if ph1:
                        s1, e1 = ph1[min(len(ph1) - 1, [p for p in d0["phases"] if p[0] == name].index((name, s, e)))][1:]
                        sel1 = [n for n, f in enumerate(d1["frames"]) if s1 <= f < e1]
                        if sel1:
                            Q1 = d1["Q"][sel1]
                            M1pol = np.transpose(d1["Mp"][sel1], (1, 0, 2))
                            pos = np.linspace(0, len(sel) - 1, len(sel1)).round().astype(int)       # frozen: demo 0's values
                            M1frz = Mpol[:, pos]
                            def stack(Mp_):
                                parts = [Mp_ if npol > 1 else Mp_[:1]]
                                if pk in ("motion", "both"):
                                    parts.append(np.repeat(PMq[:, None, :], len(sel1), 1))
                                return np.concatenate(parts)
                            mr, vr = comp(stack(M1pol))
                            mf, vf = comp(stack(M1frz))
                            r.update(nll_recomputed=nll(Q1, mr, vr), nll_frozen=nll(Q1, mf, vf),
                                     nll_policy_d1=nll(Q1, M1pol[0], np.tile(vpol, (len(sel1), 1))),
                                     rms_recomputed=float(np.sqrt(((Q1 - mr) ** 2).mean())),
                                     rms_policy_d1=float(np.sqrt(((Q1 - M1pol[0]) ** 2).mean())))
                res.setdefault(pk, []).append(dict(phase=name, n=len(sel), **r))
                if pk == "both":
                    codes.append(poe.PhaseCode(name, r["experts"], Wo, LAM))
        agg = {pk: {k: float(np.mean([x[k] for x in v if k in x])) for k in
                    ("nll_fit", "rms_fit", "nll_recomputed", "nll_frozen", "nll_policy_d1", "rms_recomputed", "rms_policy_d1")
                    if any(k in x for x in v)} for pk, v in res.items()}
        per_task[str(tid)] = dict(language=texts[tid], pools=agg, phases=res, code_bytes=int(sum(c.nbytes() for c in codes)),
                                  n_policy_rows=P)
        b = agg.get("both", {})
        print(f"[q2] {tid:>3} fit NLL policy {agg['policy']['nll_fit']:.2f} motion {agg['motion']['nll_fit']:.2f} options "
              f"{agg['options']['nll_fit']:.2f} both {b.get('nll_fit', float('nan')):.2f} leaves {agg['leaves']['nll_fit']:.2f} | demo 1: "
              f"policy {b.get('nll_policy_d1', float('nan')):.2f} recomputed {b.get('nll_recomputed', float('nan')):.2f} frozen "
              f"{b.get('nll_frozen', float('nan')):.2f} | {per_task[str(tid)]['code_bytes']} B | {texts[tid]}", flush=True)
    pools = ("policy", "motion", "options", "both", "leaves")
    summary = {pk: float(np.mean([v["pools"][pk]["nll_fit"] for v in per_task.values() if pk in v["pools"]])) for pk in pools}
    with_d1 = [v for v in per_task.values() if "nll_recomputed" in v["pools"].get("both", {})]
    beats = int(sum(v["pools"]["both"]["nll_recomputed"] < v["pools"]["both"]["nll_policy_d1"] for v in with_d1))
    options_cut = 1.0 - summary["options"] / summary["policy"] if summary["policy"] > 0 else None
    out = dict(lam=LAM, levels=list(LEVELS), nll_fit=summary, tasks_with_demo1=len(with_d1), recomputed_beats_policy=beats,
               recomputed_vs_frozen=int(sum(v["pools"]["both"]["nll_recomputed"] < v["pools"]["both"]["nll_frozen"] for v in with_d1)),
               options_nll_cut=options_cut, kill=bool((options_cut is not None and options_cut < 0.10) or beats < 12),
               code_bytes_mean=float(np.mean([v["code_bytes"] for v in per_task.values()])), per_task=per_task)
    print(f"[q2] NLL fit {summary}; options cut {options_cut}; recomputed code beats the policy on demo 1 in {beats} of "
          f"{len(with_d1)} tasks (vs frozen {out['recomputed_vs_frozen']}) -> kill {out['kill']}", flush=True)
    json.dump(out, open(os.path.join(R, "q2_explain.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("q2_explain", bench="libero_90", tag="read", seed=0,
                            metrics={k: v for k, v in out.items() if k != "per_task"},
                            bytes=dict(experts=0, tree=0, codes=int(out["code_bytes_mean"]), samples=0, teacher_copies=0,
                                       total=int(out["code_bytes_mean"])), config={},
                            script="experiments/q2_explain.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, "q2_explain.json")]))


if __name__ == "__main__":
    main()
