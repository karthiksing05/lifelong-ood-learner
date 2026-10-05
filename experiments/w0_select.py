"""Probe W0 (EXPLORATORY, offline; docs/PLAN_WANG.md): Wang et al.'s SELECTION by one query, on the factored library.

Wang et al. (arXiv 2605.07078) compose a frozen diffusion model's prototypes for an unseen combination by selecting the prototypes
that best cover ONE query; their stated limitation is no mechanism to detect, consolidate and reuse genuinely new primitives.  The
factored library consolidates and reuses, but selects its prototypes by parsing the instruction (object_key / instance_key /
goal_key).  Here the query selects: a HELD-OUT demonstration of each stream task (the second episode of its own scene's block; no
prototype has seen it) is split at its own first grasp, and every prototype is scored by how much it lowers the policy's own
flow-matching loss on the query's frames -- object and instance prototypes on the approach frames, goal prototypes on the
transport frames (the same noise and times for every prototype, so the scores are paired):
    gain(p) = L_policy(query) - L_p(query)
Reports: does the top-scoring prototype equal the concept the instruction names (recognition)?  Does a low best gain flag a factor
the library lacks (novelty: each query's gain under its own concept against its best gain under any OTHER concept)?

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/w0_select.py --record" slurm/jobs/robot_gpu.sbatch
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import primitives as P              # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SAMPLES, DEMO = 4, 1


def main():
    import torch
    import importlib.util
    from lifelong_ood.experts.lora import keys_on
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("e1", os.path.join(ROOT, "experiments", "e1_read.py"))
    E1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E1)
    texts = E1.texts90()
    plan = json.load(open(os.path.join(R, "e3_plan.json")))
    concept = {}                                         # prototype name -> (factor, key)
    for v in plan.values():
        if v["instance_key"]:
            concept[v["object_proto"]] = ("object", "I:" + v["instance_key"])
        else:
            concept[v["object_proto"]] = ("object", "O:" + v["object_key"])
        concept[v["type_proto"]] = ("object", "O:" + v["object_key"])
        concept[v["goal_proto"]] = ("goal", "G:" + v["goal_key"])
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    state, keyname = None, {}
    for n in concept:
        state, k, _ = adapt.load(bk, os.path.join(R, f"prim_{n}_0.pt"), state=state)
        keyname[n] = k
    out = {}
    for t, v in plan.items():
        eps = d90.episodes_of_task(int(t))
        if len(eps) <= DEMO:
            print(f"[w0] task {t}: no held-out demonstration", flush=True)
            continue
        ep = d90.load(eps[DEMO])
        clause = P.steps_of(texts[int(t)])[0]
        T = len(ep["actions"])
        tr = P.transport(ep["actions"], 0, T)
        if tr is None:
            continue
        phases = {"object": (0, tr[0]), "goal": tr}
        idx, _ = D.chunk_indices(T, bk.H)
        row = {}
        t0 = time.time()
        for factor, (lo, hi) in phases.items():
            obs = []
            for i in range(lo, hi):
                o = d90.observation(ep, i)
                o["prompt"] = clause
                obs.append(o)
            bank = adapt.PrefixBank(bk, obs)
            full = torch.as_tensor(np.asarray(D.pad_rows(D.to_rows(ep["actions"][idx[lo:hi]], norm), bk.H), np.float32), device=bk.device)
            gen = torch.Generator(device="cpu").manual_seed(0)
            draws = [adapt._flow_batch(hi - lo, bk.H, bk.A, gen, bk.device) for _ in range(SAMPLES)]
            names = [None] + [n for n, (f, _) in concept.items() if f == factor]
            losses = {}
            for n in names:
                on = keys_on(state, {keyname[n]} if n else set())
                tot = 0.0
                with torch.no_grad(), on():
                    for tt, eps_ in draws:
                        for b in range(0, hi - lo, 32):
                            j = np.arange(b, min(b + 32, hi - lo))
                            a = full[j]
                            xt = tt[j][:, None, None] * eps_[j] + (1 - tt[j][:, None, None]) * a
                            u = (eps_[j] - a).reshape(len(j), -1)
                            vv = bk.v_full(xt, tt[j], bank.take(j))
                            tot += float(((vv - u) ** 2).mean(1).sum())
                losses[n or "policy"] = tot / (SAMPLES * (hi - lo))
            row[factor] = dict(frames=[int(lo), int(hi)], policy=losses["policy"],
                               gain={n: losses["policy"] - l_ for n, l_ in losses.items() if n != "policy"})
            del bank
        truth = dict(object="I:" + v["instance_key"] if v["instance_key"] else "O:" + v["object_key"], goal="G:" + v["goal_key"],
                     type="O:" + v["object_key"])
        out[t] = dict(row, truth=truth, episode=int(eps[DEMO]))
        top = {f: max(row[f]["gain"], key=row[f]["gain"].get) for f in row}
        print(f"[w0] task {t} ({time.time() - t0:.0f}s): object top {top['object']} {concept[top['object']][1]} (truth {truth['object']}); "
              f"goal top {top['goal']} {concept[top['goal']][1]} (truth {truth['goal']})", flush=True)
    res = analyse(out, concept)
    json.dump(dict(rows=out, concept={n: list(c) for n, c in concept.items()}, summary=res),
              open(os.path.join(R, "w0_select.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("w0_select", bench="libero_90", tag="probe", seed=0, metrics=dict(summary=res, rows=out),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(samples=SAMPLES, demo=DEMO),
                            script="experiments/w0_select.py", command=" ".join(sys.argv)))


def analyse(out, concept):
    """Recognition: the top object prototype is the instruction's concept (an instance child when named; the type concept also
    counts as right at type level); the top goal prototype is the instruction's goal.  Novelty: per query, the gain under its own
    concept against the best gain under any other concept of the factor; AUC of 'own' against 'best other'."""
    res = {}
    for f in ("object", "goal"):
        hit, hit_type, own, other = 0, 0, [], []
        for t, r in out.items():
            g = r[f]["gain"]
            top = max(g, key=g.get)
            hit += concept[top][1] == r["truth"][f]
            if f == "object":
                hit_type += concept[top][1] in (r["truth"]["object"], r["truth"]["type"])
            mine = [x for n, x in g.items() if concept[n][1] == r["truth"][f]]
            rest = [x for n, x in g.items() if concept[n][1] != r["truth"][f] and not (f == "object" and concept[n][1] == r["truth"]["type"])]
            if mine and rest:
                own.append(max(mine))
                other.append(max(rest))
        own, other = np.array(own), np.array(other)
        auc = float(np.mean([[float(a > b) + 0.5 * float(a == b) for b in other] for a in own]))
        res[f] = dict(queries=len(out), top1=int(hit), top1_type_level=int(hit_type) if f == "object" else None,
                      own_gain_median=float(np.median(own)), best_other_gain_median=float(np.median(other)), novelty_auc=auc,
                      own_beats_other=int((own > other).sum()), pairs=len(own))
    return res


if __name__ == "__main__":
    main()
