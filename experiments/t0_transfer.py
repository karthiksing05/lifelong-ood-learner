"""Screen T0 (EXPLORATORY; docs/PLAN_WANG.md): transfer to a NEW object WITHOUT a demonstration, by Cobweb categorisation.

The streams save demonstrations mostly on tasks the policy already performs (ledger w1_stream: 24 of 27 demonstration-free keeps).
A new object has no concept yet.  Cobweb sorts a new instance under the concept whose features it best matches; here the instance's
only features are the instruction's words, read by the policy's own language model (the last-layer hidden state of the object
phrase, mean over its tokens).  Each stream task is tried with the 3 OTHER object concepts nearest to its object phrase (E3's
library type prototypes; none holds the task's demonstration), the approach only (the policy carries: fac@<object>.x), and the
learner's practice decides (stored states 0-9, 10 episodes, noise seed 13; zero-shot from e3_accept).  Wang et al.'s selection by
a query's likelihood failed (ledger w0_select); this selects by the instruction's features and keeps by practice.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/t0_transfer.py --plan" slurm/jobs/robot_gpu.sbatch
    ~/flash/miniconda3/envs/chunks/bin/python experiments/t0_transfer.py --launch
    ~/flash/miniconda3/envs/chunks/bin/python experiments/t0_transfer.py --read [--record]
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
PLAN = os.path.join(R, "t0_plan.json")
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
SEED, K, THETA = 13, 3, 8


def plan():
    import torch
    from openpi.models import tokenizer as _t
    from lifelong_ood.policy.backbone import PolicyBackbone
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    protos = {}                                                   # object key -> its type prototype (E3's library)
    donors = {}                                                   # object key -> the tasks whose demonstrations it holds
    for v in e3.values():
        protos[v["object_key"]] = v["type_proto"]
        donors[v["object_key"]] = set(v["obj_tasks"])
    bk = PolicyBackbone("pi05_libero", device="cuda")
    lm = bk.model.paligemma_with_expert.paligemma.language_model
    sp = _t.PaligemmaTokenizer(200)._tokenizer
    feats = {}
    for k in protos:
        phrase = k.split("|", 1)[1]                                # "the ketchup"
        ids = torch.tensor([sp.encode(phrase, add_bos=True)], device=bk.device)
        with torch.no_grad():
            emb = lm.get_input_embeddings()(ids)
            out = lm(inputs_embeds=emb, output_hidden_states=True)
        h = out.hidden_states[-1][0, 1:].float().mean(0).cpu().numpy()   # the phrase's tokens, without BOS
        feats[k] = h / np.linalg.norm(h)
    keys = sorted(protos)
    S = np.array([[float(feats[a] @ feats[b]) for b in keys] for a in keys])
    out = {}
    for t, v in e3.items():
        own = v["object_key"]
        cands = [(S[keys.index(own), keys.index(k)], k) for k in keys if k != own and int(t) not in donors[k]]
        top = [k for _, k in sorted(cands, reverse=True)[:K]]
        out[t] = dict(object_key=own, nearest=top, protos=[protos[k] for k in top],
                      sims=[round(float(S[keys.index(own), keys.index(k)]), 3) for k in top])
        print(f"[t0] {t} {own}: {list(zip(top, out[t]['sims']))}", flush=True)
    json.dump(dict(tasks=out, keys=keys, similarity=S.round(4).tolist()), open(PLAN, "w"), indent=1)


def arms(v):
    return [f"fac@{p}.x" for p in v["protos"]]


def tag(a):
    return a.replace("@", "_") + f"_seed{SEED}"


def launch():
    for t, v in json.load(open(PLAN))["tasks"].items():
        cmd = [PY, "experiments/launch_pack.py", "--tasks", t, "--arms", ",".join(arms(v)), "--experiment", "t0_transfer", "--first", "0",
               "--trials", "10", "--seed", str(SEED), "--bind", "track", "--partition", "overcap", "--per", "3"]
        print(t, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()[-120:], flush=True)


def succ(exp, t, a):
    f = os.path.join(R, f"{exp}_{t}_{tag(a)}.json")
    if not os.path.exists(f):
        return None
    return int(sum(bool(e["success"]) for v in json.load(open(f))["tasks"].values() for e in v["episodes"]))


def read(record):
    pl = json.load(open(PLAN))["tasks"]
    e3 = json.load(open(os.path.join(R, "e3_plan.json")))
    per, missing = {}, []
    for t, v in pl.items():
        tries = {p: succ("t0_transfer", t, a) for p, a in zip(v["protos"], arms(v))}
        if None in tries.values():
            missing.append(t)
            continue
        zs = succ("e3_accept", t, "zs")
        own = succ("e3_accept", t, f"fac@{e3[t]['object_proto']}.{e3[t]['goal_proto']}")
        best = max(tries, key=tries.get)
        per[t] = dict(zs=zs, tries=tries, best=best, best_acc=tries[best], first=tries[v["protos"][0]], own_library=own,
                      transfers=bool(tries[best] > zs and tries[best] >= THETA))
    n = len(per)
    res = dict(per_task=per, missing=missing, tasks=n,
               transfers=sum(p["transfers"] for p in per.values()),
               transfers_first=sum(p["first"] > p["zs"] and p["first"] >= THETA for p in per.values()),
               beats_policy=sum(p["best_acc"] > p["zs"] for p in per.values()),
               policy_below_theta=sum(p["zs"] < THETA for p in per.values()),
               transfers_where_policy_fails=sum(p["transfers"] for p in per.values() if p["zs"] < THETA))
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1), flush=True)
    for t, p in per.items():
        print(f"[t0] {t}: zs {p['zs']} tries {p['tries']} own library {p['own_library']}", flush=True)
    json.dump(res, open(os.path.join(R, "t0_read.json"), "w"), indent=1)
    if record and not missing:
        from lifelong_ood import ledger
        print(ledger.record("t0_transfer", bench="libero_90", tag="screen", seed=SEED, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(k=K, theta=THETA),
                            script="experiments/t0_transfer.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    if "--plan" in sys.argv:
        plan()
    if "--launch" in sys.argv:
        launch()
    if "--read" in sys.argv:
        read("--record" in sys.argv)
