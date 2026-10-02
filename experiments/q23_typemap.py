"""TYPE MAPS by consensus (exploratory, offline): the grammar of experiments/q22_grammar.py picks the instance a qualifier names from
the peaks of a prototype's cosine map, and fails online when a distractor is among the peaks (ledger q22_qual).  A type is what its
exemplars share: an instance of the type resembles EVERY stored prototype of it, a distractor only some.  Per qualifier task, from
the library's prototypes of the moved type built from OTHER tasks' demos (never the task's own):
  single  -- each donor's map alone (averaged over donors),
  mean    -- the map of the donors' mean prototype,
  min     -- the patchwise minimum of the donors' maps (an instance must match all of them).
Scored on the first frames of init states 0-9 with the simulator's segmentation: the grammar's pick among the map's n strongest
peaks (n = 3, 4) lies on the named object.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q23_typemap.py --record" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import bind_hooks as BH      # noqa: E402
from lifelong_ood.policy import factors as F          # noqa: E402
from lifelong_ood.policy import where as W            # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FR = os.path.join(R, "f0b_firstframes")


def peaks(S, n, sep=2):
    out = []
    for p in np.argsort(S.reshape(-1))[::-1]:
        ij = divmod(int(p), 16)
        if all(max(abs(ij[0] - a), abs(ij[1] - b)) > sep for a, b in out):
            out.append(ij)
        if len(out) == n:
            break
    return out


def pick(pk, word):
    axis, end = BH.GRAMMAR[word]
    order = sorted(pk, key=lambda ab: ab[0] if axis == "row" else ab[1])
    return order[0] if end == "first" else order[-1] if end == "last" else order[len(order) // 2]


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("q4", os.path.join(ROOT, "experiments", "q4_template.py"))
    q4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q4)
    gspec = importlib.util.spec_from_file_location("q22", os.path.join(ROOT, "experiments", "q22_grammar.py"))
    q22 = importlib.util.module_from_spec(gspec)
    gspec.loader.exec_module(q22)
    bk = PolicyBackbone("pi05_libero", device="cuda")
    P = np.load(os.path.join(R, "q4_object_protos.npz"), allow_pickle=True)
    lib = {int(t): str(ty) for t, ty in zip(P["tids"], P["types"])}
    tasks = sorted((x for x in F.load() if x.suite == "libero_90"), key=lambda x: x.index)
    per, agg = {}, {}
    for t in tasks:
        word = q22.qualifier(t.language, 0)
        mv = BH.referents(t)["open"]
        if word is None or not mv:
            continue
        named, ty = mv[0], t.objects.get(mv[0])
        if sum(v == ty for v in t.objects.values()) < 2:
            continue
        donors = [k for k in sorted(lib) if lib[k] == ty and k != t.index]
        f = os.path.join(FR, f"libero_90_{t.index}.npz")
        if not donors or not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        V = np.stack([P[str(k)] for k in donors])
        mean = V.mean(0)
        mean /= np.linalg.norm(mean)
        res = {}
        for k in range(len(z["states"])):
            fk = q4.feats(bk, z["img"][k], z["wimg"][k], z["state8"][k])
            truth = W.oracle_map(z["seg"][k], W.geoms_of(z, name=named)).reshape(16, 16) > 0.1
            maps = {"single": [(fk @ v).reshape(16, 16) for v in V], "mean": [(fk @ mean).reshape(16, 16)],
                    "min": [np.min(np.stack([(fk @ v).reshape(16, 16) for v in V]), 0)]}
            for name, ms in maps.items():
                for n in (3, 4):
                    hits = [int(truth[pick(peaks(S, n), word)]) for S in ms]
                    res.setdefault(f"{name}_n{n}", []).append(float(np.mean(hits)))
        per[str(t.index)] = dict(word=word, donors=donors, **{k: float(np.mean(v)) for k, v in res.items()})
        for k, v in res.items():
            agg.setdefault(k, []).append(float(np.mean(v)))
        print(f"[q23] {t.index:>3} {word:6s} donors {len(donors):>2} " + " ".join(f"{k} {np.mean(v):.2f}" for k, v in res.items()), flush=True)
    summ = {k: float(np.mean(v)) for k, v in agg.items()}
    print("[q23] mean over tasks", {k: round(v, 3) for k, v in summ.items()}, flush=True)
    out = dict(per_task=per, mean=summ, tasks=len(per))
    json.dump(out, open(os.path.join(R, "q23_typemap.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("q23_typemap", bench="libero_90", tag="offline", seed=0, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/q23_typemap.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
