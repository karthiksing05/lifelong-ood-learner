"""INTERPRETATION 3: what the motion primitives DO (exploratory, offline).  On demonstration frames, the frozen policy samples its
action chunk with a stream key's final primitive switched on and off under the SAME noise; the mean absolute difference of the
executed actions (the chunk's first 5 steps) is split by action dimension (x, y, z, rotation, gripper) and by the demonstration's
phase at that frame (approach: before the first grasp; carry: gripper closed; after: after the first release).  Frames come from
tasks of the primitive's own step key and of the other keys, so the table also shows whether a primitive acts only on its step.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/i3_behavior.py --record" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
R = os.path.join(ROOT, "results", "robot")
DIMS = ("x", "y", "z", "rot", "gripper")
STRIDE, EXEC = 5, 5


def phase_of(actions):
    g = np.asarray(actions)[:, 6] > 0
    first = int(np.argmax(g)) if g.any() else len(g)
    rel = next((i for i in range(first + 1, len(g)) if not g[i] and g[i - 1]), len(g))
    return np.array(["approach" if i < first else "carry" if i < rel else "after" for i in range(len(g))])


def main():
    import torch
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    st = json.load(open(os.path.join(R, "c2_state.json")))
    keys = {k: s for k, s in st.items() if not k.startswith("_")}
    bk = PolicyBackbone("pi05_libero", device="cuda")
    state, pkeys = None, {}
    for k, s in keys.items():
        state, key, _ = adapt.load(bk, os.path.join(R, f"prim_{s['prim']}_0.pt"), state=state)
        pkeys[k] = key
    d90 = D.Demos90()
    res = {}
    for fk, s in keys.items():                                    # frames from each key's first two library demonstrations
        for t in [x for x in s["library"] if d90.episodes_of_task(x)][:2]:
            ep = d90.load(d90.episodes_of_task(t)[0])
            ph = phase_of(ep["actions"])
            for i in range(0, len(ep["actions"]), STRIDE):
                obs = d90.observation(ep, i)
                o = bk.observation(obs)
                pre = bk.prefix(o)
                st8 = o.state[0].float().cpu().numpy()
                state["on"] = frozenset()
                with torch.no_grad():
                    off = bk.unnormalize(bk.sample(1, pre, seed=1000 + i, num_steps=10), state=st8)[0][:EXEC]
                for pk, key in pkeys.items():
                    state["on"] = frozenset({key})
                    with torch.no_grad():
                        on = bk.unnormalize(bk.sample(1, pre, seed=1000 + i, num_steps=10), state=st8)[0][:EXEC]
                    d = np.abs(np.asarray(on) - np.asarray(off)).mean(0)
                    row = [d[0], d[1], d[2], d[3:6].mean(), d[6]]
                    cell = res.setdefault(pk, {}).setdefault(fk, {}).setdefault(str(ph[i]), [])
                    cell.append(row)
                state["on"] = frozenset()
            print(f"[i3] frames of key {fk} task {t} done", flush=True)
    table = {pk: {fk: {p: dict(zip(DIMS, np.mean(v, 0).tolist()), n=len(v)) for p, v in d.items()} for fk, d in r.items()}
             for pk, r in res.items()}
    own_vs_other = {pk: dict(own=float(np.mean([np.mean(v) for p, v in res[pk][pk].items()])),
                             other=float(np.mean([np.mean(v) for fk, d in res[pk].items() if fk != pk for p, v in d.items()])))
                    for pk in res}
    out = dict(table=table, own_vs_other=own_vs_other, keys={k: s["key"] for k, s in keys.items()}, stride=STRIDE, executed=EXEC)
    print(json.dumps(own_vs_other, indent=1), flush=True)
    json.dump(out, open(os.path.join(R, "i3_behavior.json"), "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("i3_behavior", bench="libero_90", tag="read", seed=0, metrics=out,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/i3_behavior.py", command=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
