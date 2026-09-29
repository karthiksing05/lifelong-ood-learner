"""P5 (docs/PLAN_TRANSFER.md, under rules v2): the minimal continual-learning stream over the 21 held-out tasks.

A binding table grows as tasks arrive: grasp bindings keyed by the manipulated object's TYPE, place bindings keyed by the goal
RELATION (predicate + receptacle region; experiments/p3c_table.py `relation`), each pointing at the demo that supplied it.  On
arrival a task (known: its scene's objects and its BDDL goal, i.e. the task specification) is COMPOSED from the table when both its
keys are present (lifelong_ood/policy/phase.py compose; no demo requested), else it requests its ONE demo, whose v2 cut adds its
keys (first come: a key is never overwritten, so a stored program never changes and nothing is forgotten).  Tasks whose demo moves
no object (fixture tasks) always request their demo.  Orders: A = random.Random(0) shuffle, B = A reversed.

    python experiments/p5_stream.py plan      # -> results/robot/p5_stream_plan.json (per order: decisions and sources)
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]


def orders():
    a = list(HELD)
    random.Random(0).shuffle(a)
    return {"A": a, "B": a[::-1]}


def plan():
    src = json.load(open(os.path.join(R, "p3c_table.json")))["sources"]     # every task with a matched demo and an obj/rec cut
    out = {}
    for name, order in orders().items():
        grasp, place, steps = {}, {}, []
        for t in order:
            info = src.get(str(t))
            if info is not None and info["object"] in grasp and info["relation"] in place:
                g, p = grasp[info["object"]], place[info["relation"]]
                steps.append(dict(task=t, action="compose", g=g, p=p, object=info["object"], relation=info["relation"]))
                continue
            steps.append(dict(task=t, action="demo"))
            if info is not None:                                  # its demo supplies its keys (first come)
                grasp.setdefault(info["object"], t)
                place.setdefault(info["relation"], t)
        demos = sum(s["action"] == "demo" for s in steps)
        out[name] = dict(order=order, steps=steps, demos=demos, composed=len(order) - demos,
                         cumulative_demos=[sum(s["action"] == "demo" for s in steps[:i + 1]) for i in range(len(steps))])
        print(f"order {name}: {demos} demos for {len(order)} tasks; composed: "
              + ", ".join(f"{s['task']} (grasp {s['g']}, place {s['p']})" for s in steps if s["action"] == "compose"))
    json.dump(out, open(os.path.join(R, "p5_stream_plan.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["plan"]:
        plan()
