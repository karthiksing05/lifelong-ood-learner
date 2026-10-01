"""P5 (archive/2026-10-program/PLAN_TRANSFER.md, under rules v2): the minimal continual-learning stream over the 21 held-out tasks.

A binding table grows as tasks arrive: grasp bindings keyed by the manipulated object's TYPE, place bindings keyed by the goal
RELATION (predicate + receptacle region; experiments/p3c_table.py `relation`), each pointing at the demo that supplied it.  On
arrival a task (known: its scene's objects and its BDDL goal, i.e. the task specification) is COMPOSED from the table when both its
keys are present (lifelong_ood/policy/phase.py compose; no demo requested), else it requests its ONE demo, whose v2 cut adds its
keys (first come: a key is never overwritten, so a stored program never changes and nothing is forgotten).  Tasks whose demo moves
no object (fixture tasks) always request their demo.  Orders: A = random.Random(0) shuffle of the pool, B = A reversed.  Pool and
keys: experiments/p5_pool.py (results/robot/p5_pool.json).

    python experiments/p5_stream.py plan      # -> results/robot/p5_stream_plan.json (per order: decisions and sources)
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]


def orders(pool):
    a = sorted(int(t) for t in pool)
    random.Random(0).shuffle(a)
    return {"A": a, "B": a[::-1]}


def gate(min_rate=0.5):
    """Amendment 1 (ledger m3_prereg p5_stream_amend1): a demo's keys enter the table only if its v2 program succeeds on at least
    `min_rate` of its practice placements (init states 0-19) executed in simulation with no policy (experiments/p38_generate.py
    --rules v2: g27_generate.json, g28_generate.json).  -> {task: passes}"""
    ok = {}
    for f in ("g27_generate.json", "g28_generate.json"):
        for t, v in json.load(open(os.path.join(R, f))).items():
            ok[int(t)] = v["k"] / v["n"] >= min_rate
    return ok


def plan(pool_file="p5_pool.json", out_file="p5_stream_plan.json", gated=False):
    """Keys from experiments/p5_pool.py: grasp = the goal object's type, place = the whole goal with the object abstracted; a task
    supplies keys (and can be composed) only when its demo's v2 cut manipulates the goal's object relative to its receptacle."""
    pool = json.load(open(os.path.join(R, pool_file)))
    valid = gate() if gated else None
    out = {}
    for name, order in orders(pool).items():
        grasp, place, steps = {}, {}, []
        for t in order:
            info = pool[str(t)]
            ok = info["composable"]
            if ok and info["object_type"] in grasp and info["goal_key"] in place:
                steps.append(dict(task=t, action="compose", g=grasp[info["object_type"]], p=place[info["goal_key"]],
                                  object=info["object"], receptacle=info["receptacle"]))
                continue
            steps.append(dict(task=t, action="demo"))
            if ok and (valid is None or valid.get(t, False)):     # its demo supplies its keys (first come; never overwritten)
                grasp.setdefault(info["object_type"], t)
                place.setdefault(info["goal_key"], t)
        demos = sum(s["action"] == "demo" for s in steps)
        half = len(order) // 2
        out[name] = dict(order=order, steps=steps, demos=demos, composed=len(order) - demos,
                         composed_first_half=sum(s["action"] == "compose" for s in steps[:half]),
                         composed_second_half=sum(s["action"] == "compose" for s in steps[half:]),
                         cumulative_demos=[sum(s["action"] == "demo" for s in steps[:i + 1]) for i in range(len(steps))])
        print(f"order {name}: {demos} demos for {len(order)} tasks (composed {out[name]['composed_first_half']} in the first half, "
              f"{out[name]['composed_second_half']} in the second); composed: "
              + ", ".join(f"{s['task']}<-{s['g']}+{s['p']}" for s in steps if s["action"] == "compose"))
    json.dump(out, open(os.path.join(R, out_file), "w"), indent=1)
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["plan"]:
        plan(*sys.argv[2:4])
    elif sys.argv[1:2] == ["plan_gated"]:
        plan("p5_pool.json", "p5_stream_plan_gated.json", gated=True)
