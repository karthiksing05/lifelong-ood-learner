"""P3c (cross-task composition under rules v2, through the prior): which held-out tasks can be composed from OTHER tasks' phases,
and from which sources, by a fixed rule.

Keys (the plan's binding table): a grasp binding is keyed by the manipulated object's TYPE (lifelong_ood/policy/phase.py
`type_of`); a place binding by the goal RELATION = the goal predicate on the target object and the receptacle region with the
instance suffix removed (from the task's BDDL goal: In ketchup_1 basket_1_contain_region -> In basket_contain_region).  Sources:
tasks with a matched demo (results/robot/p0_demo) whose v2 cut has an approach and a carry relative to a receptacle.  Rule: grasp
source = the lowest-numbered other task with the target's object type; place source = the lowest-numbered other task with the
target's relation, other than the grasp source when there is a choice; sibling (control) = the place source's whole demo.

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/p3c_table.py
"""
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, REPO)
from lifelong_ood.policy import phase as PH                                # noqa: E402

R = os.path.join(REPO, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]


def relation(t, obj_name):
    from libero.libero import benchmark, get_libero_path
    task = benchmark.get_benchmark_dict()["libero_90"]().get_task(t)
    txt = (pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).read_text()
    goal = txt[txt.index("(:goal"):]
    m = re.search(r"\((In|On) " + re.escape(obj_name) + r" (\S+?)\)", goal)
    if not m:
        return None
    region = re.sub(r"_\d+_", "_", m.group(2))
    return f"{m.group(1)} {region}", task.language


def main():
    info = {}
    for f in sorted(glob.glob(f"{R}/p0_demo/*_match.npz")):
        t = int(os.path.basename(f).split("_")[0])
        try:
            z = PH.at_rest(PH.load_demo(R, t))
        except KeyError:
            continue
        parts = PH.cut(z)
        if parts[0][5] != "obj" or len(parts) < 2 or parts[1][5] != "rec":
            continue
        names = [str(n) for n in z["names"]]
        rel = relation(t, names[parts[0][6]])
        if rel is None:
            continue
        info[t] = dict(object=PH.type_of(names[parts[0][6]]), relation=rel[0], language=rel[1])
    table = {}
    for t in HELD:
        if t not in info:
            continue
        g = sorted(s for s in info if s != t and info[s]["object"] == info[t]["object"])
        p = sorted(s for s in info if s != t and info[s]["relation"] == info[t]["relation"])
        if not g or not p:
            continue
        gs = g[0]
        ps = next((s for s in p if s != gs), p[0])
        table[str(t)] = dict(g=gs, p=ps, sibling=ps, **info[t], grasp_source=info[gs]["language"], place_source=info[ps]["language"])
        print(f"{t} ({info[t]['language']}): grasp from {gs} ({info[gs]['language']}); place from {ps} ({info[ps]['language']})")
    json.dump(dict(sources=info, table=table), open(os.path.join(R, "p3c_table.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
