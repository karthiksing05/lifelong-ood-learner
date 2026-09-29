"""P5 pool (docs/PLAN_TRANSFER.md, under rules v2): the stream's tasks and their binding keys, from the task specification.

Per task (a LIBERO-90 task the prior fails zero-shot, 0/5; ledger zeroshot90): the BDDL goal's atoms; the target OBJECT = the first
argument of the first On/In atom whose first argument is movable (results/robot/p0_demo/<t>.npz movable flags); the RECEPTACLE = that
atom's second argument, or the object of interest whose name begins its region name, or whose type names the table region; the grasp key = the object's TYPE
(lifelong_ood/policy/phase.py type_of); the place key = the whole goal with the object written OBJ and instance numbers removed (a
compound goal such as "turn on the stove and put the pan on it" keys on both atoms).  Also whether the demo's v2 cut manipulates the
goal's object relative to its receptacle (a task composes and supplies bindings only then).

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/p5_pool.py
"""
import json
import os
import pathlib
import re
import sys

REPO = "/coc/flash10/ksingara3/lifelong-ood-learner"
sys.path.insert(0, REPO)
import numpy as np                                                          # noqa: E402
from lifelong_ood.policy import phase as PH                                # noqa: E402

R = os.path.join(REPO, "results", "robot")
POOL = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63,
        8, 11, 13, 16, 17, 18, 21, 23, 25, 26, 27, 65, 66, 69, 71, 73, 74, 75, 76, 78, 80, 81, 83, 84, 85, 86, 87, 88, 89]


def strip(name):
    return re.sub(r"_\d+(?=_|$)", "", name)


def main():
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    out = {}
    for t in POOL:
        task = suite.get_task(t)
        txt = (pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file).read_text()
        atoms = [a.split() for a in re.findall(r"\((\w+ [^()]*?)\)", txt[txt.index("(:goal"):])]
        if not os.path.exists(os.path.join(R, "p0_demo", f"{t}.npz")):
            print(f"{t:>2} (no demo poses yet)", flush=True)
            continue
        d = np.load(os.path.join(R, "p0_demo", f"{t}.npz"), allow_pickle=True)
        names = [str(n) for n in d["names"]]
        movable = {n: bool(m) for n, m in zip(names, d["movable"])}
        rel = next((a for a in atoms if a[0] in ("On", "In") and len(a) == 3 and movable.get(a[1])), None)
        obj = rel[1] if rel else None
        rec = None
        if rel:
            rec = rel[2] if rel[2] in names else next((n for n in sorted(names, key=len, reverse=True) if rel[2].startswith(n)), None)
            if rec is None:                     # a table region defined relative to another object (kitchen_table_plate_right_region)
                rec = next((n for n in names if n != obj and f"_{PH.type_of(n)}_" in f"_{rel[2]}_"), None)
        key = " & ".join(sorted(" ".join(strip(x) if x != obj else "OBJ" for x in a) for a in atoms)) if obj else None
        z = PH.at_rest(PH.load_demo(R, t))
        parts = PH.cut(z)
        cut_obj = names[parts[0][6]] if parts[0][5] == "obj" else None
        cut_rec = names[parts[1][6]] if len(parts) > 1 and parts[1][5] == "rec" and parts[1][6] is not None else None
        ok = obj is not None and rec is not None and cut_obj == obj and cut_rec == rec
        out[str(t)] = dict(language=task.language, object=obj, object_type=PH.type_of(obj) if obj else None, receptacle=rec,
                           goal_key=key, cut_object=cut_obj, cut_receptacle=cut_rec, composable=ok, atoms=[" ".join(a) for a in atoms])
        print(f"{t:>2} {'OK ' if ok else '-- '} {task.language[:60]:<60} | obj {obj} rec {rec} | cut {cut_obj} / {cut_rec} | {key}", flush=True)
    json.dump(out, open(os.path.join(R, "p5_pool.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
