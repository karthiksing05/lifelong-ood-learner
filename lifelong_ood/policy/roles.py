"""ROLE codes: a new task = a task prototype the policy performs in the same scene + a role map (plan of 2026-10-01 evening,
Bind; the observation-side counterpart of attention binding).

Pi0.5 runs a familiar scene's trained routine whatever object the instruction names (ledger f1_binding).  A task whose
relation the scene's routine already performs differs from a performed task only in WHICH object fills the moved-object
role.  The role code keeps the performed task (its instruction: the task prototype) and rebinds the role: the policy is shown
the performed task's object where the new task's object is, and the new task's object where the performed one was
(experiments/libero_client.py --swap; ORACLE masks from the simulator, a detector's in a deployable form).  The policy
generates every action; the code is the base task id and the instance pair, a few bytes:

    code = role_code(task, tasks, zeroshot, binding)   # {"base": 46, "prompt": "pick up the alphabet soup ...",
                                                         #  "pairs": [("ketchup_1", "alphabet_soup_1")]}
Base rule (fixed before any role episode): a task of the same scene whose goal equals the new task's goal with the moved
instance replaced (same target instance, every other atom identical), with one moved object, that the policy performs (a training task, or LIBERO-90 zero-shot success
>= 0.8); among those, a recorded zero-shot binding accuracy of 1.0 first (the success is the routine acting on the named
object, not a coincidence), then training tasks, then the lowest index.
"""
from __future__ import annotations

import numpy as np

from .bind_hooks import referents


def relation(task):
    return sorted(((s[0],) + tuple("" if x is None else str(x) for x in s[2:])) for s in task.slots)


def performed(t, zeroshot):
    return t.suite != "libero_90" or (zeroshot.get(t.index) or 0.0) >= 0.8


def _moved(task):
    return [a[1] for a in task.goal if a[0] in ("In", "On") and len(a) == 3]


def role_code(task, tasks, zeroshot, binding=None, exclude=()):
    """-> {"base": "<suite>:<index>", "prompt": ..., "pairs": [(new instance, base instance)]} or None.  The base's goal must
    equal the new task's goal with the moved instance replaced (every other atom, e.g. Turnon(stove), identical), so a
    trained SEQUENCE in the scene composes too: 21 'turn on the stove and put the frying pan on it' = libero_10:2 (the moka
    pot) with the frying pan in the moka pot's role."""
    binding = binding or {}
    mv = _moved(task)
    if len(set(mv)) != 1:
        return None
    mine = mv[0]
    goal = {tuple(a) for a in task.goal}
    cands = []
    for x in tasks:
        if x is task or x.scene != task.scene or x.tid in exclude or not performed(x, zeroshot):
            continue
        xm = _moved(x)
        if len(set(xm)) != 1 or xm[0] == mine or xm[0] not in task.objects:
            continue
        theirs = xm[0]
        sub = {tuple(theirs if (i == 1 and a[0] in ("In", "On")) else v for i, v in enumerate(a)) for a in goal}
        if sub != {tuple(a) for a in x.goal}:
            continue                                     # same target instance and every other atom identical
        ba = binding.get(f"{x.index}:zeroshot", {}).get("BA") if x.suite == "libero_90" else None
        cands.append(((0 if ba == 1.0 else 1 if x.suite != "libero_90" else 2), x.index, x, theirs))
    if not cands:
        return None
    _, _, base, inst = sorted(cands, key=lambda c: (c[0], c[1]))[0]
    return {"base": base.tid, "prompt": base.language, "pairs": [(mine, inst)]}


# coarse SHAPE classes of LIBERO's movable objects (for the shape-different steerability control only)
SHAPE = {"alphabet_soup": "can", "tomato_sauce": "can", "ketchup": "bottle", "bbq_sauce": "bottle", "salad_dressing": "bottle",
         "new_salad_dressing": "bottle", "milk": "carton", "orange_juice": "carton", "cream_cheese": "box", "butter": "box",
         "chocolate_pudding": "box", "akita_black_bowl": "bowl", "white_bowl": "bowl", "plate": "plate",
         "porcelain_mug": "mug", "red_coffee_mug": "mug", "white_yellow_mug": "mug", "moka_pot": "pot",
         "chefmate_8_frypan": "pan", "wine_bottle": "bottle", "black_book": "book", "yellow_book": "book"}


def placebo_code(task, code, seed=0, shape_different=False):
    """Steerability control: the base's object swapped with a movable object the task does NOT name (the policy should move
    whatever wears the base object's appearance).  `shape_different`: only objects whose coarse shape differs from the base's
    (a can's texture on another can tests texture, not identity)."""
    if code is None:
        return None
    named = {code["pairs"][0][0], code["pairs"][0][1]}
    tgt = set(referents(task)["closed"])
    others = sorted(n for n in task.objects if n not in named and n not in tgt)
    if shape_different:
        bshape = SHAPE.get(task.objects.get(code["pairs"][0][1]))
        others = [n for n in others if SHAPE.get(task.objects.get(n)) not in (None, bshape)]
    if not others:
        return None
    other = others[int(np.random.default_rng(seed).integers(len(others)))]
    return {"base": code["base"], "prompt": code["prompt"], "pairs": [(other, code["pairs"][0][1])], "placebo_object": other}


def code_bytes(code):
    """The stored code: the base id and the instance pair as text."""
    return len((code["base"] + "|" + "|".join(a + ":" + b for a, b in code["pairs"])).encode())
