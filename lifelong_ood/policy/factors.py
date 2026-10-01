"""The FACTORS of a LIBERO task, read by rule from its BDDL file (the overhaul's step 1: Read -> Compose -> Fold -> Consolidate).

A task's factors are its goal atoms with object instances replaced by their types, the instruction's qualifiers (left, right,
front, back, middle: which instance), and its scene.  Each goal atom gives a SLOT tuple:

    In(ketchup_1, wooden_tray_1_contain_region)          -> ("In", "ketchup", "wooden_tray", "contain")
    On(chocolate_pudding_1, living_room_table_plate_right_region) -> ("On", "chocolate_pudding", "plate", "right")
    Close(wooden_cabinet_1_top_region)                   -> ("Close", "wooden_cabinet", None, "top")
    Turnoff(flat_stove_1)                                -> ("Turnoff", "flat_stove", None, None)

so the OBJECT slot is the moved object's type and the RELATION slot is (predicate, target type, region).  Coverage is read
against pi0.5's 40 training tasks (LIBERO spatial / object / goal / 10): where a factor value is named, and where it is
CONTRASTED (two training instructions in one scene differ in that slot, so the policy had to read it there).  Nothing here
is tuned: the parse is fixed before any scoring and `parser_hash()` goes into every pre-registration that uses it.
"""
from __future__ import annotations

import hashlib
import os
import re
import runpy
from dataclasses import dataclass, field

LIBERO = "/coc/flash10/ksingara3/robot/openpi/third_party/libero/libero/libero"
TRAIN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
QUALIFIERS = ("left", "right", "front", "back", "middle")
DIRS = ("left", "right", "front", "back")
FAMILY = {"KITCHEN": "kitchen", "LIVING_ROOM": "living_room", "STUDY": "study"}


def task_map():
    """suite -> task names in LIBERO's benchmark order (the order of every task id in ledger/)."""
    return runpy.run_path(os.path.join(LIBERO, "benchmark", "libero_suite_task_map.py"))["libero_task_map"]


def _section(text, head):
    """The balanced-parenthesis body of `(:<head> ...)`."""
    i = text.find(f"(:{head}")
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(text):
        depth += text[j] == "("
        depth -= text[j] == ")"
        j += 1
        if depth == 0:
            break
    return text[i + len(head) + 2: j - 1]


def _typed(body):
    """'a_1 b_2 - type' lines -> {instance: type}."""
    out = {}
    for line in body.splitlines():
        if " - " in line:
            names, typ = line.rsplit(" - ", 1)
            for n in names.split():
                out[n.strip()] = typ.strip()
    return out


@dataclass
class Task:
    suite: str
    index: int
    name: str
    language: str
    scene: str
    family: str
    objects: dict
    fixtures: dict
    interest: list
    goal: list                      # [(predicate, arg, ...)]
    slots: list = field(default_factory=list)
    qualifiers: tuple = ()

    @property
    def moved(self):
        """Object types the goal moves (In / On atoms)."""
        return [s[1] for s in self.slots if s[0] in ("In", "On")]

    @property
    def relations(self):
        return [s for s in self.slots]

    @property
    def tid(self):
        return f"{self.suite}:{self.index}"


def _instances(task_objects):
    return sorted(task_objects, key=len, reverse=True)


def _target(arg, inst):
    """A goal argument -> (target type, region).  Instance, instance + region, or a table-relative region."""
    if arg in inst:
        return inst[arg], None
    for n in _instances(inst):
        if arg.startswith(n + "_"):
            rest = arg[len(n) + 1:]
            if inst[n].endswith("table") and rest.endswith("_region"):          # living_room_table_plate_right_region
                core = rest[: -len("_region")]
                parts = core.split("_")
                ref, d = (core[: -len(parts[-1]) - 1], parts[-1]) if parts[-1] in DIRS else (core, None)
                ref = next((t for t in sorted(set(inst.values())) if t == ref or t.endswith("_" + ref)), ref)   # stove -> flat_stove
                return ref, d
            return inst[n], (rest[: -len("_region")] if rest.endswith("_region") else rest)
    return arg, None


def parse(suite, index, name):
    path = os.path.join(LIBERO, "bddl_files", suite, name + ".bddl")
    text = open(path).read()
    lang = re.search(r"\(:language ([^)]*)\)", text).group(1).strip()
    objects = _typed(_section(text, "objects"))
    fixtures = _typed(_section(text, "fixtures"))
    interest = _section(text, "obj_of_interest").split()
    goal = [tuple(a.split()) for a in re.findall(r"\(([A-Z][A-Za-z]* [^()]*)\)", _section(text, "goal"))]
    m = re.match(r"((KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", name)
    if m:
        scene, family = m.group(1), FAMILY[m.group(2)]
    elif suite == "libero_object":                    # one layout per task: its scene is its object set
        scene, family = "libero_object:" + "+".join(sorted(set(objects.values()))), "object_floor"
    else:
        scene, family = suite, "kitchen"              # spatial and goal: one kitchen table each
    inst = dict(objects, **fixtures)
    slots = []
    for atom in goal:
        pred, *args = atom
        if pred in ("In", "On") and len(args) == 2:
            obj = inst.get(args[0], args[0])
            tgt, region = _target(args[1], inst)
            slots.append((pred, obj, tgt, region))
        elif len(args) == 1:
            tgt, region = _target(args[0], inst)
            slots.append((pred, tgt, None, region))
        else:
            slots.append((pred,) + tuple(args))
    quals = tuple(q for q in QUALIFIERS if re.search(rf"\b{q}\b", lang))
    return Task(suite, index, name, lang, scene, family, objects, fixtures, interest, goal, slots, quals)


def load(suites=TRAIN_SUITES + ("libero_90",)):
    tm = task_map()
    return [parse(s, i, n) for s in suites for i, n in enumerate(tm[s])]


def parser_hash():
    """Hash of this file: pre-registrations record which parse they rely on."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]


# ── coverage against the 40 training tasks ─────────────────────────────────────────────────────────────────────────
def coverage(task, train):
    """Where pi0.5's training tasks name / move / contrast this task's factors.  -> dict of booleans and counts."""
    same_scene = [t for t in train if t.scene == task.scene]
    same_fam = [t for t in train if t.family == task.family]
    moved = set(task.moved)
    rels = {s[:1] + s[2:] for s in task.slots}              # (pred, target, region): the relation without the object
    preds = {s[0] for s in task.slots}

    def has_obj(ts):
        return bool(moved) and all(any(o in t.moved for t in ts) for o in moved)

    def has_rel(ts):
        return all(any(r == s[:1] + s[2:] for t in ts for s in t.slots) for r in rels)

    def contrast_obj():
        """Training scenes where one task moves this object and another task of the same scene moves a different one."""
        k = 0
        for sc in {t.scene for t in train}:
            ts = [t for t in train if t.scene == sc]
            if any(moved & set(t.moved) for t in ts) and any(set(t.moved) - moved for t in ts):
                k += 1
        return k

    return dict(scene_seen=bool(same_scene), obj_same_scene=has_obj(same_scene), obj_same_family=has_obj(same_fam),
                obj_any=has_obj(train), rel_same_scene=has_rel(same_scene), rel_same_family=has_rel(same_fam),
                rel_any=has_rel(train), pred_any=all(any(p == s[0] for t in train for s in t.slots) for p in preds),
                contrast_obj=contrast_obj() if moved else 0,
                quals_seen=all(any(q in t.qualifiers for t in same_fam) for q in task.qualifiers))
