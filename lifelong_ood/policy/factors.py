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


def prompt_of(name):
    """The instruction the policy was trained / evaluated with: LIBERO's grab_language_from_filename (the BDDL's own
    (:language ...) differs on LIBERO-Goal, e.g. "Open the middle layer of the drawer")."""
    if name[0].isupper():
        return " ".join(name[name.find("SCENE") + (8 if "SCENE10" in name else 7):].split("_"))
    return " ".join(name.split("_"))


def parse(suite, index, name):
    path = os.path.join(LIBERO, "bddl_files", suite, name + ".bddl")
    text = open(path).read()
    lang = prompt_of(name)
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


# ── object phrases (for factor prompts) ─────────────────────────────────────────────────────────────────────────────
PHRASE_FALLBACK = {"akita_black_bowl": "black bowl", "porcelain_mug": "white mug", "white_yellow_mug": "yellow and white mug",
                   "red_coffee_mug": "red mug", "chefmate_8_frypan": "frying pan", "black_book": "book", "yellow_book": "book",
                   "cream_cheese": "cream cheese box", "new_salad_dressing": "salad dressing", "wooden_tray": "tray",
                   "flat_stove": "stove", "wooden_cabinet": "cabinet", "white_cabinet": "cabinet", "desk_caddy": "caddy"}
_PATTERNS = (r"pick up the (.+?) and (?:put|place) (?:it|them)", r"^put the (.+?) (?:in|on|to|at|under|into)\b",
             r"^put both the (.+?) and the ", r"^stack the (.+?) on ")


def object_phrases(tasks):
    """type -> the phrase LIBERO's instructions use for it, read from single-object tasks (qualifiers dropped)."""
    out = {}
    for t in tasks:
        mv = sorted(set(t.moved))
        if len(mv) != 1:
            continue
        for pat in _PATTERNS:
            m = re.search(pat, t.language)
            if m:
                head = re.split(r"\b(?:on|in|from|next|between|at|to|under)\b", m.group(1))[0]     # "book on the left" -> "book"
                words = [w for w in head.split() if w not in QUALIFIERS and w not in ("the", "of")]
                phrase = " ".join(words)
                if phrase and mv[0] not in out:
                    out[mv[0]] = phrase
                break
    for k, v in PHRASE_FALLBACK.items():
        out.setdefault(k, v)
    return out


def phrase_of(typ, phrases):
    return phrases.get(typ) or typ.replace("_", " ")


def own_phrase(task, phrases):
    """The phrase the task's own instruction uses for its moved object (read with the same patterns; the global phrase as a
    fallback): LIBERO-Goal says "the bowl" where LIBERO-Spatial says "the black bowl"."""
    mv = sorted(set(task.moved))
    if len(mv) != 1:
        return None
    for pat in _PATTERNS:
        m = re.search(pat, task.language)
        if m:
            head = re.split(r"\b(?:on|in|from|next|between|at|to|under)\b", m.group(1))[0]
            words = [w for w in head.split() if w not in QUALIFIERS and w not in ("the", "of")]
            if words and " ".join(words) in task.language:
                return " ".join(words)
    g = phrase_of(mv[0], phrases)
    return g if g in task.language else None


def substitute(task, new_type, phrases):
    """The task's own instruction with its moved object's phrase replaced by `new_type`'s (None if it cannot be found)."""
    old = own_phrase(task, phrases)
    if old is None:
        return None
    if new_type in task.moved:
        return task.language
    return task.language.replace(old, phrase_of(new_type, phrases), 1)


def scene_phrases(task, tasks):
    """Phrases as the task's own scene says them (its other tasks' instructions first), then the global map."""
    local = object_phrases([t for t in tasks if t.scene == task.scene])
    glob_ = object_phrases(tasks)
    return {**glob_, **local}


def candidates(task):
    """Movable object types in the scene that the task's object slot could name (its own targets excluded)."""
    targets = {sl[2] for sl in task.slots if len(sl) > 2 and sl[2]}
    return sorted({ty for ty in task.objects.values() if ty not in targets})
