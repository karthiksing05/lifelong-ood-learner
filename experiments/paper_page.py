"""The results site as a paper (published as the artifact https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk): the overhaul of
2026-10-01, Read -> Compose -> Fold -> Consolidate, with marked places for the reads that are still running.  Every number is
read from ledger/ (the rows the README quotes); every clip was recorded by the evaluation client (experiments/site_media.py,
stage `overhaul`).  Replaces experiments/site_page.py (the program-era site) as the writer of results/robot/site/index.html.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/site_media.py overhaul posters
    ~/flash/miniconda3/envs/chunks/bin/python experiments/paper_page.py
"""
import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger as L                                       # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SITE = os.path.join(R, "site")
MEDIA = json.load(open(os.path.join(SITE, "media", "media.json")))
esc = lambda s: html.escape(str(s))                                        # noqa: E731


def row(exp, tag):
    rows = [r for r in L.load(exp, tag=tag) if not r.get("retired")]
    return rows[-1]["metrics"] if rows else None


cov = row("f0_coverage", "read")
bind = row("f1_binding", "f1_bind_libero_90")["rows"]
bind_obj = row("f1_binding", "f1_bind_libero_object")["rows"]
probe = row("f1_factor_probe", "read")["groups"]
pilot = row("zc_pilot", "read")
zmain = row("p1_bindings", "zc_main")
zfold = row("p1_bindings", "zc_fold")
tokpilot = row("f1_binding", "f4_pilot_libero_90")["rows"]
capacity = row("f4_fold", "capacity_pilot")
library = json.load(open(os.path.join(R, "task_library.json")))
nopol = row("p1_bindings", "nopolicy")["N"]
override = row("p0_audit", "guide_override")["pi05_libero"]
sota = row("p1_bindings", "sota")
pi0 = row("p1_bindings", "pi0")
amend = row("p1_bindings", "sota_amend1")
q2m, q2ma = row("b2_q2", "b_q2mini")["rows"], row("b2_q2", "b_q2mini")["arms"]
q2f, q2fa = row("b2_q2", "b_q2full")["rows"], row("b2_q2", "b_q2full")["arms"]
w1g, w3g = row("b1_where", "read_all")["groups"], row("b1_where", "read_w3center")["groups"]
steer = row("b2_steer", "offline_class_a")["groups"]["class_a"]
scheme = row("b5_scheme", "main")
roletok = [r for r in L.load("f4_fold", tag="token_rollouts_role") if not r.get("retired")][-1]["bytes"]["total"]
q0 = row("q0_share", "read")
q1 = row("q1_options", "read")
q2x = row("q2_explain", "read")
q3p = row("q3_pilot", "read")
q4t = row("q4_template", "read")
q5 = row("b2_q2", "q5_w4")
q6 = row("q6_main", "read")
q7 = row("q7_tokpractice", "read")
q13 = row("q13_main", "read")
q16 = row("q16_library", "read")
q27 = row("q27_joint", "read")
f1 = row("f1_fresh", "read")
f1c = row("f1_fresh", "costs")
e3 = row("e3_hier", "read")
of10 = lambda key, f="BA", src=None: int(round(10 * (src or q2m)[key][f]))                     # noqa: E731
token_bytes = [r for r in L.load("f4_fold", tag="token_demo") if not r.get("retired")][-1]["bytes"]["total"]
lang = {int(k): v["language"] for k, v in json.load(open(os.path.join(R, "f0_coverage.json")))["tasks"].items()}   # the read's artifact
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
pretty = lambda typ: typ.replace("_", " ").replace("cream cheese", "cream cheese")                # noqa: E731
OBJ_OWN = {0: "alphabet_soup", 1: "cream_cheese", 2: "salad_dressing", 3: "bbq_sauce", 4: "ketchup"}
swap_ba = [v["BA"] for v in bind_obj.values() if v["named"] != OBJ_OWN.get(v["task"])]
FIG = {"n": 0}


def fig(body, caption, wide=False):
    FIG["n"] += 1
    return (f'<figure class="{"wide" if wide else ""}" id="fig{FIG["n"]}">{body}<figcaption><b>Figure {FIG["n"]}.</b> '
            f'{caption}</figcaption></figure>')


TAB = {"n": 0}


def table(head, rows, caption, num_from=1, cls=""):
    TAB["n"] += 1
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{esc(x)}</th>' for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>"
                for r in rows)
    return (f'<div class="tbl {cls}"><p class="tcap"><b>Table {TAB["n"]}.</b> {caption}</p><div class="scroll"><table>'
            f'<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div></div>')


def video(name):
    m = MEDIA["overhaul"].get(name) or MEDIA.get("bind", {})[name]
    return (f'<video src="media/{m["file"]}" poster="media/{m["file"][:-4]}.jpg" controls playsinline muted preload="metadata">'
            f'</video>')


def pending(title, text):
    return f'<aside class="pending"><p class="pl">Running</p><p class="pt">{esc(title)}</p><p>{text}</p></aside>'


def pct(x):
    return f"{100 * x:.0f}%"


# ── charts ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def coverage_chart():
    """Zero-shot success with and without each coverage feature (to scale, 0 to 0.6)."""
    F = cov["features"]
    items = [("every factor named in this scene family", F["rule_family_named"]),
             ("the object contrasted with another in one training scene", F["contrast_obj"]),
             ("the scene trained at all", F["scene_seen"])]
    W, top, bh, gap, left, right = 640, 14, 16, 30, 0, 60
    scale = (W - right) / 0.6
    out = [f'<svg viewBox="0 0 {W} {top + len(items) * (2 * bh + gap) + 24}" class="chart" role="img" '
           f'aria-label="zero-shot success with and without each feature">']
    y = top
    for lab, v in items:
        out.append(f'<text x="0" y="{y - 3}" class="lab">{esc(lab)} <tspan class="dim">(AUC {v["auc"]:.2f})</tspan></text>')
        for k, (key, cls, n) in enumerate((("success_on", "on", v["n_on"]), ("success_off", "off", v.get("n_off") or 90 - v["n_on"]))):
            val = v[key] or 0.0
            yy = y + k * (bh + 2)
            out.append(f'<rect x="0" y="{yy}" width="{val * scale:.1f}" height="{bh}" class="bar {cls}"/>')
            out.append(f'<text x="{val * scale + 6:.1f}" y="{yy + bh - 4}" class="val">{val:.2f} <tspan class="dim">'
                       f'{"with" if cls == "on" else "without"}, {int(n)} tasks</tspan></text>')
        y += 2 * bh + gap
    for t in (0, 0.2, 0.4, 0.6):
        out.append(f'<text x="{t * scale:.1f}" y="{y + 2}" class="tick" text-anchor="{"start" if t == 0 else "middle"}">{t:.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def methods_chart(rows):
    """Success rate per method on the stream's tasks (fresh start states), to scale 0-1; each bar labelled with its successes,
    the demonstrations it used and the bytes it stores.  rows: (label, successes, episodes, demos, bytes, ours)."""
    W, top, bh, gap, right = 640, 30, 18, 30, 150
    scale = (W - right)
    h = top + len(rows) * (bh + gap) + 20
    out = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img" aria-label="success rate per method with demonstrations and bytes">',
           f'<rect x="0" y="4" width="12" height="10" class="bar ours"/><text x="18" y="13" class="tick">this scheme</text>',
           f'<rect x="110" y="4" width="12" height="10" class="bar base"/><text x="128" y="13" class="tick">baselines</text>']
    y = top
    for lab, k, n, demos, nb, ours in rows:
        r = k / n if n else 0.0
        out.append(f'<text x="0" y="{y - 4}" class="lab">{esc(lab)} <tspan class="dim">({demos} demonstrations, '
                   f'{nb / 1e6:.1f} MB)</tspan></text>')
        out.append(f'<rect x="0" y="{y}" width="{max(r * scale, 2):.1f}" height="{bh}" rx="2" class="bar {"ours" if ours else "base"}">'
                   f'<title>{esc(lab)}: {k} of {n}</title></rect>')
        out.append(f'<text x="{r * scale + 6:.1f}" y="{y + bh - 4}" class="val">{r:.2f} <tspan class="dim">{k} of {n}</tspan></text>')
        y += bh + gap
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{t * scale:.1f}" y="{y + 2}" class="tick" text-anchor="{"start" if t == 0 else "middle"}">{t:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def binding_grid():
    """Living rooms 1-3: which object the policy picked up first, against the object the instruction named."""
    tasks = [t for t in range(46, 60)]
    cols = ["alphabet_soup", "cream_cheese", "butter", "tomato_sauce", "ketchup", "milk", "orange_juice", "None"]
    rows = []
    scene = {46: "1", 47: "1", 48: "1", 49: "1", 50: "2", 51: "2", 52: "2", 53: "2", 54: "2", 55: "3", 56: "3", 57: "3", 58: "3", 59: "3"}
    for t in tasks:
        v = bind[f"{t}:zeroshot"]
        cells = []
        for c in cols:
            n = v["bound_counts"].get(c, 0)
            cls = "hit" if c == v["named"] and n else ("habit" if n >= 5 else ("some" if n else ""))
            cells.append(f'<td class="g {cls}">{n if n else ""}</td>')
        rows.append(f'<tr><td class="rm">{scene[t]}</td><td class="nm">{esc(pretty(v["named"]))}</td>{"".join(cells)}'
                    f'<td class="n">{pct(v["success"])}</td></tr>')
    head = "".join(f'<th class="g">{esc(pretty(c) if c != "None" else "nothing")}</th>' for c in cols)
    return (f'<div class="scroll"><table class="grid"><thead><tr><th>room</th><th>named</th>{head}<th class="n">success</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def dot_rows(rows, arms, xmax, label_w=250, unit="of 20"):
    """rows = [(label, {arm: value}, highlight)], arms = [(key, name, css)] -> a dot plot to scale (0..xmax)."""
    W, rh, top = 720, 22, 26
    plot = W - label_w - 20
    sx = lambda v: label_w + v / xmax * plot                                 # noqa: E731
    H = top + len(rows) * rh + 22
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="per task, successes of each arm">']
    for i, (key, name, css) in enumerate(arms):
        out.append(f'<circle cx="{label_w + i * 150 + 6}" cy="10" r="5" class="dot {css}"/><text x="{label_w + i * 150 + 16}" y="14" '
                   f'class="tick">{esc(name)}</text>')
    for t in range(0, xmax + 1, 5):
        out.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 4}" y2="{H - 18}" class="gridl"/>'
                   f'<text x="{sx(t):.1f}" y="{H - 4}" class="tick" text-anchor="middle">{t}</text>')
    for j, (lab, vals, hl) in enumerate(rows):
        y = top + j * rh + rh / 2
        out.append(f'<text x="{label_w - 8}" y="{y + 4}" class="rl{" hl" if hl else ""}" text-anchor="end">{esc(lab)}</text>')
        xs = [sx(vals[k]) for k, _, _ in arms if k in vals]
        if xs:
            out.append(f'<line x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{y}" y2="{y}" class="span"/>')
        for k, _, css in arms:
            if k in vals:
                out.append(f'<circle cx="{sx(vals[k]):.1f}" cy="{y}" r="5" class="dot {css}"><title>{esc(lab)}: {vals[k]} {unit}</title></circle>')
    out.append(f'<text x="{label_w + plot}" y="{H - 4}" class="tick" text-anchor="end">successes {unit}</text>')
    out.append("</svg>")
    return "".join(out)


def short(t, n=34):
    s = lang.get(int(t), str(t))
    s = s.replace("pick up the ", "").replace("put the ", "").replace(" and put it", " →").replace(" of the cabinet", "")
    return (s[: n - 1] + "…") if len(s) > n else s


def composition_chart():
    pt = zmain["per_task"]
    rows = [(f"{t} · {short(t)}", {"z": v["z"], "c": v["c"], "p": v["p"], "f": v["f"]}, t == "48")
            for t, v in sorted(pt.items(), key=lambda kv: -kv[1]["f"])]
    return dot_rows(rows, [("z", "zero-shot", "z"), ("c", "guidance on the instruction", "c"), ("p", "placebo pair", "p"),
                           ("f", "composition", "f")], 20)


def token_chart():
    pt = zfold["per_task"]
    unc = {t: any(u[0] == "relation" for u in v["uncovered"]) for t, v in library["held_out"].items()}
    rows = [(f"{t} · {short(t)}{'  ◦' if unc.get(t) else ''}", {"tok": v["tok"], "lora": v["lora"]}, not unc.get(t))
            for t, v in sorted(pt.items(), key=lambda kv: (unc.get(kv[0], False), kv[1]["tok"] - kv[1]["lora"]))]
    return dot_rows(rows, [("tok", "factor token, 18.4 KB", "f"), ("lora", "one-demo LoRA, 18.7 MB", "c")], 20, label_w=270)


def fmt_factor(u):
    kind, val = u
    if isinstance(val, list):
        val = " ".join(str(x).replace("_", " ") for x in val if x)
    return f"{kind}: {str(val).replace('_', ' ').lower()}"


# ── content ──────────────────────────────────────────────────────────────────────────────────────────────────────────
H1, H1b, H1c = zmain["H1"], zmain["H1b"], zmain["H1c"]
H2a, H2b, Z = zfold["H2a"], zfold["H2b"], zfold["Z"]
F = cov["features"]
lr1 = [bind[f"{t}:zeroshot"]["bound_counts"].get("alphabet_soup", 0) for t in (46, 47, 48, 49)]
ex58 = library["held_out"]["58"]
ex3 = library["held_out"]["3"]
cap = capacity["totals"]

ABSTRACT = f"""
<p>A vision-language-action policy fine-tuned on forty tasks fails almost every new task in the same rooms. We ask whether such a
policy can take on tasks it has never seen, given only the task's instruction and scene and no demonstration, by recombining what
it already holds. The policy generates every action, a few bytes are stored per task, and nothing it learned changes. In a familiar
room pi0.5 carries out the room's trained routine whatever object the instruction names, so a new task usually differs from a task
it performs only in which object fills a role. We code such a task as that <b>task prototype plus a role map</b>, about 40 bytes
("task 50 with the milk in the alphabet soup's role"), and supply the binding the policy cannot read from the word, either by
giving the new object the prototype's object's appearance or by raising the action expert's attention on it. Where the scene's
training contrasted object shapes, the role swap rebinds the routine (milk and orange juice picked up first in {of10('52:role')} and
{of10('53:role')} of 10 episodes, against {of10('52:zs')} and {of10('53:zs')}); where the routine is tied to a position, only the
attention pointer redirects it. The policy's own successes, recorded with the real scene, then fold into an 18.4 KB key-value token
with every weight frozen, after which the task's own instruction suffices. In the pre-registered read on start states nobody tuned
on, practice found a form for 4 of the 14 tasks with a role code; there the tokens alone succeed
{scheme['hypotheses']['H1']['x']} of {scheme['hypotheses']['H1']['n']} times against {scheme['hypotheses']['H1']['control']} for the frozen
policy, with no demonstration and no simulator state at test. The tokens do not transfer to another room, and 10 of the 14 tasks
stay out of reach. Allowing one demonstration, used once as a query and never executed, its first grasp marks a 4 KB object
prototype in the policy's own image features, which points the attention with no simulator state at any step. A learner that keeps
whichever of the policy alone, this binding and its folded token its own fresh practice validates then succeeds
{q13['hypotheses']['H1']['x']} of {q13['hypotheses']['H1']['n']} times against {q13['hypotheses']['H1']['control']} on nine tasks in a
pre-registered read, storing {q13['kept_bytes'] / 1024:.0f} KB in all and leaving the policy unchanged where neither unit helps.
Once an object type has a prototype, later tasks with that type need no demonstration: practice chooses among the stored
prototypes, and the kept arm succeeds {q16['hypotheses']['H1']['x']} of {q16['hypotheses']['H1']['n']} times against
{q16['hypotheses']['H1']['control']} zero-shot, level with forms built from each task's own demonstration
({q16['hypotheses']['H2']['control']}). Tokens shared across tasks bind the object more often but succeed less often. Where a
step needs a motion the policy lacks, the demonstration may teach it as a reusable step: a primitive fitted on every other task
that shares the step, kept only where the learner's practice favours it, raises {len(q27['per_task'])} tasks from
{q27['hypotheses']['H1']['control']} to {q27['hypotheses']['H1']['x']} successes of {q27['hypotheses']['H1']['n']} in a
pre-registered read with no demonstration of the scored task, including a goal no binding reached. In a stream of 30 tasks
on never-used start states, these consolidated primitives beat joint, sequential and replay training of shared adapters and
reach {100 * f1['descriptive']['c2_vs_lora']['x'] / f1['descriptive']['c2_vs_lora']['y']:.0f}% of a separate one-demo adapter per task
at a {f1c['lora'] / f1c['scheme']:.0f}th of its storage; a separate adapter per task with the same acceptance test stays
stronger. Factoring each step at the policy's own grasp into an OBJECT prototype and a GOAL prototype, each consolidated over every
demonstration that shares it, with a child prototype when the instruction names an instance, closes most of that gap: on a second
set of never-used start states the kept library succeeds {e3['hypotheses']['H1']['x']} of {e3['hypotheses']['H1']['n']} times
against {e3['hypotheses']['H1']['y']} zero-shot, and {e3['hypotheses']['H2']['x']} of {e3['hypotheses']['H2']['n']} against
{e3['hypotheses']['H2']['y']} for per-task adapters with the same acceptance test (difference {e3['hypotheses']['H2']['diff']['diff']:+.3f},
95% interval {e3['hypotheses']['H2']['diff']['lo']:+.3f} to {e3['hypotheses']['H2']['diff']['hi']:+.3f}), storing
{e3['bytes']['prototypes']} prototypes in place of {e3['bytes']['loras']} adapters; non-inferiority at the pre-registered margin of
0.05 is not established.</p>"""

P_INTRO = f"""
<p>A trained robot policy already knows how to reach, grasp, carry and place. When it fails a new task in a familiar room, the
missing piece is often small: which object, which receptacle, which side. The usual repair collects demonstrations of each new task and trains
weights on them, a LoRA of 18.7 MB per task in our setting; a new task then needs a person to show it first. This report pursues a different repair,
stated in <a href="#thesis">the thesis</a>: decompose a new task into what the policy already holds, store a code when that
explains it, fold a small unit into the policy only where it does not, and keep generating every action from the policy itself.</p>
<p>Two papers shape the approach. Wang et al. discover the prototypes a frozen diffusion model holds by mode-finding and compose them
for unseen combinations. The user's ICLR paper on caption coverage shows that a diffusion model can know each object and still fail
a prompt that names several, because its captions rarely named them together; summing its single-object predictions restores
the composition. We carry both ideas to a flow-matching robot policy, pi0.5 on LIBERO-90.</p>
<div class="callout" id="thesis"><p><b>Thesis.</b> A new task is a recombination of factors the policy already reads somewhere.
Read the factors; compose the policy's own predictions when they explain the task; fold the smallest unit that makes a missing
factor readable; consolidate. Cost then grows with the vocabulary of factors, not with the number of tasks, and the policy is never
overwritten.</p></div>
<p class="contrib"><b>What this report contributes.</b> (1) A diagnosis of why the policy fails new tasks: reading, not motor skill,
for most of them (Section 2). (2) A benchmark that tests the thesis: tasks the policy has never seen, with no demonstration of them.
(3) Role codes: a new task as a task prototype the policy performs with one object slot rebound, and two ways to inject the
binding with the policy generating (Sections 3.3, 4.3). (4) Folding the policy's own successes into a token, with no demonstration
(Section 4.4), and the scheme's pre-registered read on start states nobody tuned on (Section 4.5). (5) An acceptance test on the learner's own practice that keeps the policy, the binding or the token per task; in its pre-registered read no task does worse than the frozen policy (Section 4.5). (6) A library of 4 KB object prototypes that lets later tasks with a known object type skip the demonstration (Sections 4.5, 4.7). (7) Motion primitives per step, consolidated over the arrivals that share the step, which reach tasks binding cannot (Section 4.8).</p>"""

P_WHY = f"""
<h3 id="s2-1">2.1 Success follows what the training instructions named</h3>
<p>We parse every task's goal into factors by rule: the moved object, the goal relation, the instruction's qualifiers (left, front,
back) and the scene. Against pi0.5's forty training instructions, a simple rule separates its zero-shot successes on all 90 LIBERO-90
tasks: every factor named in this scene family (Figure {FIG['n'] + 1}). The strongest single predictor is whether training contrasted the
object with another one in the same scene, so that the word had to be read.</p>
{fig('<div class="chartwrap">' + coverage_chart() + '</div>', f"Zero-shot success of pi0.5 on the 90 LIBERO-90 tasks (5 episodes each), with and without each feature. "
     f"AUC: how well the feature alone ranks successes above failures. The {cov['n_solved']} tasks it solves are mostly tasks whose factors it was "
     f"trained to read.")}
<h3 id="s2-2">2.2 In a familiar room, the policy runs the room's routine</h3>
<p>To see what the policy reads, we asked it for each object in three living rooms and recorded which object it picked up first
(the first object displaced more than 3 cm while its gripper is closed). In living room 1, trained only on <em>put both the alphabet
soup and the cream cheese box in the basket</em>, it picks up the alphabet soup first in {min(lr1)} of 10 episodes whichever object we
name. Its zero-shot successes there are coincidences of that routine: asked for the cream cheese, it carries the alphabet soup and
then the cream cheese, and the success check fires. In LIBERO-Object's own layouts, where it learned each object, an instruction
naming another object present is followed {sum(swap_ba) / len(swap_ba):.2f} of the time.</p>
{fig(video('ov_habit_48'), "Living room 1, the same start state. Left: asked for the alphabet soup. Right: asked for the ketchup. Both times the "
     "policy picks up the alphabet soup, the first object of the routine it was trained on in this room.", wide=True)}
{fig(binding_grid(), "Which object pi0.5 picked up first (10 episodes per row, practice states), by the object the instruction named. Dark: the "
     "room's routine; green: the named object. The success column counts the episodes the task's own check passed.")}
<p>The same picture comes out of the ledger as a whole: the held-out tasks are the ones whose factors the instructions named only
in other rooms, or never. Composition and folding below target exactly these two cases.</p>"""

P_METHOD = f"""
<h3 id="s3-1">3.1 What the frozen policy holds</h3>
<p><b>Prototypes.</b> Mode-finding on the policy's action distribution over its training tasks gives a hierarchy of prototypes, with
the noise level as depth. One observation's conditional is a single narrow bump, so the hierarchy lives across observations: its
coarse prototypes are verbs (reach down, close, carry, lower, release), its top split is the gripper's state, and its labels change
at a task's phase events. <b>Task prototypes.</b> Every task the policy performs is stored as its factors and its prototype chain, the
sequence of verbs its actions pass through: {library['n_prototypes']} of them so far, the forty training tasks and the living-room tasks it
solves.</p>
<h3 id="s3-2">3.2 Read: decompose the arrival into task prototypes</h3>
<p>A new task is decomposed by a greedy cover of its factors with task prototypes, the selection step of Wang et al. moved from
pixels to tasks. Each selected prototype supplies the factors it shares, and its chain says in which phase that factor acts. Factors no
prototype covers are listed; they are where a fold will be needed.</p>
{table(["new task", "task prototype", "factors it supplies"],
       [[esc(ex58['language']) if i == 0 else "", esc(s['language']), esc(", ".join(c[0] for c in s['covers']))] for i, s in enumerate(ex58['selected'])]
       + [[esc(ex3['language']) if i == 0 else "", esc(s['language']), esc(", ".join(c[0] for c in s['covers']))] for i, s in enumerate(ex3['selected'])]
       + [["", "<em>not covered</em>", esc("; ".join(fmt_factor(u) for u in ex3['uncovered']))]],
       "Two decompositions read from the task-prototype library (results/robot/task_library.json).", num_from=9)}
<h3 id="s3-3">3.3 Compose: a task prototype with its object slot rebound</h3>
<p>Most failures in a familiar room come from a task that differs from one the policy performs only in which object fills a role.
The ROLE CODE of such a task names the performed task (the base: same scene, same goal with the moved object replaced) and the
object pair. The policy runs the base's own conditional; composition must supply the binding of the base's object slot to the new
object, which the policy cannot read from the word. Two injections bind it, and in both the policy produces every action:</p>
<ul><li><b>Role swap.</b> The new object and the base's object exchange appearance in what the policy sees, while its own commanded
gripper is open; the simulator's renderer exchanges their meshes (an oracle of what an image editor would do).</li>
<li><b>Attention pointer.</b> The action expert's attention to the image tokens over the new object is raised to a target mass at every
layer and Euler step, while the gripper is open. Zero bias is the policy bit for bit.</li></ul>
<p>Practice on a few start states decides which injection a task keeps. Prompt composition, summing the policy's velocities under
several prompts by the rule that restores composition in image diffusion models, remains available: it amplifies a factor the
policy reads faintly (Section 4.1).</p>
<h3 id="s3-4">3.4 Fold: from the policy's own attempts</h3>
<p>The composed episodes that succeed are recorded with the REAL scene and relabelled with the task's own instruction; they fit a
factor token, one key and one value per layer appended to the prompt's cache, with every policy weight frozen (18.4 KB). The token
then practises alone and is refit on both rounds' successes. At test the policy receives only the instruction and the token: no
injection and no simulator state. No demonstration enters at any point.</p>
<h3 id="s3-5">3.5 Consolidate</h3>
<p>Tokens of one factor merge across arrivals; accrued tokens and any larger folds are distilled into one shared unit and deleted,
trading bytes for drift on old tasks.</p>
{pending("Consolidation", "Designed; it runs after the escalation and stream reads below.")}"""

pp = pilot["totals"]
P_COMPOSE = f"""
<p>We ran composition on every LIBERO-90 task pi0.5 fails that has a readable base in its room with the same relation: 12 tasks. A pilot
on practice states passed its pre-registered gate ({pp['fcomp_8']} of 70 at weight 8, against {pp['cfg_4']} for guidance on the instruction
alone and {pp['placebo_4']} for a placebo pair of unrelated instructions), so the main read ran on the scoring states.</p>
{table(["", "composition", "comparison", "successes", "lower bound", "verdict"],
       [["H1", "weight 8", "zero-shot", f"{H1['x']} vs {H1['control']} of {H1['n']}", f"{H1['diff']['lo90']:+.3f}", "fails (needs > +0.05)"],
        ["H1b", "weight 8", "guidance on the instruction", f"{H1b['x']} vs {H1b['control']}", f"{H1b['diff']['lo90']:+.3f}", "fails"],
        ["H1c", "weight 8", "placebo pair", f"{H1c['x']} vs {H1c['control']}", f"{H1c['diff']['lo90']:+.3f}", f"passes (Holm p {H1c['holm_p']:.2f})"]],
       "The pre-registered composition read: 12 tasks, 20 episodes each, init states 30-49 (ledger p1_bindings, tag zc_main).", num_from=3)}
{fig('<div class="chartwrap">' + composition_chart() + '</div>', "Per task, successes of 20 for each arm. The gain sits on the ketchup in living "
     "room 1, where the policy's first actions already lean toward the named object; elsewhere the named object stays unread.", wide=True)}
{fig(video('ov_48_30'), "Pick up the ketchup and put it in the basket, init state 30. Zero-shot, the policy puts the alphabet soup in the "
     "basket. Composed, it fetches the ketchup.", wide=True)}
<p>Composition is therefore real but narrow: it amplifies a factor the policy reads faintly, and cannot create one it does not
read.</p>"""

P_WHERE = f"""
<p>Binding needs to know where the named object is. We asked the policy's own vision-language backbone first: the attention from the
object's words to the image tokens, layer by layer, scored by whether its peak token lies on the object (simulator segmentation). The
pre-registered gate asked for 0.8 on LIBERO-Goal and 0.6 on the failing tasks; the best layer reaches
{w1g['validity']['w1_by_layer']['10']:.2f} and {w1g['class_a']['w1_by_layer']['10']:.2f}, and an open-vocabulary detector (OWLv2)
{w3g['validity']['w3']:.2f} and {w3g['class_a']['w3']:.2f}. The detector finds kitchen objects but never the alphabet soup. Both fail, so
the reads below take where from the simulator's segmentation and renderer: an oracle, used during practice only for the token arm.</p>"""

def bind_rows():
    out = []
    for t, name in ((48, "ketchup → basket (living room 1)"), (49, "tomato sauce → basket (living room 1)"),
                    (52, "milk → basket (living room 2)"), (53, "orange juice → basket (living room 2)")):
        out.append([esc(name), f"{of10(f'{t}:zs')}", f"{of10(f'{t}:bind_rho0.7_all')}", f"{of10(f'{t}:role')}"])
    return out


P_BIND = f"""
<p>We first checked offline that the attention pointer acts: on the failing tasks it turns the policy's first chunk toward the biased
object (contrast cosine {min(v['CC'] for v in steer['configs'].values()):.2f} to {max(v['CC'] for v in steer['configs'].values()):.2f}).
Online, on practice start states, we counted which object the policy picks up first.</p>
{table(["task", "zero-shot", "attention pointer", "role swap"], bind_rows(),
       "Episodes (of 10) in which the named object is the first picked up; practice states 0-9; oracle where (ledger b2_q2, tag b_q2mini; "
       "pre-registered).", num_from=1)}
<p>The two injections bind in different scenes. Living room 2 was trained on two instructions over cans and boxes; its routine keys
on shape, and giving the milk or the orange juice the soup can's appearance makes the policy pick them up. Showing the real object once
it is held raises success to {q2m['52:role_open']['success']:.1f} and {q2m['53:role_open']['success']:.1f} (exploratory). An unnamed carton given
the can's appearance is picked up too ({q2m['52:roleshape']['BA_placebo']:.1f} and {q2m['53:roleshape']['BA_placebo']:.1f}), an unnamed can is
not: the routine follows shape, not label. Living room 1 was trained on one instruction; its routine goes to the soup's position
whatever stands there, and only the attention pointer moves it to the ketchup. Biasing attention for whole episodes breaks grasping
on most tasks (mean success {q2fa['bind_rho0.7_all']['success_mean']:.2f}); biasing it only during the approach puts the tomato sauce in
the tray in {of10('59:bind_rho0.7_all_approach', 'success', q2f)} of 10 episodes (exploratory). Mugs that differ only in colour, and a
frying pan in a moka pot's role, stay out of reach of both.</p>
{(fig(video('bd_53_1'), "Pick up the orange juice and put it in the basket. Left: zero-shot, the policy takes the milk. Middle: the role code, "
     "as the policy sees it (the orange juice wears the soup can's mesh; task 50's instruction). Right: the folded token, the task's own "
     "instruction and the real scene.", wide=True) if 'bind' in MEDIA and 'bd_53_1' in MEDIA['bind'] else '')}
{(fig(video('bd_59_0'), "Pick up the tomato sauce and put it in the tray (living room 3). Left: zero-shot, the policy takes the alphabet soup. "
     "Right: the attention pointer on the tomato sauce while the gripper is open; the policy does the rest.", wide=True)
  if 'bind' in MEDIA and 'bd_59_0' in MEDIA['bind'] else '')}
{(fig(video('bd_48_0'), "Living room 1, the ketchup. Left: the role swap; the routine goes to the alphabet soup's place, where the ketchup's "
     "appearance now stands. Right: the attention pointer; the policy takes the ketchup and holds it, but the room's carry never starts.",
     wide=True) if 'bind' in MEDIA and 'bd_48_0' in MEDIA['bind'] else '')}"""

P_FOLD = f"""
<p>The role episodes that succeeded on practice states 10-29 fit an {roletok / 1e3:.1f} KB token per task. Given only its own
instruction, the policy then picks up the orange juice first in {of10('53:ftoken_self_role')} of 10 episodes and succeeds
{q2m['53:ftoken_self_role']['success']:.1f} of the time (zero-shot: {of10('53:zs')}, 0); the milk {of10('52:ftoken_self_role')} of 10 and
{q2m['52:ftoken_self_role']['success']:.1f} (exploratory, practice states 0-9). The token reads a word the policy could not read before,
learned from its own attempts.</p>"""


def main_read():
    if not scheme:
        return pending("The scheme's pre-registered read", "Init states 30-49 of the 14 tasks with a role code: zero-shot against the form "
                       "practice chose and against the folded token alone; the tomato-sauce token from living room 3 reused in living room 1.")
    h = scheme["hypotheses"]
    rows = []
    for k, lab in (("H2", "composed form (oracle) vs zero-shot"), ("H1", "token alone vs zero-shot"), ("H4", "transferred token vs zero-shot")):
        if k in h:
            v = h[k]
            rows.append([k, esc(lab), f"{v['x']} vs {v['control']} of {v['n']}", f"{v['diff']['diff']:+.2f}",
                         f"{v['sign_p']:.2g}", "passes" if v["passes"] else "fails"])
    pt = scheme["per_task"]
    trows = [[esc(lang.get(int(t), t)), str(v.get('zs', '')), str(v.get('form', '–')), str(v.get('token', '–'))]
             for t, v in pt.items() if 'form' in v]
    return (table(["", "comparison", "successes", "difference", "sign test p", "verdict"], rows,
                  "The scheme's pre-registered read on init states 30-49 (ledger b5_scheme, tag main; m3_prereg tag b_scheme_main).", num_from=2)
            + table(["task", "zero-shot", "composed form", "token alone"], trows, "Successes of 20 on the tasks practice gave a form.", num_from=1))


def all_reads():
    rows = []
    for lab, src, where, test, r in (("role codes, 4 tasks", "no demonstration", "simulator (practice only)", "token", scheme),
                                     ("demo prototype, 2 tasks", "1 demo as the query", "the demo's object prototype", "token", q6),
                                     ("broad screen, 7 tasks", "no demonstration", "simulator (practice only)", "token", row("q8_main", "read")),
                                     ("demo prototype, 7 tasks", "1 demo as the query", "the demo's object prototype", "token", row("q10_main", "read")),
                                     ("acceptance test, 9 tasks", "1 demo as the query", "the demo's object prototype",
                                      "what practice kept", row("q13_main", "read")),
                                     ("library, 8 tasks", "no demonstration", "other tasks' object prototypes",
                                      "what practice kept", row("q16_library", "read"))):
        if not r:
            continue
        h = r["hypotheses"]["H1"]
        rows.append([esc(lab), esc(src), esc(where), esc(test), f"{h['x']} vs {h['control']} of {h['n']}", f"{h.get('holm_p', h['sign_p']):.2g}",
                     "passes" if h["passes"] else "fails"])
    return table(["read", "demonstration", "where during practice", "at test", "successes vs zero-shot", "Holm p", "verdict"], rows,
                 "Every pre-registered read of the scheme. At test the policy gets the task's instruction and an 18.4 KB token, or, in "
                 "the acceptance test, whichever of nothing, the 4 KB form and the token its own practice kept (ledger b5_scheme, "
                 "q6_main, q8_main, q10_main, q13_main, q16_library).", num_from=4)


def accept_par():
    r = row("q13_main", "read")
    if not r:
        return ""
    h, kept = r["hypotheses"], list(r["kept"].values())
    d = h["H1"]["diff"]
    return (f"<p>Fixed in advance (m3_prereg tag q13_main): before it is scored, the learner tries the policy alone, the binding form and "
            f"the folded token 10 times each on init states 0-9, which no token was fitted on, and keeps the best (ties go to the policy, "
            f"then the token). At a new noise seed the kept arm succeeds {h['H1']['x']} of {h['H1']['n']} times against "
            f"{h['H1']['control']} zero-shot; it loses only {h['H1']['discordant'][1]} paired episodes and wins {h['H1']['discordant'][0]} "
            f"(task-level bootstrap {d['diff']:+.3f}, 95% interval {d['lo']:.3f} to {d['hi']:.3f}). It keeps the policy on "
            f"{kept.count('zs')} tasks, the token on {kept.count('token')} and the form on {kept.count('form')}: "
            f"{r['kept_bytes']:,} bytes for {len(kept)} tasks. The token alone ({h['H2']['x']}) and the form alone ({h['H3']['x']}) "
            f"also beat zero-shot at this seed, but their task-level intervals include zero: each fails on some task where the other, "
            f"or the policy, does better. The rule was proposed after the q6 and q10 reads; the episodes, the noise and the acceptance "
            f"states are new.</p>" + library_par())


def library_par():
    r = row("q16_library", "read")
    if not r:
        return ""
    h1, h2, pt = r["hypotheses"]["H1"], r["hypotheses"]["H2"], r["per_task"]
    d = h1["diff"]
    return (f"<p>The library removes the demonstration (m3_prereg tag q16_library). A new task's candidate forms use object "
            f"prototypes already stored from other tasks' demonstrations of the same object type, up to six, and the same "
            f"acceptance test keeps one or the policy alone. On the {len(r['kept'])} tasks with a stored prototype of their type, at "
            f"a third noise seed, the kept arm succeeds {h1['x']} of {h1['n']} times against {h1['control']} zero-shot "
            f"(discordant pairs {h1['discordant'][0]} to {h1['discordant'][1]}; task-level bootstrap {d['diff']:+.3f}, 95% interval "
            f"{d['lo']:.3f} to {d['hi']:.3f}), level with the forms built from each task's own demonstration ({h2['control']}). "
            f"Among three bowls, practice found prototypes of the right instance in tasks that moved it (the middle bowl: "
            f"{pt['14']['kept']} against {pt['14']['zs']} zero-shot), and missed once (the front bowl: {pt['13']['kept']} against "
            f"{pt['13']['own']} with its own demonstration). The kept unit is an index into the library.</p>")


P_MAIN = f"""
<p>Fixed in advance: on practice states 0-9 each of the 14 tasks with a role code keeps the injection with the higher success (none
if both stay below 0.2); that form practises on states 10-29, its successes fit a token, the token practises alone, and is refit on
both rounds. Scoring uses states 30-49, which no step tuned on.</p>
{main_read()}
<p>The same procedure was then run on the tasks two broad screens found (48 more failing tasks, 10 practice episodes each), with the
simulator's segmentation and, separately, with only the demonstration's object prototype as where.</p>
{all_reads()}
{accept_par()}"""

P_POE = f"""
<p>Since 2026-10-02 one demonstration may serve as the query, used once to select experts and never executed. We ported Wang et
al.'s test-time composition to the policy: the experts are the policy itself under its instruction, under other prompts and with its
attention on one of 16 image blocks (scene options), and the motion prototypes the demonstration's chunks fall into; per gripper
phase, Wang's greedy coverage selects three, and the prototypes multiply into one Gaussian that is mixed with the policy experts at
every sampling step, the policy keeping a fixed minimum share.</p>
{table(["probe", "result"],
       [["share of the policy", f"an early prototype weight still moves the output by {q0['early_effect_max']:.2f} of the gap (the pre-registered limit was 0.2), "
                                f"so weights are constant over the steps; at weight 0.5 the chunk keeps {q0['summary']['0.5_0.5']['sens']:.2f} of the policy's "
                                f"sensitivity to the observation"],
        ["options at one observation", f"none under the empty prompt; attention on an image block reaches "
                                       f"{q1['groups']['class_a']['grid_objects_mean']:.1f} objects per frame"],
        ["explaining a second demo", f"the code from demo 0 beats the policy alone on demo 1 in {q2x['recomputed_beats_policy']} of "
                                     f"{q2x['tasks_with_demo1']} tasks, but recomputing its options at demo 1's observations beats freezing them in only "
                                     f"{q2x['recomputed_vs_frozen']}"],
        ["pilot, 6 tasks x 10 episodes", f"{q3p['totals']['poe_0.5']} successes of 60 against {q3p['totals']['zs']} zero-shot: the go rule fails"]],
       "Prototype PoE planning (ledger q0_share, q1_options, q2_explain, q3_pilot; all pre-registered).", num_from=1)}
<p>A product of motion prototypes held over a whole gripper phase is an average motion: it says how the hand moves, never where.
The demonstration does carry where: the gripper's position at its first grasp marks the object in its first frame, and the policy's
own image features there are a {q4t['proto_bytes'] / 1e3:.0f} KB object prototype that finds the object in new start states
{q4t['own_pointing']:.2f} of the time. Binding the policy's attention to it (no simulator state) puts the white bowl on top of the
cabinet in {of10('43:bindw4_rho0.7_approach', 'success', row('b2_q2', 'q5_w4')['rows'])} of 10 practice episodes (zero-shot 0); in the
pre-registered read the folded tokens alone succeed {q6['hypotheses']['H1']['x']} of {q6['hypotheses']['H1']['n']} times against
{q6['hypotheses']['H1']['control']} on the two tasks practice gave a form. A round of the demo token's own practice does not help
({q7['H1']['x']} against {q7['H1']['control']} of 420; the one-demo LoRA {q7['H2']['control']}).</p>"""

q11 = row("q11_reuse", "white_bowl")
q12 = row("q12_joint", "black_bowl")
q12g = row("q12_joint", "black_bowl_grasp")


def reuse_par():
    if not (q11 and q12 and q12g):
        return ""
    t, g = q12["totals"], q12g["totals"]
    jb, zs, go, ga = t["ftoken_12_2_60_jb"], t["zs"], g["ftoken_12_2_60_jbg_open"], g["ftoken_12_2_60_jbg"]
    return (f"<p>Reuse is not yet shown (exploratory reads). A token folded on one arrival learns that task, not its object: the "
            f"white-bowl tokens of task 43, loaded unchanged for the white-bowl tasks of other rooms, succeed {q11['successes']} times "
            f"in {q11['episodes']} episodes (ledger q11_reuse). One token fitted on the policy's own successes from three black-bowl "
            f"tasks picks up the bowl first more often on four black-bowl tasks it never practised ({jb['bound']} of {jb['n']} against "
            f"{zs['bound']} zero-shot) but succeeds less ({jb['success']} against {zs['success']}). Cut at the grasp and loaded only "
            f"while the gripper is open, it binds {go['bound']} and still succeeds {go['success']}; loaded throughout, {ga['bound']} and "
            f"{ga['success']} (ledger q12_joint). The loss arises during the approach itself, so placements copied from the source tasks "
            f"do not explain it.</p>" + proto_reuse())


def proto_reuse():
    b = row("q14_borrow", "screen")
    if not b:
        return ""
    u, o, m = b["unique_borrowed"], b["unique_own"], b["multi_borrowed_same_scene"]
    return (f"<p>The 4 KB object prototype reuses where the token does not (exploratory, practice states). Where the moved object's "
            f"type has one instance in the scene, a prototype from another task's demonstration, three of the six from other rooms, "
            f"picks the object up first {u['bound']} of {u['n']} times and succeeds {u['success']} times; the task's own gives "
            f"{o['bound']} and {o['success']}. In the kitchen with three black bowls a sibling task's prototype finds the named bowl "
            f"{m['bound']} of {m['n']} times. Figure {FIG['n'] + 1} shows why: the prototype matches every bowl by type, and its best "
            f"patches fall on the bowl at its source's place.</p>"
            + fig('<img src="media/q14_maps.png" alt="cosine maps of object prototypes over first frames" style="width:100%">',
                  "Cosine similarity of an object prototype to the policy's own image features in a task's first frame (brighter: "
                  "more similar; green: the 4 patches binding attends to). Top row: each task's own prototype. Bottom row: a "
                  "prototype from another task's demonstration. The white bowl (task 43) and the chocolate pudding (task 70) are "
                  "found from other rooms; in the three-bowl kitchen (tasks 12 and 13) a borrowed prototype points at its own "
                  "source's bowl.", wide=True))


def stream_par():
    r = row("q17_stream", "read")
    if not r:
        return ""
    h = r["hypotheses"]
    A, B, E = h["H1"], h["H2"], h["H3"]
    return (f"<p>The stream (pre-registered, m3_prereg tag q17_stream). The {A['n'] // 20} failing tasks with a moved object arrive one at "
            f"a time, in a seeded order and its mirror. Only an arrival whose object type the library lacks asks for a demonstration "
            f"({r['demos']['A']} of {A['n'] // 20} in either order, {r['bytes']['A']:,} bytes of prototypes); the rest choose among the "
            f"library's prototypes of their type, and the acceptance test keeps a form or the policy. In the seeded order the kept arms "
            f"succeed {A['x']} of {A['n']} times against {A['control']} zero-shot (Holm p {A['holm_p']:.2g}, not significant); in "
            f"the mirror {B['x']} (Holm p {B['holm_p']:.2g}). Each order keeps a form on {A['kept_forms']} tasks; a demonstration for "
            f"every task keeps {E['kept_forms']} and succeeds {E['x']} times ({r['bytes']['every_demo']:,} bytes). The library holds "
            f"one prototype per type, and among three bowls it points at its own source's bowl: tasks naming the front or the "
            f"middle bowl find no candidate that beats the policy. The library has to grow below the type.</p>" + self_par()
            + inst_par())


def inst_par():
    r = row("q20_stream", "read")
    if not r:
        return ""
    A, B = r["hypotheses"]["H1"], r["hypotheses"]["H2"]
    fb = B["per_task"]["13"]
    return (f"<p>Practice can supply the instance instead (m3_prereg tag q20_stream). A type's prototype matches every object of "
            f"the type (Figure {FIG['n']}); cut its map into its 4 strongest peaks, bind one peak per candidate, and let the "
            f"acceptance test choose. The same stream, with the same {r['demos']['A']} demonstrations, then passes in both orders: "
            f"{A['x']} and {B['x']} of {A['n']} against {A['control']} zero-shot (Holm p {A['holm_p']:.2g} and {B['holm_p']:.2g}; "
            f"{A['kept_forms']} and {B['kept_forms']} forms kept). The front bowl, which no stored prototype pointed at, rises to "
            f"{fb[0]} of 20 from {fb[1]} through another kitchen's prototype at its fourth peak. The gain sits in a few tasks, the "
            f"task-level intervals include zero, and one kept form loses at test (the chocolate pudding), so ten practice episodes "
            f"sometimes keep the wrong unit.</p>" + codes_table() + grammar_par() + confirm_par() + reach_par())


def grammar_par():
    g, q = row("q22_grammar", "offline"), row("q22_qual", "screen")
    if not (g and q):
        return ""
    r = q["rows"]
    return (f"<p>The kept codes hint at a grammar: practice kept the first peak for the back bowl and the last for the front bowl. "
            f"Across all {g['tasks']} LIBERO-90 tasks whose instruction picks one of several same-type objects by a word, the named "
            f"object holds a fixed place among its siblings in the policy's own image: back the top row, front the bottom row, middle "
            f"the middle row, left the rightmost column and right the leftmost (the camera faces the robot). Learned for each word from "
            f"the other tasks that use it, the rule picks the named object in {g['loo_hits']} of {g['loo_frames']} held-out first "
            f"frames (exploratory, simulator segmentation for the analysis). Applied to the peaks of another task's prototype it "
            f"needs no instance search: the back bowl {r['12:bindw4_rho0.7_approach_src2_qual_npk3']['success']} of 10 with a "
            f"prototype from another kitchen (zero-shot {r['12:zs']['success']}), the middle bowl "
            f"{r['14:bindw4_rho0.7_approach_src2_qual_npk3']['success']} (zero-shot {r['14:zs']['success']}); but when a distractor "
            f"(the plate) is among the peaks the middle one is the wrong object "
            f"({r['14:bindw4_rho0.7_approach_src2_qual_npk4']['success']} of 10). The rule is exact; its input, the peaks, is not.</p>"
            + grammar_read_par())


def grammar_read_par():
    g, tm = row("q24_grammar", "read"), row("q23_typemap", "offline")
    if not (g and tm):
        return ""
    h1, h2, pt = g["hypotheses"]["H1"], g["hypotheses"]["H2"], g["per_task"]
    return (f"<p>A type is what its exemplars share. Averaging the library's prototypes of a type from other tasks gives a type "
            f"prototype whose peaks are more often the type's instances (the grammar's pick lands on the named object "
            f"{tm['mean']['mean_n3']:.2f} of the time offline, against {tm['mean']['single_n3']:.2f} from one donor), and how many "
            f"peaks each word orders is learned from the other tasks that use it. In the pre-registered read (m3_prereg tag "
            f"q24_grammar) on the {len(pt)} qualifier tasks, with no demonstration of the scored task, the named object is moved "
            f"first {h1['x']} of {h1['n']} times against {h1['control']} zero-shot (discordant pairs {h1['discordant'][0]} to "
            f"{h1['discordant'][1]}, Holm p {h1['holm_p']:.2g}); success is level ({h2['x']} against {h2['control']}). Five words "
            f"and the library now pick the object. The pick-and-place tasks gain (the front bowl {pt['13']['success_grammar']} "
            f"against {pt['13']['success_zs']}); the tasks with a second step bind and still fail at that step.</p>" + gaccept_par())


def gaccept_par():
    g = row("q26_grammar_accept", "read")
    if not g:
        return ""
    h1, h2 = g["hypotheses"]["H1"], g["hypotheses"]["H2"]
    k = sum(v == "grammar" for v in g["kept"].values())
    return (f"<p>Filtered by the acceptance test (m3_prereg tag q26_grammar_accept), the grammar is kept on {k} tasks, all bowl "
            f"tasks, and the policy elsewhere; the kept arms succeed {h1['x']} of {h1['n']} times against {h1['control']} "
            f"(discordant pairs {h1['discordant'][0]} to {h1['discordant'][1]}, Holm p {h1['holm_p']:.2g}). A type prototype, five "
            f"words and the learner's own practice turn better reading into more successes, with no demonstration of the scored "
            f"task.</p>")


def reach_par():
    zb, dose = row("q20_stream", "zs_binding"), row("q21_dose", "screen")
    if not (zb and dose):
        return ""
    ov = [1, 3, 4, 5, 32, 34, 36, 37, 40, 41, 42, 43, 63]
    lo = [L.load("m3_final", tag="lora", keys__tasks=str(t))[-1]["metrics"] for t in ov]
    k_lo, n_lo = sum(round(m["success_rate"] * m["n_episodes"]) for m in lo), sum(m["n_episodes"] for m in lo)
    h = row("q20_stream", "read")["hypotheses"]
    a, b = (sum(h[x]["per_task"][str(t)][0] for t in ov) for x in ("H1", "H2"))
    z = sum(h["H1"]["per_task"][str(t)][1] for t in ov)
    d = dose["rows"]
    return (f"<p>Where composition stops. {len(ov)} of the stream's tasks also have the archived one-demo LoRA, an 18.7 MB adapter "
            f"fitted on the task's demonstration: it succeeds {k_lo} of {n_lo} times there, the stream {a} and {b}, zero-shot {z}. On "
            f"{len(zb['wrong_object'])} of these the policy moves another object first, so binding is what fails, yet attention binding "
            f"takes hold on only the white bowl; raising the attention mass to 0.95 still never moves the white bowl of the second "
            f"kitchen or the frying pan first ({d['36:bindw4_rho0.95_approach']['bound']} and "
            f"{d['42:bindw4_rho0.95_approach']['bound']} of 10), and the butter, moved first {d['3:bindw4_rho0.85_approach']['bound']} "
            f"of 10 times at 0.85, never completes the task's second step. Composition reaches the tasks a frozen policy fails by "
            f"reading the wrong object and could grasp; a grasp it lacks or a step it does not chain still needs what a fitted "
            f"adapter takes from a demonstration.</p>" + seq_par())


def seq_par():
    q = row("q25_seq", "screen")
    if not q:
        return ""
    t = q["totals"]
    return (f"<p>Chaining by the instruction's grammar does not supply the missing step either (exploratory). Splitting the "
            f"two-clause instructions at their conjunction and switching to the second clause at the policy's first release gives "
            f"{t['seqzs']} successes alone and {t['bindw4_rho0.7_approach_seq']} with binding over 80 practice episodes, against "
            f"{t['zs']} zero-shot; in recorded episodes the policy carries the bound butter toward its own base, never to the "
            f"drawer. The first clause is itself not a task the policy performs, and a sequence can only chain clauses it does.</p>")


def codes_table():
    r, plan = row("q20_stream", "read"), json.load(open(os.path.join(R, "q20_plan.json")))
    if not r:
        return ""
    info, rows = plan["info"], []
    import re as _re
    for h in ("H1", "H2"):
        v = r["hypotheses"][h]
        for t, arm in r["kept"][v["condition"]].items():
            if arm == "zs":
                continue
            m = _re.match(r"bindw4@rho0.7_approach_src(\d+)(?:_inst(\d+)_npk(\d+))?", arm)
            src, inst, npk = m.group(1), m.group(2), m.group(3)
            pt = info[src]["type"].replace("_", " ").replace("akita ", "")
            where = (f"its own demonstration's {pt} prototype" if src == t else
                     f"the {pt} prototype stored for task {src} ({'same room' if info[src]['scene'] == info[t]['scene'] else 'another room'})")
            code = esc(f"\u201c{info[t]['language']}\u201d + {where}" + (f", instance {int(inst) + 1} of {npk}" if inst else "")
                       + ", attention raised while the gripper is open")
            ks, zs = v["per_task"][t]
            rows.append([esc("A" if h == "H1" else "B (mirror)"), code, "4 KB" if src == t else "an index", f"{ks}", f"{zs}"])
    return table(["order", "the kept code, read aloud", "stored", "kept (of 20)", "zero-shot (of 20)"], rows,
                 "Every form the stream with instances kept (ledger q20_stream), with its successes on init states 30-49. Each code "
                 "is the task's instruction plus a pointer into the library; a prototype stored for another task costs the new task "
                 "only its index.", num_from=3)


def confirm_par():
    c = row("q20_stream", "confirm")
    if not c:
        return ""
    h, pu = c["hypotheses"], c["confirmation"]["B:70"]
    return (f"<p>A confirmation round does not repair it (amendment q20_confirm): requiring a kept form to beat the policy again on "
            f"init states 10-19 drops the back bowl's gain and keeps the pudding's form, which wins {pu['form']} to {pu['zs']} there "
            f"while the policy alone succeeds 16 of 20 times on the test states; neither order passes ({h['H1']['x']} and "
            f"{h['H2']['x']} against {h['H1']['control']}, Holm p {h['H1']['holm_p']:.2g}). The pudding's practice states do not "
            f"represent its test states.</p>")


def self_par():
    r = row("q18_self", "read")
    if not r:
        return ""
    h = r["hypotheses"]["H1"]
    return (f"<p>Growing it from the learner's own successes, with no demonstration at all, fails (m3_prereg tag q18_self). Each of the "
            f"same tasks practises zero-shot; a success yields a prototype from the policy's own grasp, the same construction as from "
            f"a demonstration. {r['self_protos']} tasks get one, {r['forms_kept']} keep a form, and the kept arms succeed {h['x']} of "
            f"{h['n']} times against {h['control']} (sign p {h['sign_p']:.2g}). A prototype exists only where the policy already "
            f"finds the object, and there binding mostly gets in its way; the tasks binding rescues never succeed zero-shot, so they "
            f"offer nothing to learn from.</p>")


P_STREAM = f"""
{reuse_par()}
{stream_par()}
{pending("Other policies", "The same codes on pi0 and SmolVLA, each with its own conditionals and its own tokens; and CALVIN ABC→D, whose "
         "unseen scene D is the same reading problem.")}"""

def prim_par():
    c0, v2, c1, cb = row("c0_exec", "screen"), row("c0_exec", "v2"), row("c1_reuse", "screen"), row("c1b_joint", "screen")
    if not (c0 and v2 and c1 and cb):
        return ""
    q = row("q27_joint", "read")
    out = (f"<p>Binding cannot supply a motion the policy lacks, so the user allowed the one demonstration to teach a motion, as a "
           f"reusable STEP rather than a task (docs/PLAN_PRIMITIVES.md). An instruction splits into steps at its conjunction; a "
           f"step's key is its verb, preposition and target (\u201cput | in | the top drawer of the cabinet\u201d), the object "
           f"left out. A primitive is a rank-16 LoRA on the action expert (18.7 MB) fitted on the step's segment of a demonstration "
           f"with the step's clause as the prompt; the policy switches primitives at its own release. On the 13 pool tasks with "
           f"an archived one-demo LoRA, a task's own step primitives succeed {v2['own_total_with_one_step']} of 130 times against "
           f"{v2['lora_total']} for the LoRA and {c0['totals']['zs']} zero-shot (exploratory). Fitted on ONE other task's demo, a "
           f"primitive does not transfer: {c1['reused']} successes against {c1['zero_shot']} zero-shot over "
           f"{len(c1['one_step_pairs'])} pairs; it holds that object's grasp and room. Fitted on EVERY other task that shares the "
           f"step, it does: \u201con top of the cabinet\u201d gives {cb['cabinet']['joint']} of {cb['cabinet']['n']} against "
           f"{cb['cabinet']['zs']} on three black-bowl tasks in kitchens it never saw, including the one whose routine puts the "
           f"bowl in a drawer ({cb['rows']['25:prims_Jcab25.0']['success']} of 10 against 0). Where the difficulty is which of "
           f"three bowls, the pooled \u201con the plate\u201d overrides the policy's reading ({cb['plate']['joint']} of "
           f"{cb['plate']['n']} against {cb['plate']['zs']}). Consolidation across arrivals, Proposition 7's prediction, is what "
           f"makes a step reusable.</p>")
    if q:
        h1, h2 = q["hypotheses"]["H1"], q["hypotheses"]["H2"]
        out += (f"<p>Pre-registered (m3_prereg tag q27_joint): on the {len(q['per_task'])} pool tasks with a step shared by at "
                f"least three other tasks, leave-one-task-out consolidated primitives kept by the acceptance test succeed "
                f"{h1['x']} of {h1['n']} times against {h1['control']} zero-shot (Holm p {h1['holm_p']:.2g}); unfiltered, "
                f"{h2['x']} (Holm p {h2['holm_p']:.2g}). No demonstration of a scored task entered its primitives.</p>")
    c2 = row("c2_stream", "read")
    if c2:
        h1, h2 = c2["hypotheses"]["H1"], c2["hypotheses"]["H2"]
        kept = sum(v["kept"] != "zs" for v in c2["per_task"].values())
        nodemo = sum(v["kept"] != "zs" and not v["demo"] for v in c2["per_task"].values())
        out += (f"<p>The stream (pre-registered, m3_prereg tag c2_stream): {sum(c2['arrivals'].values())} single-step tasks in "
                f"{len(c2['arrivals'])} step keys arrive one at a time; each tries its key's current primitive first and requests "
                f"its demonstration only if the primitive is not kept, after which the key's primitive is refit on every "
                f"demonstration so far. It requested {sum(c2['demos'].values())} demonstrations and kept a primitive on {kept} "
                f"arrivals ({nodemo} with no demonstration of their own), storing {c2['bytes'] / 1e6:.1f} MB for all of them. The kept "
                f"arms succeed {h1['x']} of {h1['n']} times against {h1['control']} zero-shot (Holm p {h1['holm_p']:.2g}); each key's "
                f"final primitive, run on all its arrivals, {h2['x']} (Holm p {h2['holm_p']:.2g}), so refitting on later "
                f"demonstrations keeps part of the earlier arrivals' gain. Reuse saved few demonstrations: a refit primitive rarely "
                f"beat the policy on a new arrival before that arrival's own demonstration joined it.</p>")
    f = row("f1_fresh", "q27_half")
    if f:
        out += (f"<p>Every stored LIBERO-90 start state had been used by some earlier read, so the pre-registered fresh-state benchmark "
                f"(m3_prereg tag f1_fresh) draws 20 new start states per task from LIBERO's own placement sampler. On the "
                f"{f['H2']['tasks']} tasks above, the kept arms succeed {f['H2']['x']} of {f['H2']['n']} times against "
                f"{f['H2']['y']} zero-shot (discordant pairs {f['H2']['discordant'][0]} to {f['H2']['discordant'][1]}). The per-task "
                f"one-demo LoRA, fitted on each scored task's own demonstration, succeeds {f['q27_vs_lora']['y']} times: with no "
                f"demonstration of the scored task the scheme reaches about two fifths of it. A per-task adapter damages nothing "
                f"either, so the scheme's case is a frontier: success per demonstration requested and per byte stored, which the "
                f"stream measures.</p>")
    F1, FC, A2 = row("f1_fresh", "read"), row("f1_fresh", "costs"), row("f1_fresh", "amend2")
    if F1 and FC:
        d, h1 = F1["descriptive"], F1["hypotheses"]["H1"]
        nd = FC["scheme_demos"] - len(FC["lora_missing"])
        rows = [("zero-shot (the frozen policy)", h1["y"], h1["n"], 0, 0, False),
                ("sequential fine-tuning, one LoRA per key", d["c2_vs_seq"]["y"], d["c2_vs_seq"]["n"], FC["lora_tasks"], FC["bcl"]["seq"], False),
                ("experience replay, one LoRA per key", d["c2_vs_er"]["y"], d["c2_vs_er"]["n"], FC["lora_tasks"], FC["bcl"]["er"], False),
                ("joint multi-task, one LoRA per key", d["c2_vs_joint"]["y"], d["c2_vs_joint"]["n"], FC["lora_tasks"], FC["bcl"]["joint"], False),
                *([("joint multi-task + the acceptance test", A2["joint_acc_vs_zs"]["x"], A2["joint_acc_vs_zs"]["n"],
                     FC["lora_tasks"], FC["bcl"]["joint"], False)] if A2 else []),
                ("this scheme: consolidated step primitives", h1["x"], h1["n"], nd, FC["scheme"], True),
                ("a one-demo LoRA per task", d["c2_vs_lora"]["y"], d["c2_vs_lora"]["n"], FC["lora_tasks"], FC["lora"], False),
                ("a one-demo LoRA per task + the acceptance test", d["c2_vs_lora_acc"]["y"], d["c2_vs_lora_acc"]["n"], FC["lora_tasks"], FC["lora"], False)]
        out += (f"<p>Against its baselines, on start states no rule has seen (pre-registered, m3_prereg tags f1_fresh and f1_amend1), "
                f"the stream is the best of the learners that share storage across tasks, and reaches "
                f"{100 * d['c2_vs_lora']['x'] / d['c2_vs_lora']['y']:.0f}% of a separate adapter per task at a "
                f"{FC['lora'] / FC['scheme']:.0f}th of its storage (Figure {FIG['n'] + 1}). Sequential fine-tuning forgets; a separate "
                f"adapter per task given the same acceptance test remains the strongest."
                + (f" Given the same acceptance test, one adapter per step key trained on all its demonstrations reaches "
                   f"{A2['joint_acc_vs_zs']['x']}, and the stream's lead over it is not significant (sign p "
                   f"{A2['c2_vs_joint_acc']['sign_p']:.2g}): most of the gain over shared-storage baselines is the acceptance test; "
                   f"the stream adds a quarter of the storage and fewer demonstrations." if A2 else "") + "</p>"
                + fig('<div class="chartwrap">' + methods_chart(rows) + '</div>',
                      f"Success rate on the stream's {FC['lora_tasks'] + len(FC['lora_missing'])} tasks, 20 never-used start "
                      f"states each, noise seed 9, with the demonstrations each method used and the bytes it stores (task 51 has "
                      f"no demonstration, so the per-task rows cover 29 tasks). The scheme also ran {FC['practice_episodes']} "
                      f"practice episodes before test; the baselines ran none."))
    z = row("c1b_joint", "size")
    if z:
        out += (f"<p>A primitive need not be large: refit at rank 4 ({z['bytes']['r4'] / 1e6:.1f} MB) the three cabinet primitives "
                f"succeed {z['totals']['r4']} of {z['n']} times, as at rank 16 ({z['bytes']['r16'] / 1e6:.1f} MB); confined to the "
                f"expert's last two layers ({z['bytes']['l2'] / 1e6:.1f} MB) they fail ({z['totals']['l2']}). One rank-4 primitive "
                f"per step, shared by every task with that step, against an 18.7 MB adapter per task (exploratory).</p>")
    return out


P_PRIM = f"""{prim_par()}"""


def interp_par():
    i1, i2, i3 = row("i1_prototypes", "read"), row("i2_primitives", "read"), row("i3_behavior", "read")
    if not (i1 and i2 and i3):
        return ""
    u, g, ov, tb = i1["unique_share"], i2["module_share_mean"], i3["own_vs_other"], i3["table"]
    return (f"<p>A prototype is an object type at a place. Over the library's {i1['n']} prototypes, three factors explain "
            f"{100 * i1['r2_full']:.0f}% of the variance in their pairwise similarity: the type {100 * u['type']:.0f}% on its own, the "
            f"place in the image {100 * u['place']:.0f}%, the room {100 * u['room']:.0f}% (exploratory). Their hierarchy (Figure "
            f"{FIG['n'] + 1}) groups the books of three study rooms, the frying pans and the bottles, and puts two objects grasped at "
            f"the same patch side by side. That is why a borrowed prototype transfers across rooms when its type is alone and "
            f"points at its own source's instance when it is not.</p>"
            + fig('<img src="media/i1_prototype_tree.png" alt="hierarchy of the object prototypes" style="width:100%">',
                  "The library's object prototypes, clustered by 1 minus cosine similarity (average linkage); each leaf is the "
                  "object type, the room and the grasp patch (row, column) of the demonstration it came from.")
            + f"<p>A consolidated primitive encodes its step. Primitives of one step key have cosine {i2['same_key_cos']:.2f} with each "
            f"other and {i2['diff_key_cos']:.3f} with other keys' (Figure {FIG['n'] + 1}), and a key's successive refits in the "
            f"stream agree more as demonstrations accumulate. The update lives in the action expert's queries "
            f"({g['q_proj']:.2f} of its squared norm) and MLP ({g['gate_proj'] + g['up_proj'] + g['down_proj']:.2f}), almost not in "
            f"its keys and values: a primitive changes what the action tokens look at and how they transform it. What it does to the "
            f"actions matches the reads: the cabinet primitive changes the height of the carry ({tb['cab']['cab']['carry']['z']:.2f}) "
            f"and acts twice as much on its own step's frames ({ov['cab']['own']:.2f} against {ov['cab']['other']:.2f}); the plate "
            f"primitive acts on the approach, where the policy chooses the bowl, and no more on its own step's frames than on others "
            f"({ov['plate']['own']:.3f} against {ov['plate']['other']:.3f}), which is the instance gap left against per-task adapters.</p>"
            + fig('<img src="media/i2_similarity.png" alt="cosine similarity between motion primitives" style="width:100%">',
                  "Cosine similarity between the motion primitives' weight updates, exact in rank space; rows are grouped by step "
                  "key (C2: the stream's successive refits; J and Q27: leave-one-out joint fits; own: one demonstration)."
                  ) + c4_par())


def c4_par():
    c = row("c4_bind", "screen")
    if not c:
        return ""
    t = c["totals"]
    return (f"<p>The interpretation points at a repair: a primitive trained without binding moves the approach on its own, so binding "
            f"added at test breaks it. Refit with binding on during training, the plate primitives and binding compose: "
            f"{t['new_bind']} of 40 practice episodes against {t['old_bind']} for the earlier primitives with binding added, the back "
            f"bowl {c['rows']['12:prims_Jplate12b.0_bind']['success']} of 10 against {c['rows']['12:zs']['success']} zero-shot "
            f"(exploratory); on two of the four tasks the policy alone remains better ({t['zs']} zero-shot in all).</p>" + q28_par())


def q28_par():
    q = row("q28_bindprim", "read")
    if not q:
        return ""
    h1, h2, tt = q["hypotheses"]["H1"], q["hypotheses"]["H2"], q["totals"]
    return (f"<p>Pre-registered on the {len(q['kept'])} plate and basket tasks, on fresh start states (m3_prereg tags q28_bindprim "
            f"and q28_amend1; a first run was void because six tasks had no stored prototype): primitives trained under binding and "
            f"run with it succeed {tt['bind']} of {h2['n']} times against {tt['plain']} for the plain ones (Holm p "
            f"{h2['holm_p']:.2g}), so attention and adapter compose. Both stay below the policy on these keys ({tt['zs']} "
            f"zero-shot), and the kept arms reach {h1['x']} (Holm p {h1['holm_p']:.2g}, not significant): where the policy already "
            f"chooses the object well, a primitive pooled over other objects rarely improves on it.</p>")



P_INTERP = f"""{interp_par()}"""


def factored_par():
    d0, d1, dg = row("d0_goal", "screen"), row("d1_factored", "read"), row("d1_factored", "diagnosis")
    if not (d0 and d1 and dg):
        return ""
    t0 = d0["totals"]
    h1, h2, de = d1["hypotheses"]["H1"], d1["hypotheses"]["H2"], d1["descriptive"]
    kept = {k: sum(v == k for v in d1["kept"].values()) for k in ("zs", "goal", "goal_bind")}
    g = dg["totals"]
    return (f"<p>The interpretation shows a step primitive fusing two factors: which object the arm reaches for, and where it carries "
            f"it. Plan D splits each step at the policy's own grasp. Before the grasp the object factor acts (the object prototype "
            f"through attention binding); from the grasp to the release a GOAL primitive acts, a rank-4 LoRA fitted only on "
            f"demonstrations' post-grasp frames, keyed by the goal (\u201cin the basket\u201d) with the object left out and "
            f"consolidated over every other task with that goal. In a screen on {t0['tasks']} held-out tasks the goal primitive "
            f"succeeds {t0['goal']} of {10 * t0['tasks']} times against {t0['zs']} zero-shot, and no longer overrides the policy's "
            f"choice of object (exploratory).</p>"
            f"<p>Pre-registered on the stream's {h1['tasks']} tasks and fresh start states (m3_prereg tag d1_factored), the acceptance "
            f"test kept the policy on {kept['zs']} tasks, the goal primitive on {kept['goal']} and the goal primitive with binding on "
            f"{kept['goal_bind']}. The kept arms succeed {h1['x']} of {h1['n']} times against {h1['y']} zero-shot (sign p "
            f"{h1['sign_p']:.2g}). Against the per-task one-demo LoRA with the same acceptance test they reach {h2['x']} of {h2['n']} "
            f"against {h2['y']} (difference {h2['diff']['diff']:+.3f}, 95% interval {h2['diff']['lo']:+.3f} to "
            f"{h2['diff']['hi']:+.3f}): non-inferiority fails. The first grasp shows where. On the {len(dg['tasks'])} tasks where only "
            f"the LoRA succeeds, the policy grasps the wrong object first in {g['zs']['wrong']} of {g['zs']['n']} episodes (the milk "
            f"for the orange juice, the alphabet soup for the ketchup); the goal primitive inherits that grasp, and binding turns the "
            f"arm toward the named object but grasps nothing in {g['goal_bind']['none']}. The LoRA grasps the named object first in "
            f"{g['lora']['right']} and succeeds {g['lora']['success']} times. Binding says where the object is, not how to reach and "
            f"grasp it: the approach needs its own prototype.</p>")


def plan_e_par():
    e0, dg, mp, sn = row("e0_object", "screen"), row("d1_factored", "diagnosis"), row("l90_scenes", "map"), row("d1_factored", "sensitivity")
    if not (e0 and dg and mp and sn):
        return ""
    t, gap = e0["totals"], list(dg["tasks"])
    g_all = sum(e0["per_task"][x]["all"] for x in gap)
    g_lora = sum(e0["per_task"][x]["lora"] for x in gap)
    out = (f"<p>Plan E gives the approach its own prototype, symmetric to the goal's: an OBJECT primitive, a rank-4 LoRA fitted only on "
           f"demonstrations' frames before their first grasp, keyed by the step's object with the instance word left out "
           f"(\u201cput the black bowl\u201d) and consolidated over every LIBERO-90 demonstration of that object. It acts until the "
           f"policy's grasp; the goal primitive acts from the grasp to the release. A task is the composition of its object's and its "
           f"goal's prototypes: the stream's {e0['tasks']['all']} tasks need {e0['library']['object_keys']} object and "
           f"{e0['library']['goal_keys']} goal prototypes. In a screen on stored states (exploratory), with each task's one "
           f"demonstration absorbed into its two prototypes, the library succeeds {t['all']} of {10 * e0['tasks']['all']} times against "
           f"{t['zs']} zero-shot; the per-task LoRA succeeds {t['lora']} of {10 * e0['tasks']['lora']}. On the {len(gap)} wrong-object "
           f"tasks it succeeds {g_all} of {10 * len(gap)} (LoRA {g_lora}). Without any demonstration of the task the object prototypes "
           f"of other tasks transfer only for some objects ({t['loo']} against {e0['paired']['loo']['zs']} zero-shot on "
           f"{e0['tasks']['loo']} tasks).</p>")
    out += (f"<p>Reading the screen exposed a data defect. The IPEC port of LIBERO-90 merges tasks by instruction ({mp['shared_texts']} "
            f"instructions shared by {mp['shared_tasks']} tasks in different scenes), and every read had taken an instruction's first "
            f"episode as the task's demonstration: {len(mp['wrong_scene_tasks'])} tasks were given another scene's demonstration, the "
            f"per-task LoRA's included, and some leave-one-out fits contained the scored task's own demonstration through a "
            f"same-instruction task. Each task now takes its demonstration from its own scene's block of episodes, matched to its rendered "
            f"first frames. Without the {len(sn['excluded'])} affected stream tasks D1 keeps its verdicts (kept {sn['H1']['x']} against "
            f"{sn['H1']['y']} of {sn['H1']['n']} zero-shot; {sn['H2']['x']} against {sn['H2']['y']} of {sn['H2']['n']} for the LoRA "
            f"with acceptance), and the stream's two demonstrations \u201csaved\u201d were the defect's.</p>")
    e1 = row("e1_factored", "read")
    if e1:
        h1, h2, de, b = e1["hypotheses"]["H1"], e1["hypotheses"]["H2"], e1["descriptive"], e1["bytes"]
        kept = sum(v == "fac" for v in e1["kept"].values())
        out += (f"<p>Pre-registered on the corrected data (m3_prereg tag e1_factored; fresh start states, a new noise seed), the "
                f"acceptance test kept the factored library on {kept} of {h1['tasks']} tasks. The kept arms succeed {h1['x']} of "
                f"{h1['n']} times against {h1['y']} zero-shot (sign p {h1['sign_p']:.2g}). Against the per-task LoRA, refit on each "
                f"task's own scene and given the same acceptance test, they reach {h2['x']} of {h2['n']} against {h2['y']} (difference "
                f"{h2['diff']['diff']:+.3f}, 95% interval {h2['diff']['lo']:+.3f} to {h2['diff']['hi']:+.3f}): non-inferiority "
                f"{'holds' if h2['noninferior'] else 'fails'} at the pre-registered margin of 0.05. Alone, the library succeeds "
                f"{de['fac_vs_zs']['x']} times and the LoRA {de['lora_vs_zs']['x']}. The library stores {b['prototypes']} prototypes "
                f"({b['library'] / 1e6:.1f} MB) where the per-task LoRAs store {b['loras']} adapters ({b['per_task_lora'] / 1e6:.1f} "
                f"MB).</p>")
    e2, e3_ = row("e2_instance", "screen"), row("e3_hier", "read")
    if e2 and e3_:
        h1, h2, de, b = e3_["hypotheses"]["H1"], e3_["hypotheses"]["H2"], e3_["descriptive"], e3_["bytes"]
        out += (f"<p>The gap sat on tasks that name an instance (\u201cthe middle black bowl\u201d), where a prototype pooled over "
                f"14 black bowls averages their approaches. Cobweb's hierarchy answers it: a type concept has instance children, and a "
                f"step that names an instance uses the child, consolidated over the demonstrations with that object and that "
                f"qualifier. In a screen the children succeed {e2['totals']['scratch']} of 40 times on the four instance tasks against "
                f"{e2['totals']['type']} for the type prototype (exploratory). The confirmatory read (m3_prereg tag e3_hier) tested "
                f"the hierarchical library, {b['prototypes']} prototypes for {h1['tasks']} tasks, on a second set of start states no "
                f"design decision had seen. The kept arms succeed {h1['x']} of {h1['n']} times against {h1['y']} zero-shot (sign p "
                f"{h1['sign_p']:.2g}), and {h2['x']} of {h2['n']} against {h2['y']} for the per-task LoRA with acceptance (difference "
                f"{h2['diff']['diff']:+.3f}, 95% interval {h2['diff']['lo']:+.3f} to {h2['diff']['hi']:+.3f}). The interval holds "
                f"zero, but its lower bound misses the margin of 0.05, as the registration's power projection expected: the tasks "
                f"differ too much among themselves for 30 of them to establish non-inferiority. Without acceptance the library and the "
                f"LoRA are level ({de['hfac_vs_lora']['x']} against {de['hfac_vs_lora']['y']}); the instance children succeed "
                f"{de['instance_child_vs_type']['x']} of {de['instance_child_vs_type']['n']} times against "
                f"{de['instance_child_vs_type']['y']} for the type prototype. The library stores {b['library'] / 1e6:.1f} MB where "
                f"the per-task LoRAs store {b['per_task_lora'] / 1e6:.1f} MB.</p>")
    i4 = row("i4_factors", "read")
    if i4:
        sm, st = i4["similarity"], i4["stability"]
        out += (f"<p>In weight space (exploratory), each object prototype points its own way: the mean cosine between object "
                f"prototypes is {sm['object_object']:.3f}, between object and goal prototypes {sm['object_goal']:.3f}. No shared "
                f"reach-and-grasp update exists; the object factor carries the object. Both factors act in the same layers and modules "
                f"(the later layers' queries and MLP, almost nothing in keys and values), so they differ in direction, not in location. A "
                f"prototype summarises its demonstrations: refit without one task's demonstration, an object prototype keeps a cosine of "
                f"{st['same_object']:.2f} with the full one, against {st['other_objects']:.3f} with other objects' prototypes.</p>")
    return out


P_FACTOR = f"""{factored_par()}{plan_e_par()}"""


def streams_par():
    e4, w1, w0 = row("e4_stream", "read"), row("w1_stream", "read"), row("w0_select", "probe_sample")
    if not e4:
        return ""
    out = ("<p>Learned as a stream, the library starts from the demonstrations of the LIBERO-90 tasks outside the stream, and the "
           "30 tasks arrive one at a time in two orders. An arrival is composed from the current library (its instance child, else its "
           "type concept; its goal concept; a missing factor is left to the policy) and requests its one demonstration only when practice "
           "rejects the composition; the demonstration then updates every concept on its path, as Cobweb updates every node an instance "
           "passes. Each task is served at the end by the final library (pre-registered, m3_prereg tag e4_stream).</p>")
    rows_ = []
    for name, r in (("E4", e4), ("W1", w1)):
        if not r:
            continue
        for o in ("a", "b"):
            x = r[o]
            dm = x.get("damage") or {}
            rows_.append(f"<tr><td>{name}</td><td>{o}</td><td class=n>{x['demos']}</td><td class=n>{len(x['kept_without_demo'])}</td>"
                         f"<td class=n>{x['H1']['x']} / {x['H1']['y']}</td><td class=n>{x['vs_lora_acc']['x']} / {x['vs_lora_acc']['y']}</td>"
                         f"<td class=n>{(str(dm['x']) + ' / ' + str(dm['y'])) if dm else '&ndash;'}</td></tr>")
    out += ('<div class="tbl"><p class="tcap"><b>Table.</b> The library learned as a stream, per rule and order. Demonstrations requested '
            '(of the 29 that exist); arrivals kept without one; served successes against zero-shot (of 600) and against the per-task '
            'LoRA with acceptance (of 580); on the arrivals whose composition changed after they arrived, the final composition against '
            'the arrival-time one (damage). Fresh start states, set b.</p><div class="scroll"><table><thead><tr><th>rule</th><th>order</th>'
            '<th class=n>demos</th><th class=n>no demo</th><th class=n>vs zero-shot</th><th class=n>vs LoRA + acc.</th>'
            '<th class=n>final / arrival</th></tr></thead><tbody>' + "".join(rows_) + "</tbody></table></div></div>")
    a_, b_ = e4["a"], e4["b"]
    out += (f"<p>Under E4's rule (a demonstration whenever the composition does not beat the policy in practice) both orders beat the "
            f"policy by far ({a_['H1']['x']} and {b_['H1']['x']} of {a_['H1']['n']} against {a_['H1']['y']}). In order a "
            f"consolidation helped the arrivals it touched: the final library succeeds {a_['damage']['x']} times where their "
            f"arrival-time compositions succeeded {a_['damage']['y']}. In order b it did not: {b_['damage']['x']} against "
            f"{b_['damage']['y']}, and one arrival kept at 9 practice successes against the policy's 8 falls to 0 of 20 under the final "
            f"library, where the policy alone succeeds 17. The losses sit on arrivals kept without a demonstration on a thin practice "
            f"margin (2 against 1, 3 against 0): the rule saves demonstrations on evidence too weak to keep a unit, and nothing "
            f"re-checks an arrival when a concept it uses changes.</p>")
    cap = ("<b>Demonstrations requested as the stream arrives.</b> E4 requests one whenever the composition does not beat the policy"
           + ("; W1 only when neither the policy nor the composition reaches 8 of 10 practice episodes" if w1 else "")
           + ". Hollow markers: arrivals kept without a demonstration; dotted: one demonstration per arrival.")
    out += fig('<img src="media/streams_demos.png" alt="cumulative demonstrations requested against arrivals" style="width:100%">', cap)
    if w0:
        out += (f"<p>Wang et al. select a model's prototypes by how well they cover one query. Given a held-out demonstration as the "
                f"query, neither the policy's flow-matching loss nor the error of each prototype's own samples picks the instruction's "
                f"concept above chance (object concept on top for {w0['summary']['object']['top1']} of "
                f"{w0['summary']['object']['queries']} queries by sampled error; exploratory): one trajectory's motion varies more between "
                f"placements than between concepts. The learner therefore selects by the instruction's words and keeps by practice, and "
                f"what it adds to Wang et al.'s program is their missing mechanism: new primitives detected by practice, consolidated into "
                f"concepts and reused by later arrivals.</p>")
    return out


P_STREAMS = f"""{streams_par()}"""


P_LIMITS = """
<ul>
<li>Where the named object is comes from the simulator's segmentation during practice, or from the one demonstration's object
prototype: the policy's own grounding and an open-vocabulary detector both fall short of the pre-registered gate. At test only the
instruction and the kept unit remain.</li>
<li>Coverage: approach-only binding gives a form on 9 of the 48 failing tasks with a moved object; on the rest the policy fails for
reasons binding does not reach.</li>
<li>Reuse across arrivals is not yet shown: shared tokens bind more often and succeed less often.</li>
<li>A role swap binds only where the scene's routine keys on appearance (here, shape); a routine tied to a position needs the attention
pointer, and objects that differ only in colour, or whose grasps differ, stay out of reach.</li>
<li>Role codes need a base: a task the policy performs in the same scene with the same goal up to the moved object. 14 of the 71 failing
LIBERO-90 tasks have one.</li>
<li>Simulation only (LIBERO-90), one policy so far (pi0.5).</li>
</ul>"""

P_VER = f"""
<p>Every read quoted as a test was pre-registered in <code>ledger/m3_prereg</code> before its first episode (tags zc_probe, zc_bind, zc_pilot,
zc_main, b_bind_q1, b_bind_q2, b_bind_q2_amend1, b_scheme_main, q_probes, q_pilot, q6_main, q7_tokpractice, q8_main, q10_main, q13_main, q16_library, q17_stream, q18_self, q20_stream, q20_confirm, q24_grammar, q26_grammar_accept, q27_joint, c2_stream, f1_fresh, f1_amend1, f1_amend2, q28_bindprim, q28_amend1, d1_factored); reads marked exploratory were not. Every number on this page is read from <code>ledger/</code> at commit
<code>{esc(commit)}</code> by <code>experiments/paper_page.py</code>, and the README's copies are re-derived by
<code>experiments/audit_numbers.py</code>. Episodes are paired by task and init state; intervals come from a two-level bootstrap, tests from
the exact sign test, with Holm correction within a read. One pre-registered probe failed its own validity gate and is not used: the
policy's first chunk points at the named object only {probe['validity']['PA']:.2f} of the time even in LIBERO-Goal.</p>"""

NAV = [("abstract", "", "Abstract"), ("s1", "1", "Introduction"), ("s2", "2", "Why the policy fails new tasks"),
       ("s3", "3", "Method"), ("s4", "4", "Experiments"), ("s4-1", "4.1", "Prompt composition"), ("s4-2", "4.2", "Where"),
       ("s4-3", "4.3", "Binding"), ("s4-4", "4.4", "Folding"), ("s4-5", "4.5", "Main read"), ("s4-6", "4.6", "Prototype PoE"),
       ("s4-7", "4.7", "Streams, other policies"), ("s4-8", "4.8", "Motion primitives"), ("s4-9", "4.9", "What the units encode"),
       ("s4-10", "4.10", "Factored steps"), ("s4-11", "4.11", "Learned as a stream"), ("s6", "5", "Limitations"),
       ("s7", "", "Verification")]

CSS = r"""
/* a paper in one column: serif text, sans figures and captions, a contents rail on wide screens */
:root{--paper:#fafbfb;--ink:#161b1e;--ink2:#3b464c;--muted:#5d686f;--rule:#dce2e5;--wash:#eef2f3;
--accent:#1d5e86;--ours:#1d5e86;--base:#a35d2b;--good:#16804f;--bad:#b23f2f;--hold:#946a07;--hold-bg:#fbf5e6;
--serif:"Source Serif 4","Source Serif Pro",Georgia,serif;--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#121619;--ink:#e6eaec;--ink2:#c3cbcf;--muted:#97a3a9;
--rule:#2a3237;--wash:#1a2024;--accent:#7fb6db;--ours:#7fb6db;--base:#d99a6b;--good:#5cc38f;--bad:#e38a7b;--hold:#e0b65a;--hold-bg:#2a2414;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#121619;--ink:#e6eaec;--ink2:#c3cbcf;--muted:#97a3a9;--rule:#2a3237;--wash:#1a2024;--accent:#7fb6db;
--ours:#7fb6db;--base:#d99a6b;--good:#5cc38f;--bad:#e38a7b;--hold:#e0b65a;--hold-bg:#2a2414;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:17px/1.62 var(--serif);margin:0}
.shell{display:grid;grid-template-columns:minmax(0,1fr);max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px;gap:40px}
@media (min-width:1100px){.shell{grid-template-columns:220px minmax(0,1fr)}}
nav.toc{display:none;font:13px/1.5 var(--sans)}
@media (min-width:1100px){nav.toc{display:block;position:sticky;top:calc(env(safe-area-inset-top,0px) + 24px);align-self:start}}
nav.toc ol{list-style:none;margin:0;padding:0;display:grid;gap:4px}
nav.toc a{color:var(--muted);text-decoration:none;display:flex;gap:8px}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent)}
nav.toc .num{min-width:28px;font-variant-numeric:tabular-nums}
main{min-width:0;max-width:760px}
header.title{padding-bottom:18px;border-bottom:1px solid var(--rule);margin-bottom:8px}
header.title .kicker{font:600 12px/1.4 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}
h1{font:600 34px/1.18 var(--serif);margin:0 0 12px;text-wrap:balance;letter-spacing:-.01em}
.meta{font:13px/1.5 var(--sans);color:var(--muted);margin:0}
h2{font:600 24px/1.25 var(--serif);margin:44px 0 10px;text-wrap:balance}
h2 .sn{color:var(--accent);margin-right:10px;font-variant-numeric:tabular-nums}
h3{font:600 18px/1.35 var(--sans);margin:28px 0 6px;color:var(--ink2)}
p{margin:0 0 14px}
.abstract{background:var(--wash);padding:18px 22px;border-radius:4px;font-size:16px}
.abstract h2{margin:0 0 8px;font-size:15px;font-family:var(--sans);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.callout{border-left:3px solid var(--accent);padding:4px 0 4px 18px;margin:18px 0}
.contrib{font-size:16px;color:var(--ink2)}
.eq{font:17px/1.6 var(--serif);text-align:center;overflow-x:auto;padding:6px 0}
.eq var{font-style:italic}
figure{margin:26px 0}
@media (min-width:1100px){figure.wide{margin-right:-180px}}
figure video,figure img{width:100%;display:block;border-radius:3px;background:var(--wash)}
figcaption{font:14px/1.5 var(--sans);color:var(--muted);margin-top:8px}
figcaption b,.tcap b{color:var(--ink2)}
.chartwrap{overflow-x:auto}
svg.chart{width:100%;min-width:560px;height:auto;display:block;font-family:var(--sans)}
svg .lab{font-size:13px;fill:var(--ink2)} svg .val{font-size:12px;fill:var(--ink)} svg .dim{fill:var(--muted)}
svg .tick{font-size:11px;fill:var(--muted)} svg .rl{font-size:12px;fill:var(--ink2)} svg .rl.hl{fill:var(--ink);font-weight:600}
svg .bar.on{fill:var(--ours)} svg .bar.off{fill:var(--rule)} svg .bar.ours{fill:var(--ours)} svg .bar.base{fill:var(--base)}
svg .gridl{stroke:var(--rule);stroke-width:1} svg .span{stroke:var(--rule);stroke-width:2}
svg .dot{stroke:var(--paper);stroke-width:1.5} svg .dot.f{fill:var(--ours)} svg .dot.c{fill:var(--base)} svg .dot.p{fill:var(--muted)} svg .dot.z{fill:var(--ink)}
.tbl{margin:22px 0}
.tcap{font:14px/1.5 var(--sans);color:var(--muted);margin:0 0 6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font:14px/1.45 var(--sans)}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--ink2);padding:6px 10px 6px 0;vertical-align:bottom}
td{border-bottom:1px solid var(--rule);padding:6px 10px 6px 0;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.grid th.g{writing-mode:vertical-rl;transform:rotate(180deg);font-weight:500;font-size:12px;padding:4px 2px;text-align:left;height:120px}
table.grid td.g{text-align:center;font-variant-numeric:tabular-nums;width:34px;padding:4px 0;border:1px solid var(--paper)}
table.grid td.some{background:var(--wash)} table.grid td.habit{background:var(--ink2);color:var(--paper)}
table.grid td.hit{background:var(--good);color:var(--paper);font-weight:600}
table.grid td.rm{color:var(--muted);width:44px} table.grid td.nm{white-space:nowrap;padding-right:12px}
aside.pending{border:1.5px dashed var(--hold);background:var(--hold-bg);border-radius:4px;padding:12px 16px;margin:22px 0;font:15px/1.5 var(--sans)}
aside.pending .pl{font:600 11px/1.3 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--hold);margin:0 0 4px}
aside.pending .pt{font-weight:600;margin:0 0 4px;color:var(--ink)}
aside.pending p:last-child{margin:0;color:var(--ink2)}
code{font:14px var(--mono);background:var(--wash);padding:1px 4px;border-radius:3px}
a{color:var(--accent)}
footer{margin-top:48px;padding-top:14px;border-top:1px solid var(--rule);font:13px/1.5 var(--sans);color:var(--muted)}
@media (max-width:600px){body{font-size:16px}h1{font-size:27px}h2{font-size:21px}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
"""

toc = "".join(f'<li><a href="#{i}"><span class="num">{n}</span><span>{esc(t)}</span></a></li>' for i, n, t in NAV)
HTML = f"""<title>Read, Compose, Fold</title>
<meta name="description" content="Composing a frozen robot policy's own predictions to take on LIBERO-90 tasks it has never seen, with at most one demonstration used as a query.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{CSS}</style>
<div class="shell">
<nav class="toc" aria-label="Contents"><ol>{toc}</ol></nav>
<main>
<header class="title"><p class="kicker">lifelong-ood-learner · working paper</p>
<h1>Read, Compose, Fold: taking on tasks never seen, with no demonstration, from a frozen robot policy's own predictions</h1>
<p class="meta">pi0.5 on LIBERO-90 · 21 held-out tasks the policy fails · every number from the project's ledger at commit {esc(commit)}; boxes marked
<em>Running</em> fill in as their pre-registered reads land.</p></header>
<section class="abstract" id="abstract"><h2>Abstract</h2>{ABSTRACT}</section>
<section id="s1"><h2><span class="sn">1</span>Introduction</h2>{P_INTRO}</section>
<section id="s2"><h2><span class="sn">2</span>Why the policy fails new tasks</h2>{P_WHY}</section>
<section id="s3"><h2><span class="sn">3</span>Method: read, compose, fold, consolidate</h2>{P_METHOD}</section>
<section id="s4"><h2><span class="sn">4</span>Experiments</h2>
<p>Setup: pi0.5 (openpi's <code>pi05_libero</code>), frozen. 21 held-out LIBERO-90 tasks the policy fails, no demonstration of any of them,
20 scoring episodes per task at init states 30-49 that no method trained or practised on, the same states and noise
for every arm.</p>
<section id="s4-1"><h3 style="font-size:20px">4.1 Prompt composition</h3>{P_COMPOSE}</section>
<section id="s4-2"><h3 style="font-size:20px">4.2 Where an unread object is</h3>{P_WHERE}</section>
<section id="s4-3"><h3 style="font-size:20px">4.3 Binding a task prototype's slot</h3>{P_BIND}</section>
<section id="s4-4"><h3 style="font-size:20px">4.4 Folding the policy's own successes</h3>{P_FOLD}</section>
<section id="s4-5"><h3 style="font-size:20px">4.5 The scheme on unseen start states</h3>{P_MAIN}</section>
<section id="s4-6"><h3 style="font-size:20px">4.6 Prototype PoE planning, and where from one demonstration</h3>{P_POE}</section>
<section id="s4-7"><h3 style="font-size:20px">4.7 Streams and other policies</h3>{P_STREAM}</section>
<section id="s4-8"><h3 style="font-size:20px">4.8 Reusable motion primitives</h3>{P_PRIM}</section>
<section id="s4-9"><h3 style="font-size:20px">4.9 What the prototypes and primitives encode</h3>{P_INTERP}</section>
<section id="s4-10"><h3 style="font-size:20px">4.10 Factored steps: an object prototype and a goal prototype</h3>{P_FACTOR}</section>
<section id="s4-11"><h3 style="font-size:20px">4.11 The library learned as a stream</h3>{P_STREAMS}</section>
</section>
<section id="s6"><h2><span class="sn">5</span>Limitations</h2>{P_LIMITS}</section>
<section id="s7"><h2>Verification</h2>{P_VER}</section>
<footer>Simulation only (LIBERO-90). Commit {esc(commit)}. The program-era report is kept in the repository
(<code>experiments/site_page.py</code>).</footer>
</main></div>
"""
os.makedirs(SITE, exist_ok=True)
open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(HTML)
print(os.path.join(SITE, "index.html"), len(HTML))
