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
({q16['hypotheses']['H2']['control']}). Tokens shared across tasks bind the object more often but succeed less often.</p>"""

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
(Section 4.4), and the scheme's pre-registered read on start states nobody tuned on (Section 4.5). (5) An acceptance test on the learner's own practice that keeps the policy, the binding or the token per task; in its pre-registered read no task does worse than the frozen policy (Section 4.5). (6) A library of 4 KB object prototypes that lets later tasks with a known object type skip the demonstration (Sections 4.5, 4.7).</p>"""

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
            f"sometimes keep the wrong unit.</p>" + codes_table() + confirm_par())


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
zc_main, b_bind_q1, b_bind_q2, b_bind_q2_amend1, b_scheme_main, q_probes, q_pilot, q6_main, q7_tokpractice, q8_main, q10_main, q13_main, q16_library, q17_stream, q18_self, q20_stream, q20_confirm); reads marked exploratory were not. Every number on this page is read from <code>ledger/</code> at commit
<code>{esc(commit)}</code> by <code>experiments/paper_page.py</code>, and the README's copies are re-derived by
<code>experiments/audit_numbers.py</code>. Episodes are paired by task and init state; intervals come from a two-level bootstrap, tests from
the exact sign test, with Holm correction within a read. One pre-registered probe failed its own validity gate and is not used: the
policy's first chunk points at the named object only {probe['validity']['PA']:.2f} of the time even in LIBERO-Goal.</p>"""

NAV = [("abstract", "", "Abstract"), ("s1", "1", "Introduction"), ("s2", "2", "Why the policy fails new tasks"),
       ("s3", "3", "Method"), ("s4", "4", "Experiments"), ("s4-1", "4.1", "Prompt composition"), ("s4-2", "4.2", "Where"),
       ("s4-3", "4.3", "Binding"), ("s4-4", "4.4", "Folding"), ("s4-5", "4.5", "Main read"), ("s4-6", "4.6", "Prototype PoE"),
       ("s4-7", "4.7", "Streams, other policies"), ("s6", "5", "Limitations"),
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
svg .bar.on{fill:var(--ours)} svg .bar.off{fill:var(--rule)}
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
