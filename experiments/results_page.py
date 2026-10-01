"""The results page of the transfer plan (archive/2026-10-program/PLAN_TRANSFER.md, archive/2026-10-program/METHOD_program.md): one self-contained HTML file, every number read from
ledger/ (the rows the README quotes) or from the stream plans, charts drawn to scale as inline SVG.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/results_page.py     # -> results/robot/results_page.html
"""
import html
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                            # noqa: E402

R = os.path.join(ROOT, "results", "robot")


def row(exp, tag):
    rows = [r for r in ledger.load(exp, tag=tag) if not r.get("retired")]
    return rows[-1]["metrics"]


def fmt_bytes(b):
    for unit, s in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= s:
            return f"{b / s:.1f} {unit}"
    return f"{b} B"


def esc(s):
    return html.escape(str(s))


# ── the numbers ─────────────────────────────────────────────────────────────
M = [r for r in ledger.load("m3_stats", tag="final_23tasks") if not r.get("retired")][-1]["metrics"]["methods"]
meth = lambda k: M[k + ".prior_fails"]
v2, v2s = row("p1_bindings", "v2"), row("p1_bindings", "v2_synth")
al = row("p1_bindings", "alloc_v2")
p3c, p3r2 = row("p3_compose", "p3c"), row("p3_compose", "p3c_reader2")
c1 = row("c1_grammar", "read")
p5 = row("p5_stream", "read")
p5v, p5m, p5p = row("p5_stream", "validated"), row("p5_stream", "validated_multi"), row("p5_stream", "practice")
cf = row("p5_stream", "confirm")
s1, d1 = row("p1_bindings", "seed1"), row("p1_bindings", "demo1_v2")
dev = row("p5_stream", "dev_fixes")
p2 = row("p2_negatives", "read")
p4 = row("p4_growth", "rollouts")
ty = row("p6_reader", "type_transfer")
yw, ywr = row("p6_reader", "yaw"), row("p6_reader", "yaw_read")
stove = row("p0_audit", "stove_knob")
rest = row("p0_audit", "demo_at_rest")
gram, sto = row("grammar", "readout"), row("storage", "read")
rc, co, col = row("p6_reader", "compress_read"), row("p6_reader", "compress_offline"), row("p6_reader", "compress_offline_large")
gplan = json.load(open(os.path.join(R, "p5_stream_plan_gated.json")))
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
audit = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "audit_numbers.py")], capture_output=True, text=True,
                       cwd=ROOT).stdout.strip().splitlines()[-1]

# ── figure 1: success against storage (21 held-out tasks, 420 paired episodes per method) ────────────
F1 = [  # (label, method key, family, label side)
    ("v2 bindings, simulator poses", "ours_phasebind2", "bind", "r"),
    ("v2 bindings, learned reader", "ours_phasebind2_synth", "bind", "r"),
    ("v1 bindings, simulator poses", "ours_phasebind_nofold", "bind", "r"),
    ("borrowing + fold", "ours_dim1_last2", "bind", "r"),
    ("demo replay", "base_playback", "bind", "r"),
    ("one-demo LoRA", "practice_lora", "lora", "l"),
    ("LoRA + failure negatives", "nft_bal", "lora", "l"),
    ("LoRA on v2-generated demos", "gen_lora_v2", "lora", "l"),
    ("LoRA rank 64", "base_lora_r64", "lora", "r"),
    ("full fine-tuning", "base_fullft", "lora", "b"),
]
pts1 = [(lab, meth(k)["k"], meth(k)["bytes"], fam, side) for lab, k, fam, side in F1]
pts1.append(("practice allocator", al["AL1"]["x"], al["bytes_total"], "alloc", "r"))


def scatter_success_bytes():
    W, H, L, Rm, T, B = 860, 430, 64, 24, 20, 52
    x0, x1 = 4.0, 11.0                    # log10 bytes
    y0, y1 = 150, 420
    X = lambda b: L + (math.log10(b) - x0) / (x1 - x0) * (W - L - Rm)
    Y = lambda k: T + (y1 - k) / (y1 - y0) * (H - T - B)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f1t" class="chart">']
    for k in range(150, 421, 50):
        g.append(f'<line x1="{L}" x2="{W - Rm}" y1="{Y(k):.1f}" y2="{Y(k):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{Y(k) + 4:.1f}" class="tick" text-anchor="end">{k}</text>')
    for e, lab in ((4, "10 KB"), (5, "100 KB"), (6, "1 MB"), (7, "10 MB"), (8, "100 MB"), (9, "1 GB"), (10, "10 GB"), (11, "100 GB")):
        g.append(f'<line x1="{X(10 ** e):.1f}" x2="{X(10 ** e):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                 f'<text x="{X(10 ** e):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{lab}</text>')
    g.append(f'<text x="{(L + W - Rm) / 2:.0f}" y="{H - 10}" class="axis" text-anchor="middle">storage for all 21 tasks (log scale)</text>')
    g.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(T + H - B) / 2:.0f})">successes of 420</text>')
    for lab, k, b, fam, side in pts1:
        cx, cy = X(max(b, 1e4)), Y(k)
        dx = 10 if side == "r" else -10
        anc = "start" if side == "r" else "end"
        dy = 4
        if side == "b":                                   # below the point, ending at it (its row is taken on both sides)
            dx, anc, dy = 6, "end", 22
        tip = f"{lab}: {k} of 420 at {fmt_bytes(b)}"
        g.append(f'<g class="pt" tabindex="0" data-tip="{esc(tip)}"><circle cx="{cx:.1f}" cy="{cy:.1f}" r="14" class="hit"/>'
                 f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" class="dot {fam}"/></g>'
                 f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" class="lab" text-anchor="{anc}">{esc(lab)}</text>')
    g.append('</svg>')
    return "".join(g)


def cumulative_demos():
    W, H, L, Rm, T, B = 860, 330, 56, 130, 16, 46
    n = len(gplan["A"]["steps"])
    X = lambda i: L + (i - 1) / (n - 1) * (W - L - Rm)
    Y = lambda d: T + (n - d) / n * (H - T - B)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f3t" class="chart" id="cum">']
    for d in range(0, n + 1, 10):
        g.append(f'<line x1="{L}" x2="{W - Rm}" y1="{Y(d):.1f}" y2="{Y(d):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{Y(d) + 4:.1f}" class="tick" text-anchor="end">{d}</text>')
    for i in (1, 10, 20, 30, 40, 50):
        g.append(f'<text x="{X(i):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{i}</text>')
    g.append(f'<text x="{(L + W - Rm) / 2:.0f}" y="{H - 8}" class="axis" text-anchor="middle">tasks arrived</text>')
    g.append(f'<line x1="{X(1):.1f}" y1="{Y(1):.1f}" x2="{X(n):.1f}" y2="{Y(n):.1f}" class="ref"/>'
             f'<text x="{X(n) + 8:.1f}" y="{Y(n) + 4:.1f}" class="lab">a demo per task</text>')
    for o, cls in (("A", "s1"), ("B", "s2")):
        cum = gplan[o]["cumulative_demos"]
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(i + 1):.1f},{Y(c):.1f}" for i, c in enumerate(cum))
        g.append(f'<path d="{path}" class="line {cls}"/><circle cx="{X(n):.1f}" cy="{Y(cum[-1]):.1f}" r="4.5" class="dot {cls} ring"/>'
                 f'<text x="{X(n) + 8:.1f}" y="{Y(cum[-1]) + (14 if o == "B" else -4):.1f}" class="lab">order {o}: {cum[-1]}</text>')
    g.append(f'<rect x="{L}" y="{T}" width="{W - L - Rm}" height="{H - T - B}" class="hover-band" data-n="{n}" data-l="{L}" data-w="{W - L - Rm}"/>')
    g.append(f'<line class="xhair" x1="0" x2="0" y1="{T}" y2="{H - B}" visibility="hidden"/>')
    g.append('</svg>')
    return "".join(g)


def tradeoff():
    V = [("every task its own demo", {"A": (50, p5v["orders"]["A"]["all_demo"]), "B": (50, p5v["orders"]["B"]["all_demo"])}),
         ("composed as registered", {o: (p5["orders"][o]["demos"], p5["orders"][o]["stream"]) for o in "AB"}),
         ("composition checked in simulation", {o: (p5v["orders"][o]["demos"], p5v["orders"][o]["stream"]) for o in "AB"}),
         ("several sources, checked in simulation", {o: (p5m["orders"][o]["demos"], p5m["orders"][o]["stream"]) for o in "AB"}),
         ("composition checked by practice", {o: (p5p["orders"][o]["demos"], p5p["orders"][o]["stream"]) for o in "AB"}),
         ("checked by practice, fresh orders", {"A": (cf["orders"]["C"]["demos"], cf["orders"]["C"]["stream"]),
                                                "B": (cf["orders"]["D"]["demos"], cf["orders"]["D"]["stream"])})]
    W, H, L, Rm, T, B = 860, 380, 64, 250, 16, 50
    x0, x1, y0, y1 = 30, 52, 680, 900
    X = lambda d: L + (d - x0) / (x1 - x0) * (W - L - Rm)
    Y = lambda k: T + (y1 - k) / (y1 - y0) * (H - T - B)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f4t" class="chart">']
    for k in range(700, 901, 50):
        g.append(f'<line x1="{L}" x2="{W - Rm}" y1="{Y(k):.1f}" y2="{Y(k):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{Y(k) + 4:.1f}" class="tick" text-anchor="end">{k}</text>')
    for d in range(30, 51, 5):
        g.append(f'<line x1="{X(d):.1f}" x2="{X(d):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                 f'<text x="{X(d):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{d}</text>')
    g.append(f'<text x="{(L + W - Rm) / 2:.0f}" y="{H - 8}" class="axis" text-anchor="middle">demos requested for 50 tasks</text>')
    g.append(f'<text x="14" y="{(T + H - B) / 2:.0f}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(T + H - B) / 2:.0f})">stream successes of 1,000</text>')
    want = sorted(((Y(p["A"][1]) + Y(p["B"][1])) / 2, i) for i, (_, p) in enumerate(V))
    ly_of, last = {}, -1e9
    for y, i in want:                                      # label rows at least 18 px apart, in the points' order
        last = max(y, last + 18)
        ly_of[i] = last
    for i, (lab, pts) in enumerate(V):
        (da, ka), (db, kb) = pts["A"], pts["B"]
        g.append(f'<line x1="{X(da):.1f}" y1="{Y(ka):.1f}" x2="{X(db):.1f}" y2="{Y(kb):.1f}" class="pair"/>')
        for o, cls in (("A", "s1"), ("B", "s2")):
            d, k = pts[o]
            on = {"A": "C", "B": "D"}[o] if lab.endswith("fresh orders") else o
            g.append(f'<g class="pt" tabindex="0" data-tip="{esc(f"{lab}, order {on}: {k:,} of 1,000 with {d} demos")}">'
                     f'<circle cx="{X(d):.1f}" cy="{Y(k):.1f}" r="14" class="hit"/><circle cx="{X(d):.1f}" cy="{Y(k):.1f}" r="5" class="dot {cls} ring"/></g>')
        ly = ly_of[i] + 4
        g.append(f'<text x="{W - Rm + 12}" y="{ly:.1f}" class="lab">{esc(lab)}</text>'
                 f'<line x1="{max(X(da), X(db)) + 8:.1f}" y1="{(Y(ka) + Y(kb)) / 2:.1f}" x2="{W - Rm + 8}" y2="{ly - 4:.1f}" class="leader"/>')
    g.append('</svg>')
    return "".join(g)


def storage_bars():
    """Bytes a task keeps (log scale), and what 50 tasks keep."""
    med = sto["median"]
    per = [("rank-16 LoRA", sto["lora_bytes"], "s2"), ("raw LIBERO demonstration", sto["raw_demo_bytes"], "s2"),
           ("pose reader, full ridge", med["reader_full"], "s1"), ("pose reader, 512-direction head", med["reader_pca512"], "s1"),
           ("phases (the bindings)", med["program"], "s1")]
    W, L, Rm, T, rowh = 860, 250, 90, 10, 34
    H = T + rowh * len(per) + 44
    x0, x1 = 3.0, 8.0
    X = lambda b: L + (math.log10(b) - x0) / (x1 - x0) * (W - L - Rm)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f5t" class="chart">']
    for e, lab in ((3, "1 KB"), (4, "10 KB"), (5, "100 KB"), (6, "1 MB"), (7, "10 MB"), (8, "100 MB")):
        g.append(f'<line x1="{X(10 ** e):.1f}" x2="{X(10 ** e):.1f}" y1="{T}" y2="{H - 34}" class="grid"/>'
                 f'<text x="{X(10 ** e):.1f}" y="{H - 16}" class="tick" text-anchor="middle">{lab}</text>')
    for i, (lab, b, cls) in enumerate(per):
        y = T + i * rowh + 8
        g.append(f'<text x="{L - 10}" y="{y + 13}" class="lab" text-anchor="end">{esc(lab)}</text>'
                 f'<g class="pt" tabindex="0" data-tip="{esc(f"{lab}: {fmt_bytes(b)} per task")}"><rect x="{L}" y="{y}" width="{X(b) - L:.1f}" height="18" rx="3" class="{cls}"/></g>'
                 f'<text x="{X(b) + 8:.1f}" y="{y + 13}" class="tick">{fmt_bytes(b)}</text>')
    g.append('</svg>')
    return "".join(g)


def _row_or_none(exp, tag):
    rows = [r for r in ledger.load(exp, tag=tag) if not r.get("retired")]
    return rows[-1]["metrics"] if rows else None


dcr = _row_or_none("p5_stream", "dev_carries")
scheme = _row_or_none("p5_stream", "scheme")
CARRY_NAMES = {"streamc": "the place source's route", "streamh": "the prior carries", "streamhe": "the prior carries, events on arrival",
               "streamte": "lift-across-descend, on arrival", "streamtle": "the same, lifted by the source's own lift"}


def phase_bars():
    """Where each carry's practice episodes stop along the grammar (reach, lift, bring, place), stacked to 100%."""
    po = dcr["phase_outcomes"]
    order = [k for k in CARRY_NAMES if k in po]
    stages = [("reach", "var(--bad)"), ("lift", "#c77b2b"), ("bring", "var(--mix)"), ("place", "#8f8f3a"), ("success", "var(--good)")]
    W, L, Rm, T, rowh = 860, 270, 20, 26, 30
    H = T + rowh * len(order) + 30
    X = lambda f: L + f * (W - L - Rm)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f7t" class="chart">']
    for i, (st, col) in enumerate(stages):
        g.append(f'<rect x="{L + i * 118}" y="4" width="10" height="10" fill="{col}"/><text x="{L + i * 118 + 16}" y="13" class="tick">{"failed at " + st if st != "success" else st}</text>')
    for j, k in enumerate(order):
        v = po[k]
        n = v["episodes"]
        y = T + j * rowh
        g.append(f'<text x="{L - 10}" y="{y + 15}" class="lab" text-anchor="end">{esc(CARRY_NAMES[k])}</text>')
        acc = 0
        for st, col in stages:
            c = v["failed_at"][st]
            if c:
                tip = f"{CARRY_NAMES[k]}: {c} of {n} practice episodes {'failed at ' + st if st != 'success' else 'succeeded'}"
                g.append(f'<g class="pt" tabindex="0" data-tip="{esc(tip)}"><rect x="{X(acc / n):.1f}" y="{y}" width="{X((acc + c) / n) - X(acc / n):.1f}" height="20" fill="{col}"/></g>')
            acc += c
    g.append('</svg>')
    return "".join(g)


def binding_grid():
    """The pool as a grid of object types x goal relations; each task a cell: its keys held by no other task, or composable from
    other tasks' bindings, marked by whether some composition of it passes practice (>= 8 of 10) with some carry."""
    import collections
    import glob
    import re
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    comp = {t: v for t, v in pool.items() if v["composable"]}
    types = [k for k, _ in collections.Counter(v["object_type"] for v in comp.values()).most_common()]
    rels = [k for k, _ in collections.Counter(v["goal_key"] for v in comp.values()).most_common()]
    passed, tried = set(), set()
    for f in glob.glob(os.path.join(R, "dev_*_*_stream*_*_*.json")):
        m = re.match(r".*dev_[a-z0-9]+?_(\d+)_stream[a-z]*_\d+_\d+\.json", f)
        if not m or "smoke" in f or "tdiag" in f:
            continue
        t = m.group(1)
        tried.add(t)
        e = json.load(open(f))["tasks"][t]["episodes"]
        if sum(x["success"] for x in e) >= 8:
            passed.add(t)
    both = {t for t, v in comp.items() if any(u != t and w["object_type"] == v["object_type"] for u, w in comp.items())
            and any(u != t and w["goal_key"] == v["goal_key"] for u, w in comp.items())}
    cw, chh, L, T = 26, 22, 170, 150
    W, H = L + cw * len(rels) + 10, T + chh * len(types) + 10
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f8t" class="chart" style="min-width:{W}px">']
    short = lambda r: re.sub(r"\bOBJ\b ?", "", r).replace("_contain_region", "").replace("_region", "").replace("wooden_", "").replace("_", " ")
    for j, r in enumerate(rels):
        x = L + j * cw + cw / 2
        g.append(f'<text x="{x:.1f}" y="{T - 6}" class="tick" transform="rotate(-55 {x:.1f} {T - 6})">{esc(short(r)[:26])}</text>')
    for i, ty in enumerate(types):
        y = T + i * chh
        g.append(f'<text x="{L - 8}" y="{y + 15}" class="tick" text-anchor="end">{esc(ty.replace("_", " "))}</text>')
        for j in range(len(rels)):
            g.append(f'<rect x="{L + j * cw + 2}" y="{y + 2}" width="{cw - 4}" height="{chh - 4}" rx="2" class="gcell"/>')
    for t, v in comp.items():
        i, j = types.index(v["object_type"]), rels.index(v["goal_key"])
        cls = "gpass" if t in passed else ("gfail" if t in both else "gown")
        state = ("composed from other tasks, passes practice" if t in passed else
                 "composable, no composition passed practice" if t in both else "its keys are held by no other task")
        g.append(f'<g class="pt" tabindex="0" data-tip="{esc(f"task {t}: {v["language"]} ({state})")}"><rect x="{L + j * cw + 2}" y="{T + i * chh + 2}" '
                 f'width="{cw - 4}" height="{chh - 4}" rx="2" class="{cls}"/></g>')
    g.append('</svg>')
    return "".join(g), len(comp), len(both), len(passed & both), len(types), len(rels)


def verb_glyphs():
    """Each coarse prototype's mode drawn as the hand path it commands over ten steps: horizontal travel (along its own direction)
    against height, one shared scale, orange where the gripper is closed."""
    sel = [("0.55", c, v) for c, v in sorted(gram["prototypes"]["0.55"].items(), key=lambda kv: -kv[1]["frames"])]
    sel += [("0.4", c, v) for c, v in sorted(gram["prototypes"]["0.4"].items(), key=lambda kv: -kv[1]["tasks"])[:5]]
    pw, ph, gap, top = 130, 120, 12, 18
    W = len(sel) * (pw + gap)
    H = ph + top + 46
    sc = 2.2                                               # px per commanded cm
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f6t" class="chart" style="min-width:{min(W, 900)}px">']
    for i, (u, c, v) in enumerate(sel):
        x0 = i * (pw + gap)
        P = [[0.0, 0.0, 0.0]] + v["mode_path_cm"]
        d = [P[-1][0], P[-1][1]]
        nd = math.hypot(*d) or 1.0
        hz = [((p[0] * d[0] + p[1] * d[1]) / nd, p[2]) for p in P]
        cx, cy = x0 + 20, top + ph / 2
        g.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" rx="4" class="glyph"/>')
        g.append(f'<line x1="{x0 + 8}" x2="{x0 + pw - 8}" y1="{cy:.1f}" y2="{cy:.1f}" class="grid"/>')
        for j in range(1, len(hz)):
            cls = "s2" if v["mode_grip"][j - 1] > 0 else "s1"
            xa, ya = cx + hz[j - 1][0] * sc, cy - hz[j - 1][1] * sc
            xb, yb = cx + hz[j][0] * sc, cy - hz[j][1] * sc
            g.append(f'<line x1="{min(max(xa, x0 + 4), x0 + pw - 4):.1f}" y1="{ya:.1f}" x2="{min(max(xb, x0 + 4), x0 + pw - 4):.1f}" y2="{yb:.1f}" class="vpath {cls}"/>')
        name = v["name"].split(" (")[0]
        tip = f"t = {u}, prototype {c}: {name}; {v['tasks']} tasks, {v['frames']} frames"
        g.append(f'<g class="pt" tabindex="0" data-tip="{esc(tip)}"><rect x="{x0}" y="{top}" width="{pw}" height="{ph}" class="hit"/></g>')
        g.append(f'<text x="{x0 + pw / 2}" y="{top - 6}" class="tick" text-anchor="middle">t = {u} · {v["tasks"]} tasks</text>')
        g.append(f'<text x="{x0 + pw / 2}" y="{top + ph + 18}" class="lab" text-anchor="middle">{esc(name)}</text>')
    g.append('</svg>')
    return "".join(g)


def chip(state):
    icon = {"holds": "✓", "fails": "✕", "mixed": "◐"}[state]
    return f'<span class="chip {state}"><span aria-hidden="true">{icon}</span> {state}</span>'


claims = [
    ("C2 · object bindings", "holds",
     f"The demo's phases, re-anchored on the objects and followed by the frozen pi0.5, succeed {v2['V1']['x']} of 420 times "
     f"(learned pose reader, no simulator state at test: {v2s['V5']['x']}), against {v2['V2']['control']} for the best LoRA."),
    ("C1 · grammar from prototypes", "holds",
     f"The coarse level t = 0.55 changes label at {c1['levels']['0.55']['recall']:.2f} of the phase events "
     f"(random {c1['levels']['0.55']['random_recall']:.2f}); {c1['levels']['0.55']['precision']:.2f} of its changes sit at an event."),
    ("C3 · composition transfers", "mixed",
     f"Tasks built from other tasks' phases match their own demo when layouts match ({p3c['C1']['x']} vs {p3c['C1']['control']} of 120; "
     f"no demo, no simulator state: {p3r2['R2']['x']}). Across the 50-task pool unchecked compositions fail."),
    ("Interpretable parts", "holds",
     f"Coarse prototypes read as verbs; the carry's prototype string groups tasks by goal relation "
     f"({gram['string_by_key']['carry']['same_key_median']:.2f} vs {gram['string_by_key']['carry']['other_key_median']:.2f}, "
     f"p {gram['string_by_key']['carry']['perm_p']:.3f}) and the grasp's geometry by object type "
     f"({gram['grasp_offset_cm']['within_type_median']:.1f} vs {gram['grasp_offset_cm']['across_type_median']:.1f} cm)."),
    ("Storage", "holds",
     f"A task keeps {fmt_bytes(sto['median']['program'] + sto['median']['reader_pca512'])} (phases + a reader head, non-inferior: "
     f"{rc['RC1']['x']} vs {rc['RC1']['control']} of 1,000) against {fmt_bytes(sto['lora_bytes'])} for a LoRA."),
    ("C5 · practice picks the form", "holds",
     f"Phases, prototype borrowing or a LoRA, chosen by 10 practice episodes: {al['AL1']['x']} of 420 at {fmt_bytes(al['bytes_total'])}."),
    ("C4 · continual learning", "mixed",
     f"The binding vocabulary grows ({p5['S1']['only_late']} tasks compose only when late, {p5['S1']['only_early']} only when early), "
     f"but composed arrivals score {p5['S2']['x']} vs {p5['S2']['control']} of {p5['S2']['n']}; checked by practice, "
     f"{p5p['composed_vs_own']['x']} vs {p5p['composed_vs_own']['control']} of {p5p['composed_vs_own']['n']} with "
     f"{p5p['orders']['A']['demos']} and {p5p['orders']['B']['demos']} demos (exploratory); the pre-registered confirmation on fresh orders "
     f"gets {cf['composed_vs_own']['x']} vs {cf['composed_vs_own']['control']} of {cf['composed_vs_own']['n']}, not shown non-inferior."),
]

p3lang = json.load(open(os.path.join(R, "p3c_table.json")))["table"]
p3rows = "".join(
    f"<tr><td class='task'>{esc(p3lang[t]['language'])}</td><td class='n'>{p3c['per_task'][t]['own']}</td><td class='n'>{p3c['per_task'][t]['c']}</td>"
    f"<td class='n'>{p3r2['per_task'][t]['crp']}</td><td class='n'>{p3c['per_task'][t]['sib']}</td></tr>" for t in sorted(p3c["per_task"], key=int))

levels = "".join(
    f"<tr{' class=hl' if u == '0.55' else ''}><td class='n'>{float(u):.2f}</td><td class='n'>{d['prototypes']:,}</td><td class='n'>{d['change_points_per_demo']:.1f}</td>"
    f"<td class='n'>{d['recall']:.2f} <span class='dim'>({d['random_recall']:.2f})</span></td>"
    f"<td class='n'>{d['precision']:.2f} <span class='dim'>({d['random_precision']:.2f})</span></td></tr>" for u, d in c1["levels"].items())


gp = gram["programs"]
EX = ("48", "58", "13", "3", "39")
prog_rows = "".join(f"<tr><td class='task'>{esc(gp[t]['language'])}</td><td>{esc(gp[t]['sentence'])}</td><td class='n'>{gp[t]['bytes']:,}</td></tr>" for t in EX if t in gp)
p55 = sorted(gram["prototypes"]["0.55"].items(), key=lambda kv: -kv[1]["frames"])
p4v = sorted(gram["prototypes"]["0.4"].items(), key=lambda kv: -kv[1]["tasks"])[:8]
verb_rows = "".join(f"<tr{' class=hl' if u == '0.55' else ''}><td class='n'>{u}</td><td>{esc(v['name'])}</td><td class='n'>{v['tasks']}</td><td class='n'>{v['speed_cm_per_step']:.2f}</td><td class='n'>{v['dz_cm_per_step']:+.2f}</td></tr>"
                    for u, lst in (("0.55", p55), ("0.4", p4v)) for c, v in lst)
sk = gram["string_by_key"]
key_rows = (f"<tr><td>carry: prototype string</td><td class='n'>{sk['carry']['same_key_median']:.2f}</td><td class='n'>{sk['carry']['other_key_median']:.2f}</td><td class='n'>{sk['carry']['perm_p']:.3f}</td></tr>"
            f"<tr><td>grasp: prototype string</td><td class='n'>{sk['grasp']['same_key_median']:.2f}</td><td class='n'>{sk['grasp']['other_key_median']:.2f}</td><td class='n'>{sk['grasp']['perm_p']:.2f}</td></tr>"
            f"<tr><td>grasp: binding geometry (cm)</td><td class='n'>{gram['grasp_offset_cm']['within_type_median']:.1f}</td><td class='n'>{gram['grasp_offset_cm']['across_type_median']:.1f}</td><td class='n'>{gram['grasp_geometry_perm_p']:.4f}</td></tr>")

negatives = [
    ("Sharing weights across tasks", "one fold for all tasks, the other half's fold, family pooling and family pretraining all lose to per-task weights"),
    ("Failure negatives at the failed phase", f"{p2['all_21']['phase']} against {p2['all_21']['stall']} of 420 for negatives at the stall"),
    ("Growing the library with successes", f"{p4['grown']} against {p4['base']} of 120"),
    ("Warping a composed carry", f"{dev['totals']['streamw']} against {dev['totals']['streamc']} of {dev['n']} on practice states"),
    ("A generic lift-move transport", f"{dev['totals']['streamt']} against {dev['totals']['streamc']} of {dev['n']}: the route into a drawer or onto a shelf is part of the place skill"),
    ("One pose reader per object type", f"fixtures {ty['fixture_median_cm']:.2f} cm, movable objects {ty['movable_median_cm']:.2f} cm on unseen scenes"),
    ("Reading the books' yaw", f"{yw['median_deg']:.1f} degrees against {yw['fixed_median_deg']:.1f} fixed, yet {ywr['Y1']['x']} against {ywr['Y1']['control']} of 220"),
]
neg_html = "".join(f"<li><strong>{esc(a)}</strong><span>{esc(b)}</span></li>" for a, b in negatives)

steps = [
    ("Grammar", "The demo is cut into an approach, a carry and a release at the object events; the coarse prototype level marks the same boundaries."),
    ("Bindings", "Each phase is expressed relative to an object: the approach in the grasped object's frame, the carry and release in the receptacle's."),
    ("Perception", "A per-task reader, trained only on simulator-rendered placements of the task's scene, reads where the objects are from the first frame."),
    ("Execution", "At each query the phases are re-anchored on those poses and the frozen pi0.5 is sampled through the next ten planned actions as a guide."),
    ("Composition", "A task with no demo takes the grasp phase of a task with the same object type and the place phases of a task with the same goal."),
    ("Practice", "Ten practice episodes decide what a task keeps: its phases, prototype borrowing (where the prior's own skill is needed) or a LoRA."),
]
steps_html = "".join(f"<li><h4>{esc(a)}</h4><p>{esc(b)}</p></li>" for a, b in steps)

PAGE = f"""<title>Grammar × Bindings</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Instrument+Serif&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column of findings; each figure sits on its own plate with its ledger source underneath */
:root {{
  --bg: #f3f6f7; --plate: #ffffff; --ink: #0f1a1f; --ink-2: #44535c; --muted: #75848d; --rule: #d9e1e5; --grid: #e7ecef;
  --accent: #1c63ad; --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --ref: #9aa7af;
  --good: #0ca30c; --bad: #d03b3b; --mix: #b27b00; --chip-bg: #eef2f4;
  --display: "Instrument Serif", "Iowan Old Style", Georgia, serif;
  --sans: "Instrument Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1417; --plate: #151c20; --ink: #e9eef1; --ink-2: #b1bec6; --muted: #83929b; --rule: #27323a; --grid: #202a30;
  --accent: #74aeea; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --ref: #5f6d75; --mix: #e0a93a; --chip-bg: #1d262c; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0f1417; --plate: #151c20; --ink: #e9eef1; --ink-2: #b1bec6; --muted: #83929b; --rule: #27323a; --grid: #202a30;
  --accent: #74aeea; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --ref: #5f6d75; --mix: #e0a93a; --chip-bg: #1d262c; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.6 var(--sans); }}
main {{ max-width: 920px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; display: grid; gap: 56px; }}
header {{ display: grid; gap: 14px; }}
.eyebrow {{ font: 500 12px/1.4 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 400 clamp(44px, 8vw, 76px)/1 var(--display); letter-spacing: -.01em; margin: 0; text-wrap: balance; }}
h1 em {{ font-style: italic; color: var(--accent); }}
.lead {{ font-size: 19px; line-height: 1.55; color: var(--ink-2); max-width: 64ch; margin: 0; }}
h2 {{ font: 400 34px/1.15 var(--display); margin: 0 0 6px; text-wrap: balance; }}
h3 {{ font: 600 15px/1.4 var(--sans); margin: 0; }}
section {{ display: grid; gap: 16px; min-width: 0; }}
section > p {{ margin: 0; color: var(--ink-2); max-width: 68ch; }}
.claims {{ display: grid; gap: 0; border-top: 1px solid var(--rule); }}
.claim {{ display: grid; grid-template-columns: minmax(0, 15rem) 6.5rem minmax(0, 1fr); gap: 16px; padding-block: 14px; border-bottom: 1px solid var(--rule); align-items: baseline; }}
.claim h3 {{ font-weight: 600; }}
.claim p {{ margin: 0; color: var(--ink-2); font-size: 15px; }}
.chip {{ display: inline-flex; gap: 6px; align-items: center; font: 500 12px/1 var(--mono); padding: 5px 8px; border-radius: 4px; background: var(--chip-bg); width: max-content; }}
.chip.holds span {{ color: var(--good); }} .chip.fails span {{ color: var(--bad); }} .chip.mixed span {{ color: var(--mix); }}
.plate {{ background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; padding: 20px 20px 14px; display: grid; gap: 12px; min-width: 0; }}
.plate .cap {{ font-size: 14px; color: var(--ink-2); margin: 0; }}
.plate .cap strong {{ color: var(--ink); font-weight: 600; }}
.source {{ font: 12px/1.5 var(--mono); color: var(--muted); }}
.chartwrap {{ overflow-x: auto; }}
.chart {{ width: 100%; min-width: 620px; height: auto; display: block; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .tick {{ font: 12px var(--mono); fill: var(--muted); font-variant-numeric: tabular-nums; }}
.chart .axis {{ font: 12px var(--sans); fill: var(--ink-2); }}
.chart .lab {{ font: 13px var(--sans); fill: var(--ink-2); }}
.chart .dot {{ stroke: var(--plate); stroke-width: 2; }}
.chart .bind, .chart .s1 {{ fill: var(--s1); }} .chart .lora, .chart .s2 {{ fill: var(--s2); }} .chart .alloc {{ fill: var(--s3); }}
.chart .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.chart path.s1 {{ stroke: var(--s1); fill: none; }} .chart path.s2 {{ stroke: var(--s2); fill: none; }}
.chart .ref {{ stroke: var(--ref); stroke-width: 1.5; }}
.chart .pair {{ stroke: var(--rule); stroke-width: 2; }}
.chart .leader {{ stroke: var(--grid); stroke-width: 1; }}
.chart .hit {{ fill: transparent; }}
.chart .glyph {{ fill: none; stroke: var(--rule); }}
.chart .gcell {{ fill: var(--grid); opacity: .5; }} .chart .gpass {{ fill: var(--good); }} .chart .gfail {{ fill: var(--mix); }} .chart .gown {{ fill: var(--ref); }}
.chart .vpath {{ stroke-width: 3; stroke-linecap: round; }} .chart .vpath.s1 {{ stroke: var(--s1); }} .chart .vpath.s2 {{ stroke: var(--s2); }}
.chart .pt {{ cursor: default; outline: none; }}
.chart .pt:hover .dot, .chart .pt:focus-visible .dot {{ stroke: var(--ink); }}
.chart .hover-band {{ fill: transparent; }}
.chart .xhair {{ stroke: var(--muted); stroke-width: 1; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: var(--ink-2); }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.legend i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.legend i.l {{ width: 16px; height: 2px; border-radius: 1px; }}
.k1 {{ background: var(--s1); }} .k2 {{ background: var(--s2); }} .k3 {{ background: var(--s3); }} .kr {{ background: var(--ref); }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th {{ text-align: left; font: 500 12px/1.3 var(--mono); letter-spacing: .03em; color: var(--muted); padding: 8px 10px; border-bottom: 1px solid var(--rule); vertical-align: bottom; }}
td {{ padding: 8px 10px; border-bottom: 1px solid var(--grid); }}
td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }}
td.task {{ color: var(--ink-2); }}
tr.hl td {{ background: var(--chip-bg); color: var(--ink); }}
.dim {{ color: var(--muted); }}
ol.steps {{ list-style: none; counter-reset: s; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
ol.steps li {{ counter-increment: s; background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; }}
ol.steps li::before {{ content: counter(s); font: 500 12px var(--mono); color: var(--muted); }}
ol.steps h4 {{ margin: 2px 0 4px; font: 600 15px var(--sans); }}
ol.steps p {{ margin: 0; font-size: 14px; color: var(--ink-2); }}
ul.neg {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 0; border-top: 1px solid var(--rule); }}
ul.neg li {{ display: grid; grid-template-columns: minmax(0, 17rem) minmax(0, 1fr); gap: 16px; padding-block: 10px; border-bottom: 1px solid var(--rule); font-size: 15px; }}
ul.neg span {{ color: var(--ink-2); }}
.tip {{ position: fixed; pointer-events: none; background: var(--plate); color: var(--ink); border: 1px solid var(--rule); border-radius: 4px; padding: 6px 9px; font: 13px var(--sans); box-shadow: 0 2px 10px rgba(0,0,0,.12); max-width: 320px; }}
footer {{ font: 12px/1.6 var(--mono); color: var(--muted); border-top: 1px solid var(--rule); padding-top: 16px; }}
@media (max-width: 640px) {{
  .claim {{ grid-template-columns: 1fr; gap: 6px; }}
  ul.neg li {{ grid-template-columns: 1fr; gap: 2px; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
<main>
<header>
  <div class="eyebrow">lifelong-ood-learner · frozen pi0.5 · LIBERO-90 · one demo per task</div>
  <h1>Grammar <em>×</em> bindings</h1>
  <p class="lead">A task is a phase grammar (reach, carry, release) bound to particular objects and places. Keeping the policy frozen and storing
  only the bindings of one demonstration turns out to beat fine-tuning. The same bindings compose into new tasks when the scene's layout fits,
  and a few practice episodes decide what a task keeps.</p>
</header>

<section>
  <h2>What held</h2>
  <div class="claims">{"".join(f'<div class="claim"><h3>{esc(a)}</h3>{chip(s)}<p>{esc(b)}</p></div>' for a, s, b in claims)}</div>
</section>

<section>
  <h2>Success against storage</h2>
  <p>Twenty-one held-out LIBERO-90 tasks the prior fails, one demonstration each, twenty scoring episodes per task on init states it never practised.</p>
  <div class="plate">
    <div class="legend"><span><i class="k1"></i>no trained weights</span><span><i class="k2"></i>trained weights</span><span><i class="k3"></i>practice picks per task</span></div>
    <div class="chartwrap">{scatter_success_bytes()}</div>
    <p class="cap" id="f1t"><strong>Bindings through the frozen prior reach {v2['V1']['x']} of 420 at {fmt_bytes(meth('ours_phasebind2')['bytes'])} for all 21 tasks.</strong>
    A learned reader in place of the simulator's poses keeps {v2s['V5']['x']} (non-inferior, lower bound {v2s['V5']['diff']['lo90']:+.3f}); every LoRA variant stores {fmt_bytes(meth('practice_lora')['bytes'])} or more.</p>
    <p class="cap"><strong>Both replicate.</strong> With another sampling seed: v2 {s1['R1']['x']}, the reader {s1['R2']['x']}, the allocator {s1['R3']['x']},
    the LoRA with failure negatives {s1['R1']['control']}. With a second recorded demo (20 tasks): v2 {d1['D1']['x']} of {d1['D1']['n']} against
    {d1['D1']['control']} for the LoRA trained on that demo.</p>
    <div class="source">ledger/m3_stats final_23tasks · ledger/p1_bindings v2, v2_synth, alloc_v2, seed1, demo1_v2 · the v1 row predates the fix for objects still falling in the demos' first frames (up to {rest['max_cm']:.1f} cm)</div>
  </div>
</section>

<section>
  <h2>The method</h2>
  <ol class="steps">{steps_html}</ol>
</section>

<section>
  <h2>The grammar comes from the prototypes</h2>
  <p>Every frame of the 50 pool demos, labelled at each level of the 40-task prototype tree, then compared with the phase events the bindings use (within one policy query).</p>
  <div class="plate">
    <div class="tablewrap"><table><thead><tr><th class="n">level t</th><th class="n">prototypes</th><th class="n">changes per demo</th><th class="n">events at a change (random)</th><th class="n">changes at an event (random)</th></tr></thead><tbody>{levels}</tbody></table></div>
    <p class="cap"><strong>At t = 0.55 the three coarse prototypes change almost exactly at the phase events.</strong> The hierarchy says when a phase starts and ends; the scene says which object and where.</p>
    <div class="source">ledger/c1_grammar read · pre-registered m3_prereg c1_grammar</div>
  </div>
  <div class="plate">
    <div class="legend"><span><i class="l k1"></i>gripper open</span><span><i class="l k2"></i>gripper closed</span></div>
    <div class="chartwrap">{verb_glyphs()}</div>
    <p class="cap" id="f6t"><strong>What each coarse prototype does:</strong> the hand path its mode commands over ten steps, horizontal travel against height, on one scale.</p>
    <div class="tablewrap"><table><thead><tr><th class="n">level t</th><th>prototype, named by its frames</th><th class="n">tasks</th><th class="n">cm per step</th><th class="n">vertical</th></tr></thead><tbody>{verb_rows}</tbody></table></div>
    <p class="cap"><strong>The coarse prototypes are verbs shared across tasks</strong>: reaching down with the gripper open, lowering while holding, the grip change.
    A task's program is a sentence over them, anchored on its objects:</p>
    <div class="tablewrap"><table><thead><tr><th>instruction</th><th>program</th><th class="n">bytes</th></tr></thead><tbody>{prog_rows}</tbody></table></div>
    <div class="source">ledger/grammar readout · docs/GRAMMAR.md · experiments/grammar_readout.py</div>
  </div>
  <div class="plate">
    <div class="tablewrap"><table><thead><tr><th>distance between two tasks' phases</th><th class="n">same key</th><th class="n">other key</th><th class="n">permutation p</th></tr></thead><tbody>{key_rows}</tbody></table></div>
    <p class="cap"><strong>The prototypes carry the verb, the bindings carry the object.</strong> Carries with the same goal relation share a prototype string;
    grasps share a string whatever the object, and differ by the object's geometry. These are the keys of the binding table: grasps by object type, places by goal relation.</p>
    <div class="source">ledger/grammar readout · keys: object type (grasp), goal relation (place)</div>
  </div>
</section>

<section>
  <h2>What a task costs to keep</h2>
  <p>Per task, what must be stored to perform it later. The prior is never changed and each task keeps its own record, so a new task cannot alter an old one.</p>
  <div class="plate">
    <div class="legend"><span><i class="k1"></i>this method</span><span><i class="k2"></i>fine-tuning or replay</span></div>
    <div class="chartwrap">{storage_bars()}</div>
    <p class="cap" id="f5t"><strong>A task keeps {fmt_bytes(sto['median']['program'] + sto['median']['reader_pca512'])}: its phases and a reader head over a shared basis.</strong>
    The head reads positions at {co['summary']['pca512']['median_err_cm']:.3f} cm against {co['summary']['full']['median_err_cm']:.3f} cm for the full reader, and keeps success
    ({rc['RC1']['x']} against {rc['RC1']['control']} of 1,000 episodes; non-inferior, lower bound {rc['RC1']['diff']['lo90']:+.3f}). Over the 50 tasks:
    {fmt_bytes(sto['streams']['A']['totals']['lora'])} of LoRAs, {fmt_bytes(sto['streams']['A']['totals']['raw'])} of raw demos, {fmt_bytes(sto['streams']['A']['totals']['own_pca'])} here
    ({fmt_bytes(sto['basis_pca512'])} of it the basis, stored once). A seeded random projection needs no basis: {col['summary']['rand2048']['median_err_cm']:.3f} cm at {fmt_bytes(col['summary']['rand2048']['mean_bytes'])} (offline).</p>
    <div class="source">ledger/storage read · ledger/p6_reader compress_offline, compress_offline_large, compress_read · pre-registered m3_prereg reader_compress</div>
  </div>
</section>

<section>
  <h2>Tasks composed from other tasks</h2>
  <p>A task with no demo of its own takes the grasp phase of a task with the same object type and the place phases of a task with the same goal, relabelled onto its scene.</p>
  <div class="plate">
    <div class="tablewrap"><table><thead><tr><th>target task</th><th class="n">own demo</th><th class="n">composed</th><th class="n">composed, learned reader</th><th class="n">sibling's whole demo</th></tr></thead><tbody>{p3rows}</tbody></table></div>
    <p class="cap"><strong>Composed {p3c['C1']['x']} of 120 against {p3c['C1']['control']} with the task's own demo; {p3r2['R2']['x']} with no demo, no practice and no simulator state at test.</strong>
    Another object type's grasp (the sibling's demo) gets {p3c['C2']['control']}. Successes of 20 per task.</p>
    <div class="source">ledger/p3_compose p3c, p3c_reader2 · pre-registered m3_prereg p3c, p3c_amend1, p3c_reader, p3c_reader2</div>
  </div>
</section>

<section>
  <h2>Continual learning over 50 tasks</h2>
  <p>Fifty tasks the prior fails arrive one at a time in two mirrored orders. A task composes from the stored bindings when both its keys are known and otherwise asks for its one demo.</p>
  <div class="plate">
    <div class="legend"><span><i class="l k1"></i>order A</span><span><i class="l k2"></i>order B</span><span><i class="l kr"></i>a demo for every task</span></div>
    <div class="chartwrap">{cumulative_demos()}</div>
    <p class="cap" id="f3t"><strong>Composition becomes possible more often as tasks accumulate:</strong> {p5['S1']['only_late']} tasks compose only when they arrive late, {p5['S1']['only_early']} only when early (sign test p {p5['S1']['sign_p']:.4f}).</p>
    <div class="source">results/robot/p5_stream_plan_gated.json · ledger/p5_stream read · pre-registered m3_prereg p5_stream, p5_stream_amend1</div>
  </div>
  <div class="plate">
    <div class="legend"><span><i class="k1"></i>order A</span><span><i class="k2"></i>order B</span></div>
    <div class="chartwrap">{tradeoff()}</div>
    <p class="cap" id="f4t"><strong>Unchecked composition costs success: composed arrivals score {p5['S2']['x']} against {p5['S2']['control']} of {p5['S2']['n']} for their own demos.</strong>
    A stored place path keeps its source's route, which fails in another layout. Keeping only compositions that succeed in 8 of 10 practice episodes
    brings the streams back near the all-demo stream with a few demos saved; on fresh orders the composed arrivals score {cf['composed_vs_own']['x']} against
    {cf['composed_vs_own']['control']} of {cf['composed_vs_own']['n']} for their own demos, which is too few arrivals to show non-inferiority.</p>
    <div class="source">ledger/p5_stream read, validated, validated_multi, practice, confirm · the first three checked variants are exploratory (m3_prereg p5_stream_amend2-4); fresh orders = the pre-registered confirmation (m3_prereg p5_confirm; orders C and D plotted as the first and second order)</div>
  </div>
</section>

<section>
  <h2>What did not work</h2>
  <ul class="neg">{neg_html}</ul>
  <p>Open problems: a place skill whose route adapts to a new layout, and joint-state bindings (the stove's knob starts anywhere from {stove['init_min_rad']:.2f} to {stove['init_max_rad']:.2f} rad; practice hands that task to prototype borrowing).</p>
</section>

<footer>Every number on this page is read from ledger/ at commit {esc(commit)} by experiments/results_page.py; the README's copies pass experiments/audit_numbers.py ({esc(audit)}). Pre-registrations: ledger/m3_prereg. Simulation only (LIBERO-90); the rendered-scene readers and the checks use the simulator at training time.</footer>
</main>
<div class="tip" id="tip" hidden></div>
<script>
(function () {{
  const tip = document.getElementById("tip");
  function show(text, x, y) {{ tip.textContent = text; tip.hidden = false; const w = tip.offsetWidth;
    tip.style.left = Math.min(x + 14, window.innerWidth - w - 8) + "px"; tip.style.top = (y + 14) + "px"; }}
  document.querySelectorAll(".pt").forEach(function (g) {{
    g.addEventListener("pointermove", function (e) {{ show(g.dataset.tip, e.clientX, e.clientY); }});
    g.addEventListener("pointerleave", function () {{ tip.hidden = true; }});
    g.addEventListener("focus", function () {{ const r = g.getBoundingClientRect(); show(g.dataset.tip, r.right, r.top); }});
    g.addEventListener("blur", function () {{ tip.hidden = true; }});
  }});
  const cum = document.getElementById("cum");
  if (cum) {{
    const band = cum.querySelector(".hover-band"), xh = cum.querySelector(".xhair");
    const A = {json.dumps(gplan["A"]["cumulative_demos"])}, B = {json.dumps(gplan["B"]["cumulative_demos"])};
    const n = +band.dataset.n, L = +band.dataset.l, W = +band.dataset.w;
    band.addEventListener("pointermove", function (e) {{
      const pt = cum.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const p = pt.matrixTransform(cum.getScreenCTM().inverse());
      const i = Math.max(1, Math.min(n, Math.round((p.x - L) / W * (n - 1)) + 1));
      const x = L + (i - 1) / (n - 1) * W; xh.setAttribute("x1", x); xh.setAttribute("x2", x); xh.setAttribute("visibility", "visible");
      show(i + " tasks arrived: order A " + A[i - 1] + " demos, order B " + B[i - 1] + " demos (" + i + " without composition)", e.clientX, e.clientY);
    }});
    band.addEventListener("pointerleave", function () {{ tip.hidden = true; xh.setAttribute("visibility", "hidden"); }});
  }}
}})();
</script>
"""
out = os.path.join(R, "results_page.html")
open(out, "w", encoding="utf-8").write(PAGE)
print(out, len(PAGE))
