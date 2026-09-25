"""Figures for docs/REVIEW_2026-09.md, drawn from the c-series result files.

    python experiments/c_figures.py --c1 colormnist:'results/c1_cobweb_update_colormnist_F3*__seed*.json' \
        --c1 shapes3d:'results/c1_cobweb_update_shapes3d_S3*__seed*.json' \
        --c2 colormnist:'results/c2_router_loop_colormnist_F__seed*.json' --c3 results/c3_flagship.json

Palette: the dataviz reference categorical slots, validated (scripts/validate_palette.js) -- the
first three for the all-pairs scatter (worst CVD dE 9.2, normal-vision 24.0), four for lines (worst
adjacent 9.1 / 22.9).  Aqua and yellow sit under 3:1 on the surface, so every point and line carries
a direct text label (the relief rule).  Text is ink, never series colour.  One axis per panel.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# superseded in the figure (still in every table): consolidation over TAUGHT contexts only, which
# the wide variant replaces, and the lambda=30 point of the sparse arm, which sits on its lambda=10 line
SKIP = {"lora-consol@rho0.05@lam10", "cw-sparse@rho0.01@lam30"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "c")


def family(arm):
    base = arm.split("@")[0]
    if base.startswith("value-lora"):
        return "added structure (per-value adapter)", BLUE
    if base in ("ft-full", "cw-sparse-noanc"):
        return "no anchoring", AQUA
    if base == "emb-only":
        return None, None
    return "zero added parameters, anchored", ORANGE


def short(arm):
    return (arm.replace("@rho", " ρ").replace("@lam", " λ").replace("@lr", " lr")
            .replace("value-lora-r1", "value adapter r1").replace("lora-consolw", "adapter→consolidate (wide)")
            .replace("lora-consol", "adapter→consolidate"))


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)                              # the grid sits behind every mark
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def place_labels(fig, ax, pts, fs=7.5):
    """Greedy direct labels: each tries offsets around its point and takes the first whose text box
    overlaps no placed label and no point.  Boxes are measured in display pixels."""
    fig.canvas.draw()
    tr = ax.transData
    dots = [tr.transform((x, y)) for _, x, y in pts]
    placed = []
    cw, ch = fs * 0.78, fs * 1.6                       # approximate glyph width / line height (px)
    offs = [(8, 4, "left"), (8, -12, "left"), (-8, 4, "right"), (-8, -12, "right"),
            (8, 16, "left"), (8, -24, "left"), (-8, 16, "right"), (-8, -24, "right"),
            (8, 28, "left"), (-8, 28, "right"), (8, -36, "left"), (-8, -36, "right")]
    for (name, x, y), (px, py) in zip(pts, dots):
        w = cw * len(name)
        for dx, dy, ha in offs:
            x0 = px + dx if ha == "left" else px + dx - w
            box = (x0, py + dy, x0 + w, py + dy + ch)
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                      for b in placed)
            hit |= any(box[0] - 8 < qx < box[2] + 8 and box[1] - 8 < qy < box[3] + 8
                       for (qx, qy) in dots)
            if not hit:
                break
        placed.append(box)
        far = abs(dy) > 12                              # moved away: draw a leader to its point
        ax.annotate(name, (x, y), xytext=(dx, dy + 2), textcoords="offset points", fontsize=fs,
                    color=INK, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=4)
                    if far else None)


def c1_last(files):
    runs = [json.load(open(f)) for f in files]
    arms = {}
    for r in runs:
        for x in r["rows"]:
            arms.setdefault(x["arm"], {}).setdefault(r["config"]["seed"], []).append(x)
    out = {}
    for a, by_seed in arms.items():
        last = [max(rs, key=lambda x: x["task"]) for rs in by_seed.values()]
        out[a] = dict(un=[x["untaught_joint"] for x in last], ta=[x["taught_joint"] for x in last],
                      dr=[100 * x["drift_mean"] for x in last], drop=[x["old_drop"] for x in last],
                      du=[np.asarray(x["drift_u"]) * 100 for x in last],
                      u=runs[0].get("u_grid") or np.round(np.arange(0.1, 1.0, 0.1), 2).tolist())
    return out


def fig_c1(specs):
    fig, axes = plt.subplots(1, len(specs), figsize=(6.2 * len(specs), 4.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (bench, files) in zip(axes[0], specs):
        d = c1_last(files)
        style(ax, f"{bench}: never-taught combinations vs disturbance of the old classes")
        seen, pts = set(), []
        for a, m in d.items():
            fam, col = family(a)
            if fam is None or a in SKIP:
                continue
            x, y = np.mean(np.maximum(m["dr"], 1e-3)), np.mean(m["un"])
            xe = np.std(m["dr"], ddof=1) / np.sqrt(len(m["dr"])) if len(m["dr"]) > 1 else 0
            ye = np.std(m["un"], ddof=1) / np.sqrt(len(m["un"])) if len(m["un"]) > 1 else 0
            ax.errorbar(max(x, 1e-2), y, xerr=xe, yerr=ye, fmt="o", ms=7, color=col, mec=SURFACE,
                        mew=1.5, elinewidth=1, capsize=0,
                        label=fam if fam not in seen else None, zorder=3)
            seen.add(fam)
            pts.append((short(a), max(x, 1e-2), y))
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.0)
        place_labels(fig, ax, pts)
        ax.set_xlabel("ε-loss drift on the old classes, % (log; 0 plotted at 0.01)", color=INK2,
                      fontsize=8.5)
        ax.set_ylabel("joint accuracy on never-taught combinations (addressed, 72 B)", color=INK2,
                      fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "c1_frontier.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_drift(bench, files, arms):
    d = c1_last(files)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fig.patch.set_facecolor(SURFACE)
    style(ax, f"{bench}: where the update disturbs the old classes, by noise level")
    for a, col in zip(arms, (AQUA, YELLOW, ORANGE, BLUE)):
        if a not in d:
            continue
        du = np.mean(np.stack(d[a]["du"]), 0)
        u = np.asarray(d[a]["u"][:len(du)])
        ax.plot(u, np.maximum(du, 1e-2), color=col, linewidth=2)
        ax.annotate(short(a), (u[-1], max(du[-1], 1e-2)), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=INK, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("noise level u (0 = clean, 1 = pure noise)", color=INK2, fontsize=8.5)
    ax.set_ylabel("ε-loss drift on the old classes, % (log)", color=INK2, fontsize=8.5)
    ax.set_xlim(0.05, 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"c1_drift_by_u_{bench}.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_c2(bench, files):
    runs = [json.load(open(f)) for f in files]
    r = runs[0]
    rec = r["records"]
    popcol = {"in-dist": BLUE, "new-comb": ORANGE, "new-value": AQUA}
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax, f"{bench}: what the router stored, concept by concept (seed {r['config']['seed']})")
    cum = np.maximum(np.array([x["bytes_total"] for x in rec], float), 1.0)
    ax.step(range(len(rec)), cum, where="post", color=INK2, linewidth=1.2)
    tag = {"IN-DIST": "I", "ADDRESS": "A", "EXTEND": "E", "REFINE": "R", "WEIGHT": "W",
           "WEIGHT-cap": "W*"}
    for i, x in enumerate(rec):
        ax.plot(i, cum[i], "o", ms=8, color=popcol[x["pop"]], mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(tag.get(x["tier"], x["tier"]), (i, cum[i]), xytext=(0, 7),
                    textcoords="offset points", fontsize=7, color=INK, ha="center")
    for p_, c_ in popcol.items():
        ax.plot([], [], "o", ms=8, color=c_, label=p_)
    ax.set_yscale("log")
    ax.set_xlabel("arrival order (letters: I in-dist, A address, E extend, R refine, W weight)",
                  color=INK2, fontsize=8)
    ax.set_ylabel("cumulative bytes stored (log)", color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"c2_stream_{bench}.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_c5(bench, variants):
    """variants: [(label, colour, files)].  Left: megabytes stored over the stream, seed 0.  Right:
    final megabytes against the new-value concepts' end-of-stream joint, one point per seed."""
    runs = [(lab, col, [json.load(open(f)) for f in fs]) for lab, col, fs in variants if fs]
    if not runs:
        return
    name = {"colormnist": "ColorMNIST", "shapes3d": "Shapes3D"}.get(bench, bench)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    style(ax, f"{name}: storage as the stream arrives (seed 0)")
    ends = []
    for lab, col, rs in runs:
        r0 = next((r for r in rs if r["config"]["seed"] == 0), rs[0])
        rec = r0["records"]
        mb = np.array([x["bytes_total"] for x in rec], float) / 1e6
        ax.step(range(len(rec)), mb, where="post", color=col, linewidth=1.8)
        for i, x in enumerate(rec):
            if x["pop"] == "new-value":
                ax.plot(i, mb[i], "o", ms=5.5, color=col, mec=SURFACE, mew=1.2, zorder=3)
            if "merge" in x:
                ax.annotate("merge", (i, mb[i]), xytext=(0, -14), textcoords="offset points",
                            fontsize=7, color=INK, ha="center")
            if x["tier"] == "LEAF":
                ax.annotate("leaf", (i, mb[i]), xytext=(0, 6), textcoords="offset points",
                            fontsize=7, color=INK, ha="center")
        ends.append((len(rec) - 1, mb[-1], lab))
    for x_, y_, lab in ends:
        ax.annotate(lab, (x_, y_), xytext=(6, 0), textcoords="offset points", fontsize=7.5,
                    color=INK, va="center")
    ax.set_xlim(-0.5, len(runs[0][2][0]["records"]) + 6.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("arrival order (dots: concepts carrying the new value)", color=INK2, fontsize=8)
    ax.set_ylabel("megabytes stored", color=INK2, fontsize=8.5)
    ax = axes[1]
    style(ax, "storage against the new value's accuracy, per seed")
    pts = []
    for lab, col, rs in runs:
        for r in rs:
            nv = [x for x in r["records"] if x["pop"] == "new-value"]
            xb = r["summary"]["bytes_total"] / 1e6
            yj = float(np.mean([x["joint_end"] for x in nv]))
            ax.plot(xb, yj, "o", ms=8, color=col, mec=SURFACE, mew=1.5, zorder=3)
            pts.append((f"{lab.split(' (')[0]} s{r['config']['seed']}", xb, yj))
    ax.set_xlim(0, max(p_[1] for p_ in pts) * 1.35)
    ax.set_xlabel("megabytes stored at the end of the stream", color=INK2, fontsize=8.5)
    ax.set_ylabel("concepts carrying the new value: joint", color=INK2, fontsize=8.5)
    ax.set_ylim(-0.02, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    place_labels(fig, ax, pts, fs=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"c5_stream_{bench}.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_c3(path):
    s = json.load(open(path))["summary"]
    arms = [a for a in s if isinstance(s[a], dict)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(arms))[::-1]
    ax = axes[0]
    style(ax, "Shapes3D, all 40 held-out cells: joint accuracy (Wilson 95%)")
    j = np.array([s[a]["joint"] for a in arms])
    lo = np.array([s[a]["ci"][0] for a in arms])
    hi = np.array([s[a]["ci"][1] for a in arms])
    ax.barh(y, j, height=0.55, color=BLUE)
    ax.errorbar(j, y, xerr=[j - lo, hi - j], fmt="none", ecolor=INK, elinewidth=1)
    for yi, v in zip(y, j):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(arms, fontsize=8, color=INK)
    ax.set_xlim(0, 1.12)
    ax = axes[1]
    style(ax, "coverage of the real class (reference-anchored)")
    cv = np.array([s[a]["coverage"] for a in arms])
    ax.barh(y, cv, height=0.55, color=BLUE)
    for yi, v in zip(y, cv):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "c3_flagship.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c1", action="append", default=[], help="bench:glob")
    p.add_argument("--c2", action="append", default=[], help="bench:glob")
    p.add_argument("--c3", default=None)
    p.add_argument("--c5", action="append", default=[],
                   help="bench:label=glob;label=glob  (Cobweb, flat, C2 ... in that colour order)")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c1 = [(s.split(":", 1)[0], sorted(glob.glob(s.split(":", 1)[1]))) for s in args.c1]
    c1 = [(b, f) for b, f in c1 if f]
    if c1:
        fig_c1(c1)
        for b, f in c1:
            fig_drift(b, f, ["ft-full", "cw-sparse-noanc@rho0.01", "cw-sparse@rho0.01@lam10",
                             "lora-consol@rho0.05@lam10"])
    for s in args.c2:
        b, g = s.split(":", 1)
        f = sorted(glob.glob(g))
        if f:
            fig_c2(b, f)
    if args.c3 and os.path.exists(args.c3):
        fig_c3(args.c3)
    for s in args.c5:
        b, spec = s.split(":", 1)
        vs = []
        for part, col in zip(spec.split(";"), (BLUE, ORANGE, AQUA, YELLOW)):
            lab, g = part.split("=", 1)
            vs.append((lab, col, sorted(glob.glob(g))))
        fig_c5(b, vs)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
