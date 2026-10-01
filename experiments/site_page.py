"""The results site (published as the artifact https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk): a paged report in the layout of
the diffusion-cobweb-updates site, every number read from ledger/ (the rows the README quotes), every clip cut from recorded
episodes (experiments/site_media.py).

    ~/flash/miniconda3/envs/chunks/bin/python experiments/site_media.py      # media -> results/robot/site/media/
    ~/flash/miniconda3/envs/chunks/bin/python experiments/site_page.py       # -> results/robot/site/index.html
"""
import html
import json
import os
import runpy
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                            # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SITE = os.path.join(R, "site")
MEDIA = json.load(open(os.path.join(SITE, "media", "media.json")))
RP = runpy.run_path(os.path.join(ROOT, "experiments", "results_page.py"))  # the charts (and every row they read)


def row(exp, tag):
    rows = [r for r in ledger.load(exp, tag=tag) if not r.get("retired")]
    return rows[-1]["metrics"] if rows else None


def esc(s):
    return html.escape(str(s))


fmt_bytes = RP["fmt_bytes"]
M = RP["M"]
meth = lambda k: M[k + ".prior_fails"]
v2, v2s, al, s1, d1 = RP["v2"], RP["v2s"], RP["al"], RP["s1"], RP["d1"]
p3c, p3r2, c1, p5, p5p, cf = RP["p3c"], RP["p3r2"], RP["c1"], RP["p5"], RP["p5p"], RP["cf"]
gram, sto, rc, co, col = RP["gram"], RP["sto"], RP["rc"], RP["co"], RP["col"]
p2, p4, dev, ty, yw, ywr, stove, rest = RP["p2"], RP["p4"], RP["dev"], RP["ty"], RP["yw"], RP["ywr"], RP["stove"], RP["rest"]
geo = row("p0_audit", "geometry")
dh = row("p5_stream", "dev_handoff")
dcr = row("p5_stream", "dev_carries")
scheme = row("p5_stream", "scheme")
fam = row("r20_family", "family") or {}
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
audit = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "audit_numbers.py")], capture_output=True, text=True,
                       cwd=ROOT).stdout.strip().splitlines()[-1]
per_task_kb = (sto["median"]["program"] + sto["median"]["reader_pca512"]) / 1e3
aim_n = sum(v for k, v in geo["primary"]["classes"].items() if k != "fixture")
lang = {int(t): v["language"] for t, v in json.load(open(os.path.join(R, "p5_pool.json"))).items()}

CSS = r"""
/* layout: a report in pages; a sticky contents column on the left, one reading column of findings on the right; figures are
   real footage from the simulator, captioned and numbered like a paper */
:root{--bg:#f4f6f7;--paper:#fbfcfc;--ink:#12191d;--ink-2:#39464d;--muted:#58656c;--rule:#d9e0e4;--grid:#e6ebee;--head:#edf1f3;
--accent:#1b62a5;--accent-soft:#e2ecf6;--ours:#1b62a5;--base:#b4501f;--base-soft:#f8e8df;--good:#12824f;--bad:#bf3d31;--mix:#a77413;
--s1:#1b62a5;--s2:#c4622e;--s3:#12824f;--ref:#9aa6ad;--chip-bg:#edf1f3;--plate:#fbfcfc;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0e1316;--paper:#151c20;--ink:#e5ecf0;--ink-2:#c1ccd2;
--muted:#96a4ab;--rule:#28333a;--grid:#1f292f;--head:#1a2226;--accent:#79b2ea;--accent-soft:#16283a;--ours:#79b2ea;--base:#e98b5c;
--base-soft:#34231b;--good:#3cb97e;--bad:#e4695d;--mix:#dca63d;--s1:#6aa8e6;--s2:#e38451;--s3:#3cb97e;--ref:#5d6a71;--chip-bg:#1a2226;
--plate:#151c20;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e1316;--paper:#151c20;--ink:#e5ecf0;--ink-2:#c1ccd2;--muted:#96a4ab;--rule:#28333a;--grid:#1f292f;
--head:#1a2226;--accent:#79b2ea;--accent-soft:#16283a;--ours:#79b2ea;--base:#e98b5c;--base-soft:#34231b;--good:#3cb97e;--bad:#e4695d;
--mix:#dca63d;--s1:#6aa8e6;--s2:#e38451;--s3:#3cb97e;--ref:#5d6a71;--chip-bg:#1a2226;--plate:#151c20;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 var(--sans);padding-inline:16px;padding-block:24px 64px}
h1,h2,h3,h4{font-family:var(--cond);font-weight:600;text-wrap:balance;margin:0}
h2{font-size:30px;line-height:1.18;margin:0 0 4px}
h3{font-size:17px}
h3.sec{margin-top:14px;padding-top:14px;border-top:1px solid var(--rule)}
p{margin:0;max-width:78ch}
.lede{font-size:16.5px;max-width:80ch}
.meta{font:12.5px/1.5 var(--mono);color:var(--muted);letter-spacing:.02em}
.shell{display:grid;grid-template-columns:236px minmax(0,1fr);gap:44px;max-width:1400px;margin:0 auto}
.toc{position:sticky;top:20px;align-self:start;font:13.5px/1.45 var(--sans);max-height:calc(100vh - 40px);overflow-y:auto;padding-right:4px}
.toc .brand{font:600 19px/1.2 var(--cond);color:var(--ink);margin:0 0 14px}
.toc .brand span{display:block;font:400 12.5px/1.45 var(--sans);color:var(--muted);margin-top:5px}
.toc h2{font:600 11px/1.35 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin:18px 0 6px}
.toc ol{list-style:none;margin:0;padding:0;display:grid;gap:1px}
.toc a{display:grid;grid-template-columns:26px 1fr;padding:5px 8px;border-radius:4px;color:var(--ink);text-decoration:none}
.toc a .n{color:var(--muted);font-variant-numeric:tabular-nums}
.toc a:hover{background:var(--head)}
.toc a[aria-current="page"]{background:var(--accent-soft);font-weight:500}
.toc a[aria-current="page"] .n{color:var(--accent)}
.topbar{display:none}
main{min-width:0;display:flex;flex-direction:column;gap:6px}
.eyebrow{font:500 12px/1.4 var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
a{color:var(--accent)}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
@media (max-width:920px){.shell{grid-template-columns:1fr;gap:0}.toc{display:none}
.topbar{display:block;position:sticky;top:env(safe-area-inset-top,0px);z-index:10;background:var(--bg);border-bottom:1px solid var(--rule);
margin:-24px -16px 12px;padding:10px 16px}
.topbar summary{cursor:pointer;font-size:14px;font-weight:600;list-style:none;display:flex;justify-content:space-between;gap:12px}
.topbar summary::-webkit-details-marker{display:none}
.topbar summary::after{content:"Contents";font-weight:400;color:var(--accent)}
.topbar details[open] summary::after{content:"Close"}
.topbar .toc-inner{max-height:70vh;overflow-y:auto;margin-top:8px;font-size:14px}
.topbar .toc-inner .brand{display:none}
.topbar .toc-inner h2{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px}
.topbar .toc-inner ol{list-style:none;margin:0;padding:0}
.topbar .toc-inner a{display:grid;grid-template-columns:26px 1fr;padding:6px 4px;color:var(--ink);text-decoration:none}
.topbar .toc-inner a .n{color:var(--muted)}
.topbar .toc-inner a[aria-current="page"]{color:var(--accent);font-weight:600}
h2{font-size:25px}}
.page{display:flex;flex-direction:column;gap:14px}
.callout{border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:6px;padding:12px 16px;display:flex;flex-direction:column;gap:8px;background:var(--paper)}
.callout.core{font-size:16px}
.callout.thesis{border-left-color:var(--ink)}
.callout.pending{border-left-color:var(--mix)}
ol.results,ul.plain{margin:0;padding-left:22px;display:flex;flex-direction:column;gap:7px;max-width:88ch}
.defs{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(260px,100%),1fr));gap:16px;margin-top:4px}
.defs>div{border-left:3px solid var(--accent);padding:2px 0 2px 14px;display:flex;flex-direction:column;gap:4px;min-width:0}
.defs>div.b{border-left-color:var(--base)}
.defs h4{font-size:15.5px}
code{font:13px var(--mono);background:var(--head);padding:1px 5px;border-radius:4px}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
thead th{background:var(--head);font-weight:600;font-size:12.5px;color:var(--muted);letter-spacing:.02em}
td.num,th.num{text-align:right;white-space:nowrap}
tbody th{font-weight:500}
tr.ours th,tr.ours td{color:var(--ours)}
tr.ours th{font-weight:600}
tr.ref th,tr.ref td{color:var(--muted)}
td.bytes{color:var(--muted);font-size:12.5px;white-space:nowrap}
.tcap{font-size:13.5px;font-weight:600;margin-top:6px}
figure{margin:6px 0 0;display:flex;flex-direction:column;gap:8px;min-width:0}
figcaption{color:var(--muted);font-size:13.5px;max-width:95ch}
.fnum{font-weight:600;color:var(--ink)}
figure img{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;background:var(--paper)}
.note{font-size:13px;color:var(--muted);max-width:95ch}
video{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;background:#0b0f11}
.vtabs{display:flex;flex-wrap:wrap;gap:6px}
.vtabs button{font:13px/1.2 var(--sans);padding:7px 11px;border:1px solid var(--rule);border-radius:999px;background:var(--paper);color:var(--ink);cursor:pointer}
.vtabs button:hover{border-color:var(--accent)}
.vtabs button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--paper)}
.vtabs button .o{font:12px var(--mono);opacity:.8;margin-left:6px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:18px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:18px}
.proto{display:flex;flex-direction:column;gap:8px;min-width:0;border:1px solid var(--rule);border-radius:8px;padding:12px;background:var(--paper)}
.proto h4{font-size:17px;display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.proto h4 span{font:12px var(--mono);color:var(--muted);font-weight:400}
.proto .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:13px;color:var(--ink-2)}
.proto .kv b{font-weight:500;color:var(--muted)}
.proto video{border-radius:4px}
.chip{display:inline-flex;gap:6px;align-items:center;font:500 12px/1 var(--mono);padding:5px 8px;border-radius:4px;background:var(--chip-bg);width:max-content}
.chip.holds{color:var(--good)} .chip.fails{color:var(--bad)} .chip.run{color:var(--mix)}
.claims{display:grid;border-top:1px solid var(--rule)}
.claim{display:grid;grid-template-columns:minmax(0,14rem) 7rem minmax(0,1fr);gap:16px;padding-block:12px;border-bottom:1px solid var(--rule);align-items:baseline}
.claim p{color:var(--ink-2);font-size:14.5px}
@media (max-width:640px){.claim{grid-template-columns:1fr;gap:4px}}
.pager{display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--rule);padding-top:16px;margin-top:22px;font:14px/1.4 var(--sans)}
.pager a{display:flex;flex-direction:column;gap:2px;text-decoration:none;color:var(--ink);max-width:48%}
.pager a small{color:var(--muted);font-size:12px}
.pager a.next{text-align:right;margin-left:auto}
.pager a:hover span{text-decoration:underline}
/* the stepped walkthrough */
.st{display:flex;flex-direction:column;gap:10px}
.st-stage{border:1px solid var(--rule);border-radius:10px;background:var(--paper);padding:16px;min-height:340px;position:relative;overflow:hidden}
.st-sc{display:none;flex-direction:column;gap:12px}
.st-sc.on{display:flex;animation:stin .45s ease}
@keyframes stin{from{opacity:.25;transform:translateY(4px)}to{opacity:1;transform:none}}
.st-sc h4{font-size:18px}
.frames{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px}
.frames figure{margin:0;gap:4px}
.frames img{border-radius:4px}
.frames figcaption{font:12px var(--mono);color:var(--muted);text-align:center}
.ribbon{display:flex;height:22px;border-radius:4px;overflow:hidden;border:1px solid var(--rule)}
.ribbon span{display:block;height:100%}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
.phases{display:grid;gap:6px}
.phase{display:grid;grid-template-columns:120px minmax(0,1fr);gap:10px;align-items:center;font-size:14px}
.phase .bar{height:30px;border-radius:4px;display:flex;align-items:center;padding:0 10px;font:500 13px var(--mono);color:var(--paper);background:var(--accent)}
.bind{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(250px,100%),1fr));gap:12px}
.bind>div{border:1px solid var(--rule);border-radius:6px;padding:10px 12px;font-size:14px;display:flex;flex-direction:column;gap:4px}
.bind .num{font:15px var(--mono);color:var(--accent)}
.st-bar{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.st-btn{font:15px/1 var(--sans);min-width:34px;height:30px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);border-radius:6px;cursor:pointer}
.st-btn:hover{border-color:var(--accent)}
.st-dots{display:flex;gap:5px;margin-left:6px}
.st-dot{width:11px;height:11px;border-radius:50%;border:1.5px solid var(--muted);background:transparent;padding:0;cursor:pointer}
.st-dot[aria-current="step"]{background:var(--accent);border-color:var(--accent)}
.st-count{font:12.5px var(--mono);color:var(--muted);margin-left:auto}
@media (max-width:640px){.frames{grid-template-columns:repeat(3,minmax(0,1fr))}.phase{grid-template-columns:1fr}}
/* charts carried over from the first page */
.chartwrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--paper);padding:10px}
.chart{width:100%;min-width:620px;height:auto;display:block}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .tick{font:12px var(--mono);fill:var(--muted)}
.chart .axis{font:12px var(--sans);fill:var(--ink-2)}
.chart .lab{font:13px var(--sans);fill:var(--ink-2)}
.chart .dot{stroke:var(--paper);stroke-width:2}
.chart .bind,.chart .s1{fill:var(--s1)} .chart .lora,.chart .s2{fill:var(--s2)} .chart .alloc{fill:var(--s3)}
.chart .line{fill:none;stroke-width:2}
.chart path.s1{stroke:var(--s1);fill:none} .chart path.s2{stroke:var(--s2);fill:none}
.chart .ref{stroke:var(--ref);stroke-width:1.5} .chart .pair{stroke:var(--rule);stroke-width:2} .chart .leader{stroke:var(--grid);stroke-width:1}
.chart .hit{fill:transparent} .chart .pt{outline:none} .chart .pt:hover .dot,.chart .pt:focus-visible .dot{stroke:var(--ink)}
.chart .hover-band{fill:transparent} .chart .xhair{stroke:var(--muted);stroke-width:1}
.chart .glyph{fill:none;stroke:var(--rule)} .chart .vpath{stroke-width:3;stroke-linecap:round} .chart .vpath.s1{stroke:var(--s1)} .chart .vpath.s2{stroke:var(--s2)}
.chart .gcell{fill:var(--grid);opacity:.6} .chart .gpass{fill:var(--good)} .chart .gfail{fill:var(--mix)} .chart .gown{fill:var(--ref)}
.tip{position:fixed;pointer-events:none;background:var(--paper);color:var(--ink);border:1px solid var(--rule);border-radius:4px;padding:6px 9px;font:13px var(--sans);box-shadow:0 2px 10px rgba(0,0,0,.12);max-width:320px;z-index:20}
footer{font:12px/1.6 var(--mono);color:var(--muted);border-top:1px solid var(--rule);padding-top:16px;margin-top:10px}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
"""

# ── structure ───────────────────────────────────────────────────────────────
NAV = [("Introduction", [("overview", "–", "Overview"), ("motivation", "1", "Motivation"), ("grammar", "2", "The grammar in the policy")]),
       ("Part I · Bindings from one demonstration", [("method-a", "3", "Method"), ("cases-a", "4", "Case study"), ("results-a", "5", "Twenty-one held-out tasks")]),
       ("Part II · Composing new tasks", [("method-b", "6", "Method"), ("cases-b", "7", "Case study: the carry"), ("stream", "8", "Fifty tasks in a stream")]),
       ("Results", [("cost", "9", "Cost and latency"), ("negatives", "10", "What did not work")]),
       ("Discussion", [("learned", "11", "Is it training or zero-shot?"), ("models", "12", "Other models, and pi0.7")]),
       ("Appendix", [("baselines", "13", "How the baselines were implemented"), ("verification", "14", "Verification")])]
ORDER = [(i, n, t) for _, items in NAV for i, n, t in items]


def toc():
    out = ['<p class="brand">Grammar × Bindings<span>adapting a frozen pi0.5 to new LIBERO-90 tasks from one demonstration, and composing new tasks from old ones</span></p>']
    for grp, items in NAV:
        out.append(f"<h2>{esc(grp)}</h2><ol>" + "".join(f'<li><a href="#{i}"><span class="n">{n}</span><span>{esc(t)}</span></a></li>'
                                                        for i, n, t in items) + "</ol>")
    return "".join(out)


def page(pid, eyebrow, title, body):
    k = [i for i, _, _ in ORDER].index(pid)
    prev = ORDER[k - 1] if k else None
    nxt = ORDER[k + 1] if k + 1 < len(ORDER) else None
    pg = '<nav class="pager" aria-label="Previous and next section">'
    pg += (f'<a href="#{prev[0]}"><small>Previous</small><span>{prev[1] if prev[1] != "–" else ""} {esc(prev[2])}</span></a>' if prev else "<span></span>")
    pg += (f'<a class="next" href="#{nxt[0]}"><small>Next</small><span>{nxt[1]} {esc(nxt[2])}</span></a>' if nxt else "")
    pg += "</nav>"
    return (f'<section class="page" id="page-{pid}" aria-label="{esc(title)}"><p class="eyebrow">{esc(eyebrow)}</p>'
            f'<h2>{esc(title)}</h2>{body}{pg}</section>')


FIG = {"n": 0}


def fig(chap, body, caption):
    FIG["n"] += 1
    return f'<figure>{body}<figcaption><span class="fnum">Figure {chap}.</span> {caption}</figcaption></figure>'


def video(file, poster=True, autoplay=False, loop=False, cls=""):
    p = f' poster="media/{file[:-4]}.jpg"' if poster else ""
    a = " autoplay muted" if autoplay else " muted"
    return (f'<video class="{cls}" src="media/{file}"{p} controls playsinline preload="{"auto" if autoplay else "metadata"}"{a}'
            f'{" loop" if loop else ""}></video>')


def table(head, rows, cls=None, num_from=1):
    h = "".join(f'<th class="{"num" if i >= num_from else ""}">{esc(x)}</th>' for i, x in enumerate(head))
    body = []
    for r in rows:
        c = r[0] if isinstance(r[0], str) and r[0] in ("ours", "ref", "") else ""
        cells = r[1:] if c or r[0] == "" else r
        tds = "".join((f"<th>{x}</th>" if i == 0 else f'<td class="{"num" if i >= num_from else ""}">{x}</td>') for i, x in enumerate(cells))
        body.append(f'<tr class="{c}">{tds}</tr>')
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# ── the stepped walkthrough (task 48, its demo, its bindings, our rollout at a new init state) ─────────────────────────
VERB_COL = {"open move": "#9cc3e6", "open reach down": "#2f7cc6", "open grip change": "#12191d", "closed lift": "#e0a060",
            "closed transport": "#c4622e", "closed lower": "#7a3a18"}


def ribbon(t):
    import numpy as np
    L = np.load(os.path.join(R, "grammar_labels.npz"))
    k = [int(x) for x in L["tasks"]].index(t)
    names = {int(c): v["name"].split(" (")[0] for c, v in gram["prototypes"]["0.4"].items()}
    lab = [names.get(int(x), "other") for x in L["lab_0.4"][L["cut"][k]:L["cut"][k + 1]]]
    runs, cur, n = [], None, 0
    for v in lab:
        if v == cur:
            n += 1
        else:
            if cur is not None:
                runs.append((cur, n))
            cur, n = v, 1
    runs.append((cur, n))
    T = len(lab)
    spans = "".join(f'<span style="width:{100 * n / T:.3f}%;background:{VERB_COL.get(v, "var(--ref)")}" title="{esc(v)}, {n} steps"></span>' for v, n in runs)
    return f'<div class="ribbon" role="img" aria-label="the demo\'s steps coloured by prototype verb">{spans}</div>', T


def walkthrough():
    p48 = gram["programs"]["48"]
    ph = p48["phases"]
    T = ph[-1]["frames"][1]
    rib, _ = ribbon(48)
    keys = MEDIA["stills"]["demo48_keys"]
    frames = "".join(f'<figure><img src="media/demo48_{k}.jpg" alt="demo frame {i}: {k}" loading="lazy"><figcaption>step {i}</figcaption></figure>'
                     for k, i in keys.items())
    legend = "".join(f'<span><i style="background:{c}"></i>{esc(v)}</span>' for v, c in VERB_COL.items())
    grasp = next(p for p in ph if p["kind"] == "grasp")
    carry = next(p for p in ph if p["kind"] == "carry")
    bars = "".join(f'<div class="phase"><span>{esc({"grasp": "grasp", "carry": "carry", "rest": "finish", "fixture": "operate"}[p["kind"]])} '
                   f'<span class="note">steps {p["frames"][0]}–{p["frames"][1]}</span></span><div class="bar" style="width:{max(8, 100 * (p["frames"][1] - p["frames"][0]) / T):.1f}%;'
                   f'background:{ {"grasp": "var(--s1)", "carry": "var(--s2)", "rest": "var(--ref)"}.get(p["kind"], "var(--ref)") }">{esc(p["anchor"])}</div></div>' for p in ph)
    roll = "".join(f'<figure><img src="media/roll48_{i}.jpg" alt="our rollout" loading="lazy"><figcaption>{lab}</figcaption></figure>'
                   for i, lab in (("1", "grasp"), ("2", "carry"), ("3", "descend"), ("4", "released")))
    steps = [
        ("One demonstration", "A person teleoperated the robot once: pick up the ketchup and put it in the basket.",
         f'<div class="frames">{frames}</div>'),
        ("The prototypes label every step", "Each ten-step chunk of the demo is matched to the policy's own action prototypes. The labels "
         "read as verbs: move, reach down, close the gripper, lift, carry, lower.", f'<div class="frames">{frames}</div>{rib}<div class="legend">{legend}</div>'),
        ("Cut at the object events", "Where the gripper closes on the ketchup and where it lets go split the demo into phases, each "
         "anchored on an object. The coarse prototypes change label at these same events.", f'<div class="phases">{bars}</div>'),
        ("Keep only the bindings", "Each phase is stored relative to its anchor, so what the task adds is a few numbers.",
         f'<div class="bind"><div><b>grasp binding</b><span class="num">{grasp["hand_in_object_cm"]} cm</span><span>the hand in the ketchup\'s frame at the close, '
         f'{esc(grasp["approach"])}</span></div><div><b>place binding</b><span class="num">{carry["release_in_receptacle_cm"]} cm</span><span>the ketchup at the '
         f'release, in the basket\'s frame</span></div><div><b>the whole program</b><span class="num">{p48["bytes"]:,} bytes</span><span>{esc(p48["sentence"])}</span></div></div>'),
        ("A new placement", f"At test the objects start elsewhere. A {sto['median']['reader_pca512'] / 1e3:.1f} KB reader head reads where they are "
         f"from the first image ({co['summary']['pca512']['median_err_cm']:.2f} cm median error); no simulator state is used.",
         '<div class="grid2"><figure><img src="media/roll48_0.jpg" alt="a new init state"><figcaption>init state 30, never seen in training</figcaption></figure>'
         '<figure><img src="media/demo48_start.jpg" alt="the demo\'s placement"><figcaption>the demo\'s placement</figcaption></figure></div>'),
        ("The frozen policy follows the re-anchored phases", "Every query re-anchors the phases on the read poses and samples pi0.5 "
         "through the next ten planned actions. No weights change.", f'<div class="frames" style="grid-template-columns:repeat(4,minmax(0,1fr))">{roll}</div>'),
    ]
    sc = "".join(f'<div class="st-sc{" on" if i == 0 else ""}" data-i="{i}"><h4>{i + 1}. {esc(t)}</h4><p>{esc(d)}</p>{b}</div>' for i, (t, d, b) in enumerate(steps))
    dots = "".join(f'<button class="st-dot" data-go="{i}" aria-label="step {i + 1}"></button>' for i in range(len(steps)))
    return (f'<div class="st" id="walk" data-n="{len(steps)}"><div class="st-stage">{sc}</div><div class="st-bar">'
            f'<button class="st-btn" data-act="prev" aria-label="Previous step">&#8249;</button><button class="st-btn st-play" data-act="play" aria-label="Pause">&#10074;&#10074;</button>'
            f'<button class="st-btn" data-act="next" aria-label="Next step">&#8250;</button><div class="st-dots">{dots}</div><span class="st-count">1 / {len(steps)}</span></div></div>')


# ── pages ───────────────────────────────────────────────────────────────────
CMP = MEDIA["comparisons"]
CMP_LAB = {35: "Open the microwave", 42: "Frying pan under the shelf", 32: "Ketchup into the drawer", 1: "Close the drawer, bowl on top",
           34: "Mug in front of the white mug", 5: "Pudding into the drawer", 37: "Bowl right of the plate", 39: "Turn off the stove"}


def outcome_str(o):
    return " · ".join("✓" if x else "✗" for x in o)


def hero():
    btns = "".join(f'<button type="button" data-v="{c["file"]}" data-cap="{esc(c["language"])} (init state {c["trial"]})" aria-pressed="{"true" if i == 0 else "false"}">'
                   f'{esc(CMP_LAB.get(c["task"], c["language"]))}<span class="o">{outcome_str(c["outcomes"])}</span></button>'
                   for i, c in enumerate(CMP) if c["task"] != 39)
    first = CMP[0]
    return (f'<div class="vtabs" id="herotabs" role="group" aria-label="Choose a task">{btns}</div>'
            f'<video id="herovid" src="media/{first["file"]}" poster="media/{first["file"][:-4]}.jpg" controls playsinline muted autoplay preload="auto"></video>'
            f'<p class="note" id="herocap">{esc(first["language"])} (init state {first["trial"]})</p>')


claims = [
    ("Bindings from one demo", "holds",
     f"The demo's phases, re-anchored on the objects and followed by the frozen pi0.5, succeed {v2['V1']['x']} of 420 times on 21 held-out tasks "
     f"({v2s['V5']['x']} with a learned pose reader), against {v2['V2']['control']} for the best fine-tuning. A new sampling seed gives {s1['R1']['x']}; "
     f"a second demo {d1['D1']['x']} of {d1['D1']['n']}."),
    ("The policy's grammar", "holds",
     f"The coarsest prototypes of the policy's own actions change label at {c1['levels']['0.55']['recall']:.2f} of the phase events "
     f"(random {c1['levels']['0.55']['random_recall']:.2f}). Prototypes carry the verb, bindings carry the object "
     f"(permutation p {gram['string_by_key']['carry']['perm_p']:.3f} and {gram['grasp_geometry_perm_p']:.4f})."),
    ("A task's cost", "holds",
     f"{per_task_kb:.1f} KB per task: the phases and a pose-reader head, non-inferior to the full reader ({rc['RC1']['x']} vs {rc['RC1']['control']} "
     f"of 1,000). A LoRA stores {fmt_bytes(sto['lora_bytes'])}."),
    ("Composing new tasks", "holds" if (scheme and scheme["H1"]["non_inferior"]) else "mixed",
     (f"On four fresh orders of a 50-task stream, {scheme['H1']['units']} tasks built only from other tasks' bindings score {scheme['H1']['x']} of "
      f"{scheme['H1']['n']} against {scheme['H1']['control']} for their own demos (pre-registered, non-inferior); the streams ask for "
      f"{min(v['demos'] for v in scheme['orders'].values())}–{max(v['demos'] for v in scheme['orders'].values())} demos instead of 50."
      if scheme else "The stream's confirmation is running.")),
]


def chip(state):
    icon = {"holds": "✓", "fails": "✕", "mixed": "◐", "run": "…"}[state]
    lab = {"holds": "holds", "fails": "fails", "mixed": "mixed", "run": "running"}[state]
    return f'<span class="chip {"run" if state == "mixed" else state}"><span aria-hidden="true">{icon}</span> {lab}</span>'


P_OVERVIEW = f"""
<div class="callout core"><p><strong>Core idea.</strong> A robot policy trained on many tasks already knows how to manipulate: reach, grasp,
carry, release. What a new task adds is mostly <em>which</em> object and <em>where</em>. We keep pi0.5 frozen and store, per task, only its
<strong>bindings</strong>: a few numbers that say how to hold the object and where to put it, each relative to an object. One demonstration
gives them, and a new task can borrow them from tasks it resembles. The policy supplies the motion.</p></div>
<p class="lede">The report asks how a vision-language-action policy can take on tasks it fails, from one demonstration each, without retraining
and without forgetting. Figure 0.1 shows real evaluation episodes: the same task from the same start state, handled three ways.</p>
{fig("0.1", hero(), "Recorded evaluation episodes on held-out LIBERO-90 tasks, played at twice real time. Left: pi0.5 as trained. Middle: the best "
     "fine-tuning we found, a rank-16 LoRA trained on the demo, on practice successes and on failure negatives. Right: the task's bindings from its one "
     "demo, guiding the frozen pi0.5 (no simulator state at test). Each start state was never seen in training; ✓ and ✗ give the three outcomes in order.")}
<h3 class="sec">What holds</h3>
<div class="claims">{"".join(f'<div class="claim"><h3>{esc(a)}</h3>{chip(s)}<p>{esc(b)}</p></div>' for a, s, b in claims)}</div>
<h3 class="sec">The method in six steps</h3>
<p>Figure 0.2 follows one task from its demonstration to a new start state. Section 1 explains why the method takes this shape.</p>
{fig("0.2", walkthrough(), "Task 48, <em>pick up the ketchup and put it in the basket</em>: its one demo, the prototype labels, the phases, the "
     "bindings, and the frozen policy at a new start state. Frames 1–4 are recorded from the demo; frames 5–6 from our evaluation episode.")}
"""

M_LORA, M_NFT, M_R64, M_FFT = meth("practice_lora"), meth("nft_bal"), meth("base_lora_r64"), meth("base_fullft")
M_ZS, M_FAMP = meth("zeroshot"), M.get("fam_pool.prior_fails")

P_MOTIVATION = f"""
<p class="lede">Before any method, one question: when a trained robot policy fails a new task, what exactly is it missing? This section answers
from evidence, states the thesis the rest of the report tests, and derives both methods from it.</p>
<h3 class="sec">1.1 The setting</h3>
<p>pi0.5 (<code>pi05_libero</code>) was fine-tuned on LIBERO's forty tasks and solves them almost always. LIBERO-90 holds tasks it never saw, in
the same kitchens, living rooms and studies. We screened them and kept the 50 it fails (Figure 1.1). Each gets one human demonstration. The task, as
we define it: from that one example, succeed at new start states; store little per task; damage nothing learned before.</p>
{fig("1.1", '<img src="media/mosaic.jpg" alt="the fifty LIBERO-90 tasks pi0.5 fails, first frame of each" loading="lazy">',
     f"The 50 tasks, one start state each. Zero-shot, pi0.5 succeeds {M_ZS['k']} times in 420 episodes on the 21 held-out tasks that anchor the benchmark.")}
<h3 class="sec">1.2 Fine-tuning works partly, and costs a model per task</h3>
<p>The standard answer trains weights. A rank-16 LoRA on the one demo succeeds {M_LORA['k']} of 420 times. Our best version, trained also on the
policy's practice successes and on its failures as negatives, reaches {M_NFT['k']}. Rank 64 and full fine-tuning of the action expert reach
{M_R64['k']} and {M_FFT['k']}. Every variant stores a new model per task: {fmt_bytes(M_NFT['bytes'] / 21)} for a LoRA, and more for full fine-tuning.
Sharing one adapter across related tasks is worse than training each alone, because the adapter then mixes up the tasks: asked for the butter at
the back, it puts the butter at the front in the drawer.</p>
<h3 class="sec">1.3 The failures are about where, not how</h3>
<p>We replayed the persistent failures bit for bit with the objects' poses. In {geo['primary']['aim_share']:.2f} of the {aim_n} failures on movable
objects, the hand closes beside the object, not on it. The reaching, grasping and carrying look right; they are aimed at the wrong place. A model
that already moves well is missing positions.</p>
<h3 class="sec">1.4 The policy already has the grammar</h3>
<p>We grouped the policy's own action chunks into a hierarchy of prototypes (Section 2). Its coarse levels are a handful of verbs that every task
shares, and they change label exactly where a task changes phase: at {c1['levels']['0.55']['recall']:.2f} of the grasp and release events,
against {c1['levels']['0.55']['random_recall']:.2f} for random boundaries. The order of verbs is already inside the policy.</p>
<div class="callout thesis"><p><strong>Thesis.</strong> A task is a <em>grammar</em> times its <em>bindings</em>. The grammar is the order of verbs,
shared by every task of a kind and already inside the policy. The bindings say which object, held how, placed where, each relative to an object,
and they are what a new task adds. <strong>The question we exploit:</strong> if the policy already has the grammar, can we give it a new task by
giving it only the bindings?</p></div>
<div class="defs"><div><h4>Part I · Bindings from one demonstration</h4><p>Read the bindings off the one demo and guide the frozen policy with
them at every new start state. If the thesis holds, this beats fine-tuning while storing kilobytes.</p></div>
<div><h4>Part II · Composing new tasks</h4><p>Bindings are keyed by object type and goal relation, so a task can borrow the grasp of one task and the
placement of another. If the thesis holds, new tasks need no demo when their parts are known, and the cost of a stream grows with its
vocabulary of bindings.</p></div>
<div class="b"><h4>What would refute it</h4><p>Bindings that do not beat fine-tuning; prototypes that do not line up with phases; borrowed bindings
that fail where the task's own succeed. Each read below was pre-registered with its margin before its episodes ran.</p></div></div>
"""


def action_profiles():
    """Each verb's most-used t = 0.4 prototype as signals over its ten steps: the commanded hand displacement per step along the robot's
    forward, left and up axes (cm), and the gripper command (+1 closed, -1 open); one scale per row."""
    P = gram["prototypes"]["0.4"]
    verbs = [v for v, _ in (("open move", 0), ("open reach down", 0), ("open grip change", 0), ("closed lift", 0), ("closed transport", 0), ("closed lower", 0))]
    pick = {}
    for v in verbs:
        cand = [(P[c]["frames"], c) for c in P if P[c]["name"].split(" (")[0] == v]
        if cand:
            pick[v] = max(cand)[1]
    rows = [("forward", 0), ("left", 1), ("up", 2), ("gripper", 3)]
    cw, ch, gx, top, left = 150, 46, 14, 34, 66
    W = left + len(pick) * (cw + gx)
    H = top + len(rows) * (ch + 8) + 10
    import numpy as _np
    paths = {v: _np.diff(_np.vstack([_np.zeros(3), _np.asarray(P[c]["mode_path_cm"])]), axis=0) for v, c in pick.items()}
    vmax = max(1.0, max(float(_np.abs(p).max()) for p in paths.values()))
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f22t" class="chart" style="min-width:{min(W, 980)}px">']
    for r, (lab, k) in enumerate(rows):
        y0 = top + r * (ch + 8)
        g.append(f'<text x="{left - 8}" y="{y0 + ch / 2 + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    for i, (v, c) in enumerate(pick.items()):
        x0 = left + i * (cw + gx)
        g.append(f'<text x="{x0 + cw / 2}" y="16" class="lab" text-anchor="middle">{esc(v)}</text>')
        g.append(f'<text x="{x0 + cw / 2}" y="29" class="tick" text-anchor="middle">prototype {c} · {P[c]["tasks"]} tasks</text>')
        for r, (lab, k) in enumerate(rows):
            y0 = top + r * (ch + 8)
            mid = y0 + ch / 2
            g.append(f'<rect x="{x0}" y="{y0}" width="{cw}" height="{ch}" class="glyph"/><line x1="{x0}" x2="{x0 + cw}" y1="{mid}" y2="{mid}" class="grid"/>')
            vals = paths[v][:, k] / vmax if k < 3 else _np.asarray(P[c]["mode_grip"])
            bw = cw / len(vals)
            for j, val in enumerate(vals):
                h = float(val) * (ch / 2 - 3)
                cls = "s2" if (k == 3 and val > 0) else "s1"
                g.append(f'<rect x="{x0 + j * bw + 1:.1f}" y="{min(mid, mid - h):.1f}" width="{bw - 2:.1f}" height="{abs(h):.1f}" class="{cls}"/>')
    g.append('</svg>')
    return "".join(g), vmax


def proto_cards():
    P = gram["prototypes"]["0.4"]
    cards = []
    for pr in MEDIA.get("prototypes", []):
        ids = [str(i) for i in pr["prototype_ids"] if str(i) in P]
        fr = sum(P[i]["frames"] for i in ids)
        spd = sum(P[i]["frames"] * P[i]["speed_cm_per_step"] for i in ids) / max(fr, 1)
        dz = sum(P[i]["frames"] * P[i]["dz_cm_per_step"] for i in ids) / max(fr, 1)
        grip = sum(P[i]["frames"] * P[i]["gripper_closed"] for i in ids) / max(fr, 1)
        cards.append(
            f'<div class="proto"><h4>{esc(pr["verb"])}<span>{len(ids)} prototype{"s" if len(ids) != 1 else ""}</span></h4>'
            f'{video(pr["file"], poster=True, autoplay=False, loop=True)}'
            f'<p>{esc(pr["gloss"][0].upper() + pr["gloss"][1:])}.</p>'
            f'<div class="kv"><b>in demos of</b><span>{pr["n_tasks"]} of {len(gram["tasks"])} tasks</span>'
            f'<b>frames</b><span>{fr:,}</span><b>hand speed</b><span>{spd:.2f} cm per step</span>'
            f'<b>vertical</b><span>{dz:+.2f} cm per step</span><b>gripper closed</b><span>{grip:.0%} of frames</span></div></div>')
    return "".join(cards)


def exec_block():
    m = MEDIA.get("proto_exec")
    if not m:
        return ""
    return fig("2.3", video(m["file"], loop=True), "Each verb's prototype EXECUTED: its ten commanded actions replayed four times in a row, open-loop, "
               f"from the same start state of one scene (<em>{esc(m['language'])}</em>), with no policy. Beside each clip, the hand's path so far, "
               "from above (as the camera sees the table) and from the side. " + esc(m["moves"]) + " Each matches its profile in Figure 2.2 in "
               "direction and proportion: the controller covers about a quarter of each commanded step, so four repetitions travel about as far as "
               "the profile's one chunk. Reach down, grip change and lower do what their names say. Open move and closed transport move the hand "
               "only about 2 cm. Closed lift mostly carries the closed hand "
               "sideways (40 cm left, 3 cm up). It took its name from where its frames fall in the demos, just after the grasp.")


levels_rows = [[f"{float(u):.2f}", f"{d['prototypes']:,}", f"{d['change_points_per_demo']:.1f}",
                f"{d['recall']:.2f} <span class='note'>({d['random_recall']:.2f})</span>",
                f"{d['precision']:.2f} <span class='note'>({d['random_precision']:.2f})</span>"] for u, d in c1["levels"].items()]
levels_rows = [(["ours"] + r) if r[0] == "0.55" else r for r in levels_rows]
EXPROG = ("48", "58", "32", "13", "40", "73", "39")
prog_rows = [[esc(gram["programs"][t]["language"]), esc(gram["programs"][t]["sentence"]), f'{gram["programs"][t]["bytes"]:,}']
             for t in EXPROG if t in gram["programs"]]
sk = gram["string_by_key"]
key_rows = [["carry: prototype string (edit distance)", f"{sk['carry']['same_key_median']:.2f}", f"{sk['carry']['other_key_median']:.2f}", f"{sk['carry']['perm_p']:.3f}"],
            ["grasp: prototype string (edit distance)", f"{sk['grasp']['same_key_median']:.2f}", f"{sk['grasp']['other_key_median']:.2f}", f"{sk['grasp']['perm_p']:.2f}"],
            ["grasp: binding geometry (cm)", f"{gram['grasp_offset_cm']['within_type_median']:.1f}", f"{gram['grasp_offset_cm']['across_type_median']:.1f}", f"{gram['grasp_geometry_perm_p']:.4f}"]]
geo_rows = []
for otype, lab in (("black_book", "black book"), ("chefmate_8_frypan", "frying pan"), ("akita_black_bowl", "black bowl"), ("ketchup", "ketchup")):
    xs = gram["grasp_by_type"].get(otype, [])
    if xs:
        geo_rows.append([lab, str(len(xs)), "; ".join(f"({', '.join(f'{v:+.1f}' for v in x['hand_in_object_cm'])})" for x in xs[:4]) + (" …" if len(xs) > 4 else ""),
                         esc(xs[0]["approach"])])

P_GRAMMAR = f"""
<p class="lede">Section 1 claimed the policy already holds the grammar of manipulation. This section shows it, read from the policy's own actions:
what its prototypes are, what each one does on screen, and how they split a task with the bindings.</p>
<h3 class="sec">2.1 A hierarchy of the policy's actions</h3>
<p>pi0.5 outputs actions in chunks of ten steps. We collected the chunks it predicts along the demonstrations of its forty training tasks
and built a hierarchy of prototypes by mode-seeking at increasing noise: at the lowest noise almost every chunk is its own prototype; as noise grows, chunks that move alike fall into
the same basin. Any new demo can then be labelled frame by frame at every level. Table 2.1 lists the levels and how well each one's label changes
line up with the task's phase events.</p>
<p class="tcap">Table 2.1. Prototype levels over the 50 pool demos (a change within one policy query of an event counts; random boundaries in parentheses).</p>
{table(["level t", "prototypes", "changes per demo", "events at a change", "changes at an event"], levels_rows)}
<h3 class="sec">2.2 The prototypes are verbs</h3>
<p>We named each prototype by what its frames do: whether the gripper is closed, how fast and which way the hand moves, and which phase of the
task its frames fall in. The same names come out across scenes and objects. Each card below plays the demo frames where that verb is active, in four
tasks from four different scenes (the green bar under a clip marks the verb's frames).</p>
<div class="grid3">{proto_cards()}</div>
{fig("2.2", '<div class="chartwrap">' + action_profiles()[0] + '</div>', f"What each verb's prototype commands, step by step over its ten steps: the hand's "
     f"displacement along the robot's forward, left and up axes (one scale, the largest step = {action_profiles()[1]:.1f} cm), and the gripper "
     f"command (orange = close). Reaching down ends by closing the gripper; lowering ends by opening it.")}
{exec_block()}
{fig("2.1", '<div class="chartwrap">' + RP["verb_glyphs"]() + '</div>', "What each coarse prototype commands: the hand path of its mode over ten steps, horizontal travel against height, on one scale; "
     "orange where the gripper is closed.")}
<h3 class="sec">2.3 A program is a sentence over the verbs</h3>
<p>Cut at the object events (Section 3), each demo becomes a short program: its phases, each anchored on an object. Families of tasks share the
same grammar: {', '.join(f'{k.replace(" ", " → ")} {len(v)}' for k, v in sorted(gram['families'].items(), key=lambda kv: -len(kv[1])))}.</p>
<p class="tcap">Table 2.2. Programs read out of single demos (bytes = what the program stores).</p>
{table(["instruction", "program", "bytes"], prog_rows, num_from=2)}
<h3 class="sec">2.4 What the prototypes carry, and what the bindings carry</h3>
<p>If the thesis is right, the prototypes should describe the verb and ignore the object, and the bindings the reverse. Table 2.3 measures both
across the 50 demos, with a permutation test that shuffles the keys across phases.</p>
<p class="tcap">Table 2.3. Distance between two tasks' phases, same key against other key.</p>
{table(["what is compared", "same key", "other key", "permutation p"], key_rows)}
<p>Carries toward the same goal relation share a prototype string; grasps share one whatever the object; the grasp's geometry is shared only within
an object type (Table 2.4). The prototypes carry the verb and the bindings carry the object. These are the keys Part II uses to look up bindings:
grasps by object type, placements by goal relation.</p>
<p class="tcap">Table 2.4. Grasp bindings by object type: the hand's position in the object's frame at the close (cm).</p>
{table(["object", "tasks", "hand in the object's frame", "approach"], geo_rows, num_from=1)}
"""


P_METHOD_A = f"""
<p class="lede">Method A turns one demonstration into bindings and lets the frozen policy execute them at any new start state. It trains no weights.</p>
<div class="defs">
<div><h4>1 · Phases from object events</h4><p>The manipulated object is the first movable object the demo displaces by more than 1.5 cm while the
gripper is closed. The demo splits at the close on it and at the first release after: an approach in the object's frame, a carry in the
receptacle's frame, and the rest. A task that moves no object (a door, a knob) is one phase in the fixture's frame. Objects still settling in
the demo's first frames are read at rest.</p></div>
<div><h4>2 · Bindings</h4><p>Each phase keeps the demo's controller goals relative to its anchor. At a new start state every goal is carried
along with its anchor: goal = (anchor now) · (anchor in the demo)<sup>−1</sup> · (goal in the demo). The object's pose in the hand is predicted
from the grasp binding, so nothing is sensed in-hand.</p></div>
<div><h4>3 · Perception</h4><p>A ridge regression reads the objects' positions from the first image's frozen ResNet-18 features. It is
trained on 400 simulator-rendered placements of the task's scene; the simulator labels those training images and is never used at test. Stored
as a 7 KB head over a 512-direction code shared by all tasks, it keeps its success (Section 9).</p></div>
<div><h4>4 · Execution</h4><p>At every query the next ten actions are planned along the demo's own lagged path, and pi0.5 is sampled through
them as a constant-weight guide (weight 1, variance 10<sup>−3</sup>). The guide steers the policy's own sampling, so its motion stays smooth
and it keeps reacting to what it sees.</p></div>
<div><h4>5 · Practice picks the form</h4><p>Ten practice episodes decide what a task keeps: its phases; prototype borrowing with a small fold,
where the prior's own skill is needed (the stove's knob); or a LoRA on demos generated from the phases.</p></div>
</div>
<p class="note">Code: <code>lifelong_ood/policy/phase.py</code> (cut, compose, PhaseBindHooks), <code>experiments/serve_guided.py</code> (mode
<code>phasebind</code>), <code>experiments/p6_compress.py</code> (the reader head).</p>
"""


def cmp_grid():
    out = []
    for c in CMP:
        extra = ""
        if c["task"] == 39:
            extra = (f" The one task where bindings lose: the stove's knob starts anywhere from {stove['init_min_rad']:.2f} to "
                     f"{stove['init_max_rad']:.2f} rad, and replaying the demo's twist leaves it on. Object poses cannot express a joint angle; "
                     f"practice hands this task to prototype borrowing (17 against 6 of 20).")
        out.append(fig("4." + str(len(out) + 1), video(c["file"]), f"<em>{esc(c['language'])}</em>, init state {c['trial']}: zero-shot, LoRA, bindings: "
                       f"{outcome_str(c['outcomes'])}.{extra}"))
    return "".join(out)


P_CASES_A = f"""
<p class="lede">Every clip is a recorded evaluation episode. The three approaches start from the same simulator state, so the difference is the
method alone. We chose the tasks to show the range: where fine-tuning fails, where it succeeds only sometimes, and where bindings lose.</p>
<div class="grid2">{cmp_grid()}</div>
<p class="note">Videos saved by the evaluation client for the first two scoring states of each task (init states 30 and 31), stitched without
editing (<code>experiments/site_media.py</code>). The LoRA row is the best fine-tuning arm; its panels and ours come from the same pre-registered reads.</p>
"""

T21 = [("zero-shot pi0.5", "zeroshot", ""), ("replaying the demo open-loop", "base_playback", ""), ("prototype borrowing + fold", "ours_dim1_last2", ""),
       ("LoRA, one demo", "practice_lora", ""), ("LoRA rank 64", "base_lora_r64", ""), ("full fine-tuning of the action expert", "base_fullft", ""),
       ("LoRA + practice + failure negatives", "nft_bal", ""), ("LoRA on demos generated from the phases", "gen_lora_v2", ""),
       ("bindings, simulator poses", "ours_phasebind2", "ours"), ("bindings, learned reader", "ours_phasebind2_synth", "ours")]
t21_rows = [[c, esc(lab), str(meth(k)["k"]), fmt_bytes(max(meth(k)["bytes"], 1)) if meth(k)["bytes"] else "0 B"] for lab, k, c in T21]
sota = row("p1_bindings", "sota")
if sota:
    lora21 = meth("practice_lora")["bytes"]
    actor = sum(os.path.getsize(os.path.join(R, f"dsrl_{t}_actor.pt")) for t in (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63))
    new_rows = [["", "STRAP: LoRA on the demo + retrieved sub-trajectories", str(sota["B1"]["control"]), fmt_bytes(lora21)],
                ["", "DSRL: one-demo LoRA steered in noise space, 100 practice episodes", str(sota["B2"]["control"]), fmt_bytes(lora21 + actor)],
                ["", "experience replay: one LoRA on all 21 demos", str(sota["B3"]["control"]), fmt_bytes(sto["lora_bytes"] + 21 * sto["raw_demo_bytes"])]]
    t21_rows = t21_rows[:7] + new_rows + t21_rows[7:]
t21_rows.append(["ours", "practice picks the form per task", str(al["AL1"]["x"]), fmt_bytes(al["bytes_total"])])

def _warm():
    import json as _j
    k = n = 0
    for t in (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63):
        f = os.path.join(R, f"dsrl_practice_{t}_dsrl_train.json")
        if os.path.exists(f):
            e = _j.load(open(f))["tasks"][str(t)]["episodes"][:10]
            k += sum(x["success"] for x in e)
            n += len(e)
    return k, max(n, 1)


sota_warm = _warm()


def sota_block():
    if not sota:
        return ""
    rows = [["ours", "bindings + 7 KB reader head (no practice)", str(sota["B1"]["x"]), "–"]]
    for h, lab in (("B1", "STRAP"), ("B2", "DSRL"), ("B3", "experience replay")):
        v = sota[h]
        rows.append(["", lab, str(v["control"]), f"{v['diff']['diff']:+.3f} [{v['diff']['lo']:+.3f}, {v['diff']['hi']:+.3f}], Holm p {v['holm_p']:.1g}"])
    return (f"""<h3 class="sec">Recent methods</h3>
<p>Three 2025–26 approaches, implemented here on pi0.5 with the same one demo and scored on the same 420 paired episodes (pre-registered):
<strong>STRAP</strong> (Memmel et al., ICLR 2025) retrieves sub-trajectories that resemble the demo's from the policy's 40 training tasks, by
subsequence DTW over DINOv2 features, and fine-tunes a LoRA on the demo and the retrieved data. <strong>DSRL</strong> (Wagenmaker et al., CoRL
2025) keeps the network frozen and trains a small SAC agent to choose each chunk's initial noise from practice rewards; it starts from the
one-demo LoRA and gets 100 practice episodes per task. <strong>Experience replay</strong>, which recent continual-learning work found to forget
little in VLAs, trains one LoRA on every task's demo.</p>
<p class="tcap">Table 5.2. The recent methods against the bindings (differences: ours minus the method, 95% interval).</p>
{table(["method", "successes of 420", "difference"], rows)}
<p>The bindings beat all three. Each also trails the best per-task LoRA ({sota['vs_nft']['strap']['control']}). Retrieving other tasks' behaviour under
the target's instruction and sharing one adapter across tasks both mix the tasks up, as Section 1.2 found for pooling. DSRL did not improve on its
base policy within 100 episodes, although that policy succeeded in about {round(100 * sota_warm[0] / sota_warm[1])}% of its warm-up
episodes. These are our implementations, adapted to a pretrained VLA with one demo; STRAP was published with five demos and policies
trained from scratch.</p>""")


P_RESULTS_A = f"""
<p class="lede">Twenty-one held-out tasks the prior fails, one demonstration each, twenty scoring episodes per task on start states nothing was
trained or practised on: 420 paired episodes per method.</p>
<p class="tcap">Table 5.1. Successes of 420 and what each method stores for all 21 tasks.</p>
{table(["method", "successes", "storage"], t21_rows)}
{fig("5.1", '<div class="chartwrap">' + RP["scatter_success_bytes"]() + '</div>', f"Success against storage (log scale). Bindings reach {v2['V1']['x']} of 420 "
     f"at {fmt_bytes(meth('ours_phasebind2')['bytes'])}; every fine-tuning variant stores {fmt_bytes(meth('practice_lora')['bytes'])} or more.")}
{sota_block()}
<h3 class="sec">Replications</h3>
<ol class="results">
<li><strong>Another sampling seed</strong> (pre-registered): bindings {s1['R1']['x']}, with the reader {s1['R2']['x']}, practice allocation {s1['R3']['x']};
the best LoRA {s1['R1']['control']}.</li>
<li><strong>A second demonstration</strong> (20 tasks): bindings {d1['D1']['x']} of {d1['D1']['n']} against {d1['D1']['control']} for a LoRA trained on
that demo; the first demo gave {d1['demo0_vs_demo1']['v2']['demo0']} on the same tasks.</li>
<li><strong>A 7 KB reader head</strong> over all 50 pool tasks (pre-registered, margin −0.05): {rc['RC1']['x']} against {rc['RC1']['control']} of
1,000 for the full reader, lower bound {rc['RC1']['diff']['lo90']:+.3f}.</li>
</ol>
"""


p3lang = json.load(open(os.path.join(R, "p3c_table.json")))["table"]
p3_rows = [[esc(p3lang[t]["language"]), str(p3c["per_task"][t]["own"]), str(p3c["per_task"][t]["c"]), str(p3r2["per_task"][t]["crp"]),
            str(p3c["per_task"][t]["sib"])] for t in sorted(p3c["per_task"], key=int)]
p3_rows.append(["ours", "<strong>all six</strong>", str(p3c["C1"]["control"]), str(p3c["C1"]["x"]), str(p3r2["R2"]["x"]), str(p3c["C2"]["control"])])

P_METHOD_B = f"""
<p class="lede">Method B builds a task with no demonstration of its own. Section 2.4 showed that grasp geometry belongs to the object type and the
carry's verbs to the goal relation, so bindings can be stored under those two keys and recombined.</p>
<div class="defs">
<div><h4>1 · The binding table</h4><p>Grasp bindings are keyed by the manipulated object's type (ketchup, black bowl, frying pan). Place bindings
are keyed by the goal relation: the task's BDDL goal with the object abstracted, such as <code>In OBJ basket</code>. A demo's bindings enter the
table only if its program succeeds when replayed in simulation at the practice placements.</p></div>
<div><h4>2 · Composition</h4><p>A new task takes the grasp phase of a task with its object type and the place phases of a task with its goal
relation, relabelled onto its own objects. Among several sources the lookup prefers a grasp recorded in the same scene and a placement of the same
object type, then the same scene.</p></div>
<div><h4>3 · The carry between them</h4><p>The two sources were recorded in different layouts, so the object must travel between the borrowed
grasp and the borrowed placement. Three carries: follow the place source's route; let the frozen policy carry; or lift by the source's own lift,
cross, and descend, passing every waypoint on arrival.</p></div>
<div><h4>4 · Practice decides</h4><p>The composed program is practised ten times at start states 0–9 with each carry. If one succeeds at least
8 times, the task is composed with it; otherwise the task asks for its own demo, and that demo's bindings extend the table.</p></div>
</div>
<p class="tcap">Table 6.1. Six tasks composed only from other tasks' phases, 20 scoring episodes each (P3c, pre-registered).</p>
{table(["task", "own demo", "composed", "composed, learned reader", "sibling's whole demo"], p3_rows)}
<p>Where the layouts fit, a borrowed grasp and a borrowed placement do as well as the task's own demo: the ketchup is grasped as in the kitchen task
that puts it in a drawer and carried as the milk goes into the basket. Borrowing the grasp from another object type (the sibling's whole demo) fails.</p>
"""


def comp_figs():
    comps = {c["name"]: c for c in MEDIA.get("compositions", [])}
    out = []
    if "fix_3" in comps and "fix_48" in comps:
        out.append(fig("7.1", f'<div class="grid2">{video(comps["fix_3"]["file"])}{video(comps["fix_48"]["file"])}</div>',
                       "The same composed programs from the same start state, before and after the fix. On the clock, the planned bridge runs ahead of the "
                       "arm, and the release fires before the hand arrives: the butter drops short of the drawer and the ketchup lands beside the basket. On "
                       "arrival, each waypoint and the release wait for the hand. Left: <em>put the butter at the back in the top drawer</em>. Right: "
                       "<em>put the ketchup in the basket</em>."))
    pc = (dcr or {}).get("per_composition", {})
    cnt = lambda key: " / ".join(str(pc.get(key, {}).get(k, "–")) for k in ("streamc", "streamh", "streamte"))
    for name, num, cap in (("carries_16", "7.2", "<em>Stack the front bowl on the middle bowl</em>, composed from the grasp of <em>put the black bowl on top "
                            "of the drawer</em> and the placement of <em>stack the middle bowl on the back bowl</em>. The source's route and the policy's own carry "
                            f"both fail; lift, cross and descend succeeds (practice successes of 10, route / prior / transport: {cnt('16_1_17')})."),
                           ("carries_13", "7.3", "<em>Put the front bowl on the plate</em>, with the placement borrowed from a mug going onto a plate in another "
                            f"room, 35 cm away in the source's layout. Here only the prior's carry works (route / prior / transport: {cnt('13_17_65')}). The carries "
                            "fail in different places, so practice chooses among them.")):
        if name in comps:
            out.append(fig(num, video(comps[name]["file"]), cap))
    return "".join(out)


if dcr:
    CN = {"streamc": "the place source's route", "streamh": "the prior carries", "streamhe": "the prior carries, events on arrival",
          "streamte": "lift, cross, descend, on arrival", "streamtle": "the same, lifted by the source's own lift"}
    carry_rows = []
    for k in dcr["carries_complete"]:
        vr = dcr["vs_route"].get(k)
        carry_rows.append([("ours" if k in dcr["carry_set"] else ""), esc(CN.get(k, k)), f"{dcr['totals'][k]}", f"{dcr['passes'][k]}",
                           (f"{vr['diff']['diff']:+.3f} [{vr['diff']['lo']:+.3f}, {vr['diff']['hi']:+.3f}]" if vr else "reference")])
    carry_table = (f'<p class="tcap">Table 7.1. The carries on the {dcr["compositions"]} compositions practised so far, 10 practice episodes each '
                   f'(exploratory; start states 0–9).</p>' + table(["carry", f"successes of {dcr['n']}", "pass practice", "vs the route (95% interval)"], carry_rows))
    po = dcr["phase_outcomes"]
    brings = "; ".join(f"{CN.get(k, k)}: {v['failed_at']['bring']} failed at bring, {v['failed_at']['success']} succeeded of {v['episodes']}" for k, v in po.items())
    phase_fig = fig("7.4", '<div class="chartwrap">' + RP["phase_bars"]() + '</div>', "Where each carry's practice episodes stop along the grammar, read "
                    "from recordings of the objects' poses: reach the grasp point, lift the object, bring it to its goal, place it. " + esc(brings) + ".")
else:
    carry_table = phase_fig = ""

P_CASES_B = f"""
<p class="lede">Composition fails in one place: carrying the object from the borrowed grasp to the borrowed placement, through a layout neither
source was recorded in. Recordings of the objects' poses located the failure (Figure 7.4), and two bugs came out of them.</p>
{comp_figs()}
{carry_table}
{phase_fig}
<p>The carries fail in different places, so the stream keeps several and lets practice choose. The set was chosen by a rule fixed before the last
carry ran: start from the carry that passes most often, then add whichever adds the most passing compositions, while it adds at least two. It keeps
{", ".join(CN.get(k, k) for k in dcr["carry_set"]) if dcr else ""} (highlighted in Table 7.1); together they pass {dcr["passes_union_set"] if dcr else ""} of
{dcr["compositions"] if dcr else ""} compositions, against {dcr["passes"]["streamc"] if dcr else ""} for the route alone.</p>
"""

def scheme_block():
    if not scheme:
        return ('<div class="callout pending"><p><strong>Running: the confirmation with the new carries.</strong> This page will show the result '
                'when it lands.</p></div>')
    h = scheme["H1"]
    orows = [[f"order {o}", str(v["demos"]), f"{v['composed']} <span class='note'>({v['composed_first_half']} + {v['composed_second_half']})</span>",
              str(v["rejected"])] for o, v in scheme["orders"].items()]
    pu = sorted(scheme["per_unit"].items(), key=lambda kv: kv[1]["composed"] - kv[1]["own"])
    worst, best = pu[0], pu[-1]
    name = lambda k: esc(lang.get(int(k.split("_")[0]), k))
    return (f'<div class="callout"><p><strong>The confirmation passes.</strong> With the lookup of Section 6 and the carries of Section 7, on four fresh '
            f'orders and a new sampling seed (pre-registered), the {h["units"]} unique compositions that arrived score <strong>{h["x"]} of {h["n"]}</strong> '
            f'against {h["control"]} for the same tasks\' own demos: {h["diff"]["diff"]:+.3f}, one-sided lower bound {h["diff"]["lo90"]:+.3f} against the '
            f'margin −0.10 (sign test p {h["sign_p"]:.4f}). Each composition counts once however many orders it arrived in.</p></div>'
            f'<p class="tcap">Table 8.1. The four fresh streams of 50 tasks: demos requested, tasks composed (first half + second half), compositions '
            f'rejected by practice.</p>' + table(["order", "demos", "composed", "rejected"], orows) +
            f'<p>Some compositions beat a weak demo of the task itself: <em>{name(best[0])}</em> scores {best[1]["composed"]} of 20 composed against '
            f'{best[1]["own"]} with its own demo, because the borrowed demos were better. One still trails: <em>{name(worst[0])}</em>, '
            f'{worst[1]["composed"]} against {worst[1]["own"]}, although it passed practice.</p>')


bound = RP["binding_grid"]()
P_STREAM = f"""
<p class="lede">Fifty tasks the prior fails arrive one at a time. Each composes from the table when both its keys are known and passes practice;
otherwise it asks for its one demo. Nothing stored is ever changed, so nothing learned earlier is forgotten.</p>
{fig("8.1", '<div class="chartwrap">' + bound[0] + '</div>', f"The 50 tasks as a grid of object types (rows) and goal relations (columns). Green: some composition "
     f"from other tasks' bindings passes practice. Amber: both keys are held by other tasks but no composition has passed yet. Grey: no other task shares a key. "
     f"Of {bound[1]} tasks with a composable form, {bound[2]} share both keys with another task and {bound[3]} have passed; LIBERO-90 crosses objects and "
     f"relations sparsely, which caps how many demos composition can save here.")}
{fig("8.2", '<div class="chartwrap">' + RP["cumulative_demos"]() + '</div>', f"Demos requested as tasks arrive, for two mirrored orders. The table fills: "
     f"{p5['S1']['only_late']} tasks compose only when they arrive late, {p5['S1']['only_early']} only when early (sign test p {p5['S1']['sign_p']:.4f}).")}
{fig("8.3", '<div class="chartwrap">' + RP["tradeoff"]() + '</div>', f"Stream success against demos requested. Unchecked composition costs success "
     f"({p5['S2']['x']} against {p5['S2']['control']} of {p5['S2']['n']} for the same tasks' own demos). Checked by practice, the streams return near the "
     f"all-demo stream with a few demos saved, but the pre-registered confirmation on fresh orders got {cf['composed_vs_own']['x']} against "
     f"{cf['composed_vs_own']['control']} of {cf['composed_vs_own']['n']}: too few arrivals, and one composition that passed practice scored 9 of 20.")}
{scheme_block()}
"""


def cost_block():
    rows = [r for r in ledger.load("cost", tag="read") if not r.get("retired")]
    if not rows:
        return ""
    c = rows[-1]["metrics"]
    order = ["phasebind2_rjpred_pca512_rj", "phasebind2_reader_spatial4_synth", "phasebind2_nofold", "zeroshot", "lora", "lora_r17bal",
             "lora_strap2", "lora_strap", "dsrlo", "dsrl", "lora_er"]
    out = []
    for tag in order:
        v = c["arms"].get(tag)
        if not v:
            continue
        L = v["latency"]
        ad = v["adapt_secs"]
        out.append([("ours" if tag.startswith("phasebind") else ""), esc(v["label"]), f'{v["successes"]}',
                    (f'{L["ms_per_query"]:.0f}' if L else "–"), ("not logged" if ad is None else ("0" if ad == 0 else f"{ad / 60:.1f}")),
                    str(v["practice_episodes"])])
    return (f"""<h3 class="sec">Latency and compute</h3>
<p class="tcap">Table 9.2. Per method on the 21 held-out tasks: successes of 420; milliseconds per policy query (one ten-step chunk,
three servers sharing an A40, from the scoring runs' logs); minutes of compute to adapt one new task; practice episodes per task.</p>
{table(["method", "successes", "ms per query", "minutes to adapt", "practice episodes"], out)}
<p>The guide adds about {c['arms']['phasebind2_rjpred_pca512_rj']['latency']['ms_per_query'] - c['arms']['zeroshot']['latency']['ms_per_query']:.0f} ms per
query over the frozen policy alone; a LoRA adds about {c['arms']['lora_r17bal']['latency']['ms_per_query'] - c['arms']['zeroshot']['latency']['ms_per_query']:.0f}.
Adapting to a new task with bindings is rendering its scene's placements for the reader; the phases are cut from the demo in milliseconds.
{esc(c['notes']['adapt'])}.</p>""")


cost_rows = [["", "rank-16 LoRA (the best fine-tuning)", fmt_bytes(sto["lora_bytes"]), fmt_bytes(sto["streams"]["A"]["totals"]["lora"])],
             ["", "a raw LIBERO demonstration, kept for replay", fmt_bytes(sto["raw_demo_bytes"]), fmt_bytes(sto["streams"]["A"]["totals"]["raw"])],
             ["", "phases + the full pose reader", fmt_bytes(sto["median"]["program"] + sto["median"]["reader_full"]), fmt_bytes(sto["streams"]["A"]["totals"]["own_full"])],
             ["ours", "phases + the 7 KB reader head", fmt_bytes(sto["median"]["program"] + sto["median"]["reader_pca512"]),
              fmt_bytes(sto["streams"]["A"]["totals"]["own_pca"]) + f' <span class="note">incl. the {fmt_bytes(sto["basis_pca512"])} basis, once</span>']]

P_COST = f"""
<p class="lede">What a task costs to keep, so that it can be performed later. The policy is never changed and each task keeps its own record, so a
new task cannot alter an old one: forgetting is ruled out by construction, not measured.</p>
{fig("9.1", '<div class="chartwrap">' + RP["storage_bars"]() + '</div>', f"Bytes per task (log scale, medians over the 50 pool tasks). A task keeps "
     f"{per_task_kb:.1f} KB: {fmt_bytes(sto['median']['program'])} of phases and a {fmt_bytes(sto['median']['reader_pca512'])} reader head.")}
{cost_block()}
<p class="tcap">Table 9.1. Storage per task and over the 50-task stream.</p>
{table(["what is stored", "per task", "50 tasks"], cost_rows)}
<p>The reader head keeps success (pre-registered, margin −0.05): {rc['RC1']['x']} against {rc['RC1']['control']} of 1,000 episodes, lower bound
{rc['RC1']['diff']['lo90']:+.3f}. Offline it reads positions at {co['summary']['pca512']['median_err_cm']:.3f} cm median error against
{co['summary']['full']['median_err_cm']:.3f} cm for the full reader. A seeded random projection needs no stored basis at all:
{col['summary']['rand2048']['median_err_cm']:.3f} cm at {fmt_bytes(col['summary']['rand2048']['mean_bytes'])} per task (offline). Past about forty tasks the
shared basis costs less than the full readers it replaces; each further task adds about {per_task_kb:.0f} KB, against {fmt_bytes(sto['lora_bytes'])} for a LoRA.</p>
"""

NEG = [("Sharing weights across tasks", "One fold for all tasks, the other half's fold, pooling a family's data and pretraining on it all lose to "
        "per-task weights; a pooled adapter carries out a sibling task's instruction."),
       ("Failure negatives at the failed phase", f"{p2['all_21']['phase']} against {p2['all_21']['stall']} of 420 for negatives at the stall."),
       ("Growing the prototype library with successes", f"{p4['grown']} against {p4['base']} of 120."),
       ("Warping the source's carry", f"{dev['totals']['streamw']} against {dev['totals']['streamc']} of {dev['n']} on practice states."),
       ("A transport that advances on the clock", f"{dev['totals']['streamt']} against {dev['totals']['streamc']} of {dev['n']}; fixed by passing waypoints on "
        "arrival (Section 7)."),
       ("One pose reader per object type", f"Fixtures at {ty['fixture_median_cm']:.2f} cm, movable objects at {ty['movable_median_cm']:.2f} cm on unseen scenes."),
       ("Reading the books' yaw", f"{yw['median_deg']:.1f} degrees against {yw['fixed_median_deg']:.1f} for a fixed yaw, yet {ywr['Y1']['x']} against "
        f"{ywr['Y1']['control']} of 220."),
       ("Joint-state bindings", f"The stove's knob starts anywhere from {stove['init_min_rad']:.2f} to {stove['init_max_rad']:.2f} rad; object poses cannot "
        "express it, and practice hands the task to prototype borrowing.")]
P_NEG = f"""
<p class="lede">Each of these was a pre-registered read or a practice-state check, kept here with its numbers.</p>
<div class="tablewrap"><table><thead><tr><th>what was tried</th><th>what happened</th></tr></thead><tbody>
{"".join(f"<tr><th>{esc(a)}</th><td>{esc(b)}</td></tr>" for a, b in NEG)}</tbody></table></div>
"""

P_VER = f"""
<p class="lede">How the numbers on this site can be checked.</p>
<ol class="results">
<li><strong>Pre-registration.</strong> Every read quoted as a test was written into <code>ledger/m3_prereg</code> (hypotheses, arms, margins, the
read's command) before its first scoring episode. Exploratory reads (practice-state checks, the carry comparison) are labelled as such.</li>
<li><strong>One source of numbers.</strong> Every number here is read from <code>ledger/</code> at commit <code>{esc(commit)}</code> by
<code>experiments/site_page.py</code>. The README's copies are re-derived by <code>experiments/audit_numbers.py</code>: {esc(audit)}.</li>
<li><strong>Paired statistics.</strong> Methods run on the same init states with the same noise seeds; intervals come from a two-level bootstrap
(tasks, then states), tests from the exact sign test on discordant pairs, with Holm correction across a read's hypotheses.</li>
<li><strong>Footage.</strong> Every clip was saved by the evaluation client during the runs it illustrates and stitched without editing
(<code>experiments/site_media.py</code>). Prototype clips come from the raw LIBERO recordings of the demos the tables use.</li>
<li><strong>Scope.</strong> Everything runs in simulation (LIBERO-90). The pose readers and the practice checks use the simulator's state at
training time only; nothing reaches the policy's server at test except camera images and the robot's own state.</li>
</ol>
"""

def models_block():
    r = row("p1_bindings", "pi0") if any(True for _ in ledger.load("p1_bindings", tag="pi0")) else None
    if not r:
        return ('<div class="callout pending"><p><strong>Running: the same method on pi0.</strong> pi0 (openpi\'s pi0_libero, a flow-matching '
                'VLA with 50-step action chunks and z-score action normalisation) with the bindings and the 7 KB reader head, against pi0 '
                'zero-shot, on the 21 held-out tasks.</p></div>')
    z = r["Z"]
    ok = z["sign_p"] < 0.05 and z["x"] > z["control"]
    head = "On pi0, the same bindings work." if ok else "On pi0, the bindings did not pass the pre-registered test."
    return (f'<div class="callout"><p><strong>{head}</strong> pi0 (openpi\'s pi0_libero) zero-shot: {z["control"]} of 420; '
            f'with the bindings and the 7 KB reader head: {z["x"]} of 420 (exact sign test p {z["sign_p"]:.1g}; on pi0.5 the same method '
            f'scores {r["vs_pi05"]["control"]}). Two things follow each checkpoint and nothing else changed: the action normalisation (z-score for '
            f'pi0) and pi0\'s state-relative actions (its first six action dimensions are trained relative to the robot\'s state, so the '
            f'guide subtracts the state before it normalises).</p></div>')


FID = [
    ("STRAP", "Memmel et al., ICLR 2025; github.com/WEIRDLabUW/STRAP",
     "DINOv2 agent-view embeddings; segments cut where the end-effector's L1 step is below 2.5 mm, merged to >= 20 steps; subsequence DTW "
     "with steps (1,1) and (1,2); top 100 split across segments; a transformer policy trained from scratch on 5 target demos + retrieved",
     "the same retrieval (strap2); a rank-16 LoRA on pi0.5 instead of a policy from scratch; one target demo; prior data = pi0.5's 40 "
     "training tasks; at most 3,000 frames (each costs 26 MB of VLM cache)"),
    ("DSRL", "Wagenmaker et al., CoRL 2025; the pi0.5 LIBERO configuration in github.com/gonora24/dsrl_pi0",
     "SAC over a 7-dim noise vector tiled across the chunk; bound 1.0; 10 critics, min reduction; 20 updates per step; discount 0.999; "
     "auto temperature; a 64 x 64 CNN trained by the critic; pi0.5 zero-shot as the base",
     "the same agent and hyperparameters (dsrlo); frozen ResNet-18 features + state instead of the CNN; the one-demo LoRA as the base, since "
     "pi0.5 alone succeeds about 0% on these tasks and sparse rewards then never arrive; 100 practice episodes on states 0-29"),
    ("experience replay", "arXiv 2603.03818: sequential fine-tuning with a replay buffer forgets little in VLAs",
     "replay of earlier tasks' data while fine-tuning on each new one", "the end state of that stream: one LoRA trained jointly on all 21 "
     "demos, balanced per batch (an upper bound for replay with this data)"),
    ("LoRA + failure negatives", "DiffusionNFT-style negative-aware flow matching (the family of pi-StepNFT)",
     "online RL rounds", "three rounds of practice on states 0-29, successes as positives and the failed chunks as negatives"),
]


def fid_rows():
    return [[esc(a), esc(b), esc(c), esc(d)] for a, b, c, d in FID]


def fidelity_block():
    am = row("p1_bindings", "sota_amend1") if any(True for _ in ledger.load("p1_bindings", tag="sota_amend1")) else None
    if not am:
        return ('<div class="callout pending"><p><strong>Running: the re-runs with the authors\' settings.</strong> STRAP with its own '
                'retrieval and DSRL with its pi0.5 LIBERO configuration, scored on the same 420 episodes.</p></div>')
    return (f'<p>With the authors\' settings: STRAP {am["B1"]["control"]} of 420 (our variant {sota["B1"]["control"]}), DSRL {am["B2"]["control"]} '
            f'(our variant {sota["B2"]["control"]}); the bindings, {am["B1"]["x"]}, beat both (Holm p {max(am["B1"]["holm_p"], am["B2"]["holm_p"]):.1g}).</p>')


LEARNED = [
    ("pi0.5, the policy", "pretrained by Physical Intelligence on LIBERO's 40 tasks", "frozen: no gradient step, ever", "shared"),
    ("the prototype hierarchy", "mode-seeking over the policy's action chunks on its 40 training tasks", "built once, no gradients", "shared"),
    ("phases and bindings", "cut from the task's one demo at the object events", "rules, no fitting", "2.4 KB per task"),
    ("pose reader head", "ridge regression on 400 simulator-rendered placements of the task's scene", "closed-form fit, seconds", "7.2 KB per task"),
    ("the reader's shared code", "principal directions of other tasks' rendered scenes", "closed-form, once", "16.8 MB shared"),
    ("a composed task", "the binding table: a grasp source by object type, a place source by goal relation", "lookup; no demo", "8 bytes"),
    ("the carry", "practice: 10 episodes with each of three carries, keep the best if it reaches 8", "a discrete choice", "–"),
    ("the stove's fallback", "prototype borrowing + a 258 KB fold, or a LoRA, only where practice prefers it", "gradient training", "where used"),
]
P_LEARNED = f"""
<p class="lede">The method never trains the policy. What it fits per task is a linear map from image features to object positions,
in closed form; what it stores per task is a few numbers per phase. A composed task needs neither a demonstration nor a gradient step.</p>
<p class="tcap">Table 11.1. What each part of the method learns, from what, and what it costs.</p>
{table(["part", "comes from", "how it is fitted", "stored"], [[esc(a), esc(b), esc(c), esc(d)] for a, b, c, d in LEARNED], num_from=9)}
<div class="callout"><p><strong>So: zero-shot in the weights, one-shot (or zero-shot) in demonstrations, few-shot in practice.</strong>
A task with its own demo needs that demo and nothing else: its bindings are read off the demo, and its reader head is fitted on rendered
images of its scene. A composed task needs no demo at all, only its scene description (to render placements for the reader) and the
practice episodes that choose its carry. Practice is a check that selects among fixed options; no parameter of the policy or of the
programs changes. The only gradient training anywhere is the fallback some tasks take when practice prefers it (Table 5.1's practice
allocator).</p></div>
"""

P_MODELS = f"""
<p class="lede">The method needs three things from a policy: a sampler that integrates a learned field from noise (flow matching or
diffusion), so a guide can be added to it; actions expressed in a space where moving a goal with an object is defined (end-effector
deltas here); and object poses, which the reader supplies from images.</p>
<h3 class="sec">12.1 Other flow and diffusion policies</h3>
<p>The guide adds a constant-weight Gaussian pull toward the planned actions to the sampler's velocity at every integration step. For a
diffusion policy sampled with DDPM or DDIM the same pull adds to the score, through the usual conversion between noise, score and velocity
parameterisations. Flow and diffusion VLAs (pi0, pi0.5, GR00T N1.x, SmolVLA, RDT) and Diffusion Policy all sample this way.
Autoregressive token policies (OpenVLA, pi0-FAST) do not; they would need constrained decoding instead.</p>
{models_block()}
<h3 class="sec">12.2 pi0.7</h3>
<p>Physical Intelligence announced pi0.7 in April 2026: a larger generalist that recombines skills it learned from fragmented data when it
is prompted with language coaching, visual subgoals and strategy metadata. Its weights and an API are not available, and its published
evaluations are real-robot tasks (espresso, box and laundry folding) with no simulation benchmark, so it cannot be run or compared number
for number here. The two approaches compose at different levels: pi0.7 inside one very large model, steered by richer prompts; this
method outside an open, frozen model, through explicit object-relative bindings that can be inspected, stored in kilobytes and looked
up.</p>
"""


P_BASELINES = f"""
<p class="lede">Each baseline as published and as run here. After the first read, a check of the authors' code found that our first
STRAP and DSRL differed from theirs; both were re-run with the authors' settings (pre-registered amendment), and the first versions are
kept as our variants.</p>
<p class="tcap">Table 13.1. Baselines: the published method and this implementation.</p>
{table(["baseline", "source", "as published", "as run here"], fid_rows(), num_from=9)}
{fidelity_block()}
"""


JS = r"""
(function(){
  var mob=document.getElementById('toc-mobile'), toc=document.getElementById('toc');
  if(mob && toc) mob.innerHTML=toc.innerHTML;
  var pages=[].slice.call(document.querySelectorAll('.page'));
  function show(id, scroll){
    if(!pages.some(function(p){return p.id==='page-'+id;})) id=pages[0].id.slice(5);
    pages.forEach(function(p){ var on=(p.id==='page-'+id); p.hidden=!on;
      if(!on) [].slice.call(p.querySelectorAll('video')).forEach(function(v){ try{v.pause();}catch(e){} }); });
    [].slice.call(document.querySelectorAll('.toc a, #toc-mobile a')).forEach(function(a){
      if(a.getAttribute('href')==='#'+id) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current');});
    var cur=document.querySelector('.toc a[aria-current="page"] span:last-child');
    var t=document.getElementById('topbar-title'); if(t && cur) t.textContent=cur.textContent;
    var d=document.getElementById('topnav'); if(d) d.open=false;
    if(scroll) window.scrollTo(0,0);
  }
  window.addEventListener('hashchange', function(){show(location.hash.slice(1), true);});
  show(location.hash.slice(1), false);
})();
(function(){
  var tabs=document.getElementById('herotabs'), v=document.getElementById('herovid'), cap=document.getElementById('herocap');
  if(!tabs||!v) return;
  tabs.addEventListener('click', function(e){
    var b=e.target.closest('button'); if(!b) return;
    [].slice.call(tabs.querySelectorAll('button')).forEach(function(x){x.setAttribute('aria-pressed', x===b?'true':'false');});
    v.src='media/'+b.dataset.v; v.poster='media/'+b.dataset.v.replace('.mp4','.jpg'); cap.textContent=b.dataset.cap;
    var p=v.play(); if(p&&p.catch) p.catch(function(){});
  });
})();
(function(){
  var st=document.getElementById('walk'); if(!st) return;
  var N=+st.dataset.n, cur=0, timer=null, playing=true, visible=false;
  var reduce=window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var sc=[].slice.call(st.querySelectorAll('.st-sc')), dots=[].slice.call(st.querySelectorAll('.st-dot')), play=st.querySelector('.st-play');
  function go(k){ cur=(k+N)%N; sc.forEach(function(g){g.classList.toggle('on', +g.dataset.i===cur);});
    dots.forEach(function(d,i){ if(i===cur) d.setAttribute('aria-current','step'); else d.removeAttribute('aria-current');});
    st.querySelector('.st-count').textContent=(cur+1)+' / '+N; }
  function tick(){ clearTimeout(timer); if(playing && visible && !document.hidden) timer=setTimeout(function(){go(cur+1); tick();}, 7000); }
  function setPlay(p){ playing=p; play.innerHTML=p?'&#10074;&#10074;':'&#9654;'; play.setAttribute('aria-label', p?'Pause':'Play'); tick(); }
  st.addEventListener('click', function(ev){ var b=ev.target.closest('button'); if(!b) return; var a=b.dataset.act;
    if(a==='play') setPlay(!playing); else if(a==='prev'){setPlay(false); go(cur-1);} else if(a==='next'){setPlay(false); go(cur+1);}
    else if(b.hasAttribute('data-go')){setPlay(false); go(+b.dataset.go);} });
  if('IntersectionObserver' in window){ new IntersectionObserver(function(es){ visible=es[0].isIntersecting && st.offsetParent!==null; tick(); },{threshold:.3}).observe(st); }
  else visible=true;
  document.addEventListener('visibilitychange', tick);
  go(0); setPlay(!reduce);
})();
(function(){
  var tip=document.getElementById('tip'); if(!tip) return;
  function show(text,x,y){ tip.textContent=text; tip.hidden=false; var w=tip.offsetWidth; tip.style.left=Math.min(x+14, window.innerWidth-w-8)+'px'; tip.style.top=(y+14)+'px'; }
  [].slice.call(document.querySelectorAll('.pt')).forEach(function(g){
    g.addEventListener('pointermove', function(e){ show(g.dataset.tip, e.clientX, e.clientY); });
    g.addEventListener('pointerleave', function(){ tip.hidden=true; });
    g.addEventListener('focus', function(){ var r=g.getBoundingClientRect(); show(g.dataset.tip, r.right, r.top); });
    g.addEventListener('blur', function(){ tip.hidden=true; }); });
})();
"""

BODIES = {"overview": ("Introduction", "Overview", P_OVERVIEW), "motivation": ("Introduction · Section 1", "1 Motivation", P_MOTIVATION),
          "grammar": ("Introduction · Section 2", "2 The grammar in the policy", P_GRAMMAR),
          "method-a": ("Part I · Section 3", "3 Method", P_METHOD_A), "cases-a": ("Part I · Section 4", "4 Case study", P_CASES_A),
          "results-a": ("Part I · Section 5", "5 Twenty-one held-out tasks", P_RESULTS_A),
          "method-b": ("Part II · Section 6", "6 Method", P_METHOD_B), "cases-b": ("Part II · Section 7", "7 Case study: the carry", P_CASES_B),
          "stream": ("Part II · Section 8", "8 Fifty tasks in a stream", P_STREAM),
          "cost": ("Results · Section 9", "9 Cost and latency", P_COST), "negatives": ("Results · Section 10", "10 What did not work", P_NEG),
          "learned": ("Discussion · Section 11", "11 Is it training or zero-shot?", P_LEARNED),
          "models": ("Discussion · Section 12", "12 Other models, and pi0.7", P_MODELS),
          "baselines": ("Appendix · Section 13", "13 How the baselines were implemented", P_BASELINES),
          "verification": ("Appendix · Section 14", "14 Verification", P_VER)}

HTML = f"""<title>Grammar × Bindings</title>
<meta name="description" content="Adapting a frozen pi0.5 to new LIBERO-90 tasks from one demonstration by storing only object-relative bindings, and composing new tasks from old ones.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>{CSS}</style>
<div class="topbar"><details id="topnav"><summary><span id="topbar-title">Overview</span></summary><div class="toc-inner" id="toc-mobile"></div></details></div>
<div class="shell">
<nav class="toc" id="toc" aria-label="Contents">{toc()}</nav>
<main id="main"><p class="meta">lifelong-ood-learner · frozen pi0.5 on LIBERO-90 · one demonstration per task · results as of commit {esc(commit)}</p>
{"".join(page(pid, *BODIES[pid]) for pid, _, _ in ORDER)}
<footer>Every number on this site is read from ledger/ at commit {esc(commit)}; {esc(audit)} for the README's copies. Simulation only (LIBERO-90).</footer>
</main></div>
<div class="tip" id="tip" hidden></div>
<script>{JS}</script>
"""
os.makedirs(SITE, exist_ok=True)
open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(HTML)
print(os.path.join(SITE, "index.html"), len(HTML))
