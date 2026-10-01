"""Q1a of the Bind plan: does the policy's OWN model know WHERE a named object is, and does its action expert look there?

On the first frames with oracle segmentation (experiments/f0b_firstframes_seg.py; practice states 0-9 only), for every task
with one moved object and at least two candidate objects, and every candidate c:
  W1  pi0.5's PaliGemma attention from c's word tokens to the agent-view tokens, with the task's instruction rewritten to
      name c (factors.substitute), per layer: does its peak token lie on c (oracle segmentation)?
  W3  OWLv2's best box for c's phrase on the 224 image: same pointing test (fallback source)
  EX  the action expert's attention MASS on each candidate's tokens while it samples the first chunk under the task's own
      instruction (no bias; attn_bias logging path): which object does the expert attend?
Groups: validity (LIBERO-Goal; gate >= 0.8), object (LIBERO-Object), class_a (the scene's relation with another object),
other LIBERO-90.  Pre-registered gates (ledger m3_prereg tag b_bind_q1): a source enters Q3 only if it points >= 0.8 on
validity and >= 0.6 on class_a.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/b1_where_probe.py [--only libero_90_48] [--no-owl] [--record]" \
        slurm/jobs/robot_gpu.sbatch
-> results/robot/b1_where_probe.json
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import attn_bias as AB          # noqa: E402
from lifelong_ood.policy import factors as F             # noqa: E402
from lifelong_ood.policy import where as W               # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FRAMES = os.path.join(R, "f0b_firstframes")
CLASS_A = (18, 26, 48, 49, 52, 53, 58, 59, 60, 62, 65, 66, 71, 38, 70)
MIN_COVER = 0.1           # a token belongs to an object's oracle map if the object covers >= 10% of it


def group(suite, tid):
    if suite == "libero_goal":
        return "validity"
    if suite == "libero_object":
        return "object"
    if suite == "libero_90" and tid in CLASS_A:
        return "class_a"
    return "other_" + suite


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    from openpi.models import tokenizer as _tok
    bk = PolicyBackbone("pi05_libero", device="cuda")
    tok = _tok.PaligemmaTokenizer(200)
    owl = None if "--no-owl" in sys.argv else W.OwlDetector()
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    only = sys.argv[sys.argv.index("--only") + 1].split("+") if "--only" in sys.argv else None
    per_task = {}
    for f in sorted(glob.glob(os.path.join(FRAMES, "*.npz"))):
        key = os.path.basename(f)[:-4]
        if only and key not in only:
            continue
        suite, tid = key.rsplit("_", 1)
        tid = int(tid)
        t = by.get((suite, tid))
        if t is None or len(set(t.moved)) != 1:
            continue
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        types = {n: t.objects.get(n) for n in names if n in t.objects}
        cands = [c for c in F.candidates(t) if c in set(types.values())]
        if len(cands) < 2:
            continue
        ph = F.scene_phrases(t, tasks)
        cands = [c for c in cands if c in ph]                 # objects no instruction names have no phrase to read
        prompts = {c: F.substitute(t, c, ph) for c in cands}
        if any(p is None for p in prompts.values()):
            continue
        geoms = {c: np.concatenate([W.geoms_of(z, name=n) for n, ty in types.items() if ty == c]) for c in cands}
        recs = []
        for k in range(len(z["states"])):
            seg = z["seg"][k]
            truth = {c: W.oracle_map(seg, geoms[c]) for c in cands}
            visible = [c for c in cands if truth[c].max() >= MIN_COVER]
            base = {"observation/image": z["img"][k], "observation/wrist_image": z["wimg"][k],
                    "observation/state": np.asarray(z["state8"][k], np.float32)}
            rec = dict(state=int(z["states"][k]), w1={}, w3={}, ex={}, cover={c: float(truth[c].sum()) for c in cands})
            for c in visible:
                obs = dict(base, prompt=prompts[c])
                if "--skip-w1" not in sys.argv:
                    try:
                        g = W.vlm_grounding(bk, obs, ph[c], tokenizer=tok)
                    except ValueError as e:
                        print(f"[b1] {key} {c}: {e}", flush=True)
                        continue
                    rec["w1"][c] = {int(l): int(W.points_at(m["agent"], truth[c], MIN_COVER)) for l, m in g.items()}
                if owl is not None:
                    mo, sc = owl(z["img"][k], ph[c])
                    ctr = owl.last_center
                    rec["w3"][c] = dict(hit=int(ctr is not None and truth[c][ctr] > MIN_COVER),          # box centre on the object
                                        hit_peak=int(W.points_at(mo, truth[c], MIN_COVER)), score=sc, center=ctr)
            # the expert's own attention under the task's OWN instruction, one logging pass per candidate's oracle map
            pre = bk.prefix(bk.observation(dict(base, prompt=t.language)))
            for c in ([] if "--skip-ex" in sys.argv else visible):
                log = []
                bk.sample(1, pre, seed=0, step_gate=AB.bias_gate(bk, {"agent": (truth[c] >= MIN_COVER).astype(np.float32)}, pre,
                                                                 mode="fixed", beta=0.0, log=log))
                rec["ex"][c] = float(np.mean([a for _, a, _ in log]))
            recs.append(rec)
        own = t.moved[0]
        layers = sorted({l for r in recs for v in r["w1"].values() for l in v})
        w1 = {l: float(np.mean([v[l] for r in recs for v in r["w1"].values() if l in v])) for l in layers} if layers else {}
        w3 = float(np.mean([v["hit"] for r in recs for v in r["w3"].values()])) if any(r["w3"] for r in recs) else None
        ex_top = [max(r["ex"], key=r["ex"].get) for r in recs if r["ex"]]
        per_task[key] = dict(group=group(suite, tid), language=t.language, candidates=cands, own=own, w1_by_layer=w1, w3=w3,
                             ex_own_top=float(np.mean([x == own for x in ex_top])) if ex_top else None,
                             ex_top=ex_top, states=recs)
        best = max(w1, key=w1.get) if w1 else None
        print(f"[b1] {key:18s} {group(suite, tid):10s} W1 best layer {best} {w1.get(best, float('nan')):.2f} "
              f"W3 {w3 if w3 is None else round(w3, 2)} | expert attends own object {per_task[key]['ex_own_top']} | {t.language}",
              flush=True)
    groups = {}
    for gname in sorted({v["group"] for v in per_task.values()}):
        vs = [v for v in per_task.values() if v["group"] == gname]
        layers = sorted({l for v in vs for l in v["w1_by_layer"]})
        groups[gname] = dict(tasks=len(vs),
                             w1_by_layer={l: float(np.mean([v["w1_by_layer"][l] for v in vs if l in v["w1_by_layer"]])) for l in layers},
                             w3=float(np.mean([v["w3"] for v in vs if v["w3"] is not None])) if any(v["w3"] is not None for v in vs) else None,
                             ex_own_top=float(np.mean([v["ex_own_top"] for v in vs if v["ex_own_top"] is not None])))
        print(f"[b1] GROUP {gname}: {groups[gname]}", flush=True)
    res = dict(min_cover=MIN_COVER, groups=groups, per_task=per_task)
    TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    out = os.path.join(R, f"b1_where_probe{'_' + TAG if TAG else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("b1_where", bench="libero", tag="read" + ("_" + TAG if TAG else ""), seed=0, metrics=dict(min_cover=MIN_COVER, groups=groups),
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/b1_where_probe.py", command=" ".join(sys.argv),
                            artifacts=[out]))


if __name__ == "__main__":
    main()
