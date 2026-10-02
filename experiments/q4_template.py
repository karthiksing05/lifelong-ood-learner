"""W4: WHERE from the one demonstration, as a visual object prototype in the policy's own vision features (exploratory probe).

The demo is the query (Wang et al.): at its first grasp (the gripper command closes) the end effector sits on the moved object, so
projecting that position into the agent view of the demo's FIRST frame marks the object's patch; the policy's own SigLIP image
features at that patch (and its 4 neighbours, averaged) are the object's prototype (a 2048-d vector, ~4 KB fp16).  In a new start
state the patch most similar to the prototype is where the object is.  Scored with the oracle segmentation of the first frames
(experiments/f0b_firstframes_seg.py; practice states 0-9): pointing = the best patch is covered >= 0.1 by the moved object.
REUSE: the prototype from one task's demo, applied in OTHER tasks' scenes that contain an object of the same type.
--goal: the GOAL prototype instead -- the end effector at the demo's first release (the gripper opens after the first grasp),
projected into the first frame, marks where the object goes (the goal slot); stored in q4_goal_protos.npz, pointing scored against
the target fixture's segmentation (a coarse check: the whole fixture, not its goal region).

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/q4_template.py --record" slurm/jobs/robot_gpu.sbatch
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood.policy import bind_hooks as BH     # noqa: E402
from lifelong_ood.policy import data as D            # noqa: E402
from lifelong_ood.policy import factors as F         # noqa: E402
from lifelong_ood.policy import where as W           # noqa: E402

R = os.path.join(ROOT, "results", "robot")
FR = os.path.join(R, "f0b_firstframes")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
CLASS_A = [18, 26, 48, 49, 52, 53, 58, 59, 60, 62, 65, 66, 71]


def feats(bk, img, wimg, state):
    import torch
    obs = bk.observation({"observation/image": img, "observation/wrist_image": wimg, "observation/state": state, "prompt": ""})
    images, _, _, _, _ = bk.model._preprocess_observation(obs, train=False)
    with torch.no_grad():
        f = bk.model.paligemma_with_expert.embed_image(images[0])[0].float()
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()                                          # [256, D] unit rows


def neighbours(i, j):
    return [(a, b) for a, b in ((i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if 0 <= a < 16 and 0 <= b < 16]


def main():
    from lifelong_ood.policy.backbone import PolicyBackbone
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    tasks = F.load()
    by = {(t.suite, t.index): t for t in tasks}
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    protos, per = {}, {}
    todo = ([int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].split("+")] if "--tasks" in sys.argv else sorted(set(HELD + CLASS_A)))
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    goal = "--goal" in sys.argv
    PF = os.path.join(R, "q4_goal_protos.npz" if goal else "q4_object_protos.npz")
    if goal:
        tag = tag or "goal"
    if tag and os.path.exists(PF):          # extend the stored prototypes, never drop them
        old = np.load(PF)
        tids_old, types_old = [int(x) for x in old["tids"]], [str(x) for x in old["types"]]
        for k in old.files:
            if k not in ("types", "tids"):
                kk = int(k)
                protos[kk] = dict(vec=old[k], type=types_old[tids_old.index(kk)] if kk in tids_old else "", token=None)
    for tid in todo:
        t = by[("libero_90", tid)]
        moved = BH.referents(t)["closed" if goal else "open"]
        if not moved or tid not in [x.index for x in tasks if x.suite == "libero_90"]:
            continue
        ep = d90.load(d90.episodes_of_text(texts[tid])[0])
        closed = np.where(ep["actions"][:, 6] > 0)[0]
        if len(closed) == 0:
            continue
        g = int(closed[0])
        if goal:                                                      # the first release after the first grasp
            opened = np.where((ep["actions"][:, 6] <= 0) & (np.arange(len(ep["actions"])) > g))[0]
            if len(opened) == 0:
                continue
            g = int(opened[0])
        z = np.load(os.path.join(FR, f"libero_90_{tid}.npz"), allow_pickle=True)
        rc = W.project(ep["state"][g][:3], z["K"], z["E"][0])
        i, j = int(np.clip(rc[0] // 16, 0, 15)), int(np.clip(rc[1] // 16, 0, 15))
        ep["image"], ep["wrist_image"] = d90.frames(ep["episode_index"])
        f0 = feats(bk, ep["image"][0], ep["wrist_image"][0], ep["state"][0])
        proto = np.mean([f0[a * 16 + b] for a, b in neighbours(i, j)], 0)
        proto /= np.linalg.norm(proto)
        mtype = t.objects.get(moved[0], t.fixtures.get(moved[0], moved[0]) if isinstance(t.fixtures, dict) else moved[0])
        protos[tid] = dict(vec=proto, type=mtype, token=(i, j))
        hits, top4 = [], []
        for k in range(len(z["states"])):
            fk = feats(bk, z["img"][k], z["wimg"][k], z["state8"][k])
            sim = fk @ proto
            pk = int(np.argmax(sim))
            top4.append([int(x) for x in np.argsort(sim)[::-1][:4]])
            truth = W.oracle_map(z["seg"][k], W.geoms_of(z, name=moved[0]))
            hits.append(int(truth.reshape(-1)[pk] > 0.1))
        per[str(tid)] = dict(language=texts[tid], moved=moved[0], type=mtype, grasp_frame=g, demo_token=[i, j],
                             pointing=float(np.mean(hits)), top4=top4)
        print(f"[q4] {tid:>3} {mtype:22s} demo token {i},{j} pointing {np.mean(hits):.2f} | {texts[tid]}", flush=True)
    # reuse: a prototype in OTHER scenes with an object of the same type (LIBERO-90 and LIBERO-Object first frames)
    reuse = []
    for src, p in ([] if tag else protos.items()):
        for f in sorted(os.listdir(FR)):
            suite, tb = f[:-4].rsplit("_", 1)
            if suite not in ("libero_90", "libero_object") or (suite == "libero_90" and int(tb) == src):
                continue
            tt = by.get((suite, int(tb)))
            if tt is None:
                continue
            inst = [n for n, ty in tt.objects.items() if ty == p["type"]]
            if len(inst) != 1:
                continue
            z = np.load(os.path.join(FR, f), allow_pickle=True)
            hits = []
            for k in range(min(5, len(z["states"]))):
                fk = feats(bk, z["img"][k], z["wimg"][k], z["state8"][k])
                pk = int(np.argmax(fk @ p["vec"]))
                truth = W.oracle_map(z["seg"][k], W.geoms_of(z, name=inst[0]))
                hits.append(int(truth.reshape(-1)[pk] > 0.1))
            reuse.append(dict(src=src, dst=f[:-4], type=p["type"], pointing=float(np.mean(hits))))
    own = float(np.mean([v["pointing"] for v in per.values()]))
    ru = float(np.mean([r["pointing"] for r in reuse])) if reuse else None
    print(f"[q4] own-scene pointing {own:.3f} over {len(per)} tasks; reuse in other scenes {ru} over {len(reuse)} pairs", flush=True)
    res = dict(own_pointing=own, reuse_pointing=ru, n_tasks=len(per), n_reuse=len(reuse), per_task=per, reuse=reuse,
               proto_bytes=int(2 * len(next(iter(protos.values()))["vec"])) if protos else 0)
    json.dump(res, open(os.path.join(R, f"q4_template{'_' + tag if tag else ''}.json"), "w"), indent=1)
    np.savez(PF, **{str(k): v["vec"] for k, v in protos.items()},
             types=np.array([protos[k]["type"] for k in protos]), tids=np.array(list(protos)))
    if "--record" in sys.argv:
        from lifelong_ood import ledger
        print(ledger.record("q4_template", bench="libero", tag="read" + ("_" + tag if tag else ""), seed=0, metrics={k: v for k, v in res.items() if k != "per_task"},
                            bytes=dict(experts=0, tree=0, codes=res["proto_bytes"], samples=0, teacher_copies=0, total=res["proto_bytes"]),
                            config={}, script="experiments/q4_template.py", command=" ".join(sys.argv),
                            artifacts=[os.path.join(R, f"q4_template{'_' + tag if tag else ''}.json")]))


if __name__ == "__main__":
    main()
