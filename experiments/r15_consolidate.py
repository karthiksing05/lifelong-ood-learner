"""COMPOSE -> PRACTISE -> CONSOLIDATE, step 3: learn from the practice rollouts of the first composition.

Inputs: the demo (IPEC episode 0) and the recorded practice episodes (experiments/libero_client.py --record) of the first
composition on instances = init states 0-19 (results/robot/m5_practice_<task>_<arm>_rec/).  Per task:
  * MEMORY: every SUCCESSFUL rollout becomes a stored composition in the demo's format (its executed action sequence and
    states, windows as chunk rows; fold windows around its own gripper events) plus a 2 KB recall key (ImageNet ResNet-18
    features of its first frames, both cameras) -- the agent recognises an instance from its cameras, since LIBERO starts
    the robot from the same pose on every instance.  Failed rollouts are not stored.
  * FOLD: the gated expert refit on the contact windows of the demo AND of every success (rank 2, last two layers, as the
    demo's fold; anchored on library frames).
  * BASELINE (--lora): a rank-16 LoRA on every frame of the demo and of the successes (stride), always on -- the same data.
Writes results/robot/r15_<task>_{memory.npz,expert.pt,prog<k>.npz[,lora.pt]} and ledger rows r15_consolidate.

    python experiments/r15_consolidate.py --task 39            # memory + refit fold
    python experiments/r15_consolidate.py --task 39 --lora     # the LoRA baseline on the same data
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402
from lifelong_ood.policy import adapt                                   # noqa: E402
from lifelong_ood.policy import data as D                               # noqa: E402
from lifelong_ood.policy.compose import DemoProgram                     # noqa: E402

R = "results/robot"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def windows(actions, pre=30, post=10):
    T = len(actions)
    w = np.zeros(T, bool)
    for e in D.gripper_events(actions):
        w[max(0, e - pre): e + post + 1] = True
    return w


def recall_keys(images, wrist, device="cuda"):
    """ImageNet ResNet-18 features of (agentview, wrist), cosine-normalised and concatenated: [n, 1024]."""
    from lifelong_ood.policy.baselines import VinnPolicy
    v = VinnPolicy(np.zeros((1, 7), np.float32), images[:1], wrist[:1], device=device)
    return v._embed(np.asarray(images), np.asarray(wrist)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--practice", default="m5_practice")
    ap.add_argument("--library", default="results/robot/r2b_libero_40.npz")
    ap.add_argument("--steps", type=int, default=1000, help="as the demo's fold and the per-task LoRA (pre-registered: same steps)")
    ap.add_argument("--fold-stride", type=int, default=2, help="every k-th frame of the successes' contact windows")
    ap.add_argument("--lora", action="store_true", help="train ONLY the LoRA baseline on the same data (a separate run: a fresh model)")
    ap.add_argument("--lora-stride", type=int, default=5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-lora-frames", type=int, default=1000, help="uniform subsample above this (GPU memory)")
    ap.add_argument("--lora-scope", default="expert", help="expert | last2 (rank 2 on last2 = the fold's size, 258 KB)")
    ap.add_argument("--max-fold-frames", type=int, default=700, help="uniformly subsample the fold's frames above this (GPU memory; "
                    "2026-09-26: only task 3, 19 successes with long windows, exceeded it)")
    ap.add_argument("--n-anchor", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="r15", help="output prefix (a smoke test writes elsewhere); ledger tags gain it when not r15")
    a = ap.parse_args()
    os.chdir(ROOT)
    log = lambda s: print(s, flush=True)                                  # noqa: E731
    t = a.task
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r4 = load_module("r4", os.path.join(ROOT, "experiments", "r4_fold.py"))
    texts, _ = r3p.libero90_texts()
    demos, d90, norm = D.Demos(), D.Demos90(), D.Norm.load()
    ep = d90.load(d90.episodes_of_text(texts[t])[0])
    T0 = len(ep["actions"])
    demo_rows = D.to_rows(ep["actions"][D.chunk_indices(T0, 10)[0]], norm)
    img0, wr0 = d90.frames(ep["episode_index"])
    # --practice may list several rounds (comma-separated experiments): every recorded rollout of every round is pooled
    recs = sorted(f for p in a.practice.replace("+", ",").split(",") for f in glob.glob(f"{R}/{p}_{t}_*_rec/task{t}_trial*.npz"))
    eps = [dict(np.load(f)) for f in recs]
    succ = [e for e in eps if bool(e["success"])]
    log(f"[r15] task {t} '{texts[t]}': {len(eps)} practice rollouts, {len(succ)} successes")
    # MEMORY: the demo's composition (dimension-wise borrowing, its r4 fold windows) + every success as a stored composition
    grasps = len(D.gripper_events(ep["actions"])) > 0
    demo_prog = DemoProgram.load(f"{R}/r7_dim_{t}_s1.npz")
    demo_prog.fold = DemoProgram.load(f"{R}/r4_{t}_approach_last2_program.npz").fold if grasps else None
    demo_prog.save(f"{R}/{a.name}_{t}_prog0.npz")
    firsts_img, firsts_wr, prog_bytes = [img0[0]], [wr0[0]], [demo_prog.bytes()]
    fold_obs, fold_rows = [], []
    lora_obs, lora_rows = [], []
    win0 = windows(ep["actions"])
    for i in np.where(win0)[0]:
        fold_obs.append(d90.observation(ep, int(i))); fold_rows.append(demo_rows[i])
    for i in range(0, T0):
        lora_obs.append(d90.observation(ep, int(i))); lora_rows.append(demo_rows[i])
    for k, e in enumerate(succ, 1):
        acts, sts = e["action"], e["state"]
        T = len(acts)
        rows = D.to_rows(acts[D.chunk_indices(T, 10)[0]], norm)
        prog = DemoProgram.from_demo(None, rows, sts, variant="raw")
        w = windows(acts)
        prog.fold = w if w.any() else None
        prog.save(f"{R}/{a.name}_{t}_prog{k}.npz")
        prog_bytes.append(prog.bytes())
        firsts_img.append(e["image"][0]); firsts_wr.append(e["wrist"][0])
        text = texts[t]
        mk = lambda i: {"observation/image": e["image"][i], "observation/wrist_image": e["wrist"][i],   # noqa: E731
                        "observation/state": sts[i], "prompt": text}
        for i in np.where(w)[0][:: a.fold_stride]:
            fold_obs.append(mk(int(i))); fold_rows.append(rows[i])
        for i in range(0, T, a.lora_stride):
            lora_obs.append(mk(int(i))); lora_rows.append(rows[i])
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    if a.lora:
        if len(lora_obs) > a.max_lora_frames:
            keep = np.linspace(0, len(lora_obs) - 1, a.max_lora_frames).round().astype(int)
            log(f"[r15] {len(lora_obs)} LoRA frames > {a.max_lora_frames}: uniformly subsampled")
            lora_obs, lora_rows = [lora_obs[i] for i in keep], [lora_rows[i] for i in keep]
        t0 = time.time()
        bank = adapt.PrefixBank(bk, lora_obs)
        state, banks = adapt.inject(bk, scope=a.lora_scope)
        key = f"lora_learn_{t}"
        params = r4.add_node(banks, key, a.lora_rank)
        hist = adapt.train(bk, state, key, params, bank, D.pad_rows(np.stack(lora_rows), bk.H), steps=a.steps, lr=1e-4, seed=a.seed, log=log)
        lb = adapt.save(banks, key, f"{R}/{a.name}_{t}_lora.pt", meta=dict(task=t, rank=a.lora_rank, scope=a.lora_scope, learned=True))
        log(f"[r15] LoRA on the same data ({len(lora_obs)} frames) in {time.time() - t0:.0f}s: {lb} B")
        ledger.record("r15_consolidate", bench="libero_90", tag="_lora" + ("" if a.name == "r15" else "_" + a.name), seed=a.seed, keys=dict(task=t),
                      metrics=dict(task=t, frames=len(lora_obs), successes=len(succ), curve=hist),
                      bytes=dict(experts=lb, tree=0, codes=0, samples=0, teacher_copies=0, total=lb), config=vars(a),
                      script="experiments/r15_consolidate.py", command=" ".join(sys.argv), artifacts=[f"{R}/{a.name}_{t}_lora.pt"])
        return
    # frames of different sizes (the demo's 256 px, recorded 224 px): embed each composition's first frames on their own
    keys = np.concatenate([recall_keys(np.asarray(i)[None], np.asarray(w)[None]) for i, w in zip(firsts_img, firsts_wr)]).astype(np.float16)
    np.savez(f"{R}/{a.name}_{t}_memory.npz", keys=keys, n=len(prog_bytes), prog_bytes=np.asarray(prog_bytes), grasps=grasps,
             successes=np.asarray([int(e["trial"]) for e in succ]), practice=np.asarray([int(e["trial"]) for e in eps]),
             practice_success=np.asarray([bool(e["success"]) for e in eps]))
    mem_bytes = int(sum(prog_bytes) + keys.nbytes)
    log(f"[r15] memory: {len(prog_bytes)} compositions (demo + {len(succ)} successes), {mem_bytes} B with recall keys")
    lib = np.load(a.library)
    rng = np.random.default_rng(a.seed)
    eb, hist = 0, None
    if grasps and fold_obs:
        aj = rng.choice(len(lib["episode"]), size=min(a.n_anchor, len(lib["episode"])), replace=False)
        anc_bank = adapt.PrefixBank(bk, r4.library_obs(demos, lib["episode"][aj], lib["frame"][aj]))
        anc_chunks = D.pad_rows(lib["demo"][aj], bk.H)
        if len(fold_obs) > a.max_fold_frames:
            keep = np.linspace(0, len(fold_obs) - 1, a.max_fold_frames).round().astype(int)
            log(f"[r15] {len(fold_obs)} fold frames > {a.max_fold_frames}: uniformly subsampled")
            fold_obs, fold_rows = [fold_obs[i] for i in keep], [fold_rows[i] for i in keep]
        t0 = time.time()
        bank = adapt.PrefixBank(bk, fold_obs)
        state, banks = adapt.inject(bk, scope="last2")
        key = f"fold_learn_{t}"
        params = r4.add_node(banks, key, 2)
        hist = adapt.train(bk, state, key, params, bank, D.pad_rows(np.stack(fold_rows), bk.H), steps=a.steps, lr=1e-4, seed=a.seed,
                           anchor=(anc_bank, anc_chunks), anchor_weight=1.0, log=log)
        eb = adapt.save(banks, key, f"{R}/{a.name}_{t}_expert.pt", meta=dict(task=t, rank=2, scope="last2", learned=True))
        log(f"[r15] fold refit on {len(fold_obs)} frames in {time.time() - t0:.0f}s: {eb} B")
    ledger.record("r15_consolidate", bench="libero_90", tag="" if a.name == "r15" else "_" + a.name, seed=a.seed, keys=dict(task=t),
                  metrics=dict(task=t, text=texts[t], practice=len(eps), successes=len(succ), compositions=len(prog_bytes),
                               memory_bytes=mem_bytes, fold_frames=len(fold_obs), curve=hist),
                  bytes=dict(experts=eb, tree=0, codes=mem_bytes, samples=0, teacher_copies=0, total=eb + mem_bytes),
                  config=vars(a), script="experiments/r15_consolidate.py", command=" ".join(sys.argv),
                  artifacts=[f"{R}/{a.name}_{t}_memory.npz"] + ([f"{R}/{a.name}_{t}_expert.pt"] if eb else []))


if __name__ == "__main__":
    main()
