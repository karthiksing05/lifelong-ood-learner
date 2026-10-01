"""Baseline: STRAP (Memmel et al., ICLR 2025, "Robot Sub-Trajectory Retrieval for Augmented Policy Learning") on pi0.5.

The task's ONE demo is split into sub-trajectories where the end-effector nearly stops (speed below its 20th percentile, pieces of at
least 20 steps); each is matched by subsequence DTW over DINOv2 features of the agent-view camera against every episode of the prior
data -- the 1,693 demonstrations of pi0.5's 40 training tasks -- and the K best matches (K = 100 in all, split evenly across the
sub-trajectories; one match per prior episode and sub-trajectory) are kept.  A rank-16 LoRA on the action expert is trained on the
demo and the retrieved sub-trajectories together, sampled uniformly from their union, every frame under the target instruction
(STRAP's single-task policy).  Differences from the paper: a LoRA on a pretrained VLA instead of a policy trained from scratch, and
one target demo instead of five.

    stage dump     (openpi env)  : the 21 target demos' frames          -> results/robot/strap_targets.npz
    stage embed    (chunks env, GPU): DINOv2-B features, prior + targets  -> results/robot/strap_emb.npz
    stage retrieve (chunks env)  : S-DTW, top K per task                 -> results/robot/strap_<t>_retrieved.json
    stage train    (openpi env, GPU): the LoRA                            -> results/robot/strap_<t>_lora.pt (arm lora@strap)
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]


def texts():
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    return r3p.libero90_texts()[0]


def dump():
    from lifelong_ood.policy import data as D
    from PIL import Image
    tx = texts()
    d90 = D.Demos90()
    out = {}
    for t in HELD:
        ep = d90.load(d90.episodes_of_text(tx[t])[0])
        img, _ = d90.frames(ep["episode_index"])
        out[f"img_{t}"] = np.stack([np.asarray(Image.fromarray(x).resize((224, 224), Image.BICUBIC)) for x in img])
        out[f"eef_{t}"] = ep["state"][:, :3]
        out[f"ep_{t}"] = np.array(ep["episode_index"])
        print(f"[strap] dump task {t}: {len(img)} frames", flush=True)
    np.savez_compressed(os.path.join(R, "strap_targets.npz"), **out)


def _decode(b):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BICUBIC), np.uint8)


def embed():
    import torch
    import timm
    from multiprocessing import Pool
    from lifelong_ood.policy import data as D
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def feats(imgs):
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                x = torch.as_tensor(np.asarray(imgs[i:i + 256]), device="cuda").permute(0, 3, 1, 2).float() / 255.0
                out.append(m(((x - mean) / std).half()).float().cpu().numpy())
        return np.concatenate(out).astype(np.float16)
    dm = D.Demos()
    eps = [e["episode_index"] for e in dm.eps]
    E, F, Fr = [], [], []
    t0 = time.time()
    with Pool(6) as pool:
        for k, e in enumerate(eps):
            ep = dm.load(e, images=True)
            imgs = pool.map(_decode, ep["image"], chunksize=16)
            f = feats(imgs)
            E.append(np.full(len(f), e, np.int32)); Fr.append(np.arange(len(f), dtype=np.int32)); F.append(f)
            if k % 200 == 0:
                print(f"[strap] embedded {k} / {len(eps)} prior episodes ({time.time() - t0:.0f}s)", flush=True)
    tg = np.load(os.path.join(R, "strap_targets.npz"))
    tf = {f"t_{t}": feats(tg[f"img_{t}"]) for t in HELD}
    np.savez(os.path.join(R, "strap_emb.npz"), ep=np.concatenate(E), frame=np.concatenate(Fr), f=np.concatenate(F), **tf)
    print(f"[strap] {sum(len(x) for x in F)} prior frames embedded in {time.time() - t0:.0f}s", flush=True)


def segments(eef, min_len=20):
    """Split where the end-effector speed falls below its 20th percentile, keeping pieces of at least min_len steps."""
    sp = np.linalg.norm(np.diff(eef, axis=0), axis=-1)
    thr = np.percentile(sp, 20)
    cuts = [0]
    for i in range(1, len(sp)):
        if sp[i] < thr and i - cuts[-1] >= min_len:
            cuts.append(i)
    if len(eef) - cuts[-1] < min_len and len(cuts) > 1:
        cuts.pop()
    cuts.append(len(eef))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def segments_official(eef, threshold=2.5e-3, min_len=20):
    """STRAP's own rule (strap/utils/retrieval_utils.py segment_trajectory_by_derivative + merge_short_segments): cut after every step
    whose L1 end-effector displacement is below 2.5 mm, then merge pieces shorter than min_len into the next (the last into the previous)."""
    d = np.abs(np.diff(eef, axis=0)).sum(1)
    stops = np.where(d < threshold)[0]
    cuts, start = [], 0
    for st in stops:
        cuts.append((start, st + 1))
        start = st + 1
    if start < len(eef):
        cuts.append((start, len(eef)))
    merged, cur = [], list(cuts[0])
    for a, b in cuts[1:]:
        if cur[1] - cur[0] < min_len:
            cur[1] = b
        else:
            merged.append(tuple(cur))
            cur = [a, b]
    if cur[1] - cur[0] < min_len and merged:
        merged[-1] = (merged[-1][0], cur[1])
    else:
        merged.append(tuple(cur))
    return merged


def sdtw21(Q, P, lens):
    """STRAP's subsequence DTW (compute_accumulated_cost_matrix_subsequence_dtw_21): steps (1, 1) and (1, 2) only -- the prior episode
    advances one or two frames per query frame, never stalls; free start and end.  -> (cost [E], start [E], end [E])."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=dev)
    Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)
    E, L, _ = Pt.shape
    m = len(Qt)
    C = torch.cdist(Qt.unsqueeze(0).expand(E, -1, -1), Pt)
    mask = torch.arange(L, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]
    C = C.masked_fill(mask[:, None, :], float("inf"))
    inf = torch.full((E, 2), float("inf"), device=dev)
    Dm = torch.empty((E, m, L), device=dev)
    Dm[:, 0] = C[:, 0]
    for i in range(1, m):
        prev = torch.cat([inf, Dm[:, i - 1]], 1)                       # prev[:, j + 2] = D[i-1, j]
        Dm[:, i] = C[:, i] + torch.minimum(prev[:, 1:L + 1], prev[:, 0:L])   # D[i-1, j-1], D[i-1, j-2]
    cost, end = Dm[:, m - 1].min(1)
    Dn = Dm.cpu().numpy()
    start = np.zeros(E, int)
    for e in range(E):
        j = int(end[e])
        for i in range(m - 1, 0, -1):
            a = Dn[e, i - 1, j - 1] if j >= 1 else np.inf
            b = Dn[e, i - 1, j - 2] if j >= 2 else np.inf
            j = j - 1 if a <= b else j - 2
        start[e] = max(j, 0)
    return cost.cpu().numpy() / m, start, end.cpu().numpy()


def sdtw(Q, P, lens):
    """Subsequence DTW of query Q [m, d] against every prior episode P [E, L, d] (padded; lens [E]).  -> (cost [E], start [E], end [E])."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=dev)
    Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)
    E, L, _ = Pt.shape
    m = len(Qt)
    C = torch.cdist(Qt.unsqueeze(0).expand(E, -1, -1), Pt)                  # [E, m, L]
    mask = torch.arange(L, device=dev)[None, :] >= torch.as_tensor(lens, device=dev)[:, None]
    C = C.masked_fill(mask[:, None, :], float("inf"))
    Dm = torch.full((E, m, L), float("inf"), device=dev)
    Dm[:, 0] = C[:, 0]                                                       # free start anywhere in P
    for i in range(1, m):
        prev = Dm[:, i - 1]
        diag = torch.cat([torch.full((E, 1), float("inf"), device=dev), prev[:, :-1]], 1)
        best_ij = torch.minimum(prev, diag) + C[:, i]
        row = best_ij.clone()                                                # horizontal moves: a running min along j
        for j in range(1, L):
            row[:, j] = torch.minimum(row[:, j], row[:, j - 1] + C[:, i, j])
        Dm[:, i] = row
    cost, end = Dm[:, m - 1].min(1)
    Dn, Cn = Dm.cpu().numpy(), C.cpu().numpy()
    start = np.zeros(E, int)
    for e in range(E):
        i, j = m - 1, int(end[e])
        while i > 0:
            opts = [(Dn[e, i - 1, j - 1] if j > 0 else np.inf, i - 1, j - 1), (Dn[e, i - 1, j], i - 1, j), (Dn[e, i, j - 1] if j > 0 else np.inf, i, j - 1)]
            _, i, j = min(opts)
        start[e] = j
    return cost.cpu().numpy() / m, start, end.cpu().numpy()


def retrieve(K=100, official=False):
    Z = np.load(os.path.join(R, "strap_emb.npz"))
    tg = np.load(os.path.join(R, "strap_targets.npz"))
    ep, F = Z["ep"], Z["f"].astype(np.float32)
    ids = np.unique(ep)
    lens = np.array([(ep == e).sum() for e in ids])
    P = np.zeros((len(ids), lens.max(), F.shape[1]), np.float32)
    for k, e in enumerate(ids):
        P[k, :lens[k]] = F[ep == e]
    for t in HELD:
        q = Z[f"t_{t}"].astype(np.float32)
        segs = segments_official(tg[f"eef_{t}"]) if official else segments(tg[f"eef_{t}"])
        per = int(K / len(segs)) if official else int(np.ceil(K / len(segs)))
        picks = []
        for a, b in segs:
            cost, st, en = (sdtw21 if official else sdtw)(q[a:b], P, lens)
            for k in np.argsort(cost)[:per]:
                picks.append(dict(segment=[int(a), int(b)], episode=int(ids[k]), start=int(st[k]), end=int(en[k]), cost=float(cost[k])))
        picks = sorted(picks, key=lambda x: x["cost"]) if official else sorted(picks, key=lambda x: x["cost"])[:K]
        name = "strap2" if official else "strap"
        json.dump(dict(task=t, segments=[list(map(int, x)) for x in segs], K=K, picks=picks, official=official),
                  open(os.path.join(R, f"{name}_{t}_retrieved.json"), "w"), indent=1)
        print(f"[strap] task {t}: {len(segs)} sub-trajectories, {len(picks)} retrieved, {sum(p['end'] - p['start'] + 1 for p in picks)} frames", flush=True)


def train(t, steps=2000, rank=16, lr=1e-4, batch=32, seed=0, name="strap"):
    sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
    from lifelong_ood import ledger
    from lifelong_ood.experts.lora import add_node
    from lifelong_ood.policy import adapt
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy.backbone import PolicyBackbone
    tx = texts()
    bk = PolicyBackbone("pi05_libero", device="cuda")
    norm = D.Norm.load()
    d90, dm = D.Demos90(), D.Demos()
    ep = d90.load(d90.episodes_of_text(tx[t])[0])
    T = len(ep["actions"])
    idx, _ = D.chunk_indices(T, bk.H)
    obs = [d90.observation(ep, i) for i in range(T)]
    rows = [D.pad_rows(D.to_rows(ep["actions"][idx], norm), bk.H)]
    ret = json.load(open(os.path.join(R, f"{name}_{t}_retrieved.json")))
    for p in ret["picks"]:
        pe = dm.load(p["episode"], images=True)
        n = len(pe["actions"])
        pidx, _ = D.chunk_indices(n, bk.H)
        fr = list(range(p["start"], p["end"] + 1))
        for i in fr:
            o = dm.observation(pe, i)
            o["prompt"] = ep["task"]                                         # STRAP's single-task policy: the target instruction
            obs.append(o)
        rows.append(D.pad_rows(D.to_rows(pe["actions"][pidx[fr]], norm), bk.H))
    full = np.concatenate(rows)
    if len(obs) > 3000:                                                   # host memory: at most 3,000 frames (~78 GB);
        keep = np.r_[np.arange(T), T + np.linspace(0, len(obs) - T - 1, 3000 - T).round().astype(int)]   # every demo frame,
        obs, full = [obs[i] for i in keep], full[keep]                     # the retrieved ones evenly thinned
    bank = adapt.PrefixBank(bk, obs, store="cpu")
    print(f"[strap] task {t}: demo {T} frames + retrieved {bank.n - T} frames", flush=True)
    state, banks = adapt.inject(bk, scope="expert")
    key = f"{name}_{t}"
    params = add_node(banks, key, rank)
    t1 = time.time()
    hist = adapt.train(bk, state, key, params, bank, full, steps=steps, lr=lr, batch=batch, seed=seed)
    secs = time.time() - t1
    path = os.path.join(R, f"{name}_{t}_lora.pt")
    nbytes = adapt.save(banks, key, path, meta=dict(task=t, rank=rank, scope="expert", retrieved=len(ret["picks"])))
    m = dict(task=t, demo_frames=int(T), retrieved_frames=int(bank.n - T), retrieved=len(ret["picks"]), segments=ret["segments"],
             final_loss=hist["loss"][-1], first_loss=hist["loss"][0], curve=hist, train_secs=secs)
    print(ledger.record("b_baselines", bench="libero_90", tag=name, seed=seed, keys=dict(task=t), metrics=m,
                        bytes=dict(experts=nbytes, tree=0, codes=0, samples=0, teacher_copies=0, total=nbytes),
                        config=dict(steps=steps, rank=rank, lr=lr, batch=batch, K=ret["K"]), script="experiments/b_strap.py",
                        command=" ".join(sys.argv), artifacts=[path]))
    print(f"[strap] saved {path}; loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f} in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dump", "embed", "retrieve", "train"))
    ap.add_argument("--task", type=int)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--official", action="store_true", help="STRAP's own segmentation, DTW steps and per-segment top-k (-> strap2)")
    a = ap.parse_args()
    {"dump": dump, "embed": embed, "retrieve": lambda: retrieve(official=a.official)}.get(
        a.stage, lambda: train(a.task, steps=a.steps, name="strap2" if a.official else "strap"))()
