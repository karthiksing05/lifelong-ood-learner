"""N26 — batch addition on a PRETRAINED backbone: K concepts in one update, not K updates.

The lifelong claim this project makes is not "concepts can be added" but "concepts can be added
efficiently, forever".  Batch addition is the efficiency half: a stream that arrives in groups should
be absorbed in groups, or the cost of the manifold is the cost of its history.  N4 established this
on our own 6.8M UNet -- one consolidation of K is 1.7x faster than K consolidations, at identical
accuracy and with retention still exactly 1.000, and it produces genuinely SHARED nodes (6 against 3)
because a batch trains a shared ancestor on all of its members at once where a sequential run can
only hand a later concept a frozen warm start.

This asks the same question of a model we did not train, and it needs its own trainer rather than
`consolidate_batch`: that method samples our cosine noise schedule, and a pretrained DDPM has its own
(linear-beta) schedule.  Training it on ours optimizes it at noise levels it never saw -- the same
class of error as the green-cast sampler in N15.

Three arms:

  `sequential`   K concepts, K optimizations.  The baseline every per-query LoRA method implies.
  `batch`        K concepts, ONE optimization, minibatches interleaved across concepts.
  `batch-shared` the same, but the K concepts are given a shared group node, so the batch can put
                 what they have in common somewhere they all read.  This is the arm that cannot be
                 simulated sequentially.

Measured: wall clock, per-concept attribute rate, Vendi, and -- the part that must not move --
whether every concept is still bit-identical after all K are present.

    python experiments/n26_batch_pretrained.py --attrs Eyeglasses Smiling Male Wearing_Hat
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                              # noqa: E402
from n25_pretrained_case import show, to_data, to_model                 # noqa: E402

from lifelong_ood.celeba import find_celeba                             # noqa: E402
from lifelong_ood.evals import SoftmaxProbe, save_json, vendi_score      # noqa: E402
from lifelong_ood.neural.adapters import HierarchicalAdapterBank        # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained              # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_attr_groups(attrs, img=256, n_each=128, n_probe=4000, probe_img=128, max_rows=60000):
    """-> dict(attr -> [n, 3*img*img] CHW-flat of images WITH that attribute), plus a probe set."""
    import pyarrow.parquet as pq
    from PIL import Image
    from lifelong_ood.celeba import _parquet_shards
    root = find_celeba()
    want = {a: [] for a in attrs}
    pX, pY, n, done = [], [], 0, False
    for sp in _parquet_shards(root):
        f = pq.ParquetFile(sp)
        for batch in f.iter_batches(batch_size=512, columns=["image"] + list(attrs)):
            d = batch.to_pydict()
            for k in range(len(d["image"])):
                lab = [1 if d[a][k] else 0 for a in attrs]
                n += 1
                need = [a for a, v in zip(attrs, lab) if v and len(want[a]) < n_each]
                if not need and len(pX) >= n_probe:
                    continue
                im = Image.open(io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
                w, h = im.size
                top = max(0, (h - w) // 2)
                sq = im.crop((0, top, w, top + w))
                if need:
                    arr = (np.asarray(sq.resize((img, img), Image.BICUBIC),
                                      np.float32) / 255.0).transpose(2, 0, 1).ravel()
                    for a in need:
                        want[a].append(arr)
                if len(pX) < n_probe:
                    pX.append((np.asarray(sq.resize((probe_img, probe_img), Image.BICUBIC),
                                          np.float32) / 255.0).transpose(2, 0, 1).ravel())
                    pY.append(lab)
            done = all(len(want[a]) >= n_each for a in attrs) and len(pX) >= n_probe
            if done or n >= max_rows:
                break
        if done or n >= max_rows:
            break
    out = {a: (np.stack(v) if v else np.zeros((0, 3 * img * img), np.float32))
           for a, v in want.items()}
    out["_probe_X"] = np.stack(pX)
    out["_probe_Y"] = np.array(pY, np.int8)
    return out


def train_group(field, bank, names, pools, img, epochs, batch, lr, t_min, seed=0, verbose=True):
    """ONE optimizer over the union of `names`' parameters, minibatches interleaved across concepts.

    This is what a batch update means here and why it is not K runs in a loop: a node shared by
    several of the K accumulates gradient from all of them within a single optimization, so the batch
    can place what the group has in common somewhere every member reads.  K sequential runs can only
    ever give a later concept a frozen warm start.
    """
    params, seen = [], set()
    for nm in names:
        emb_p = bank._emb_extra[nm]
        with torch.no_grad():
            emb_p.zero_()                     # the pretrained time embedding is not a concept slot
        for q in bank.trainable(nm):
            if q is emb_p or id(q) in seen:
                continue
            seen.add(id(q)); params.append(q)
    opt = torch.optim.Adam(params, lr=lr)
    X = {nm: torch.from_numpy(to_model(p)).to(field.device).reshape(-1, 3, img, img)
         for nm, p in zip(names, pools)}
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = len(field._abar)
    ab = torch.from_numpy(field._abar).float().to(field.device)
    field.model.train()
    t0 = time.time()
    for ep in range(epochs):
        order = []
        for nm in names:
            perm = torch.randperm(len(X[nm]), generator=g)
            order += [(nm, perm[i:i + batch]) for i in range(0, len(X[nm]), batch)]
        tot, nb = 0.0, 0
        for j in torch.randperm(len(order), generator=g).tolist():
            nm, idx = order[j]
            x0 = X[nm][idx.to(field.device)]
            # NATIVE trained timesteps, and only the high-noise range: our images are CelebA
            # upsampled from 178px and this model is CelebA-HQ, so low-t training transfers our blur
            # rather than the attribute (RESULTS 12b).
            ts = torch.randint(int(t_min), T, (len(x0),), generator=g).to(field.device)
            a = ab[ts].view(-1, 1, 1, 1)
            e = torch.randn(x0.shape, generator=g).to(field.device)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            with bank.activate(nm):
                pred = field.model.forward_t(xt, ts, emb_override=None)
            loss = torch.nn.functional.mse_loss(pred, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        if verbose and (ep + 1) % 10 == 0:
            print(f"      [{'+'.join(names)}] ep {ep + 1}/{epochs}: {tot / max(nb, 1):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    field.model.eval()
    return time.time() - t0


def sample_concept(field, bank, names, n, steps, seed, w):
    """CFG against the frozen base, over the union of `names`' paths (see n25)."""
    prev = list(bank.state["path"])
    path = sorted({p for nm in names for p in bank.concepts[nm]["path"]})

    def guided(x, u):
        bank.state["path"] = path
        try:
            return field.eps(x, u)
        finally:
            bank.state["path"] = prev
    try:
        return field.sample(n=n, steps=steps, seed=seed, clip=(-1, 1),
                            guide_eps=guided, guide_w=lambda u: float(w))
    finally:
        bank.state["path"] = prev


def fingerprint(field, bank, name, seed=0, n=2, steps=8):
    """A cheap deterministic signature of what `name` currently generates."""
    S = sample_concept(field, bank, [name], n=n, steps=steps, seed=seed, w=1.0)
    return np.asarray(S, np.float64).round(10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    p.add_argument("--attrs", nargs="+",
                   default=["Eyeglasses", "Smiling", "Wearing_Hat", "Male"])
    p.add_argument("--arms", nargs="+", default=["sequential", "batch", "batch-shared"])
    p.add_argument("--n-train", type=int, default=96)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--t-min", type=int, default=600)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--probe-img", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_train, args.epochs, args.n_samples, args.steps = 16, 2, 2, 20
        args.attrs = args.attrs[:2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field0, img = load_pretrained(args.model, device=dev, batch=args.batch)
    K = len(args.attrs)
    print(f"[n26] {args.model} {img}x{img} | {K} concepts {args.attrs} | device {dev}", flush=True)
    data = load_attr_groups(args.attrs, img=img, n_each=args.n_train, probe_img=args.probe_img)
    for a in args.attrs:
        print(f"[n26]   {a}: {len(data[a])} real images", flush=True)

    PX, PY = data["_probe_X"], data["_probe_Y"]
    cut = int(0.9 * len(PX))
    probes, ok = {}, {}
    for j, a in enumerate(args.attrs):
        pr = SoftmaxProbe(2, seed=args.seed, epochs=30).fit(PX[:cut], PY[:cut, j].astype(int))
        yv, pv = PY[cut:, j].astype(int), pr.predict(PX[cut:])
        tpr = float((pv[yv == 1] == 1).mean()) if (yv == 1).any() else 0.0
        tnr = float((pv[yv == 0] == 0).mean()) if (yv == 0).any() else 0.0
        probes[a], ok[a] = pr, (tpr >= 0.5 and 0.5 * (tpr + tnr) >= 0.65)
        print(f"[n26]   probe {a}: TPR {tpr:.2f} TNR {tnr:.2f} "
              f"{'usable' if ok[a] else 'NOT usable'}", flush=True)

    P = args.probe_img

    def to_probe(X01):
        n = len(X01)
        Z = np.asarray(X01, np.float32).reshape(n, 3, img, img)
        f = max(img // P, 1)
        return Z.reshape(n, 3, P, f, P, f).mean((3, 5)).reshape(n, -1)

    rows, panels = [], []
    for arm in args.arms:
        field, _ = load_pretrained(args.model, device=dev, batch=args.batch)
        bank = HierarchicalAdapterBank(field.model, rank=args.rank, device=dev,
                                       flat=(arm != "batch-shared"), targets=("linear",),
                                       exclude=("time_emb_proj", "time_embedding"))
        field.bank = bank
        pools = [data[a] for a in args.attrs]
        t0 = time.time()
        if arm == "sequential":
            for a, pool in zip(args.attrs, pools):
                bank.add_concept(a, to_model(pool).mean(0))
                train_group(field, bank, [a], [pool], img, args.epochs, args.batch, args.lr,
                            args.t_min, seed=args.seed)
        else:
            # ONE call registers all K -- with `group=True` they may be given a shared ancestor
            bank.add_concepts(list(args.attrs), [to_model(p).mean(0) for p in pools],
                              group=(arm == "batch-shared"))
            train_group(field, bank, list(args.attrs), pools, img, args.epochs, args.batch,
                        args.lr, args.t_min, seed=args.seed)
        wall = time.time() - t0

        marks = {a: fingerprint(field, bank, a, seed=args.seed) for a in args.attrs}
        hits, vds = {}, []
        for a in args.attrs:
            S = to_data(sample_concept(field, bank, [a], args.n_samples, args.steps,
                                       args.seed, args.guidance))
            vds.append(vendi_score(S))
            if ok[a]:
                hits[a] = float((probes[a].predict(to_probe(S)) == 1).mean())
            if a == args.attrs[0]:
                panels.append((f"{arm}\n{a} {hits.get(a, float('nan')):.2f}", S))
        exact = float(np.mean([float(np.array_equal(fingerprint(field, bank, a, seed=args.seed),
                                                    marks[a])) for a in args.attrs]))
        n_nodes = len({n for a in args.attrs for n in bank.concepts[a]["path"]})
        shared = len([1 for a in args.attrs for n in bank.concepts[a]["path"]]) - n_nodes
        row = dict(arm=arm, wall_seconds=wall, params=bank.n_params(), n_nodes=n_nodes,
                   shared_nodes=shared, retention_exact=exact,
                   mean_attr=float(np.mean([hits[a] for a in hits])) if hits else float("nan"),
                   vendi=float(np.mean(vds)), **{f"hit_{a}": v for a, v in hits.items()})
        rows.append(row)
        print(f"[n26] {arm:14s} wall {wall:6.0f}s  attr {row['mean_attr']:.2f}  "
              f"vendi {row['vendi']:.2f}  nodes {n_nodes} (+{shared} shared)  "
              f"exact {exact:.2f}  {bank.n_params() / 1e6:.2f}M", flush=True)
        del field, bank
        torch.cuda.empty_cache()

    save_json(dict(rows=rows, config=vars(args)),
              os.path.join(RESULTS, f"n26_batch_pretrained{args.tag}.json"), seed=args.seed)
    print(f"\n[n26] {K} concepts into {args.model}")
    print(f"    {'arm':14s}{'wall(s)':>9s}{'attr':>7s}{'vendi':>7s}{'nodes':>7s}"
          f"{'shared':>8s}{'exact':>7s}")
    for r in rows:
        print(f"    {r['arm']:14s}{r['wall_seconds']:9.0f}{r['mean_attr']:7.2f}{r['vendi']:7.2f}"
              f"{r['n_nodes']:7d}{r['shared_nodes']:8d}{r['retention_exact']:7.2f}")
    if len(rows) > 1 and rows[0]["arm"] == "sequential":
        for r in rows[1:]:
            print(f"    -> {r['arm']} is {rows[0]['wall_seconds'] / max(r['wall_seconds'], 1):.2f}x "
                  f"faster than sequential")

    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(panels), args.n_samples,
                                 figsize=(1.7 * args.n_samples + 2.0, 1.75 * len(panels)))
        axes = np.atleast_2d(axes)
        for r, (lab, S) in enumerate(panels):
            for c in range(args.n_samples):
                ax = axes[r][c]; ax.axis("off")
                if c < len(S):
                    ax.imshow(show(S[c], img))
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp in axes[r][0].spines.values():
                sp.set_visible(False)
        fig.suptitle(f"{K} concepts added to {args.model}: one update vs {K}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(FIG, f"n26_batch_pretrained{args.tag}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"[n26] wrote {out}")


if __name__ == "__main__":
    main()
