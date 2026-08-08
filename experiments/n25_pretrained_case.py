"""N25 — the case study on a REAL pretrained model: `google/ddpm-celebahq-256`, 113.7M params, 256px.

Why this experiment exists.  Every case-study figure in this repository has been drawn from a UNet we
trained ourselves (6.8M params at 32x32, 9.3M at 64x64), and at that scale the samples are legible
but not good.  A reader cannot tell a compositional failure from a capacity failure, which makes the
figures weak evidence however sound the numbers behind them are.

N15 tried this and stopped at a diagnosis: conditioning a pretrained unconditional model through a
single additive vector on its 512-d time embedding does not move it (`per-combo` 0.167 against
`uncond` 0.146 -- fitting ONE combination directly barely beat doing nothing).  The bottleneck was
the conditioning channel, not the composer.  `CondUNetShim` supplies the capacity that was missing:
the same insert-only LoRA bank used everywhere else, attached to someone else's UNet.
`tests/test_pretrained_bank.py` certifies that attaching it changes the pretrained model by exactly
0.000e+00 and that preservation still holds.

WHAT IS AND IS NOT CLAIMED HERE.  This model was trained on CelebA-HQ by someone else, so we cannot
say it never saw a smiling face in glasses -- the out-of-distribution claim of N9/N20 rests on
backbones whose training set we controlled, and it is not restated here.  What IS controlled is what
OUR CONCEPTS saw:

    concept A is trained only on images with (attr_a = 1, attr_b = 0)
    concept B is trained only on images with (attr_b = 1, attr_a = 0)

so the PAIR is held out of both concepts by construction, and composing them is a genuine held-out
combination at the level of the representations this method builds.  That is exactly Composable
Diffusion's setting, which is why it is the baseline that matters here.

Rows in the figure, and what each isolates:
  `real`            real CelebA faces with both attributes -- the target.
  `pretrained`      the untouched model.  What the backbone does with no concept at all.
  `A only` / `B only`   each concept alone: did the concept learn its own attribute?
  `ours`            the two concepts composed in conditioning space, then consolidated.
  `composable-diff` the same two concepts composed by ADDING their score residuals (Liu et al.).

    python experiments/n25_pretrained_case.py --attrs Eyeglasses Smiling --n-train 96
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.celeba import find_celeba                            # noqa: E402
from lifelong_ood.evals import save_json, vendi_score                  # noqa: E402
from lifelong_ood.neural.adapters import HierarchicalAdapterBank       # noqa: E402
from lifelong_ood.neural.composable import composable_eps              # noqa: E402
from lifelong_ood.neural.pretrained import load_pretrained             # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_hires(attrs_wanted, img=256, n_each=96, n_probe=4000, max_rows=60000, seed=0,
               probe_img=128):
    """Real CelebA at `img` px, split into the groups this experiment needs, plus a probe set.

    Reads the requested attribute COLUMNS straight from the parquet rather than going through
    `celeba.ATTRS`, which is a 14-attribute subset chosen for the compositional-class construction
    and does not contain every attribute worth a case study (`Eyeglasses`, for one).  The probes are
    trained here too, at 64px on the same source, so this experiment does not depend on a probe set
    fitted for a different attribute list.

    -> dict(a_only, b_only, both) at `img` px, and (probe_X64, probe_Y) for fitting.
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image
    from lifelong_ood.celeba import _parquet_shards
    root = find_celeba()
    A, B = attrs_wanted
    want = {"a_only": [], "b_only": [], "both": []}
    need = dict(a_only=n_each, b_only=n_each, both=32)
    pX, pY, n, done = [], [], 0, False
    for sp in _parquet_shards(root):
        f = pq.ParquetFile(sp)
        for batch in f.iter_batches(batch_size=512, columns=["image", A, B]):
            d = batch.to_pydict()
            for k in range(len(d["image"])):
                a = 1 if d[A][k] else 0
                b = 1 if d[B][k] else 0
                n += 1
                im = None
                grp = ("both" if (a and b) else
                       "a_only" if (a and not b) else
                       "b_only" if (b and not a) else None)
                if grp is not None and len(want[grp]) < need[grp]:
                    im = Image.open(io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
                    w, h = im.size
                    top = max(0, (h - w) // 2)
                    sq = im.crop((0, top, w, top + w))
                    # CHW-flat, NOT HWC: `field.sample` returns [n, 3, img, img] flattened, and
                    # training reshapes flat rows to (3, img, img).  Storing real images as HWC would
                    # feed the model transposed data and render every generation as a tiled mosaic --
                    # the same layout bug this repo already had once, caught the same way, by looking.
                    want[grp].append((np.asarray(sq.resize((img, img), Image.BICUBIC),
                                                 np.float32) / 255.0).transpose(2, 0, 1).ravel())
                if len(pX) < n_probe:
                    if im is None:
                        im = Image.open(io.BytesIO(d["image"][k]["bytes"])).convert("RGB")
                        w, h = im.size
                        top = max(0, (h - w) // 2)
                        sq = im.crop((0, top, w, top + w))
                    pX.append((np.asarray(sq.resize((probe_img, probe_img), Image.BICUBIC),
                                          np.float32) / 255.0).transpose(2, 0, 1).ravel())
                    pY.append([a, b])
            done = all(len(want[g]) >= need[g] for g in want) and len(pX) >= n_probe
            if done or n >= max_rows:
                break
        if done or n >= max_rows:
            break
    out = {g: (np.stack(v) if v else np.zeros((0, 3 * img * img), np.float32))
           for g, v in want.items()}
    out["probe_X"] = (np.stack(pX) if pX else
                      np.zeros((0, 3 * probe_img * probe_img), np.float32))
    out["probe_Y"] = np.array(pY, np.int8) if pY else np.zeros((0, 2), np.int8)
    return out


def to_model(X01):
    return np.asarray(X01, np.float32) * 2.0 - 1.0


def to_data(Xm):
    return np.clip((np.asarray(Xm, np.float32) + 1.0) / 2.0, 0.0, 1.0)


def show(row, img):
    """CHW-flat -> HWC for imshow.  Everything in this file is CHW-flat; only display transposes."""
    return np.clip(np.asarray(row, np.float32).reshape(3, img, img).transpose(1, 2, 0), 0, 1)


def sample_concept_cfg(field, names, args):
    """Sample a concept (or a union of concepts) with GUIDANCE against the frozen base model.

    This is the lever that makes a pretrained backbone workable, and it follows from something this
    repository already relies on: the unconditional branch IS the frozen model with an empty adapter
    path.  No CFG dropout has to be trained for, and no unconditional token is needed.

    So instead of training the adapter hard enough to move a 113.7M-parameter model on its own --
    which degrades it, measured -- the adapter is trained GENTLY and amplified at sampling time:

        eps_hat = eps_frozen + w * ( eps_adapter - eps_frozen )

    A delta too small to damage the model is still a perfectly good DIRECTION, and w decides how far
    along it to travel.  Passing several names activates their paths together, which is how a
    composed concept is sampled.
    """
    bank = field.bank
    prev = list(bank.state["path"])
    path = sorted({n for nm in names for n in bank.concepts[nm]["path"]})

    def guided(x, u):
        bank.state["path"] = path
        try:
            return field.eps(x, u)
        finally:
            bank.state["path"] = prev

    try:
        return field.sample(n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1),
                            guide_eps=guided, guide_w=lambda u: float(args.concept_guidance))
    finally:
        bank.state["path"] = prev


def train_concept(field, bank, name, X01, img, epochs=6, batch=4, lr=1e-4, seed=0,
                  t_min=0, verbose=True):
    """Consolidate one concept into the bank: a LoRA leaf, base frozen.

    The concept EMBEDDING is deliberately held at zero here, unlike on our own backbone.  Our UNet
    was trained with a class vector added to its time embedding, so that slot means something to it.
    This model is unconditional: the only thing its time embedding encodes is the noise level, and an
    offset on it is read as "you are at a different timestep".  Training that vector therefore
    corrupts denoising globally -- measured, and visible as melted faces in the v1 figure -- while
    carrying no concept information the adapter path is not already carrying.
    """
    emb_p = bank._emb_extra[name]
    with torch.no_grad():
        emb_p.zero_()
    params = [q for q in bank.trainable(name) if q is not emb_p]
    opt = torch.optim.Adam(params, lr=lr)
    X = torch.from_numpy(to_model(X01)).to(field.device).reshape(-1, 3, img, img)
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = len(field._abar)
    ab = torch.from_numpy(field._abar).float().to(field.device)
    field.model.train()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot, nb = 0.0, 0
        for i0 in range(0, len(X), batch):
            x0 = X[perm[i0:i0 + batch].to(field.device)]
            # sample NATIVE trained timesteps: this model has its own (linear-beta) schedule and
            # training it on our cosine grid would optimize it at noise levels it never saw
            ts = torch.randint(int(t_min), T, (len(x0),), generator=g).to(field.device)
            a = ab[ts].view(-1, 1, 1, 1)
            e = torch.randn(x0.shape, generator=g).to(field.device)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            with bank.activate(name):
                pred = field.model.forward_t(xt, ts, emb_override=bank.embedding(name))
            loss = torch.nn.functional.mse_loss(pred, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        if verbose:
            print(f"      [{name}] epoch {ep + 1}/{epochs}: {tot / max(nb, 1):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    field.model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/ddpm-celebahq-256")
    p.add_argument("--attrs", nargs=2, default=["Eyeglasses", "Smiling"],
                   help="the two attributes; their CO-OCCURRENCE is held out of both concepts")
    p.add_argument("--n-train", type=int, default=96, help="real images per concept")
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--probe-img", type=int, default=128,
                   help="resolution the attribute probes are fit and evaluated at; 64 "
                        "is too coarse for fine attributes (Eyeglasses TPR 0.38 there)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--targets", nargs="+", default=["linear"],
                   help="layer types to adapt.  `linear` only (attention and the "
                        "projections) is where the LoRA personalization literature puts "
                        "it; adapting every conv of a 113.7M model as well melts the "
                        "samples.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=100, help="DDIM steps")
    p.add_argument("--comp-scale", type=float, default=1.0)
    p.add_argument("--concept-guidance", type=float, default=3.0,
                   help="CFG weight for a concept against the FROZEN base.  Lets the "
                        "adapter stay small enough not to damage the model while still "
                        "steering it.")
    p.add_argument("--t-min", type=int, default=0,
                   help="lowest training timestep.  Our images are CelebA upsampled from "
                        "178px and the model is CelebA-HQ, so low-t training teaches it "
                        "our blur rather than the attribute.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_train, args.epochs, args.n_samples, args.steps = 16, 1, 3, 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, img = load_pretrained(args.model, device=dev, batch=args.batch)
    nbase = sum(q.numel() for q in field.unet.parameters())
    print(f"[n25] {args.model}: {img}x{img}, {nbase / 1e6:.1f}M params, device {dev}", flush=True)

    A, B = args.attrs
    data = load_hires(args.attrs, img=img, n_each=args.n_train, seed=args.seed,
                      probe_img=args.probe_img)
    print(f"[n25] real images: {A}-only {len(data['a_only'])}, {B}-only {len(data['b_only'])}, "
          f"BOTH {len(data['both'])} (the target, never shown to either concept)", flush=True)

    bank = HierarchicalAdapterBank(field.model, rank=args.rank, device=dev, flat=True,
                                   targets=tuple(args.targets))
    field.bank = bank
    got = {}

    got["real"] = data["both"][:args.n_samples] if len(data["both"]) else None
    got["pretrained"] = to_data(field.sample(n=args.n_samples, steps=args.steps,
                                             seed=args.seed, clip=(-1, 1)))

    # ── learn one concept per attribute, each blind to the other ─────────────
    for nm, key in ((A, "a_only"), (B, "b_only")):
        mu = to_model(data[key]).mean(0)
        bank.add_concept(nm, mu)
        print(f"[n25] consolidating `{nm}` from {len(data[key])} real images "
              f"({A if key == 'a_only' else B}=1, other=0)", flush=True)
        train_concept(field, bank, nm, data[key], img, epochs=args.epochs, batch=args.batch,
                      lr=args.lr, seed=args.seed, t_min=args.t_min)
        got[f"{nm} only"] = to_data(sample_concept_cfg(field, [nm], args))

    # ── OURS: compose in conditioning space, over the union of both paths ────
    cA = cB = c_comp = None                    # conditioning is carried by the adapter paths
    got["ours"] = to_data(sample_concept_cfg(field, [A, B], args))

    # ── Composable Diffusion: add the two concepts' score residuals ──────────
    # each conditional is evaluated on its OWN adapter path, which is the faithful reading of
    # "compose two independently learned concepts" for a bank rather than a single shared model
    def eps_branch(nm, emb):
        def f(x, u):
            return field.eps(x, u, branch=nm, emb=emb)
        return f

    e_null = lambda x, u: field.eps(x, u)                                    # noqa: E731
    parts = [(eps_branch(A, None), float(args.comp_scale)),
             (eps_branch(B, None), float(args.comp_scale))]

    def comp_eps(x, u):
        e0 = e_null(x, u)
        out = e0.copy()
        for f, w in parts:
            out = out + w * (f(x, u) - e0)
        return out

    got["composable-diff"] = to_data(field.sample(
        n=args.n_samples, steps=args.steps, seed=args.seed, clip=(-1, 1),
        guide_eps=comp_eps, guide_w=lambda u: 1.0))

    # ── score with the CelebA attribute probes (trained at 64px) ─────────────
    rows = []
    probes, probe_ok = {}, {}
    PX, PY = data["probe_X"], data["probe_Y"]
    if len(PX) > 200:
        from lifelong_ood.evals import SoftmaxProbe
        cut = int(0.9 * len(PX))
        for j, nm in enumerate(args.attrs):
            pr = SoftmaxProbe(2, seed=args.seed, epochs=30).fit(PX[:cut], PY[:cut, j].astype(int))
            yv = PY[cut:, j].astype(int)
            pv = pr.predict(PX[cut:])
            tpr = float((pv[yv == 1] == 1).mean()) if (yv == 1).any() else 0.0
            tnr = float((pv[yv == 0] == 0).mean()) if (yv == 0).any() else 0.0
            probes[nm] = pr
            probe_ok[nm] = (tpr >= 0.5 and 0.5 * (tpr + tnr) >= 0.65)
            print(f"[n25] probe {nm}: TPR {tpr:.2f} TNR {tnr:.2f} balanced {0.5*(tpr+tnr):.2f} "
                  f"{'(usable)' if probe_ok[nm] else '(NOT usable -- not reported)'}", flush=True)
    else:
        print("[n25] too few probe images; reporting images only", flush=True)

    P = args.probe_img

    def to_probe(X01):
        """CHW-flat at `img` px -> CHW-flat at the probe resolution, by mean-pooling."""
        n = len(X01)
        Z = np.asarray(X01, np.float32).reshape(n, 3, img, img)
        f = max(img // P, 1)
        return Z.reshape(n, 3, P, f, P, f).mean((3, 5)).reshape(n, -1)

    for arm, S in got.items():
        if S is None or not len(S):
            continue
        r = dict(arm=arm, vendi=vendi_score(S))
        if probes:
            S64 = to_probe(S)
            for nm in args.attrs:
                if probe_ok.get(nm):
                    r[nm] = float((probes[nm].predict(S64) == 1).mean())
            if all(probe_ok.get(nm) for nm in args.attrs):
                pa = probes[A].predict(S64) == 1
                pb = probes[B].predict(S64) == 1
                r["both"] = float((pa & pb).mean())
        rows.append(r)

    # ── the figure ──────────────────────────────────────────────────────────
    order = [a for a in ["real", "pretrained", f"{A} only", f"{B} only", "composable-diff", "ours"]
             if got.get(a) is not None and len(got[a])]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = len(order), args.n_samples
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.62 * nrow))
    axes = np.atleast_2d(axes)
    for r, arm in enumerate(order):
        S = got[arm]
        sc = next((x for x in rows if x["arm"] == arm), {})
        for c in range(ncol):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(S):
                ax.imshow(show(S[c], img))
        lab = arm
        if "both" in sc:
            lab += f"\nboth {sc['both']:.2f}"
        axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8.5)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for sp in axes[r][0].spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{args.model} — `{A}` and `{B}` learned separately, composed\n"
                 f"(their co-occurrence was withheld from BOTH concepts)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, f"n25_pretrained_case{args.tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    save_json(dict(rows=rows, config=vars(args), base_params=nbase),
              os.path.join(RESULTS, f"n25_pretrained_case{args.tag}.json"), seed=args.seed)
    print(f"\n[n25] {args.model}, `{A}` x `{B}`")
    hdr = f"    {'arm':18s}{A[:10]:>12s}{B[:10]:>12s}{'both':>8s}{'vendi':>8s}"
    print(hdr)
    for r in rows:
        print(f"    {r['arm']:18s}{r.get(A, float('nan')):12.2f}{r.get(B, float('nan')):12.2f}"
              f"{r.get('both', float('nan')):8.2f}{r['vendi']:8.2f}")
    print(f"[n25] wrote {out}")


if __name__ == "__main__":
    main()
