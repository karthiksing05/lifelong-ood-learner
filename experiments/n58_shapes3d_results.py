"""N58 -- the Shapes3D result set, on the PRETRAINED backbone.

Four from-scratch conditional Shapes3D models failed the same way (excellent samples, ignored label,
a NEGATIVE null gap).  N57 instead fine-tunes conditioning into a model that already renders the
domain -- `eurecom-ds/scoresdeve-ema-shapes3d-64`, bridged from variance-exploding to our
variance-preserving stack by `neural/ve_bridge.py`.  This produces the results.

Every number here is one the other two datasets also report, so Shapes3D can sit beside ColorMNIST
and CelebA rather than being described in prose:

    CONDITIONING GAP   loss(wrong class) - loss(right class) on TRAINING images, and the null gap.
                       This is the check that failed on every from-scratch attempt, and it comes
                       first: if the label is still decorative, nothing below means anything.
    BASIS R^2          does the class table organise compositionally (ColorMNIST 0.870, CelebA 0.620)
    ADDRESS            joint accuracy on held-out COMBINATIONS, against nearest-class and the
                       unconditional floor
    ROUTER             s_novel separation, held-out vs trained (ColorMNIST 0.980)
    CASE STUDY         figures/case_shapes3d.png, held-out combination NAMED on every block

    python experiments/n58_shapes3d_results.py --n-cases 12
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.evals import (factor_accuracy, fit_factor_probes,   # noqa: E402
                                precision_recall, probe_readable, save_json)
from lifelong_ood.data.factored import decode_slot, load_factored          # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize          # noqa: E402
from lifelong_ood.neural.router import auroc, conditional_losses       # noqa: E402
from lifelong_ood.neural.ve_bridge import load_ve_field                # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402

FIG = os.path.join(HERE, "figures")


def mode_ascent(field, x_img, u, n_starts=4, iters=120, lr=0.05, seed=0, init="hybrid"):
    """Adam ascent on log p_t against the bridged field -> (x0-space modes, provenance).

    The same Stage-1 discovery the ColorMNIST and CelebA case studies show in their `prototypes` row,
    so the three figures can be read side by side.  Hybrid initialisation: half the chains from the
    noised query, half from pure noise -- a pool grown only from the query explores one basin, and
    composition needs experts that disagree with it.
    """
    dev = field.device
    a = float(cosine_abar(np.array([u]))[0])
    g = torch.Generator(device=dev).manual_seed(seed)
    xq = torch.as_tensor(np.asarray(x_img, np.float32), device=dev).reshape(1, 3, field.img,
                                                                            field.img)
    half = n_starts // 2 if init == "hybrid" else n_starts
    parts, src = [], []
    if half:
        e = torch.randn((half, *xq.shape[1:]), generator=g, device=dev)
        parts.append((a ** 0.5) * xq.expand(half, -1, -1, -1) + ((1 - a) ** 0.5) * e)
        src += ["query"] * half
    if n_starts - half:
        parts.append(torch.randn((n_starts - half, *xq.shape[1:]), generator=g, device=dev))
        src += ["random"] * (n_starts - half)
    xt = torch.cat(parts).clone().requires_grad_(True)
    opt = torch.optim.Adam([xt], lr=lr)
    for _ in range(iters):
        eps = field.model(xt, torch.full((len(xt),), float(u), device=dev))
        loss = (eps.detach() * xt).sum() / max((1 - a) ** 0.5, 1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        eps = field.model(xt, torch.full((len(xt),), float(u), device=dev))
        x0 = (xt - ((1 - a) ** 0.5) * eps) / (a ** 0.5)
    return x0.detach().permute(0, 2, 3, 1).reshape(len(x0), -1).cpu().numpy(), src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_shapes3d_pre.pt"))
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="",
                   help="suffix for the JSON and the case-study figure.  Both output paths are "
                        "otherwise FIXED, so evaluating a sweep of checkpoints through this script "
                        "would overwrite the shipped figures/case_shapes3d.png with whichever rung "
                        "ran last -- silently, since nothing errors.  Pass a tag for any run that "
                        "is not the canonical backbone.")
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    field, ck = load_ve_field(args.ckpt, device=dev)
    sp = ck["split"]
    data = load_factored("shapes3d", n_per_slot=sp["n_per_slot"], seed=sp["seed"],
                         value_stride=sp["value_stride"], n_holdout=sp["n_holdout"],
                         fix_nuisance=sp["fix_nuisance"])
    sizes, names = data["sizes"], data["names"]
    X = np.asarray(data["X"], np.float32)
    slots = np.asarray(data["slot"])
    remap = ck["remap"]
    slots_seen = np.array(sorted(int(k) for k in remap))
    C = np.asarray(ck["ema_cemb"])[[remap[int(s)] for s in slots_seen]]
    print(f"[n58] {os.path.basename(args.ckpt)} <- {ck['model']}; {len(slots_seen)} seen "
          f"compositions over {names} {sizes}, wall_hue pinned", flush=True)

    probes = fit_factor_probes(data, seed=args.seed, epochs=30)
    readable = [n for n in names if probe_readable(probes, n)]
    print(f"[n58] probes readable {readable} of {names}", flush=True)

    # ── 1. IS IT CONDITIONAL?  the check every from-scratch attempt failed ───
    rng = np.random.default_rng(args.seed)
    null = np.asarray(ck["ema_cemb"])[ck["null_id"]]
    gaps, nulls = [], []
    for s in rng.permutation(slots_seen)[:12]:
        idx = np.where((slots == int(s)) & data["seen"])[0][:4]
        right = C[list(slots_seen).index(int(s))]
        wrong = C[int(rng.integers(len(C)))]
        for i in idx:
            L = conditional_losses(field, X[i], np.stack([right, wrong, null]),
                                   u_grid=(0.15, 0.30, 0.45), n_noise=4, seed=args.seed)
            gaps.append(float(L[1] - L[0]))
            nulls.append(float(L[2] - L[0]))
    gap, ngap = float(np.mean(gaps)), float(np.mean(nulls))
    print(f"\n[n58] CONDITIONING GAP  wrong-right {gap:+.5f}   null-right {ngap:+.5f}")
    print(f"[n58]   [from scratch: gap +0.00082, null gap -0.00088 = the label was decorative]")
    print(f"[n58]   [ColorMNIST, which works: gap +0.00140, null gap +0.00018]")
    # A SIGN TEST IS NOT ENOUGH.  The first version asked only `gap > 0 and ngap > 0` and printed
    # CONDITIONAL on a gap of +0.00003 -- 47x smaller than ColorMNIST's +0.00140 and indistinguishable
    # from zero.  A backbone is only usefully conditional if the right class beats a wrong one by a
    # margin comparable to one that demonstrably works, so the bar is a fraction of ColorMNIST's.
    REF_GAP = 0.00140                      # ColorMNIST, which conditions and composes
    conditional = gap > 0.25 * REF_GAP and ngap > 0
    verdict = ("CONDITIONAL" if conditional else
               "NOT USEFULLY CONDITIONAL" if gap > 0 else "NOT CONDITIONAL")
    print(f"[n58]   -> {verdict}  (gap is {gap / REF_GAP:.2f}x ColorMNIST's; bar is 0.25x)",
          flush=True)

    fac = factorize(C, decode_slot(slots_seen, sizes))
    print(f"[n58] basis R^2 {fac['r2']:.3f}   [ColorMNIST 0.870, CelebA 0.620]", flush=True)

    # ── 2. ADDRESS on held-out combinations ─────────────────────────────────
    held = np.array(sorted(set(int(s) for s in np.unique(slots)) - set(slots_seen.tolist())))
    cases = rng.permutation(held)[:args.n_cases]
    rows, sig_h, sig_s = [], [], []
    for ci, s in enumerate(cases):
        tgt = decode_slot(np.array([int(s)]), sizes)[0]
        idx = np.where(slots == int(s))[0]
        R = X[idx[:24]]
        c_addr = compose(fac, tuple(int(v) for v in tgt))
        d2 = ((C - c_addr[None]) ** 2).sum(1)
        arms = {
            "ours (ADDRESS)": field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                           emb_override=c_addr),
            "nearest-class": field.sample(n=args.n_samples, steps=args.steps, seed=args.seed,
                                          branch=remap[int(slots_seen[int(np.argmin(d2))])]),
            "unconditional": field.sample(n=args.n_samples, steps=args.steps,
                                          seed=args.seed + 7919 * (ci + 1),
                                          branch=ck["null_id"]),
        }
        for nm, S in arms.items():
            a = factor_accuracy(probes, np.asarray(S, np.float32), tgt)
            pr = precision_recall(np.asarray(S, np.float32), R)
            rows.append(dict(slot=int(s), arm=nm, joint=a["joint"], recall=pr["recall"],
                             **{k: v for k, v in a.items() if k in names}))
        print(f"[n58] {ci + 1:2d}/{len(cases)} slot {int(s):3d}  "
              + "  ".join(f"{nm.split()[0]} {[r for r in rows if r['slot'] == int(s) and r['arm'] == nm][0]['joint']:.2f}"
                          for nm in arms), flush=True)
        L = conditional_losses(field, X[idx[0]], C, u_grid=(0.10,), n_noise=4, seed=args.seed)
        sig_h.append(float(L.min()))
    for s in rng.permutation(slots_seen)[:len(cases)]:
        i = int(np.where(slots == int(s))[0][0])
        L = conditional_losses(field, X[i], C, u_grid=(0.10,), n_noise=4, seed=args.seed)
        sig_s.append(float(L.min()))

    print("\n[n58] " + "=" * 66)
    summary = []
    for nm in ("ours (ADDRESS)", "nearest-class", "unconditional"):
        R_ = [r for r in rows if r["arm"] == nm]
        m = {k: float(np.nanmean([r[k] for r in R_])) for k in
             ["joint", "recall"] + [n for n in names if n in R_[0]]}
        se = float(np.nanstd([r["joint"] for r in R_]) / max(len(R_) ** 0.5, 1))
        summary.append(dict(arm=nm, n=len(R_), se=se, **m))
        print(f"[n58] {nm:16s} joint {m['joint']:.3f} +/- {se:.3f} | "
              + " ".join(f"{k} {m[k]:.3f}" for k in names if k in m)
              + f" | recall {m['recall']:.3f}")
    a_nov = auroc(sig_h, sig_s)
    m_ours = float(np.mean([r["joint"] for r in rows if r["arm"] == "ours (ADDRESS)"]))
    m_unc = float(np.mean([r["joint"] for r in rows if r["arm"] == "unconditional"]))
    ratio = m_ours / max(m_unc, 1e-9)
    print(f"[n58] VERDICT: ADDRESS {m_ours:.3f} vs unconditional floor {m_unc:.3f} = {ratio:.1f}x "
          f"-> {'COMPOSES' if ratio > 2 and m_ours > 0.2 else 'DOES NOT COMPOSE'}")
    print(f"[n58] router s_novel AUROC (held-out vs trained) {a_nov:.3f}   [ColorMNIST 0.980]"
          + ("   <- ROUTING does not transfer here even though COMPOSITION does"
             if a_nov < 0.7 else ""))

    save_json(dict(rows=rows, summary=summary, basis_r2=float(fac["r2"]), gap=gap,
                   null_gap=ngap, conditional=bool(conditional), novel_auroc=float(a_nov),
                   readable=readable, names=names, sizes=sizes, backbone=ck["model"],
                   config=vars(args)),
              os.path.join(RESULTS, f"n58_shapes3d{args.tag}.json"), seed=args.seed)

    # ── 3. the case study ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["real", "prototypes", "ours (ADDRESS)", "nearest-class", "unconditional"]
    show = cases[:4]
    fig, axes = plt.subplots(len(show) * len(order), args.n_show,
                             figsize=(1.5 * args.n_show + 4.4, 1.5 * len(show) * len(order)))
    axes = np.atleast_2d(axes)
    r = 0
    for ci, s in enumerate(show):
        tgt = decode_slot(np.array([int(s)]), sizes)[0]
        idx = np.where(slots == int(s))[0]
        n_tr = int(np.sum((slots == int(s)) & data["seen"]))
        c_addr = compose(fac, tuple(int(v) for v in tgt))
        d2 = ((C - c_addr[None]) ** 2).sum(1)
        # the prototypes Stage 1 discovers, at the same noise levels f9 shows
        P, psrc = [], []
        for ui, uu in enumerate((0.05, 0.15, 0.30)):
            m, sr = mode_ascent(field, X[idx[0]].reshape(64, 64, 3).transpose(2, 0, 1), uu,
                                n_starts=2, seed=args.seed + 9973 * int(s) + 131 * ui)
            P.append(m); psrc += [f"{q}-init u={uu:g}" for q in sr]
        got = {"real": X[idx[:args.n_show]],
               "prototypes": np.concatenate(P)[:args.n_show],
               "ours (ADDRESS)": field.sample(n=args.n_show, steps=args.steps, seed=args.seed,
                                              emb_override=c_addr),
               "nearest-class": field.sample(n=args.n_show, steps=args.steps, seed=args.seed,
                                             branch=remap[int(slots_seen[int(np.argmin(d2))])]),
               "unconditional": field.sample(n=args.n_show, steps=args.steps,
                                             seed=args.seed + 7919 * (ci + 1),
                                             branch=ck["null_id"])}
        combo = ", ".join(f"{n}={int(v)}" for n, v in zip(names, tgt))
        for nm in order:
            S = np.asarray(got[nm], np.float32)
            a = (factor_accuracy(probes, S, tgt) if nm != "prototypes"
                 else {"joint": float("nan")})
            for k in range(args.n_show):
                axes[r][k].axis("off")
                if k < len(S):
                    axes[r][k].imshow(np.clip(S[k].reshape(64, 64, 3), 0, 1))
            lab = (f"real - HELD OUT\n{combo}\nslot {int(s)} - {n_tr} training images"
                   if nm == "real" else
                   "prototypes\n(mode ascent)" if nm == "prototypes" else nm)
            axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6.5)
            axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
            for sp_ in axes[r][0].spines.values():
                sp_.set_visible(False)
            if nm == "prototypes":
                for k in range(min(args.n_show, len(psrc))):
                    axes[r][k].set_title(psrc[k], fontsize=5.5)
            else:
                axes[r][args.n_show - 1].set_title(f"joint {a['joint']:.2f}", fontsize=6.5,
                                                   loc="right")
            r += 1
    fig.suptitle(
        f"Shapes3D on a PRETRAINED unconditional backbone ({ck['model']}), conditioning fine-tuned "
        f"in.\nFour from-scratch attempts left the label decorative; this one has conditioning gap "
        f"{gap:+.5f} and null gap {ngap:+.5f}. Basis R^2 {fac['r2']:.3f}.\nComposition factors are "
        f"{names}; wall hue is PINNED and scale/orientation vary within every block.", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    o = os.path.join(FIG, f"case_shapes3d{args.tag}.png")
    fig.savefig(o, dpi=130, bbox_inches="tight")
    print(f"[n58] wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
