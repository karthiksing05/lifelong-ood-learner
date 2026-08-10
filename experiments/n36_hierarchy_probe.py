"""N36 -- which instances FORMED each node's adapter, drawn as the hierarchy.

The bank's justification is that an internal node holds structure shared by its descendants -- the
Cobweb inheritance, where an internal node is a genuine concept ("cyan-on-dark") and not a routing
decision.  That has been argued, priced and certified, but never LOOKED AT.  Existing diagnostics
report delta norms (`diag_shared_node`: is ||BA|| nonzero) or paths and gradients (`diag_paths`);
none shows what a node is made of.

**The measurement: provenance, not response.**  Asking what a node "attends to" is ill-posed -- a
LoRA node is not an attention module and has no attention map.  What is well posed is which training
instances SHAPED it, and the training signal itself answers that.  Consolidation minimizes an
eps-prediction loss over the members' distillation pools, so for pool instance x_i and node n:

    infl(n, x_i) = || d L(x_i) / d (A_n, B_n) ||_F        at the trained parameters

is exactly "how strongly does this instance still pull on this node's parameters" -- one backward
pass per instance, using the SAME loss, conditioning and noise schedule `consolidate_batch` used, so
it is attribution against the real objective rather than a proxy.  Reported honestly as an
AT-CONVERGENCE influence: it is not integrated over the training trajectory, and a perfectly fitted
instance can score low precisely because the node already explains it.

That gives the picture the hierarchy claims to have:

  * a LEAF should be shaped by its own concept's pool and nothing else -- it is private;
  * an INTERNAL node should be shaped by instances from SEVERAL of its members, and its top
    instances should be spread across them rather than dominated by one.

`member_entropy` makes that quantitative: the normalized entropy of the top-k instances' concept
membership, which is ~1.0 when a node is genuinely shared and ~0.0 when one member dominates it.
A shared node whose entropy is near zero would mean the "joint structure" story is false -- the node
would simply belong to whichever concept shouted loudest.

Two supporting probes:

  `alone`          generate with root..node active and NO leaf, conditioned on its descendants' mean
                   embedding.  The direct test of "internal nodes are genuine concepts".  **Stated
                   in advance:** ancestors are only ever trained with a full path active, so acting
                   alone is off-distribution for them, and incoherence here is a real negative
                   result rather than a bug.
  `layer_profile`  ||B_nA_n||_F / ||W_base||_F per wrapped layer, in depth order.  Falsifiable
                   prediction: shared ancestors act at COARSE resolutions, private leaves at fine.
                   Same profile for both would mean depth is not organizing abstraction.

Writes `figures/f13_hierarchy_probe.png` -- the tree drawn with each node's formative instances,
colour-coded by which concept they came from -- and `f13b_layer_profiles.png`.

    python experiments/n36_hierarchy_probe.py --n-concepts 8
"""
import argparse
import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                                # noqa: E402

from lifelong_ood.datasets import rgb_img, load_colormnist                     # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                           # noqa: E402
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,            # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.ddpm import CH, _as_img                               # noqa: E402
from lifelong_ood.score import cosine_abar                                     # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


@contextlib.contextmanager
def forced(bank, name, path, emb=None):
    """Temporarily make `name` resolve to `path`, and optionally to a different embedding.

    Both overrides go through `bank`, and they have to: `activate()` re-reads
    `concepts[name]["path"]` on every `eps` call (so setting `state["path"]` from outside is silently
    overwritten -- that is how an off-concept drift metric once read a constant 0.0000), and
    `_emb(branch)` re-reads `_emb_extra[name]` the same way.

    Why not `inversion.sample_embedding`, which conditions on a raw vector?  It calls
    `field.eps(x, u, None, emb=c)` with branch=None, and branch=None activates the EMPTY path -- it
    would switch off the very node being probed.  Overriding the concept's own stored embedding and
    sampling with `branch=name` keeps the forced path active AND applies the chosen conditioning.
    """
    import torch as _t
    old_p = bank.concepts[name]["path"]
    old_e = bank._emb_extra.get(name)
    bank.concepts[name]["path"] = list(path)
    if emb is not None:
        bank._emb_extra[name] = _t.as_tensor(np.asarray(emb), dtype=_t.float32,
                                             device=bank.device)
    try:
        yield
    finally:
        bank.concepts[name]["path"] = old_p
        if old_e is not None:
            bank._emb_extra[name] = old_e


def influence(L, field, name, pool, node_ids, u_grid, seed):
    """|| dL(x_i)/d(A_n, B_n) || for each pool instance and each node on this concept's path.

    Mirrors `consolidate_batch`'s loss exactly: eps-prediction MSE, conditioned on the concept's
    learned embedding, with the concept's path active.  -> {node_id: [n_pool] float}
    """
    bank, model = L.bank, field.model
    emb = bank.embedding(name)
    g = torch.Generator(device=field.device).manual_seed(seed)
    out = {nid: np.zeros(len(pool)) for nid in node_ids}
    params = {nid: [w.A[str(nid)] for w in bank.wrapped if str(nid) in w.A]
                   + [w.B[str(nid)] for w in bank.wrapped if str(nid) in w.A]
              for nid in node_ids}
    with bank.activate(name):
        for i, x in enumerate(pool):
            x0 = _as_img(np.atleast_2d(x), field.device, field.img)
            for u in u_grid:
                a = float(cosine_abar(np.array([u]))[0])
                e = torch.randn(x0.shape, generator=g, device=field.device)
                xt = (a ** 0.5) * x0 + ((1 - a) ** 0.5) * e
                ub = torch.full((1,), float(u), device=field.device)
                pred = model(xt, ub, emb_override=emb)
                loss = torch.nn.functional.mse_loss(pred, e)
                flat = [p for nid in node_ids for p in params[nid]]
                grads = torch.autograd.grad(loss, flat, retain_graph=False, allow_unused=True)
                k = 0
                for nid in node_ids:
                    n2 = 0.0
                    for _ in params[nid]:
                        gr = grads[k]
                        k += 1
                        if gr is not None:
                            n2 += float(gr.detach().pow(2).sum())
                    out[nid][i] += n2 ** 0.5
    for nid in node_ids:
        out[nid] /= max(len(u_grid), 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--n-concepts", type=int, default=8)
    p.add_argument("--n-pool", type=int, default=32)
    p.add_argument("--n-show", type=int, default=6)
    p.add_argument("--n-per-slot", type=int, default=60)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--u-grid", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.n_concepts, args.n_pool, args.n_per_slot = 3, 12, 20
        args.epochs, args.steps = 3, 20
        args.u_grid = [0.5]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    qs = ood_queries(data, n_slots=args.n_concepts, seed=args.seed)[:args.n_concepts]
    names = [f"c{q['slot']}" for q in qs]
    xqs = [to_model(data["X"][q["query"]]).ravel() for q in qs]

    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    # batched: sequential insertion of a handful of concepts often gives a star, and a star has no
    # internal node to probe.  Done in three explicit steps so the POOLS are kept -- the instances
    # whose provenance this experiment is about.
    teachers = L.compose_batch(xqs)
    pools = L.sample_teacher_batch(teachers, n=args.n_pool, seed=args.seed)
    L.consolidate_batch([(n, pl, t["mu"]) for n, pl, t in zip(names, pools, teachers)],
                        epochs=args.epochs, group=True)
    bank = L.bank
    paths = {n: list(bank.concepts[n]["path"]) for n in names}
    nodes = sorted({nid for pth in paths.values() for nid in pth})
    members = {nid: [n for n in names if nid in paths[n]] for nid in nodes}
    print(f"[n36] {len(names)} concepts, {len(nodes)} nodes")
    for nid in nodes:
        print(f"    node {nid:3d}: {len(members[nid])} member(s)  {members[nid]}")
    internal = [nid for nid in nodes if len(members[nid]) > 1]
    if not internal:
        print("[n36] NOTE: no node is shared -- the tree is a star, so there is no internal-node "
              "structure to probe on this run")

    # ── provenance: influence of every member's pool instance on every node ──
    infl = {nid: [] for nid in nodes}        # node -> list of (concept, pool_index, value)
    for n, pool in zip(names, pools):
        got = influence(L, field, n, pool, paths[n], args.u_grid, args.seed)
        for nid, vals in got.items():
            infl[nid] += [(n, i, float(v)) for i, v in enumerate(vals)]
        print(f"[n36] influence computed for {n} ({len(pool)} instances x {len(paths[n])} nodes)",
              flush=True)

    def layer_profile(nid):
        k = str(nid)
        out = []
        for w in bank.wrapped:
            if k not in w.A:
                out.append(0.0)
                continue
            A, B = w.A[k].detach(), w.B[k].detach()
            d = (torch.einsum("or,rikl->oikl", B[:, :, 0, 0], A) if w.is_conv else B @ A)
            out.append(float(d.norm() / (w.base.weight.detach().norm() + 1e-12)))
        return out

    rows = []
    for nid in nodes:
        items = sorted(infl[nid], key=lambda t: -t[2])
        top = items[:args.n_show]
        # how spread across members are the top instances?  1.0 = evenly shared, 0.0 = one member
        cnt = {}
        for c, _, _ in items[:max(args.n_show, 8)]:
            cnt[c] = cnt.get(c, 0) + 1
        pr = np.array(list(cnt.values()), float)
        pr /= pr.sum()
        ent = float(-(pr * np.log(pr + 1e-12)).sum() / np.log(len(members[nid])))\
            if len(members[nid]) > 1 else float("nan")
        prof = layer_profile(nid)
        rows.append(dict(node=int(nid), n_members=len(members[nid]), members=members[nid],
                         is_internal=len(members[nid]) > 1,
                         member_entropy=ent,
                         top=[dict(concept=c, idx=int(i), infl=float(v)) for c, i, v in top],
                         mean_infl=float(np.mean([v for _, _, v in items])) if items else 0.0,
                         contributing_members=sorted(cnt),
                         layer_profile=prof,
                         profile_centroid=float(np.average(np.arange(len(prof)), weights=prof))
                         if np.sum(prof) > 0 else float("nan")))
        print(f"[n36] node {nid:3d} {'INTERNAL' if len(members[nid]) > 1 else 'leaf    '} "
              f"mean_infl {rows[-1]['mean_infl']:.3e}  top instances from "
              f"{sorted(cnt)}  entropy {ent:.2f}", flush=True)

    # ── alone-generation for internal nodes ─────────────────────────────────
    alone = {}
    for nid in internal:
        host = members[nid][0]
        anc = paths[host][:paths[host].index(nid) + 1]
        fam = np.mean([bank._emb_extra[n].detach().cpu().numpy() for n in members[nid]], 0)
        with forced(bank, host, anc, emb=fam):
            alone[nid] = to_data(field.sample(n=args.n_show, steps=args.steps, seed=args.seed,
                                              branch=host, clip=(-1, 1))).astype(np.float32)
        print(f"[n36] node {nid}: alone-generation done", flush=True)

    ic = [r["profile_centroid"] for r in rows if r["is_internal"] and r["profile_centroid"] == r["profile_centroid"]]
    lc = [r["profile_centroid"] for r in rows if not r["is_internal"] and r["profile_centroid"] == r["profile_centroid"]]
    verdict = "no internal nodes to compare"
    if ic and lc:
        verdict = (f"internal centroid {np.mean(ic):.1f} vs leaf {np.mean(lc):.1f} -> "
                   + ("ancestors act EARLIER (coarser), as predicted" if np.mean(ic) < np.mean(lc)
                      else "ancestors do NOT act earlier -- depth is not organizing abstraction"))
    ents = [r["member_entropy"] for r in rows if r["is_internal"] and r["member_entropy"] == r["member_entropy"]]
    share = ("no internal nodes" if not ents else
             f"mean top-instance member entropy {np.mean(ents):.2f} -> "
             + ("genuinely shared across members" if np.mean(ents) > 0.6 else
                "DOMINATED by one member -- the joint-structure story does not hold here"))
    print(f"\n[n36] sharing: {share}")
    print(f"[n36] layer depth: {verdict}")

    save_json(dict(rows=rows, paths=paths, members={str(k): v for k, v in members.items()},
                   sharing_verdict=share, depth_verdict=verdict, config=vars(args)),
              os.path.join(RESULTS, f"n36_hierarchy_probe{args.tag}.json"), seed=args.seed)

    # ── figure: the tree, each node showing the instances that formed it ────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    palette = ["#1B7F4B", "#B03A2E", "#2C6FA8", "#8A6D1F", "#6A3D9A", "#0F7B7B", "#A0522D", "#444"]
    cof = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    order = sorted(rows, key=lambda r: (not r["is_internal"], r["node"]))
    nrow, ncol = len(order), args.n_show
    fig, axes = plt.subplots(nrow, ncol + 1 + ncol,
                             figsize=(2.05 * ncol + 6.0, 1.4 * nrow + 1.6),
                             gridspec_kw=dict(width_ratios=[1] * ncol + [0.3] + [1] * ncol))
    axes = np.atleast_2d(axes)
    for ri, r in enumerate(order):
        nid = r["node"]
        for c in range(ncol):
            ax = axes[ri][c]
            ax.axis("off")
            if c < len(r["top"]):
                t = r["top"][c]
                ci = names.index(t["concept"])
                ax.imshow(rgb_img(to_data(pools[ci][t["idx"]][None])[0]))
                ax.add_patch(Rectangle((0, 0), 31, 31, fill=False, lw=2.5,
                                       edgecolor=cof[t["concept"]]))
        axes[ri][ncol].axis("off")
        blk = alone.get(nid)
        for c in range(ncol):
            ax = axes[ri][ncol + 1 + c]
            ax.axis("off")
            if blk is not None and c < len(blk):
                ax.imshow(rgb_img(blk[c]))
        tag = f"INTERNAL x{r['n_members']}" if r["is_internal"] else "leaf"
        ent = "" if r["member_entropy"] != r["member_entropy"] else f"\nentropy {r['member_entropy']:.2f}"
        axes[ri][0].set_ylabel(f"node {nid}\n({tag}){ent}\nlayer {r['profile_centroid']:.0f}",
                               rotation=0, ha="right", va="center", fontsize=7.5)
        axes[ri][0].axis("on")
        axes[ri][0].set_xticks([]); axes[ri][0].set_yticks([])
        for sp in axes[ri][0].spines.values():
            sp.set_visible(False)
    handles = [plt.Line2D([], [], color=cof[n], lw=3, label=n) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(names), 8), fontsize=8,
               frameon=False, title="border colour = which concept's pool the instance came from",
               title_fontsize=8)
    fig.suptitle("LEFT: the pool instances that most shape each node's adapter "
                 "(||dL/d(A,B)||, the consolidation loss itself).\n"
                 "RIGHT: what the node generates ALONE -- root..node active, no leaf, its "
                 "descendants' mean conditioning.", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    out = os.path.join(FIG, f"f13_hierarchy_probe{args.tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[n36] wrote {out}")

    fig2, ax = plt.subplots(figsize=(9.5, 4.2))
    seen_lbl = set()
    for r in rows:
        lbl = "internal (shared)" if r["is_internal"] else "leaf (private)"
        ax.plot(range(len(r["layer_profile"])), r["layer_profile"],
                lw=2.2 if r["is_internal"] else 1.0, alpha=0.95 if r["is_internal"] else 0.45,
                color="#1B7F4B" if r["is_internal"] else "#8A8A8A",
                label=None if lbl in seen_lbl else lbl)
        seen_lbl.add(lbl)
    ax.set_xlabel("wrapped layer, in module order (early = coarse resolution)")
    ax.set_ylabel(r"$\|B_nA_n\|_F\ /\ \|W_{base}\|_F$")
    ax.set_title("Where in the network does each node act?\n" + verdict, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig2.tight_layout()
    out2 = os.path.join(FIG, f"f13b_layer_profiles{args.tag}.png")
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"[n36] wrote {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
