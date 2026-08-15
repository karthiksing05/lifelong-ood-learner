"""N69 -- the N67 read comparison on CelebA, where the reads are NOT saturated.

N67 answered its question on Color-MNIST and the answer was "this dataset cannot tell":

    read accuracy over 12 held-out slots -- inversion 12/12, probe-query 12/12, mu_T 11/12

and because `compose(fac, values)` is deterministic given the read, the arms that read identically
ARE identical -- `probe-query` came out bit-for-bit equal to `ours-inversion-hard`.  Nothing about
discovery could show up there.  What DID move was hard-vs-soft composition (joint 0.493 -> 0.642).

CelebA is the setting where the question is live:

  * the read is not saturated -- mu_T's attribute read is **0.750**, not 1.000 (N68);
  * the attributes OVERLAP spatially, so a probe cannot separate them the way it separates a digit
    from a background;
  * and the prototype pool COLLAPSES under the paper's own CelebA column (selected spread 0.34x of
    real-image spread, with **zero** query-agnostic candidates), which `CELEBA_DISCOVERY` repairs
    to 0.82x by switching `init` to hybrid.

So the same four reads are scored here, plus the collapse repair as its own arm, so that the effect
of FIXING discovery is separated from the effect of CHOOSING a read:

  poe-guided-paper     their discovery (their CelebA column) + their generation
  poe-guided-fixed     their discovery with hybrid init      + their generation
  ours-muT-paper       read mu_T from the COLLAPSED pool     + our generation
  ours-muT-fixed       read mu_T from the REPAIRED pool      + our generation
  probe-query          probe the query directly              + our generation
  unconditional        the floor

The comparison that matters for the thesis is `ours-muT-fixed` against `probe-query`: if reading a
repaired aggregate beats reading the query, the paper's discovery earns its place on the dataset
where reading is hard, and the "best of both" is real rather than an artefact of an easy benchmark.
If it does not, the honest conclusion is that our contribution is the GENERATION half and the
composition, and the discovery half is theirs to keep.

    python experiments/n69_celeba_reads.py --n-cases 4 --n-samples 16
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.compose import (CELEBA_DISCOVERY, PAPER_CELEBA,      # noqa: E402
                                  PAPER_POE_CELEBA, candidates_ascent, poe_teacher)
from lifelong_ood.data.celeba import ATTRS, load_celeba                # noqa: E402
from lifelong_ood.evals import (fit_attr_probes, precision_recall,     # noqa: E402
                                save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,    # noqa: E402
                                 sample_embedding, to_data, to_model)

ARMS = ("poe-guided-paper", "poe-guided-fixed", "ours-muT-paper", "ours-muT-fixed",
        "probe-query", "unconditional")


def read_attrs(probes, img_row, names):
    """-> a 0/1 vector over `names`, read off ONE image by the per-attribute probes."""
    x = np.asarray(img_row, np.float32).reshape(1, -1)
    return np.array([int(probes[n].predict(x)[0]) if n in probes else 0 for n in names], int)


def attr_score(probes, S, tgt, names):
    """Fraction of attributes correct, averaged over samples -- CelebA's analogue of `joint`."""
    S = np.asarray(S, np.float32)
    hit = []
    for i, n in enumerate(names):
        if n not in probes:
            continue
        hit.append((probes[n].predict(S) == int(tgt[i])).mean())
    return float(np.mean(hit)) if hit else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_celeba128.pt"))
    p.add_argument("--n-cases", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_cases, args.n_samples, args.steps = 1, 6, 40
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    field = load_field(args.ckpt, device=dev)
    d = load_celeba(img=int(ck.get("img", 128)), min_count=int(ck.get("min_count", 120)),
                    seed=int(ck.get("seed", 0)))
    cls = np.asarray(d["cls"] if "cls" in d else d["slot"])
    X = np.asarray(d["X"], np.float32)
    A = np.asarray(d["attrs"])
    probes = fit_attr_probes(d, seed=args.seed)
    names = [n for n in ATTRS if n in probes]
    print(f"[n69] {len(names)} readable attributes: {names}", flush=True)
    hold = [int(h) for h in np.asarray(ck.get("hold_ids", [])).ravel()]
    cases = [h for h in hold if (cls == h).any()][:args.n_cases]
    print(f"[n69] {len(cases)} held-out classes: {cases}", flush=True)

    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    disc_paper = dict(PAPER_CELEBA)
    disc_fixed = dict(CELEBA_DISCOVERY)
    if args.quick:
        for dd in (disc_paper, disc_fixed):
            dd.update(u_grid=(0.05, 0.2), n_starts=4, iters=20)

    rows, reads = [], []
    for ci, h in enumerate(cases):
        idx = np.where(cls == h)[0]
        xq = to_model(X[int(idx[0])]).ravel()
        R = X[idx[1:1 + 60]]
        tgt = A[int(idx[0])][:len(ATTRS)]
        got = {}
        got["unconditional"] = to_data(field.sample(n=args.n_samples, steps=args.steps,
                                                    seed=args.seed + 31 * (ci + 1), clip=(-1, 1),
                                                    branch=None))
        rd = {}
        for tag, disc in (("paper", disc_paper), ("fixed", disc_fixed)):
            M, V, meta = candidates_ascent(field, xq, seed=args.seed, **disc)
            t = poe_teacher(xq, M, V, K=L.cfg["K"], tau=L.cfg["tau"])
            got[f"poe-guided-{tag}"] = to_data(L.sample_teacher(
                t, n=args.n_samples, seed=args.seed + ci, **PAPER_POE_CELEBA))
            r = read_attrs(probes, to_data(t["mu"]), names)
            rd[tag] = r
            # our generation from the read: the factored CelebA backbone conditions on an ATTRIBUTE
            # vector directly, so the read IS the conditioning -- no separate factor basis needed.
            emb = field.model.attr_embedding(torch.as_tensor(
                np.array([[int(v) for v in r]]), dtype=torch.long))
            got[f"ours-muT-{tag}"] = to_data(sample_embedding(
                field, emb.detach().cpu().numpy().ravel(), n=args.n_samples, steps=args.steps,
                seed=args.seed + ci))
            print(f"[n69]   case {ci + 1} {tag}: pool {len(M)}, "
                  f"query-agnostic {int(np.sum(np.asarray(meta['src']) == 'random'))}, "
                  f"attr-read {float(np.mean(r == tgt[:len(names)])):.3f}", flush=True)

        rq = read_attrs(probes, to_data(xq), names)
        embq = field.model.attr_embedding(torch.as_tensor(
            np.array([[int(v) for v in rq]]), dtype=torch.long))
        got["probe-query"] = to_data(sample_embedding(
            field, embq.detach().cpu().numpy().ravel(), n=args.n_samples, steps=args.steps,
            seed=args.seed + ci))

        tt = tgt[:len(names)]
        reads.append(dict(cls=int(h), muT_paper=float(np.mean(rd["paper"] == tt)),
                          muT_fixed=float(np.mean(rd["fixed"] == tt)),
                          probe_query=float(np.mean(rq == tt))))
        print(f"[n69] case {ci + 1}/{len(cases)} class {h} | read acc: "
              f"muT-paper {reads[-1]['muT_paper']:.3f} | muT-fixed {reads[-1]['muT_fixed']:.3f} | "
              f"probe-query {reads[-1]['probe_query']:.3f}", flush=True)

        for arm in ARMS:
            S = np.asarray(got[arm], np.float32)
            pr = precision_recall(S, R)
            rows.append(dict(arm=arm, cls=int(h), attr=attr_score(probes, S, tgt, names),
                             precision=pr["precision"], recall=pr["recall"],
                             vendi=float(vendi_score(S))))

    print(f"\n[n69] {'=' * 74}")
    print(f"{'arm':20s}{'attr':>9s}{'recall':>9s}{'precision':>11s}{'F1':>8s}{'vendi':>8s}")
    summ = {}
    for arm in ARMS:
        g = [r for r in rows if r["arm"] == arm]
        if not g:
            continue
        s_ = {k: float(np.mean([r[k] for r in g])) for k in
              ("attr", "precision", "recall", "vendi")}
        s_["f1"] = float(2 * s_["precision"] * s_["recall"] / max(s_["precision"] + s_["recall"],
                                                                  1e-9))
        summ[arm] = s_
        print(f"{arm:20s}{s_['attr']:>9.3f}{s_['recall']:>9.3f}{s_['precision']:>11.3f}"
              f"{s_['f1']:>8.3f}{s_['vendi']:>8.2f}")

    mp = float(np.mean([r["muT_paper"] for r in reads]))
    mf = float(np.mean([r["muT_fixed"] for r in reads]))
    pq = float(np.mean([r["probe_query"] for r in reads]))
    print(f"\n[n69] READ accuracy -- mu_T(paper) {mp:.3f} | mu_T(fixed) {mf:.3f} | "
          f"probe-query {pq:.3f}")
    print("[n69] VERDICT")
    if mf > pq + 0.02:
        print("[n69]   READING A REPAIRED AGGREGATE BEATS READING THE QUERY. Their discovery earns")
        print("[n69]   its place where reading is hard, and 'best of both' is a real combination.")
    elif mf > mp + 0.02:
        print("[n69]   Repairing the pool improves the read but does not beat probing the query;")
        print("[n69]   discovery is worth fixing, not worth preferring, on this backbone.")
    else:
        print("[n69]   The read does not improve. Our contribution is the GENERATION half and the")
        print("[n69]   composition; the discovery half is theirs and we should say so plainly.")
    save_json(dict(rows=rows, summary=summ, reads=reads, config=vars(args)),
              os.path.join(RESULTS, "n69_celeba_reads.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
