"""N34 -- is a shared ancestor actually REUSED, where a standalone LoRA has nothing to reuse?

The bank's stated advantage over the companion paper's one-LoRA-per-query is not just that concepts
are kept -- it is that related concepts share structure: a batch of siblings trains one group node
together, and a LATER family member descends through that frozen node, so its forward pass starts
from the family's joint structure before its own leaf has learned anything.  A standalone adapter
starts from zero every time, structurally as well as literally.  That claim has been priced (N24:
what reuse saves in bytes) but never measured as TRANSFER: does the newcomer actually learn faster
or better because the ancestor is there?

**Design.**  A family = held-out slots sharing a colour pair (colour dominates pixel distance, so
these are the concepts the tree genuinely groups; same-digit families do not cluster in data space).

  phase 1   batch-add the first K members -> one shared group node, trained by all K, then sealed
  phase 2   the (K+1)-th family member arrives.  ONE pool is sampled for it, once (compose reads
            only the frozen base, so the pool is identical across arms), and the SAME pool is
            consolidated three ways at each epoch budget:

  `reuse`        inserted into the family bank by its natural placement; if that path runs through
                 the frozen group node, the ancestor's delta is active in the forward pass while
                 ONLY the new leaf trains.  Whether it actually landed there is RECORDED, not
                 assumed -- if the tree does not group the family, the premise fails visibly.
  `wrong-family` THE CONTROL THAT MAKES THIS A TEST.  Same newcomer, same pool, same budget, but
                 forced through the ancestor of a DIFFERENT family.  If a mismatched ancestor helps
                 as much as the matched one, then "reuse" is buying nothing but extra capacity in the
                 forward pass and the shared-semantics story is false.  Any claim about hierarchy has
                 to clear this arm, not just `standalone`.
  `no-reuse`     identical bank, identical pool, but placement forced flat for this one insertion --
                 the single difference is that the ancestor's delta is absent from the forward pass.
                 Isolates the ancestor's contribution from everything else about being in a bank.
  `standalone`   a fresh bank on a fresh field: the prior solution, one adapter from scratch.

The three comparisons answer three different questions, and only the first two are about hierarchy:

    reuse vs wrong-family   does the SHARED SEMANTICS help, or is any ancestor as good?
    reuse vs no-reuse       does having an ancestor at all help?
    reuse vs standalone     does the bank beat the prior solution end to end?

Between trials the newcomer is deleted with `remove_concept`, which N31 certified restores every
survivor bit-identically -- so one phase-1 build serves the whole grid, and the deletion machinery
gets exercised under load as a side effect.  The family's fingerprints are asserted unchanged at the
very end, across all insert/train/delete cycles.

Swept over epoch budgets, so reuse shows up where it should: as the LEFT end of a learning curve.
At large budgets the leaf can learn everything itself and the arms should converge; if they do not,
that is worth knowing too.

    python experiments/n34_reuse.py --k-family 3
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, RESULTS, ood_queries                                # noqa: E402

from lifelong_ood.datasets import load_colormnist                             # noqa: E402
from lifelong_ood.evals import (composition_accuracy, fit_probes,             # noqa: E402
                                precision_recall, save_json, vendi_score)
from lifelong_ood.neural import (NeuralLifelongLearner, load_field,           # noqa: E402
                                 to_data, to_model)
from lifelong_ood.neural.adapters import fingerprint                          # noqa: E402


def pick_family(data, seed, k_needed):
    """Held-out slots sharing (fg, bg), largest family first; falls back to shared bg alone."""
    qs = ood_queries(data, seed=seed)
    by_pair, by_bg = {}, {}
    for q in qs:
        by_pair.setdefault((q["f"], q["b"]), []).append(q)
        by_bg.setdefault(q["b"], []).append(q)
    fam = max(by_pair.values(), key=len)
    how = f"fg={fam[0]['f']},bg={fam[0]['b']}"
    if len(fam) < k_needed:
        fam = max(by_bg.values(), key=len)
        how = f"bg={fam[0]['b']} (no colour pair had {k_needed} held-out slots)"
    return fam[:k_needed], how


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(CACHE, "ddpm_colormnist.pt"))
    p.add_argument("--k-family", type=int, default=3)
    p.add_argument("--epoch-grid", type=int, nargs="+", default=[3, 6, 12, 25, 40])
    p.add_argument("--n-pool", type=int, default=48)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-per-slot", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--rule", choices=["ll", "cu"], default="ll",
                   help="placement criterion for the bank's tree (§24j)")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.tag = args.tag or "_smoke"
        args.k_family, args.n_pool, args.n_per_slot = 2, 16, 20
        args.epoch_grid, args.steps = [2, 5], 20
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=8 if args.quick else 30)
    fam, how = pick_family(data, args.seed, args.k_family + 1)
    if len(fam) < args.k_family + 1:
        print(f"[n34] only {len(fam)} family members available; need {args.k_family + 1}")
        return 1
    first, newcomer = fam[:args.k_family], fam[args.k_family]
    # a SECOND family, as unlike the first as the held-out set allows -- the source of the
    # mismatched ancestor the control routes through
    others = [q for q in ood_queries(data, seed=args.seed)
              if (q["f"], q["b"]) != (fam[0]["f"], fam[0]["b"])]
    other_fam = {}
    for q in others:
        other_fam.setdefault((q["f"], q["b"]), []).append(q)
    second = max(other_fam.values(), key=len)[:args.k_family] if other_fam else []
    print(f"[n34] family {how}: slots {[q['slot'] for q in first]} first, "
          f"slot {newcomer['slot']} (digit {newcomer['d']}) arrives later")
    print(f"[n34] control family fg={second[0]['f']},bg={second[0]['b']}: "
          f"slots {[q['slot'] for q in second]}\n" if second else
          "[n34] WARNING: no second family available -- the wrong-family control cannot run\n",
          flush=True)
    R = data["X"][newcomer["rest"][:100]].astype(np.float32)
    nm = f"c{newcomer['slot']}"

    # ── phase 1: the family, as one sealed batch ─────────────────────────────
    field = load_field(args.ckpt, device=dev)
    L = NeuralLifelongLearner(field, steps=args.steps, seed=args.seed, device=dev)
    L.bank.tree.rule = args.rule
    names = [f"c{q['slot']}" for q in first]
    L.learn_concepts_batch([to_model(data["X"][q["query"]]).ravel() for q in first], names,
                           n_pool=args.n_pool, epochs=args.epoch_grid[-1])
    fam_nodes = {nid for n in names for nid in L.bank.concepts[n]["path"]} - {0}
    # the control family goes in as its OWN sealed batch, so it owns a separate group node
    names2 = [f"x{q['slot']}" for q in second]
    if second:
        L.learn_concepts_batch([to_model(data["X"][q["query"]]).ravel() for q in second], names2,
                               n_pool=args.n_pool, epochs=args.epoch_grid[-1])
    wrong_nodes = ({nid for n in names2 for nid in L.bank.concepts[n]["path"]} - {0} - fam_nodes
                   if second else set())
    fam_fp = {n: fingerprint(field, n, seed=args.seed) for n in names + names2}
    print(f"[n34] family consolidated: shared/family nodes {sorted(fam_nodes)}, "
          f"{L.bank.n_params() / 1e6:.2f}M adapter params\n", flush=True)

    # the standalone arm's field, built once and reset between trials the same way
    field_s = load_field(args.ckpt, device=dev)
    L_s = NeuralLifelongLearner(field_s, steps=args.steps, seed=args.seed, device=dev)
    L_s.bank.tree.rule = args.rule

    # ── phase 2: ONE pool for the newcomer, shared by every arm ──────────────
    xq = to_model(data["X"][newcomer["query"]]).ravel()
    teacher = L.compose(xq)                       # reads the FROZEN base only: arm-independent
    pool = L.sample_teacher(teacher, n=args.n_pool, seed=args.seed)

    def snap_tree(t):
        """Placement state, deep-copied.

        `remove_concept` reclaims parameters but deliberately does NOT revert `_touch`'s running
        means -- an incremental mean is not exactly reversible in floating point, and §22 says so.
        That is harmless for the deletion certificate and fatal here: without restoring it, each
        insert/delete cycle drifts the ancestors' means and LATER TRIALS GET DIFFERENT PLACEMENTS,
        which is exactly what made the first run of this experiment report identical numbers for
        every arm at e>=6.
        """
        return dict(mean={k: (None if v is None else v.copy()) for k, v in t.mean.items()},
                    count=dict(t.count), children={k: list(v) for k, v in t.children.items()},
                    _next=t._next)

    def restore_tree(t, s_):
        t.mean = {k: (None if v is None else v.copy()) for k, v in s_["mean"].items()}
        t.count = dict(s_["count"])
        t.children = {k: list(v) for k, v in s_["children"].items()}
        t._next = s_["_next"]

    def trial(learner, epochs, flat, force_parent=None):
        """Consolidate the newcomer once.  `force_parent` grafts its leaf under a chosen node.

        The graft is done by REGISTERING first and rewriting the path before any optimizer is built,
        so the forced ancestor is genuinely in the forward pass during training rather than only at
        sampling time -- otherwise the control would test nothing.
        """
        was = learner.bank.tree.flat
        learner.bank.tree.flat = flat
        try:
            if force_parent is None:
                learner.consolidate(pool, nm, mu=teacher["mu"], epochs=epochs)
            else:
                learner.bank.add_concept(nm, np.ravel(teacher["mu"]))
                rec = learner.bank.concepts[nm]
                leaf = rec["leaf"]
                anc = learner.bank.tree.path_of(force_parent)
                for nid in anc:                      # ensure every forced ancestor is allocated
                    for w in learner.bank.wrapped:
                        w.add_node(nid, learner.bank.device)
                rec["path"] = list(anc) + [leaf]
                learner.consolidate(pool, nm, mu=teacher["mu"], epochs=epochs,
                                    already_registered=True)
        finally:
            learner.bank.tree.flat = was
        path = list(learner.bank.concepts[nm]["path"])
        S = to_data(learner.sample_concept(nm, n=args.n_samples, steps=args.steps,
                                           seed=args.seed)).astype(np.float32)
        marginal = sum(w.A[str(n)].numel() + w.B[str(n)].numel()
                       for n in path for w in learner.bank.wrapped if str(n) in w.A
                       if n not in fam_nodes and n != 0)
        acct = learner.bank.remove_concept(nm)     # exact for PARAMETERS -- certified by N31
        return S, path, marginal, acct

    rows = []
    for epochs in args.epoch_grid:
        arms = [("reuse", L, False, None), ("no-reuse", L, True, None),
                ("standalone", L_s, False, None)]
        if wrong_nodes:
            arms.insert(1, ("wrong-family", L, False, max(wrong_nodes)))
        for arm, learner, flat, forced in arms:
            snap = snap_tree(learner.bank.tree)
            S, path, marginal, _ = trial(learner, epochs, flat, force_parent=forced)
            restore_tree(learner.bank.tree, snap)      # placement state back to phase 1 exactly
            # node ids are per-bank and both banks number from 1, so `standalone`'s path must NEVER
            # be tested against the family bank's ids -- that comparison reported through_family=True
            # for a fresh bank that contains no family at all.
            own_fam = fam_nodes if learner is L else set()
            own_wrong = wrong_nodes if learner is L else set()
            a = composition_accuracy(probes, S, newcomer["d"], newcomer["f"], newcomer["b"])
            pr = precision_recall(S, R)
            through = bool(set(path) & own_fam)
            through_wrong = bool(set(path) & own_wrong)
            rows.append(dict(arm=arm, epochs=int(epochs), joint=a["joint"], digit=a["digit"],
                             fg=a["fg"], bg=a["bg"], recall=pr["recall"], f1=pr["f1"],
                             vendi=vendi_score(S), through_family=through,
                             through_wrong_family=through_wrong,
                             marginal_params=int(marginal)))
            print(f"[n34] e={epochs:3d} {arm:11s} joint {a['joint']:.3f} f1 {pr['f1']:.3f} "
                  f"through_family={through} (+{marginal / 1e3:.0f}K params)", flush=True)

    # the reuse arm must actually have run through the family; otherwise the premise failed
    ru = [r for r in rows if r["arm"] == "reuse"]
    if not all(r["through_family"] for r in ru):
        print("\n[n34] WARNING: natural placement did NOT route the newcomer through the family "
              "node in every trial -- the transfer comparison is not measuring what it claims")

    # after every insert/train/delete cycle, the family must be bit-identical
    drift = max(float(np.max(np.abs(fingerprint(field, n, seed=args.seed) - fam_fp[n])))
                for n in names)
    print(f"\n[n34] family drift across {len(rows)} insert/train/delete cycles: {drift:.3e} "
          f"({'bit-identical' if drift == 0.0 else 'NOT bit-identical -- reuse is not read-only'})")

    save_json(dict(rows=rows, family=how, family_slots=[q["slot"] for q in first],
                   newcomer_slot=int(newcomer["slot"]), family_drift=drift, config=vars(args)),
              os.path.join(RESULTS, f"n34_reuse{args.tag}.json"), seed=args.seed)

    arm_names = ["reuse"] + (["wrong-family"] if wrong_nodes else []) + ["no-reuse", "standalone"]
    print(f"\n[n34] the newcomer's learning curve (joint / F1)")
    print(f"    {'epochs':>8s}" + "".join(f"{a:>16s}" for a in arm_names))
    for e in args.epoch_grid:
        cells = []
        for arm in arm_names:
            r = next((x for x in rows if x["arm"] == arm and x["epochs"] == e), None)
            cells.append(f"{r['joint']:.2f} / {r['f1']:.2f}" if r else "--")
        print(f"    {e:8d}" + "".join(f"{c:>16s}" for c in cells))

    # the comparison that decides whether HIERARCHY (not capacity) is doing the work
    if wrong_nodes:
        def mean_f1(a):
            v = [r["f1"] for r in rows if r["arm"] == a]
            return float(np.mean(v)) if v else float("nan")
        m_r, m_w, m_n = mean_f1("reuse"), mean_f1("wrong-family"), mean_f1("no-reuse")
        print(f"\n[n34] mean F1 over the budget grid: reuse {m_r:.3f}  wrong-family {m_w:.3f}  "
              f"no-reuse {m_n:.3f}")
        print(f"[n34] shared SEMANTICS worth {m_r - m_w:+.3f} F1 over a mismatched ancestor; "
              f"having ANY ancestor worth {m_w - m_n:+.3f} over none")
        if m_r <= m_w:
            print("[n34] => a mismatched ancestor helps as much as the matched one: on this "
                  "evidence the gain is CAPACITY, not shared structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
