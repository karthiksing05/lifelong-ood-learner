"""LifelongLearner: lifelong learning as repeated composition of prototypes.

The reframing this repository exists to test:

    lifelong learning  =  repeat {
        1. meet one OOD query x_q
        2. COMPOSE known prototypes into an OOD prototype        (compose.poe_teacher)
        3. SAMPLE the OOD prototype through the manifold          (ScoreField.sample + CFG teacher)
           -- guidance keeps the pool on the learned manifold, so the recreated instances are
              in-style renderings of the novel composition, not copies of the query
        4. CONSOLIDATE the pool into the manifold as a frozen-base additive adapter
    }

Step 4 is the poe-diffusion paper's LoRA distillation made structural.  LoRA freezes W and trains a
low-rank delta on samples from the PoE teacher; we freeze the base tree and grow a SEPARATE adapter
tree on samples from the PoE teacher, then merge snapshots (backend.merge_snapshots).  Because the
score field is a mixture, the adapter is EXACTLY additive: every base statistic is bit-identical
after any number of consolidations, and the only global effect is the mixture-weight
renormalization, which bounds old-data log-likelihood drift by log(1 + n_adapter/n_base).  The
manifold therefore cannot degrade over time by construction -- E3 measures the curves anyway, with
classic absorb-into-the-base ifit as the destructive control.

Consolidated concepts become ordinary prototypes: later compositions can select them, so composition
is closed under consolidation, which is what makes the loop lifelong rather than one-shot.
"""
import numpy as np

from .backend import make_tree, merge_snapshots
from .compose import (candidates_ascent, candidates_tree, cfg_weight, poe_teacher,
                      query_only_teacher, teacher_eps, topk_teacher)
from .score import ScoreField


class LifelongLearner:
    def __init__(self, tree=None, prior_var=0.05, min_count=3, mode="leaves", backend="auto",
                 K=3, tau=0.05, w0=1.2, wmax=2.0, wmin=0.8, schedule="anneal",
                 u_start=0.4, steps=50, seed=0):
        # K, tau, schedule, wmin/wmax and u_start are all set by experiments/e0_tune.py, not guessed:
        # tau=0.05 adaptive takes teacher accuracy to the probe ceiling (0.20 -> 0.75 joint), and the
        # annealed schedule keeps 71% of that through sampling where the paper's variance schedule
        # keeps 21%.  See PLAN.md 3.1 for the measurements.
        self.tree = tree if tree is not None else make_tree(prior_var, backend=backend)
        self.adapters = []              # [(name, tree)] -- separate trees, base stays frozen
        self.cfg = dict(min_count=min_count, mode=mode, K=K, tau=tau, w0=w0, wmax=wmax,
                        wmin=wmin, schedule=schedule, u_start=u_start, steps=steps, seed=seed)
        self._field = None
        self.history = []

    # ── the manifold: frozen base + additive adapters, one merged field ──────
    @property
    def field(self):
        if self._field is None:
            c = self.cfg
            base = self.tree.snapshot(c["min_count"])
            ads = [(n, t.snapshot(min(c["min_count"], 2))) for n, t in self.adapters]
            self._field = ScoreField(merge_snapshots(base, ads), mode=c["mode"])
        return self._field

    def invalidate(self):
        self._field = None

    def fit(self, X, order=None, seed=None, verbose=False):
        self.tree.fit(X, order=order, seed=self.cfg["seed"] if seed is None else seed, verbose=verbose)
        self.invalidate()
        return self

    # ── steps 1-2: compose an OOD prototype from one query ───────────────────
    def compose(self, xq, discovery="tree", K=None, tau=None, **disc_kw):
        c = self.cfg
        if discovery == "tree":
            M, V, meta = candidates_tree(self.field.snap, xq, **disc_kw)
        elif discovery == "ascent":
            M, V, meta = candidates_ascent(self.field, xq, seed=c["seed"], **disc_kw)
        else:
            raise ValueError(discovery)
        t = poe_teacher(xq, M, V, K=K or c["K"], tau=tau or c["tau"])
        t["meta"] = meta
        return t

    def baseline_teacher(self, xq, kind):
        if kind == "query_only":
            return query_only_teacher(xq)
        if kind.startswith("top"):
            return topk_teacher(self.field.snap, xq, k=int(kind[3:]))
        raise ValueError(kind)

    # ── step 3: recreate OOD instances by sampling the composed prototype ────
    def sample_teacher(self, teacher, n=64, steps=None, seed=None, w0=None, wmax=None,
                       wmin=None, schedule=None, u_start=None):
        """Guided reverse diffusion from a NOISED TEACHER DRAW rather than pure noise (SDEdit-style).

        The teacher already knows the composition (mu_T, Sigma_T); what it lacks is manifold detail.
        So each chain starts at x_{u_start} = forward-noised draw from q_T and the reverse process,
        with the teacher as the CFG conditional branch (their Eq. 12-14), carries it back onto the
        learned manifold.  u_start=1.0 reproduces the paper's pure-noise protocol and also works,
        but discards the analytic teacher's exact x0-space density and needs stronger guidance."""
        c = self.cfg
        seed = c["seed"] if seed is None else seed
        rng = np.random.default_rng(seed)
        x0 = teacher["mu"] + np.sqrt(teacher["var"]) * rng.standard_normal((n, len(teacher["mu"])))
        gw = cfg_weight(teacher, w0 if w0 is not None else c["w0"],
                        wmax if wmax is not None else c["wmax"],
                        wmin if wmin is not None else c["wmin"],
                        schedule or c["schedule"])
        return self.field.sample(n=n, steps=steps or c["steps"], seed=seed,
                                 u_start=c["u_start"] if u_start is None else u_start,
                                 x_init=np.clip(x0, 0, 1),
                                 guide_eps=teacher_eps(teacher), guide_w=gw)

    # ── step 4: consolidate the recreated instances into the manifold ────────
    def consolidate(self, pool, name, strategy="graft"):
        """strategy='graft': grow a separate adapter tree on the pool; the base stays frozen and the
        merged field gains only new components (the LoRA move).
        strategy='ifit': classic Cobweb absorption into the base -- every statistic on the descent
        path mutates.  This is the destructive control E3 compares against."""
        if strategy == "graft":
            same = "cobweb" if getattr(self.tree, "backend", "numpy") == "cobweb-private" else "numpy"
            ad = make_tree(getattr(self.tree, "prior_var", 0.05), backend=same)
            ad.fit(np.asarray(pool), seed=self.cfg["seed"])
            self.adapters.append((name, ad))
        elif strategy == "ifit":
            self.tree.fit(np.asarray(pool), seed=self.cfg["seed"])
        else:
            raise ValueError(strategy)
        self.history.append(name)
        self.invalidate()
        return self

    def consolidate_batch(self, items, strategy="graft"):
        """Add K concepts at once.  `items` = [(name, pool), ...].

        On the analytic backbone a batch is exactly K grafts -- each adapter is its own tree and the
        mixture is order-independent, so there is nothing to gain from joint optimization and nothing
        to lose either.  The field is rebuilt ONCE at the end instead of K times, which is the whole
        saving here; the neural backbone is where batching changes the result rather than the cost.
        """
        for name, pool in items:
            self.consolidate(pool, name, strategy=strategy)
        return self

    def learn_concept(self, xq, name, n_pool=64, discovery="tree", strategy="graft", **kw):
        """The whole loop for one OOD query: compose -> sample -> consolidate."""
        teacher = self.compose(xq, discovery=discovery)
        pool = self.sample_teacher(teacher, n=n_pool, **kw)
        self.consolidate(pool, name, strategy=strategy)
        return teacher, pool

    def learn_concepts_batch(self, xqs, names, n_pool=64, discovery="tree", strategy="graft", **kw):
        """Compose, sample and consolidate K OOD queries, rebuilding the field once at the end."""
        out = []
        for xq, name in zip(xqs, names):
            teacher = self.compose(xq, discovery=discovery)
            out.append((teacher, self.sample_teacher(teacher, n=n_pool, **kw)))
        self.consolidate_batch([(n, p) for n, (_, p) in zip(names, out)], strategy=strategy)
        return [t for t, _ in out], [p for _, p in out]

    # ── using what was learned ───────────────────────────────────────────────
    def sample_concept(self, name, n=16, steps=None, seed=None, guidance=2.0):
        """Conditional generation from a consolidated concept: the adapter's own components form the
        conditional branch, the full mixture the unconditional one -- CFG without any c_new training."""
        c = self.cfg
        f = self.field
        cond = lambda x, u: f.eps(x, u, branch=name)
        return f.sample(n=n, steps=steps or c["steps"], seed=c["seed"] if seed is None else seed,
                        guide_eps=cond, guide_w=lambda u: guidance)

    def loglik(self, X, branch=None):
        return self.field.logp(X, u=0.0, branch=branch)
