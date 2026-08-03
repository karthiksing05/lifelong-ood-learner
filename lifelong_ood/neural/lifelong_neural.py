"""NeuralLifelongLearner — the same four-step loop, on a trained DDPM.

    meet an OOD query -> COMPOSE prototypes into a PoE teacher -> SAMPLE it through the manifold
    -> CONSOLIDATE the pool into a hierarchical adapter

Steps 1-3 are literally `lifelong_ood.compose`, unchanged: the neural field satisfies the same
`BaseScoreField` interface, so mode-ascent discovery, submodular selection, the per-dimension PoE and
the guided sampler are the identical code paths the analytic track uses.  What differs is step 4.

Consolidation here is the paper's LoRA distillation (their Eq. 15) with one change that turns a
one-shot trick into a lifelong mechanism: the adapter is not a fresh flat `(theta_LoRA, c_new)` pair
per query but a new LEAF in a growing hierarchy whose ancestors are frozen (see adapters.py).  So
consolidations accumulate, share, and provably cannot disturb each other.
"""
import numpy as np
import torch

from ..compose import (candidates_ascent, cfg_weight, poe_teacher, query_only_teacher,
                       teacher_eps)
from ..score import cosine_abar
from .adapters import HierarchicalAdapterBank
from .ddpm import CH, IMG


class NeuralLifelongLearner:
    def __init__(self, field, rank=8, K=3, tau=0.05, wmin=0.8, wmax=2.0, schedule="anneal",
                 u_start=1.0, teacher_var_scale=30.0, steps=50, seed=0, device="cuda", flat=False,
                 freeze_ancestors=True, var=0.1, branch_max=4, max_depth=3):
        # teacher_var_scale corrects a systematic bias, measured in experiments/n0_tune.py.  The PoE
        # covariance is built from LAPLACE estimates at modes, which measure local curvature rather
        # than how much a class actually varies -- on ColorMNIST the teacher's sigma comes out 0.0117
        # against a real intra-class spread of 0.1595, a 13.6x gap (186x in variance).  Sampling such
        # a sharp teacher collapses the pool: every chain lands on mu_T, and distilling 64 copies of
        # one image teaches an adapter one image.  Inflating the covariance recovers spread along a
        # measured tradeoff (scale 1 -> diversity 0.13x reference at joint 0.94; scale 30 -> 0.61x at
        # 0.71; scale 100 -> 0.78x at 0.70).  30 is the default because pools feed distillation,
        # where covering the class matters more than every sample being individually correct.
        self.field = field
        self.device = device
        self.bank = HierarchicalAdapterBank(field.model, rank=rank, device=device, flat=flat,
                                            freeze_ancestors=freeze_ancestors, var=var,
                                            branch_max=branch_max, max_depth=max_depth)
        field.bank = self.bank
        self.cfg = dict(K=K, tau=tau, wmin=wmin, wmax=wmax, schedule=schedule,
                        u_start=u_start, teacher_var_scale=teacher_var_scale,
                        steps=steps, seed=seed)
        self.history = []

    # ── steps 1-2 ────────────────────────────────────────────────────────────
    def compose(self, xq, K=None, tau=None, **disc):
        """Discovery is mode ascent on the UNet's own score field -- the paper's route, since a
        neural backbone has no concept table to rank."""
        c = self.cfg
        M, V, meta = candidates_ascent(self.field, xq, seed=c["seed"], **disc)
        t = poe_teacher(xq, M, V, K=K or c["K"], tau=tau or c["tau"])
        t["meta"] = meta
        return t

    def baseline_teacher(self, xq, kind="query_only"):
        if kind == "query_only":
            return query_only_teacher(xq, var=0.01)
        raise ValueError(kind)

    # ── step 3 ───────────────────────────────────────────────────────────────
    def sample_teacher(self, teacher, n=64, steps=None, seed=None, u_start=None,
                       var_scale=None, **kw):
        c = self.cfg
        seed = c["seed"] if seed is None else seed
        rng = np.random.default_rng(seed)
        vs = c["teacher_var_scale"] if var_scale is None else var_scale
        if teacher.get("fixed_var"):        # baselines whose variance is defined, not estimated
            vs = 1.0
        teacher = dict(teacher, var=teacher["var"] * float(vs))
        x0 = teacher["mu"] + np.sqrt(teacher["var"]) * rng.standard_normal((n, len(teacher["mu"])))
        gw = cfg_weight(teacher, wmin=kw.get("wmin", c["wmin"]), wmax=kw.get("wmax", c["wmax"]),
                        schedule=kw.get("schedule", c["schedule"]))
        return self.field.sample(n=n, steps=steps or c["steps"], seed=seed,
                                 u_start=c["u_start"] if u_start is None else u_start,
                                 x_init=np.clip(x0, -1, 1), clip=(-1, 1),
                                 guide_eps=teacher_eps(teacher), guide_w=gw)

    # ── step 4: distil the pool into a new hierarchical adapter ──────────────
    def consolidate(self, pool, name, mu=None, epochs=10, batch=16, lr=1e-3, p_drop=0.1,
                    seed=None, verbose=False):
        """The paper's Eq. 15 loss, optimizing ONLY this concept's leaf delta and class embedding.

        `mu` (the teacher mean) is what places the concept in the hierarchy; pass it so related
        concepts land near each other and can share ancestors.
        """
        seed = self.cfg["seed"] if seed is None else seed
        torch.manual_seed(seed)
        self.bank.add_concept(name, np.ravel(pool.mean(0) if mu is None else mu))
        params = self.bank.trainable(name)
        opt = torch.optim.Adam(params, lr=lr)
        X = torch.from_numpy(np.asarray(pool, np.float32)).to(self.device)
        X = X.reshape(-1, CH, IMG, IMG)
        model = self.field.model
        model.train()
        n = len(X)
        g = torch.Generator(device="cpu").manual_seed(seed)
        curve = []
        with self.bank.activate(name):
            for ep in range(epochs):
                tot, nb = 0.0, 0
                perm = torch.randperm(n, generator=g)
                for i0 in range(0, n, batch):
                    x0 = X[perm[i0:i0 + batch]]
                    u = torch.rand(len(x0), device=self.device).clamp_(1e-3, 1.0)
                    a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(self.device)
                    a = a.float().view(-1, 1, 1, 1)
                    e = torch.randn_like(x0)
                    xt = a.sqrt() * x0 + (1 - a).sqrt() * e
                    # No CFG dropout is needed here, unlike the paper's Eq. 15.  They train one
                    # adapter into the shared weights, so the unconditional branch has to be
                    # protected by dropping c_new sometimes.  Here the unconditional branch IS the
                    # frozen base with an empty adapter path -- structurally untouched by this
                    # optimization -- so it stays exactly usable with no dropout at all.
                    loss = torch.nn.functional.mse_loss(
                        model(xt, u, emb_override=self.bank.embedding(name)), e)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    tot += float(loss); nb += 1
                curve.append(tot / max(nb, 1))
                if verbose:
                    print(f"      distil {name} ep {ep + 1}/{epochs}: {curve[-1]:.4f}", flush=True)
        model.eval()
        self.history.append(name)
        # The COST of learning this concept, which is how closure is measured on a dataset whose
        # attributes are spatially separable (see PLAN.md 3.2): if the hierarchy shares anything
        # useful, a concept landing near existing relatives should start lower and end lower than an
        # isolated one, even though its final accuracy may be no different.
        self.last_curve = dict(name=name, curve=curve, first=curve[0] if curve else None,
                               final=curve[-1] if curve else None,
                               depth=len(self.bank.tree.path_of(self.bank.concepts[name]["leaf"])) - 1)
        return self

    def learn_concept(self, xq, name, n_pool=64, epochs=10, **kw):
        teacher = self.compose(xq)
        pool = self.sample_teacher(teacher, n=n_pool, **kw)
        self.consolidate(pool, name, mu=teacher["mu"], epochs=epochs)
        return teacher, pool

    # ── using what was learned ───────────────────────────────────────────────
    def sample_concept(self, name, n=16, steps=None, seed=None, guidance=2.0, u_start=1.0):
        """Conditional generation from a consolidated concept: its adapter path is active and its
        learned embedding conditions the model, contrasted against the frozen base (CFG)."""
        c = self.cfg
        # `field.eps(x, u, name)` activates this concept's adapter path and embedding; the
        # unconditional branch inside sample() is `field.eps(x, u, None)`, which activates the empty
        # path -- the frozen base.  Two genuinely different functions, which is what makes this real
        # classifier-free guidance rather than a guided model guiding itself.
        return self.field.sample(
            n=n, steps=steps or c["steps"], seed=c["seed"] if seed is None else seed,
            u_start=u_start, clip=(-1, 1),
            guide_eps=lambda x, u: self.field.eps(x, u, name), guide_w=lambda u: guidance)

    def fingerprint(self, name, **kw):
        from .adapters import fingerprint
        return fingerprint(self.field, name, **kw)
