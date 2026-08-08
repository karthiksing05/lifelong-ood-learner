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

from ..compose import (candidates_ascent, candidates_ascent_batch, cfg_weight, poe_teacher,
                       query_only_teacher, stacked_cfg_weight, stacked_teacher_eps, teacher_eps)
from ..score import cosine_abar
from .adapters import HierarchicalAdapterBank
from .ddpm import CH, IMG


class NeuralLifelongLearner:
    def __init__(self, field, rank=8, K=3, tau=0.05, wmin=0.8, wmax=2.0, w0=1.2,
                 schedule="anneal", u_start=1.0, teacher_var_scale=30.0,
                 steps=50, seed=0, device="cuda", flat=False,
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
        self.cfg = dict(K=K, tau=tau, wmin=wmin, wmax=wmax, w0=w0, schedule=schedule,
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

    def compose_batch(self, xqs, K=None, tau=None, **disc):
        """Discover and compose K OOD prototypes in one pass.  -> list of teachers."""
        c = self.cfg
        pools = candidates_ascent_batch(self.field, xqs, seed=c["seed"], **disc)
        out = []
        for xq, (M, V, meta) in zip(xqs, pools):
            t = poe_teacher(np.ravel(xq), M, V, K=K or c["K"], tau=tau or c["tau"])
            t["meta"] = meta
            out.append(t)
        return out

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
        gw = cfg_weight(teacher, w0=kw.get("w0", c["w0"]), wmin=kw.get("wmin", c["wmin"]),
                        wmax=kw.get("wmax", c["wmax"]),
                        schedule=kw.get("schedule", c["schedule"]))
        return self.field.sample(n=n, steps=steps or c["steps"], seed=seed,
                                 u_start=c["u_start"] if u_start is None else u_start,
                                 x_init=np.clip(x0, -1, 1), clip=(-1, 1),
                                 guide_eps=teacher_eps(teacher), guide_w=gw)

    def sample_teacher_batch(self, teachers, n=64, steps=None, seed=None, u_start=None,
                             var_scale=None, **kw):
        """Sample K pools of `n` in ONE reverse-diffusion run.  -> list of K arrays [n, D].

        Eq. 12 is elementwise across the batch, so the K teachers are stacked one row per sample and
        the whole K*n batch denoises together.  The unconditional branch is evaluated once per step
        for the whole batch rather than once per concept, which is where the wall-clock saving comes
        from -- the sampler is the dominant cost of the loop.
        """
        c = self.cfg
        seed = c["seed"] if seed is None else seed
        rng = np.random.default_rng(seed)
        vs = c["teacher_var_scale"] if var_scale is None else var_scale
        MU, VAR = [], []
        for t in teachers:
            s = 1.0 if t.get("fixed_var") else float(vs)
            MU.append(np.tile(t["mu"], (n, 1)))
            VAR.append(np.tile(t["var"] * s, (n, 1)))
        MU, VAR = np.concatenate(MU), np.concatenate(VAR)
        x0 = np.clip(MU + np.sqrt(VAR) * rng.standard_normal(MU.shape), -1, 1)
        gw = stacked_cfg_weight(VAR, w0=kw.get("w0", c["w0"]),
                                wmin=kw.get("wmin", c["wmin"]),
                                wmax=kw.get("wmax", c["wmax"]),
                                schedule=kw.get("schedule", c["schedule"]))
        S = self.field.sample(n=len(MU), steps=steps or c["steps"], seed=seed,
                              u_start=c["u_start"] if u_start is None else u_start,
                              x_init=x0, clip=(-1, 1), eta=kw.get("eta", c.get("eta", 0.0)),
                              guide_eps=stacked_teacher_eps(MU, VAR), guide_w=gw)
        return [S[i * n:(i + 1) * n] for i in range(len(teachers))]

    # ── step 4: distil the pool into a new hierarchical adapter ──────────────
    def consolidate(self, pool, name, mu=None, epochs=10, batch=16, lr=1e-3, p_drop=0.1,
                    seed=None, verbose=False, project=None, rehearse=None,
                    freeze_embedding=False, already_registered=False, ewc=None):
        """The paper's Eq. 15 loss, optimizing ONLY this concept's leaf delta and class embedding.

        `mu` (the teacher mean) is what places the concept in the hierarchy; pass it so related
        concepts land near each other and can share ancestors.

        `rehearse` = dict(x0, conds, w) trains the delta to be a NO-OP off-concept.  A LoRA delta is
        conditional by construction: it only applies while this concept's path is active, so it is
        never asked what it does to other conditionings.  Merging it into the base makes it
        unconditionally on, which is why a naive fold collapses the model (n12: ACC 0.818 -> 0.062).
        With this, each step also sees known compositions under their own conditionings and is
        penalized for changing what the model predicted for them BEFORE this concept began -- so the
        delta is trained for the always-on job it is about to be given.

        `project` is a {layer: P} map from `activation_subspace`.  When given, the update's
        input-side factor is projected onto the complement of the activations that already-known
        conditionings produce, after every step.  That is what allows the delta to be MERGED into
        the base weights afterwards instead of being carried as a private adapter: the write lands
        where nothing already known reads.
        """
        seed = self.cfg["seed"] if seed is None else seed
        torch.manual_seed(seed)
        if not already_registered:
            self.bank.add_concept(name, np.ravel(pool.mean(0) if mu is None else mu))
        params = self.bank.trainable(name)
        if freeze_embedding:
            emb_p = self.bank._emb_extra[name]
            params = [q for q in params if q is not emb_p]
        opt = torch.optim.Adam(params, lr=lr)
        # regularisation baselines that anchor to the PRE-TASK weights (C-LoRA) need a
        # snapshot before the first step; EWC does not define this hook and is unaffected.
        if ewc is not None and hasattr(ewc, "begin_task"):
            ewc.begin_task(params)
        X = torch.from_numpy(np.asarray(pool, np.float32)).to(self.device)
        X = X.reshape(-1, CH, self.field.img, self.field.img)
        model = self.field.model
        model.train()
        n = len(X)
        g = torch.Generator(device="cpu").manual_seed(seed)
        curve = []
        reh = None
        if rehearse is not None:
            Xr = torch.from_numpy(np.asarray(rehearse["x0"], np.float32)).to(self.device)
            Xr = Xr.reshape(-1, CH, self.field.img, self.field.img)
            Cr = [torch.as_tensor(np.asarray(c, np.float32), device=self.device)
                  for c in rehearse["conds"]]
            reh = dict(X=Xr, C=Cr, w=float(rehearse.get("w", 1.0)))
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
                    if reh is not None:
                        # what does this delta do to conditionings it is not for?  It should do
                        # nothing, and the reference is the FROZEN BASE -- reachable by simply
                        # deactivating the path, so no teacher copy of the model is needed.
                        j = torch.randint(0, len(reh["X"]), (min(batch, len(reh["X"])),),
                                          generator=g).to(self.device)
                        xr0 = reh["X"][j]
                        cr = reh["C"][int(torch.randint(0, len(reh["C"]), (1,), generator=g))]
                        ur = torch.rand(len(xr0), device=self.device).clamp_(1e-3, 1.0)
                        ar = torch.from_numpy(cosine_abar(ur.detach().cpu().numpy())).to(self.device)
                        ar = ar.float().view(-1, 1, 1, 1)
                        er = torch.randn_like(xr0)
                        xrt = ar.sqrt() * xr0 + (1 - ar).sqrt() * er
                        with torch.no_grad(), self.bank.activate(None):
                            ref = model(xrt, ur, emb_override=cr)
                        loss = loss + reh["w"] * torch.nn.functional.mse_loss(
                            model(xrt, ur, emb_override=cr), ref)
                    if ewc is not None:
                        pen = ewc.penalty(params)
                        if pen is not None:
                            loss = loss + pen
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    # SI accumulates its path integral from (grad, step) pairs, so it needs to see
                    # every step; regularizers without the hooks are unaffected.
                    if ewc is not None and hasattr(ewc, "pre_step"):
                        ewc.pre_step(params)
                    opt.step()
                    if ewc is not None and hasattr(ewc, "post_step"):
                        ewc.post_step(params)
                    if project:
                        # keep the update out of the directions old conditionings read, so the
                        # delta can be folded into the base weights without overwriting them
                        from .adapters import project_out
                        project_out(self.bank, self.bank.concepts[name]["path"], project)
                    tot += float(loss); nb += 1
                curve.append(tot / max(nb, 1))
                if verbose:
                    print(f"      distil {name} ep {ep + 1}/{epochs}: {curve[-1]:.4f}", flush=True)
        model.eval()
        self.history.append(name)
        # The COST of learning this concept, which is how closure is measured on a dataset whose
        # attributes are spatially separable (see RESULTS.md 3.2): if the hierarchy shares anything
        # useful, a concept landing near existing relatives should start lower and end lower than an
        # isolated one, even though its final accuracy may be no different.
        self.last_curve = dict(name=name, curve=curve, first=curve[0] if curve else None,
                               final=curve[-1] if curve else None,
                               depth=len(self.bank.tree.path_of(self.bank.concepts[name]["leaf"])) - 1)
        return self

    def consolidate_batch(self, items, epochs=10, batch=16, lr=1e-3, seed=None, group=True,
                          replay=None, verbose=False, ewc=None, buffer=None, replay_distill=False):
        """Add K concepts to the manifold in ONE consolidation.  `items` = [(name, pool, mu), ...].

        All K leaves and any shared group node they were given are optimized TOGETHER: one Adam over
        the union of their parameters, with minibatches interleaved across concepts so a shared node
        accumulates gradient from every member.  That is the difference from K sequential
        consolidations, which can only ever hand a later concept a frozen warm start.

        The preservation guarantee is untouched.  `add_concepts` seals every node that existed before
        this call, so no parameter any earlier concept depends on is in the optimizer at all -- and
        that is true whatever K is, which is what makes the batch size a free efficiency knob rather
        than a safety trade.
        """
        seed = self.cfg["seed"] if seed is None else seed
        torch.manual_seed(seed)
        names = [n for n, _, _ in items]
        mus = [np.ravel(m if m is not None else p.mean(0)) for _, p, m in items]
        self.bank.add_concepts(names, mus, group=group)
        # `replay` = [(existing_name, pool), ...] trained alongside the new concepts WITHOUT being
        # re-registered.  This is generative replay (Shin et al.): the pools are sampled from the
        # model's own earlier concepts, so old knowledge defends itself through the loss.  It only
        # does anything when ancestors are trainable -- with frozen ancestors nothing shared can
        # move, so there is nothing for replay to protect, which is the point of the comparison.
        #
        # `replay_distill` switches those same pools to GENERATIVE DISTILLATION (Masip et al.,
        # CoLLAs 2025).  The distinction is only in the target, and it matters enormously: plain
        # generative replay regresses a self-generated sample onto the noise that was drawn for it,
        # so any error already in the sample is treated as ground truth and compounds -- which is why
        # it is measured at FID 211.8 against finetuning's 68.8 for a DDPM, i.e. WORSE than no
        # protection at all.  Distillation instead matches the PRE-TASK model's own eps-prediction on
        # the same (x_t, t), transferring the whole reverse process rather than one endpoint, and
        # recovers FID 23.5 against a joint-training bound of 15.5.  Reporting naive generative
        # replay without this arm would be reporting a known-broken configuration as a baseline.
        replay = list(replay or [])
        new_names, names = names, names + [n for n, _ in replay]
        n_new = len(new_names)
        params, seen_ids = [], set()
        for p in self.bank.trainable_batch(names):
            if id(p) not in seen_ids:
                seen_ids.add(id(p)); params.append(p)
        opt = torch.optim.Adam(params, lr=lr)
        # same pre-task snapshot for the batch path (see consolidate)
        if ewc is not None and hasattr(ewc, "begin_task"):
            ewc.begin_task(params)
        pools = [torch.from_numpy(np.asarray(p, np.float32)).to(self.device).reshape(-1, CH, self.field.img, self.field.img)
                 for _, p, _ in items]
        pools += [torch.from_numpy(np.asarray(p, np.float32)).to(self.device).reshape(-1, CH, self.field.img, self.field.img)
                  for _, p in replay]
        model = self.field.model
        model.train()
        g = torch.Generator(device="cpu").manual_seed(seed)
        curves = {n: [] for n in names}
        # The distillation teacher is this model at its PRE-TASK weights.  Snapshotting the trainable
        # tensors is enough to reconstruct it -- everything else is frozen by construction -- so no
        # second copy of the network is ever allocated.
        teacher_vals = ([p.detach().clone() for p in params] if (replay_distill and replay) else None)

        def _teacher_eps(xt, u, emb):
            cur = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, tv in zip(params, teacher_vals):
                    p.data.copy_(tv)
                ref = model(xt, u, emb_override=emb).detach()
                for p, cv in zip(params, cur):
                    p.data.copy_(cv)
            return ref
        for ep in range(max(int(epochs), 0)):
            # one pass over every concept's pool per epoch, concepts interleaved
            order = []
            for j, P in enumerate(pools):
                perm = torch.randperm(len(P), generator=g)
                order += [(j, perm[i0:i0 + batch]) for i0 in range(0, len(P), batch)]
            for i in torch.randperm(len(order), generator=g).tolist():
                j, idx = order[i]
                x0 = pools[j][idx]
                u = torch.rand(len(x0), device=self.device).clamp_(1e-3, 1.0)
                a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(self.device)
                a = a.float().view(-1, 1, 1, 1)
                e = torch.randn_like(x0)
                xt = a.sqrt() * x0 + (1 - a).sqrt() * e
                with self.bank.activate(names[j]):
                    emb_j = self.bank.embedding(names[j])
                    # replay pools are the tail of `pools`; only those get the distillation target
                    tgt = (_teacher_eps(xt, u, emb_j)
                           if (teacher_vals is not None and j >= n_new) else e)
                    pred = model(xt, u, emb_override=emb_j)
                    loss = torch.nn.functional.mse_loss(pred, tgt)
                if buffer is not None and j < n_new:
                    # DER's "dark knowledge" is the logit as it was MID-TRAJECTORY, so the example
                    # must be banked here, during training, not from a pass at the task boundary.
                    buffer.add(x0, emb_j, names[j], pred, u, e,
                               p_insert=1.0 / max(int(epochs), 1))
                if ewc is not None:
                    pen = ewc.penalty(params)
                    if pen is not None:
                        loss = loss + pen
                if buffer is not None:
                    # DER++ adds its two memory terms to EVERY step, not once per task, because the
                    # buffer is sampled i.i.d. rather than at task boundaries (Buzzega et al.).
                    bl = buffer.loss(model, self.bank, self.device, CH, self.field.img, g)
                    if bl is not None:
                        loss = loss + bl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if ewc is not None and hasattr(ewc, "pre_step"):
                    ewc.pre_step(params)
                opt.step()
                if ewc is not None and hasattr(ewc, "post_step"):
                    ewc.post_step(params)
                curves[names[j]].append(float(loss))
            if verbose:
                last = {n: (np.mean(v[-4:]) if v else float("nan")) for n, v in curves.items()}
                print(f"      batch distil ep {ep + 1}/{epochs}: "
                      f"mean {np.mean(list(last.values())):.4f}", flush=True)
        if ewc is not None:
            for j, nm in enumerate(new_names):
                ewc.consolidate(model, self.bank, nm, pools[j], params, self.device,
                                self.field.img, seed=seed)
        model.eval()
        self.history += new_names            # replayed concepts were already in the history
        self.last_batch_curves = {
            n: dict(name=n, curve=v, first=float(np.mean(v[:4])) if v else None,
                    final=float(np.mean(v[-4:])) if v else None,
                    depth=len(self.bank.tree.path_of(self.bank.concepts[n]["leaf"])) - 1)
            for n, v in curves.items() if n in new_names}
        self.last_curve = self.last_batch_curves[new_names[-1]]
        return self

    def learn_concept_inverted(self, xq, name, n_pool=64, epochs=10, inv_steps=400, inv_lr=0.05,
                               anchor=0.0, init_from_teacher=True, seed=None, **kw):
        """The loop with the manifold generating: compose -> INVERT -> sample -> consolidate.

        The difference from `learn_concept` is step 3.  There, the pool is drawn by guiding the
        sampler toward the analytic teacher, which RESULTS 5e showed collapses to near-copies of
        mu_T (recall 0.00).  Here the teacher instead determines a point in the model's own
        conditioning space (`inversion.learn_embedding`), and the pool is drawn from the model
        conditioned on it -- so the pool inherits the conditional's diversity (N7: recall 0.83,
        content diversity 1.21x real) and the concept that gets consolidated is a region rather
        than a point.

        Consolidation is unchanged, so every preservation guarantee carries over untouched: this
        changes what is handed to `consolidate`, not what `consolidate` does.
        """
        from .inversion import learn_embedding, sample_embedding, teacher_init_alpha
        c = self.cfg
        seed = c["seed"] if seed is None else seed
        teacher = self.compose(xq, **kw.pop("disc", {}) or {})
        a0 = teacher_init_alpha(self.field, teacher, xq, seed=seed) if init_from_teacher else None
        c_new, alpha, hist = learn_embedding(self.field, np.atleast_2d(xq), n_steps=inv_steps,
                                             lr=inv_lr, seed=seed, init_alpha=a0,
                                             anchor=anchor, anchor_alpha=a0)
        pool = sample_embedding(self.field, c_new, n=n_pool, steps=c["steps"], seed=seed,
                                guidance=kw.pop("guidance", 1.0))
        self.consolidate(pool, name, mu=teacher["mu"], epochs=epochs)
        self.bank._emb_extra[name].data = c_new.detach().clone()   # start the adapter AT the concept
        self.last_inversion = dict(alpha=alpha, final_loss=float(np.mean(hist[-20:])) if hist else None)
        return teacher, pool

    # ── the composer that works: factor basis + calibrated per-factor gain ───
    def attach_factorization(self, C, labels, gains=None):
        """Fit the ideal-words decomposition of the class-embedding table and keep it.

        `C` are the trained class embeddings and `labels` their factor values (both from SEEN data
        only).  Held-out combinations then become sums of vectors the model has already learned --
        measured R^2 = 0.816 on ColorMNIST, i.e. the table really is approximately additive.
        """
        from .idealwords import factorize
        self.fac = factorize(np.asarray(C), np.asarray(labels))
        self.fac_gains = list(gains) if gains is not None else [1.0] * len(self.fac["V"])
        return self.fac

    def compose_factored(self, xq, inv_steps=400, inv_lr=0.05, seed=None, gains=None):
        """Read the composition off a single query IN THE FACTOR BASIS, then build the concept.

        This is the composer the investigation converged on, and each part is there for a measured
        reason.  The factor basis, because a free weighting over entangled class embeddings cannot
        place a held-out pairing even given 64 real images of it, while ~18 factor weights recover
        (digit, fg, bg) exactly on 7 of 8 slots from ONE query.  The per-factor gain, because the
        model under-responds to its weakest factor direction -- unit gain gives bg 0.38 where
        doubling that one gain gives 0.87.

        -> (c_new, per-factor weights, the argmax read of the composition).
        """
        from .idealwords import learn_factored
        if not hasattr(self, "fac"):
            raise RuntimeError("call attach_factorization() first")
        seed = self.cfg["seed"] if seed is None else seed
        c_new, ws, hist = learn_factored(self.field, np.atleast_2d(xq), self.fac,
                                         n_steps=inv_steps, lr=inv_lr, seed=seed,
                                         gains=gains if gains is not None else self.fac_gains)
        return c_new, ws, [int(np.argmax(w)) for w in ws]

    def learn_concept_factored(self, xq, name, n_pool=64, epochs=10, inv_steps=400, inv_lr=0.05,
                               seed=None, gains=None, guidance=1.0, **kw):
        """The full loop with the best composer: read -> compose -> sample the manifold -> consolidate.

        The teacher still discovers (mode ascent supplies the prototypes the factorization is read
        against); the manifold generates, so the pool inherits the model conditional's diversity
        rather than the teacher Gaussian's collapse; and consolidation is untouched, so every
        preservation guarantee carries over.
        """
        from .idealwords import sample_factored_pool
        c_new, ws, read = self.compose_factored(xq, inv_steps=inv_steps, inv_lr=inv_lr,
                                                seed=seed, gains=gains)
        pool = sample_factored_pool(self.field, c_new, n=n_pool, steps=self.cfg["steps"],
                                    seed=self.cfg["seed"] if seed is None else seed,
                                    guidance=guidance)
        # Pin the embedding to the composed vector BEFORE consolidating, and hold it there.
        #
        # This used to run the other way round: consolidation trained the embedding along with the
        # adapter, and then `c_new` was written over it afterwards.  The adapter was therefore fit
        # against an embedding that no longer existed at sampling time -- and since the composition
        # lives in the embedding (RESULTS 5n), that is the conditioning the adapter most needs to
        # have seen.  Freezing it makes the adapter a residual correction to the actual composition
        # rather than to a discarded intermediate.
        self.bank.add_concept(name, np.asarray(pool).mean(0))
        self.bank._emb_extra[name].data = c_new.detach().clone()
        self.consolidate(pool, name, mu=np.asarray(pool).mean(0), epochs=epochs,
                         freeze_embedding=True, already_registered=True)
        self.last_read = read
        return c_new, pool

    def learn_concept(self, xq, name, n_pool=64, epochs=10, **kw):
        teacher = self.compose(xq)
        pool = self.sample_teacher(teacher, n=n_pool, **kw)
        self.consolidate(pool, name, mu=teacher["mu"], epochs=epochs)
        return teacher, pool

    def learn_concept_disc(self, xq, name, n_pool=64, epochs=10, disc=None, **kw):
        """`learn_concept` with the discovery configuration made explicit.

        `learn_concept` calls `compose(xq)` with no discovery arguments, so it silently uses
        `candidates_ascent`'s defaults while `learn_concepts_batch` uses whatever `disc` it is
        handed.  Any experiment comparing the two modes was therefore varying discovery as well as
        batching.  This exists so a sequential run can be given the batch path's discovery settings
        and the comparison isolates batching alone.
        """
        teacher = self.compose(xq, **(disc or {}))
        pool = self.sample_teacher(teacher, n=n_pool, **kw)
        self.consolidate(pool, name, mu=teacher["mu"], epochs=epochs)
        return teacher, pool

    def learn_concepts_batch(self, xqs, names, n_pool=64, epochs=10, group=True,
                             disc=None, **kw):
        """The whole loop for K OOD queries at once: compose -> sample -> consolidate, all batched.

        This is the efficiency path.  Discovery shares its query-agnostic modes across the batch,
        the K pools come out of a single reverse-diffusion run, and consolidation is one joint
        optimization -- while what reaches the manifold is the same K additive concepts, and
        everything already in it stays bit-identical.
        """
        teachers = self.compose_batch(xqs, **(disc or {}))
        pools = self.sample_teacher_batch(teachers, n=n_pool, **kw)
        self.consolidate_batch([(n, p, t["mu"]) for n, p, t in zip(names, pools, teachers)],
                               epochs=epochs, group=group)
        return teachers, pools

    # ── using what was learned ───────────────────────────────────────────────
    def sample_concept(self, name, n=16, steps=None, seed=None, guidance=1.0, u_start=1.0):
        """Conditional generation from a consolidated concept: its adapter path is active and its
        learned embedding conditions the model, contrasted against the frozen base (CFG).

        `guidance=1.0` -- follow the adapter exactly -- rather than the usual CFG value above 1, and
        this is measured, not conservative (tests/diag_sample_concept.py): joint accuracy runs 0.38
        at w=1.0, 0.12 at 1.5, and 0.00 at 2.0 and above.  Standard CFG extrapolates away from a
        WEAK unconditional model, which sharpens; here the "unconditional" branch is the frozen base
        DDPM, which is a strong generative model of the seen slots, so pushing away from it leaves
        the image manifold entirely.  The diagnostic also confirms the adapter itself is fine --
        post-hoc loss on its pool is 0.0041 with the adapter active against 0.0131 for the frozen
        base -- so what looked like a failure to learn was purely a failure to sample.
        """
        c = self.cfg
        # `field.eps(x, u, name)` activates this concept's adapter path and embedding; the
        # unconditional branch inside sample() is `field.eps(x, u, None)`, which activates the empty
        # path -- the frozen base.  Two genuinely different functions, which is what makes this real
        # classifier-free guidance rather than a guided model guiding itself.
        return self.field.sample(
            n=n, steps=steps or c["steps"], seed=c["seed"] if seed is None else seed,
            u_start=u_start, clip=(-1, 1),
            guide_eps=lambda x, u: self.field.eps(x, u, name), guide_w=lambda u: guidance)

    def sample_composed(self, names, n=16, steps=None, seed=None, scale=1.0, weights=None,
                        u_start=1.0):
        """Sample SEVERAL consolidated concepts at once, composed in SCORE space.

            eps_hat = eps_base + sum_i w_i ( eps_concept_i - eps_base )

        Use this rather than activating several paths together, which is what the bank did before
        and which is measurably worse.  Two adapters activated at once SUM two deltas that were each
        trained to be applied alone, and nothing in either one's training says what it should do
        while the other is on.  Score composition never evaluates an adapter outside its own
        conditioning: each `eps_concept_i` is the concept exactly as it was consolidated, and only
        their PREDICTIONS are combined.

        Measured on a pretrained backbone with two independently learned attributes (RESULTS 12e):
        both attributes correct 0.50 for score composition against 0.17 for path union, at Vendi
        1.56 against 1.37.  This is the failure the whole LoRA-merging literature exists to address
        (ZipLoRA, Orthogonal Adaptation, Custom Diffusion's constrained merge), and the bank had
        been doing the naive version of it.

        `scale` is the per-concept weight (Liu et al.'s w_i, which are meant to ADD -- see
        composable.py; 1/n would make this an average and a strictly weaker operator).  Preservation
        is untouched: paths stay insert-only and every concept remains bit-identical.
        """
        c = self.cfg
        names = list(names)
        w = ([float(scale)] * len(names) if weights is None
             else [float(x) for x in weights])

        def eps(x, u):
            e0 = self.field.eps(x, u, None)          # the frozen base: no path, no concept
            out = e0.copy()
            for nm, wi in zip(names, w):
                out = out + wi * (self.field.eps(x, u, nm) - e0)
            return out

        return self.field.sample(
            n=n, steps=steps or c["steps"], seed=c["seed"] if seed is None else seed,
            u_start=u_start, clip=(-1, 1),
            guide_eps=eps, guide_w=lambda u: 1.0)

    def fingerprint(self, name, **kw):
        from .adapters import fingerprint
        return fingerprint(self.field, name, **kw)
