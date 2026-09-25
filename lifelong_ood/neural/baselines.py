"""Continual-learning baselines, implemented on the same backbone so only the mechanism differs.

Every arm here adapts the SAME frozen-base UNet with the SAME adapter parameterization and the same
optimizer budget.  What changes is only how each one tries to avoid destroying what it already knew,
which is what makes the comparison about the method rather than about capacity or tuning.

  finetune   train one shared adapter on each incoming task in sequence.  No protection at all --
             the catastrophic-forgetting control every CL paper reports against.
  ewc        Elastic Weight Consolidation (Kirkpatrick et al. 2017).  After each task, estimate the
             diagonal Fisher information of the parameters on that task's data and add
             `lambda/2 * sum_i F_i (theta_i - theta*_i)^2` to every later task's loss, so directions
             the old tasks cared about become expensive to move.  A soft, learned version of what
             freezing does structurally.
  replay     Generative replay (Shin et al. 2017): before learning task t, sample from the model's
             OWN earlier concepts and mix those samples into the objective.  Unusually natural here
             because every consolidated concept is a sampler, so no exemplars are stored.
  ttt        Test-time training: keep no per-task state at all, and instead adapt a fresh copy of the
             adapter to each query at evaluation time.  Nothing can be forgotten because nothing is
             retained -- which makes it the right control for asking whether retention is doing any
             work, and the wrong method for anything that must accumulate.
  ours       frozen ancestors in an insert-only hierarchy (see adapters.py).

EWC and replay both try to PROTECT shared parameters; freezing REMOVES the shared parameters from
the optimizer entirely.  That is the axis the comparison is really about.
"""
import numpy as np
import torch

from ..backbones.schedule import cosine_abar


def _batch_loss(model, bank, name, X, idx, dev, img):
    x0 = X[idx]
    u = torch.rand(len(x0), device=dev).clamp_(1e-3, 1.0)
    a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
    e = torch.randn_like(x0)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * e
    with bank.activate(name):
        pred = model(xt, u, emb_override=bank.embedding(name))
    return torch.nn.functional.mse_loss(pred, e)


class CLoRA:
    """C-LoRA (Smith et al., *Continual Diffusion*, TMLR 2024, arXiv:2304.06027).

    The canonical continual-CUSTOMIZATION baseline for diffusion, and the one most directly
    comparable to this repository: it is replay-free, stores no user data, and adds only LoRA
    parameters -- the same budget our adapter bank spends -- but keeps ONE accumulating model
    instead of an insert-only bank.

    The mechanism is a self-regularization that needs no Fisher and no held-out data.  Where EWC
    weights the anchor by an estimated Fisher, C-LoRA weights it by the accumulated MAGNITUDE of the
    past tasks' own LoRA updates: parameters that previous concepts wrote to are expensive to move
    again.  With `A_prev`/`B_prev` the summed absolute updates from earlier tasks and `A`,`B` the
    current ones,

        L_reg = lam * || |A_prev @ B_prev| (*) (A @ B) ||_F^2

    i.e. an elementwise product of the past-usage map with the current update, penalised.  It is
    cheaper than EWC (no gradient pass to estimate curvature) and, unlike EWC, its anchor is defined
    purely by what previous tasks CHANGED.

    Included because a reviewer will ask why a continual-diffusion paper compares only against
    classifier-derived CL baselines.  The structural prediction is that it forgets less than plain
    finetuning and still cannot reach 0.000, because a single accumulating model has no mechanism
    that makes an earlier concept's function bit-identical.
    """

    def __init__(self, lam=1e4):
        self.lam = float(lam)
        self.usage = {}       # id(param) -> accumulated |delta| from PAST tasks
        self._start = {}      # id(param) -> value at the start of the current task

    def begin_task(self, params):
        self._start = {id(p): p.detach().clone() for p in params}

    def penalty(self, params):
        if not self.usage:
            return None
        tot = None
        for p in params:
            k = id(p)
            if k not in self.usage or k not in self._start:
                continue
            cur = p - self._start[k]                     # what THIS task is changing
            d = (self.usage[k] * cur.abs()) ** 2         # weighted by what past tasks changed
            tot = d.sum() if tot is None else tot + d.sum()
        return None if tot is None else self.lam * tot

    def consolidate(self, model, bank, name, X, params, dev, img, seed=0):
        """Fold this task's update magnitude into the past-usage map."""
        for p in params:
            k = id(p)
            if k not in self._start:
                continue
            delta = (p.detach() - self._start[k]).abs()
            self.usage[k] = delta if k not in self.usage else self.usage[k] + delta
        self._start = {}


class SI:
    """Synaptic Intelligence (Zenke, Poole, Ganguli, ICML 2017, arXiv:1703.04200).

    The path-integral regularizer.  Where EWC estimates curvature at the end of a task with a
    separate Fisher pass, SI accumulates each parameter's contribution to the loss drop ONLINE,
    during the very steps that produce it:

        omega_k   -= sum_t  g_k(t) * delta_theta_k(t)      accumulated every optimizer step
        Omega_k    = sum_past omega_k / ((theta_k_end - theta_k_start)^2 + xi)
        L_total    = L + c * sum_k Omega_k (theta_k - theta_k_star)^2

    Chosen because it is the cleanest transfer in its family: it never inspects the output SEMANTICS,
    only gradients and weight trajectories, so nothing about it assumes a softmax or a label.  It
    therefore applies to an eps-regression objective without reinterpretation -- unlike RWalk, whose
    KL-based denominator is only meaningful for categorical outputs.

    Expected to behave like every other regularizer here: less forgetting than finetuning, and never
    zero, because a single accumulating model has no mechanism that makes an old concept bit-exact.
    The benchmark paper for this setting (CLoG) reports regularization methods as barely separable
    from naive training for image GENERATION, which is the comparison this arm is here to make.
    """

    def __init__(self, lam=1.0, xi=1e-3):
        self.lam = float(lam)
        self.xi = float(xi)
        self.omega = {}       # id -> big-Omega, the consolidated importance from past tasks
        self.star = {}        # id -> reference weights (end of previous task)
        self._w = {}          # id -> running path integral for the CURRENT task
        self._start = {}      # id -> weights at the start of the current task
        self._prev = {}       # id -> (value, grad) captured just before the last opt.step()

    def begin_task(self, params):
        self._start = {id(p): p.detach().clone() for p in params}
        self._w = {id(p): torch.zeros_like(p) for p in params}
        self._prev = {}

    def pre_step(self, params):
        """Capture (theta, grad) before the optimizer moves anything."""
        self._prev = {id(p): (p.detach().clone(), p.grad.detach().clone())
                      for p in params if p.grad is not None}

    def post_step(self, params):
        """Accumulate -g . dtheta, the loss decrease this step is credited with."""
        for p in params:
            k = id(p)
            if k not in self._prev or k not in self._w:
                continue
            old, gr = self._prev[k]
            self._w[k] -= gr * (p.detach() - old)
        self._prev = {}

    def penalty(self, params):
        if not self.omega:
            return None
        tot = None
        for p in params:
            k = id(p)
            if k not in self.omega:
                continue
            d = self.omega[k] * (p - self.star[k]) ** 2
            tot = d.sum() if tot is None else tot + d.sum()
        return None if tot is None else self.lam * tot

    def consolidate(self, model, bank, name, X, params, dev, img, seed=0):
        for p in params:
            k = id(p)
            if k not in self._start:
                continue
            delta = p.detach() - self._start[k]
            contrib = (self._w.get(k, torch.zeros_like(p)) / (delta ** 2 + self.xi)).clamp_(min=0.0)
            self.omega[k] = contrib if k not in self.omega else self.omega[k] + contrib
            self.star[k] = p.detach().clone()
        self._start, self._w, self._prev = {}, {}, {}


class MAS:
    """Memory Aware Synapses (Aljundi et al., ECCV 2018, arXiv:1711.09601).

    Importance is the sensitivity of the SQUARED OUTPUT NORM to each parameter:

        Omega_ij = (1/N) sum_k || d( ||F(x_k; theta)||^2 ) / d theta_ij ||
        L(theta) = L_new + lam * sum_ij Omega_ij (theta_ij - theta_star_ij)^2

    This is the best-motivated regularizer for a diffusion model in the whole family, and the reason
    is worth stating: MAS was derived for an ARBITRARY vector-valued output F, with no labels and no
    softmax anywhere in its definition.  Setting F = eps_theta(x_t, t, c) is not an adaptation of the
    method, it is the method.  EWC's empirical Fisher, by contrast, is a likelihood object we are
    reinterpreting for a regression head.

    lam = 1 is the paper's untuned default across every vision task they report.
    """

    def __init__(self, lam=1.0, n_est=32, batch=8):
        self.lam = float(lam)
        self.n_est = int(n_est)
        self.batch = int(batch)
        self.omega = {}
        self.star = {}

    def penalty(self, params):
        if not self.omega:
            return None
        tot = None
        for p in params:
            k = id(p)
            if k not in self.omega:
                continue
            d = self.omega[k] * (p - self.star[k]) ** 2
            tot = d.sum() if tot is None else tot + d.sum()
        return None if tot is None else self.lam * tot

    @torch.enable_grad()
    def consolidate(self, model, bank, name, X, params, dev, img, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        acc = {id(p): torch.zeros_like(p) for p in params}
        n = 0
        for _ in range(self.n_est):
            idx = torch.randint(0, len(X), (min(self.batch, len(X)),), generator=g).to(dev)
            x0 = X[idx]
            u = torch.rand(len(x0), device=dev).clamp_(1e-3, 1.0)
            a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
            e = torch.randn_like(x0)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            with bank.activate(name):
                out = model(xt, u, emb_override=bank.embedding(name))
            # the UNSUPERVISED signal: squared L2 norm of the output, no target involved
            sq = (out ** 2).sum() / len(x0)
            grads = torch.autograd.grad(sq, params, retain_graph=False, allow_unused=True)
            for p, gr in zip(params, grads):
                if gr is not None:
                    acc[id(p)] += gr.detach().abs()
            n += 1
        for p in params:
            k = id(p)
            o = acc[k] / max(n, 1)
            self.omega[k] = o if k not in self.omega else self.omega[k] + o
            self.star[k] = p.detach().clone()


class DERPlusPlus:
    """DER++ (Buzzega et al., NeurIPS 2020, arXiv:2004.07211) for an eps-prediction objective.

        L = L_task
            + alpha * E_(x,t,eps_stored)~M || eps_stored - eps_theta(x_t, t, c) ||^2   (dark knowledge)
            + beta  * E_(x,eps_true)~M     || eps_true   - eps_theta(x_t, t, c) ||^2   (plain replay)

    The alpha term is what makes this more than experience replay: the target is the model's OWN
    prediction recorded at INSERTION time, mid-trajectory rather than at a task boundary, which the
    paper is explicit is deliberate.  The beta term degenerates to standard exemplar replay, so this
    single arm subsumes ER and is the stronger of the two to report.

    Two properties matter for the comparison this repository is making.  First, it is the strongest
    simple baseline in modern continual learning, so beating it is the meaningful result.  Second --
    and this is the point the growth/storage columns exist to make -- it BUYS its retention with a
    permanent buffer of real images, where an insert-only bank stores no data at all.  Reservoir
    sampling (Vitter 1985) fills that buffer without task boundaries.
    """

    def __init__(self, size=200, alpha=0.5, beta=0.5, batch=8, seed=0):
        self.size = int(size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.batch = int(batch)
        self.n_seen = 0
        # `noise` is stored, not redrawn.  DER's target z = h(x) is a function of a FIXED input; here
        # the network's input is x_t = sqrt(abar) x0 + sqrt(1-abar) e, so the stored eps-prediction is
        # only a valid target for the very draw of e that produced it.  Redrawing e at replay time
        # asks the model to reproduce a prediction made about a different image -- an incoherent
        # target that actively destroys the concept (measured: ACC 0.250 against finetune's 0.750).
        self.x, self.emb, self.name, self.eps, self.u, self.noise = [], [], [], [], [], []
        self._g = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        return len(self.x)

    def add(self, x0, emb, name, eps_logit, u, noise, p_insert=1.0):
        """Reservoir-sample one example, storing the eps-prediction made when it was inserted.

        `p_insert` exists because DER is defined for a SINGLE-epoch stream, where an example's
        insertion time is simply when it was trained on -- which is what makes the stored logit
        "mid-trajectory" rather than converged.  Training here runs many epochs, so inserting every
        example every epoch would both inflate the reservoir's stream length and bias the stored
        logits late.  Offering each example a 1/epochs chance per visit gives it one insertion in
        expectation, at a uniformly random point in training, which is the faithful analogue.
        """
        for i in range(len(x0)):
            if p_insert < 1.0 and float(torch.rand((), generator=self._g)) >= p_insert:
                continue
            self.n_seen += 1
            item = (x0[i].detach().clone(), emb.detach().clone(), name,
                    eps_logit[i].detach().clone(), float(u[i]), noise[i].detach().clone())
            if len(self.x) < self.size:
                self.x.append(item[0]); self.emb.append(item[1]); self.name.append(item[2])
                self.eps.append(item[3]); self.u.append(item[4]); self.noise.append(item[5])
            else:
                j = int(torch.randint(0, self.n_seen, (1,), generator=self._g))
                if j < self.size:
                    self.x[j], self.emb[j], self.name[j] = item[0], item[1], item[2]
                    self.eps[j], self.u[j], self.noise[j] = item[3], item[4], item[5]

    def loss(self, model, bank, dev, ch, img, g=None):
        """One memory term per step, grouped by concept so each distinct adapter path costs one
        forward pass rather than one per sampled example."""
        if not self.x:
            return None
        k = min(self.batch, len(self.x))
        sel = torch.randperm(len(self.x), generator=self._g)[:k].tolist()
        by_name = {}
        for i in sel:
            by_name.setdefault(self.name[i], []).append(i)
        tot, n = None, 0
        for nm, idxs in by_name.items():
            x0 = torch.stack([self.x[i] for i in idxs])
            u = torch.tensor([self.u[i] for i in idxs], device=dev, dtype=torch.float32)
            a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
            # the SAME noise draw the stored prediction was made about, so both terms describe one
            # coherent input rather than two different ones
            e = torch.stack([self.noise[i] for i in idxs])
            xt = a.sqrt() * x0 + (1 - a).sqrt() * e
            stored = torch.stack([self.eps[i] for i in idxs])
            with bank.activate(nm):
                pred = model(xt, u, emb_override=self.emb[idxs[0]])
            # alpha: match the logit recorded at insertion.  beta: match true noise (= plain ER).
            d = len(idxs) * (self.alpha * torch.nn.functional.mse_loss(pred, stored)
                             + self.beta * torch.nn.functional.mse_loss(pred, e))
            tot = d if tot is None else tot + d
            n += len(idxs)
        return None if tot is None else tot / max(n, 1)


class EWC:
    """Diagonal-Fisher penalty accumulated over tasks.

    `lam` is the only knob.  The Fisher is estimated from squared gradients of the diffusion loss on
    the task's own data, which is the standard empirical-Fisher approximation; the penalty then
    anchors each parameter to its post-task value in proportion to how much that task's loss depended
    on it.
    """

    def __init__(self, lam=5000.0, n_fisher=32):
        self.lam = float(lam)
        self.n_fisher = int(n_fisher)
        self.fisher = {}      # id -> tensor
        self.star = {}
        self._params = {}

    def penalty(self, params):
        if not self.fisher:
            return None
        tot = None
        for p in params:
            k = id(p)
            if k not in self.fisher:
                continue
            d = (p - self.star[k]) ** 2 * self.fisher[k]
            tot = d.sum() if tot is None else tot + d.sum()
        return None if tot is None else 0.5 * self.lam * tot

    @torch.enable_grad()
    def consolidate(self, model, bank, name, X, params, dev, img, seed=0):
        """Accumulate the Fisher for the task just learned and snapshot the parameters."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        fish = {id(p): torch.zeros_like(p) for p in params}
        n = 0
        for _ in range(self.n_fisher):
            idx = torch.randint(0, len(X), (min(8, len(X)),), generator=g).to(dev)
            loss = _batch_loss(model, bank, name, X, idx, dev, img)
            grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
            for p, gr in zip(params, grads):
                if gr is not None:
                    fish[id(p)] += gr.detach() ** 2
            n += 1
        for p in params:
            k = id(p)
            f = fish[k] / max(n, 1)
            self.fisher[k] = f if k not in self.fisher else self.fisher[k] + f
            self.star[k] = p.detach().clone()


def test_time_train(learner, target, name, epochs=10, batch=8, lr=1e-3, seed=0):
    """Adapt the EXISTING adapter for `name` to `target` at evaluation time, then restore it.

    Test-time training keeps no cross-task state, so it cannot forget and cannot accumulate.  Running
    it here answers a specific question: how much of our numbers come from retention at all, versus
    from simply being able to re-fit the current task on demand?  Anything TTT matches is not
    evidence for a lifelong mechanism -- but note what it costs, which the table also shows: TTT
    must keep the task's DATA forever and pay a full optimization every time the task is queried,
    where a consolidated concept is a fixed set of weights that is simply read.

    Trains the concept's already-registered parameters in place and rolls them back, so no new nodes
    are created -- re-registering the same name would grow the bank on every evaluation.
    """
    bank = learner.bank
    if name not in bank.concepts:
        raise KeyError(f"{name} must be registered before test-time training")
    params = bank.trainable_batch([name])
    backup = [p.detach().clone() for p in params]
    model = learner.field.model
    dev = learner.device
    img = learner.field.img
    X = torch.from_numpy(np.asarray(target, np.float32)).to(dev).reshape(-1, 3, img, img)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(max(int(epochs), 0)):
        perm = torch.randperm(len(X), generator=g)
        for i0 in range(0, len(X), batch):
            idx = perm[i0:i0 + batch].to(dev)
            loss = _batch_loss(model, bank, name, X, idx, dev, img)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()

    def restore():
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.data.copy_(b)
    return restore
