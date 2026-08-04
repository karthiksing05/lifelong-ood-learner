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

from ..score import cosine_abar


def _batch_loss(model, bank, name, X, idx, dev, img):
    x0 = X[idx]
    u = torch.rand(len(x0), device=dev).clamp_(1e-3, 1.0)
    a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(dev).float().view(-1, 1, 1, 1)
    e = torch.randn_like(x0)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * e
    with bank.activate(name):
        pred = model(xt, u, emb_override=bank.embedding(name))
    return torch.nn.functional.mse_loss(pred, e)


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
