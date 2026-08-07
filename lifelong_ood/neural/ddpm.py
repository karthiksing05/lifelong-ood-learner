"""DDPM training and `NeuralScoreField` — a trained UNet behind the SAME interface as the tree.

`NeuralScoreField` subclasses `BaseScoreField`, so reverse sampling, mode ascent on p_t, and the
Hutchinson Laplace covariance are the inherited implementations: the neural backbone changes what
`eps` is, and nothing else.  Every line of `compose.py` and `LifelongLearner` therefore runs
unchanged on it, which is what makes "the lifelong mechanism, not the backbone" a testable claim.

WHY A NEURAL BACKBONE AT ALL.  The analytic Cobweb field measurably struggles at one specific thing,
and it is the thing this project is about.  A Gaussian mixture over tree concepts assigns
*essentially zero* density to an unseen (fg, bg) pairing -- no component sits there -- so during
reverse sampling the unconditional score actively pulls a composed sample back toward the nearest
SEEN slot.  E0 measured the damage: the paper's variance-aware guidance retained only 21% of the
teacher's composition accuracy, and an annealed schedule was needed to reach 71%.  A convolutional
UNet shares weights across colours and positions, so it assigns real density to unseen combinations
and its unconditional branch cooperates with composition rather than fighting it.  Same method, a
backbone that can actually hold the OOD region.

CONVENTION.  This field's x-space is the model's [-1, 1] space, as flat [N, 3072] float arrays.
`to_model` / `to_data` convert to and from the [0, 1] pixel space the datasets and probes use;
neural experiments convert at their boundaries.  (The analytic field works directly in [0, 1].)
"""
import numpy as np
import torch

from ..score import BaseScoreField, cosine_abar

IMG, CH = 32, 3
DIM = CH * IMG * IMG


def to_model(x01):
    """[0,1] pixels, flat or image-shaped -> [-1,1] flat.  Resolution-agnostic."""
    x = np.asarray(x01, np.float64)
    return 2.0 * x.reshape(len(x) if x.ndim > 1 else 1, -1) - 1.0


def to_data(xm):
    """[-1,1] flat -> [0,1] flat, clipped.  Resolution-agnostic."""
    x = np.asarray(xm, np.float64)
    return np.clip((x.reshape(len(x) if x.ndim > 1 else 1, -1) + 1.0) / 2.0, 0.0, 1.0)


def _as_img(X, device, img=IMG):
    X = np.atleast_2d(np.asarray(X, np.float32))
    return torch.from_numpy(X).to(device).reshape(-1, CH, img, img)


# ── training ─────────────────────────────────────────────────────────────────
def train_ddpm(model, X01, labels, epochs=60, batch=128, lr=2e-4, wd=1e-6, p_drop=0.1,
               device="cuda", ema_decay=0.999, seed=0, log_every=200, u_min=1e-3, tf32=False):
    """Standard epsilon-prediction training with classifier-free-guidance dropout.

    Matches the paper's backbone recipe (AdamW, lr 2e-4, weight decay 1e-6, grad clip 1.0, null-token
    dropout p=0.1, EMA on the weights) on the cosine schedule this repository uses throughout, with
    `u ~ U(0,1)` standing in for their discrete t ~ U{1..T}.  Returns the EMA weights, which are what
    every downstream field is built from.
    """
    if tf32:
        # Ampere-class tensor cores for conv/matmul: ~2x on the CelebA backbone, which trains on 96k
        # 64x64 images.  OFF by default so the ColorMNIST numbers already in RESULTS keep their
        # exact arithmetic; the CelebA runs opt in.  The precision change is far inside the
        # backbone-to-backbone spread RESULTS 5p measured (up to 0.07 F1 between two runs of the
        # SAME command), so it is a speed knob, not a silent change of method.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sd = model.state_dict()
    ema = {k: v.detach().clone().float() for k, v in sd.items()}
    # fused EMA over the float tensors only; buffers (ints) are copied at the end.  Doing this with
    # _foreach_ ops rather than a Python loop over the state dict matters: the loop version runs
    # once per optimizer step and dominated step time at this model size.
    fkeys = [k for k, v in sd.items() if v.dtype.is_floating_point]
    ema_f = [ema[k] for k in fkeys]
    mod_f = [sd[k] for k in fkeys]
    img = int(getattr(model, 'img', IMG))
    # uint8 input is kept uint8 and converted per batch: the CelebA run trains on ~100k 64x64 images
    # and the float32 copy of that is 5 GB of host RAM, against 1.2 GB as uint8.
    u8 = np.asarray(X01).dtype == np.uint8
    if u8:
        # flat rows are HWC (that is how the loaders ravel them), so go through HWC before NCHW
        Xt = torch.from_numpy(np.ascontiguousarray(
            np.asarray(X01).reshape(-1, img, img, CH).transpose(0, 3, 1, 2)))
    else:
        Xt = torch.from_numpy(to_model(X01).astype(np.float32)).reshape(-1, CH, img, img)
    Y = torch.from_numpy(np.asarray(labels, np.int64))
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i0 in range(0, n, batch):
            idx = perm[i0:i0 + batch]
            x0 = Xt[idx].to(device)
            if u8:
                x0 = x0.float().div_(127.5).sub_(1.0)          # uint8 [0,255] -> [-1,1]
            c = Y[idx].to(device).clone()
            drop = torch.rand(len(c), device=device, generator=None) < p_drop
            c[drop] = model.null_id
            u = torch.rand(len(c), device=device).clamp_(u_min, 1.0)
            a = torch.from_numpy(cosine_abar(u.detach().cpu().numpy())).to(device).float()
            a = a.view(-1, 1, 1, 1)
            noise = torch.randn_like(x0)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            loss = torch.nn.functional.mse_loss(model(xt, u, c), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_f, ema_decay)
                torch._foreach_add_(ema_f, mod_f, alpha=1 - ema_decay)
            tot += float(loss); nb += 1; step += 1
            if log_every and step % log_every == 0:
                print(f"    ep {ep + 1} step {step}: loss {tot / max(nb, 1):.4f}", flush=True)
        print(f"  epoch {ep + 1}/{epochs}: loss {tot / max(nb, 1):.4f}", flush=True)
    for k, v in model.state_dict().items():          # non-float buffers, verbatim
        if not v.dtype.is_floating_point:
            ema[k] = v.detach().clone()
    return ema


class NeuralScoreField(BaseScoreField):
    """A trained UNet as a diffusion manifold.  See module docstring for the x-space convention.

    `branch` selects the conditioning: None -> the null token (the unconditional field), an int ->
    that class id, a string -> a concept registered in `self.concepts` (used by the lifelong bank,
    which maps a concept name to a class id and optionally an adapter to activate).
    """
    dim = DIM

    def __init__(self, model, device="cuda", batch=256, concepts=None, bank=None):
        self.model = model.to(device).eval()
        # resolution comes from the MODEL, not a module constant -- CelebA runs at 64x64 and a
        # class-level DIM bound at import time silently kept the sampler at 32x32
        self.img = int(getattr(model, "img", IMG))
        self.dim = CH * self.img * self.img
        self.device = device
        self.batch = int(batch)
        self.concepts = dict(concepts or {})     # name -> class id
        self.bank = bank                         # optional AdapterBank, see adapters.py

    def _cond(self, n, branch):
        """Resolve `branch` to a class-embedding row.

        None -> the null token (the frozen unconditional model).  An int -> that trained class.  A
        name registered in the adapter bank -> ALSO the null token, because `bank.activate` has
        temporarily written that concept's learned embedding into the null row; the concept's
        identity travels through the active adapter path and that row, not through a new table entry.
        """
        cid = self.model.null_id
        if branch is not None:
            if isinstance(branch, str):
                if self.bank is not None and branch in self.bank.concepts:
                    cid = self.model.null_id          # identity comes from the adapter + embedding
                elif branch in self.concepts:
                    cid = self.concepts[branch]
                else:
                    raise KeyError(f"unknown branch {branch!r}")
            else:
                cid = int(branch)
        return torch.full((n,), int(cid), device=self.device, dtype=torch.long)

    def _emb(self, branch):
        if self.bank is None or not isinstance(branch, str):
            return None
        return self.bank.embedding(branch) if branch in self.bank.concepts else None

    @torch.no_grad()
    def eps(self, X, u, branch=None, emb=None):
        """`emb` conditions on a raw embedding vector directly, bypassing the class table.

        That is how a concept learned by inversion is used: it lives in embedding space rather than
        as a table row, and conditioning on it is what lets the MODEL generate the concept instead of
        the sampler being steered toward an analytic distribution.
        """
        X = np.atleast_2d(np.asarray(X, np.float64))
        out = np.empty_like(X)
        ctx = self.bank.activate(branch) if (self.bank is not None) else _null_ctx()
        emb = self._emb(branch) if emb is None else emb
        with ctx:
            for i0 in range(0, len(X), self.batch):
                xb = _as_img(X[i0:i0 + self.batch], self.device, self.img)
                ub = torch.full((len(xb),), float(u), device=self.device)
                e = self.model(xb, ub, self._cond(len(xb), branch), emb_override=emb)
                out[i0:i0 + self.batch] = e.reshape(len(xb), -1).double().cpu().numpy()
        return out

    def loss_logp_proxy(self, X, u_grid=(0.1, 0.3, 0.5, 0.7), branch=None, seed=0):
        """A stand-in for `logp`, which a UNet does not have in closed form.

        Returns the NEGATIVE mean epsilon-prediction error over a grid of noise levels -- the same
        quantity the paper uses to rank trained classes for its top-k baseline ("score every trained
        class by the average DDPM loss of x_q under that conditional").  It is monotone in model fit,
        which is all the forgetting curves in the lifelong experiments need: higher is better.
        """
        rng = np.random.default_rng(seed)
        X = np.atleast_2d(np.asarray(X, np.float64))
        tot = 0.0
        for u in u_grid:
            a = float(cosine_abar(u))
            e = rng.standard_normal(X.shape)
            xt = np.sqrt(a) * X + np.sqrt(1 - a) * e
            tot += float(np.mean((self.eps(xt, u, branch) - e) ** 2))
        return -tot / len(u_grid)


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def load_field(path, device="cuda", batch=256, **kw):
    """Rebuild a NeuralScoreField from a checkpoint written by experiments/n1_train.py."""
    from .unet import UNet
    ck = torch.load(path, map_location=device, weights_only=False)
    model = UNet(n_classes=ck["n_classes"], **ck.get("arch", {}))
    model.load_state_dict(ck["ema"])
    return NeuralScoreField(model, device=device, batch=batch, concepts=ck.get("concepts"), **kw)
