"""Does the pretrained CelebA-HQ model actually work behind our interface, and is the added
conditioning slot a true no-op when unused?

Three things have to hold before anything is built on this: sampling through OUR DDIM loop and OUR
u-grid produces faces rather than noise (the schedules are reconciled, not merely both present);
setting the conditioning vector to zero reproduces the pretrained model exactly; and a nonzero
vector actually changes the output, so there is something to learn into.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import RESULTS
from lifelong_ood.evals import save_grid
from lifelong_ood.neural.pretrained import load_pretrained

dev = "cuda" if torch.cuda.is_available() else "cpu"
field, img = load_pretrained(device=dev, batch=2)
print(f"  loaded: {img}x{img}, emb_dim {field.emb_dim}, "
      f"{sum(p.numel() for p in field.model.parameters())/1e6:.1f}M params", flush=True)

X = np.random.default_rng(0).standard_normal((2, field.dim)) * 0.5
e0 = field.eps(X, 0.5)
ez = field.eps(X, 0.5, emb=np.zeros(field.emb_dim, np.float32))
er = field.eps(X, 0.5, emb=np.random.default_rng(1).standard_normal(field.emb_dim).astype(np.float32) * 0.5)
print(f"  zero-vector conditioning vs unconditional: max|delta| {np.abs(ez-e0).max():.3e} "
      f"(must be 0 -- the slot is additive)")
print(f"  random-vector conditioning vs unconditional: max|delta| {np.abs(er-e0).max():.3e} "
      f"(must be > 0 -- there is something to learn into)")

S = field.sample(n=4, steps=50, seed=0)
S01 = np.clip((S + 1) / 2, 0, 1).astype(np.float32)
px = S01.reshape(4, 3, img, img).transpose(0, 2, 3, 1).reshape(4, -1)
save_grid(px, os.path.join(RESULTS, "n15_pretrained_uncond.png"), ncol=4,
          title=f"google/ddpm-celebahq-256 through OUR DDIM loop ({img}x{img})")
print(f"  sampled 4 images: mean {S01.mean():.3f}, std {S01.std():.3f}, "
      f"range [{S01.min():.2f}, {S01.max():.2f}]")
print(f"  wrote {RESULTS}/n15_pretrained_uncond.png")
ok = np.abs(ez - e0).max() < 1e-9 and np.abs(er - e0).max() > 1e-3 and 0.05 < S01.std() < 0.5
print(f"\npretrained wrapper: {'ok' if ok else 'CHECK THE ABOVE'}")
