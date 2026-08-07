"""Green cast on otherwise correct faces: is it the sampler or the display path?

Per-channel statistics say which.  If R/G/B means are comparable the model output is fine and the
grid writer is mis-ordering; if green is genuinely high the sampling math is still wrong.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import RESULTS
from lifelong_ood.neural.pretrained import load_pretrained

dev = "cuda" if torch.cuda.is_available() else "cpu"
field, img = load_pretrained(device=dev, batch=2)
S = field.sample(n=2, steps=50, seed=0)
S01 = np.clip((S + 1) / 2, 0, 1).astype(np.float32).reshape(2, 3, img, img)
for c, nm in enumerate("RGB"):
    print(f"  channel {nm}: mean {S01[:, c].mean():.3f}  std {S01[:, c].std():.3f}")

# write with PIL directly, bypassing save_grid entirely
from PIL import Image
arr = (S01[0].transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(arr).save(os.path.join(RESULTS, "n15_direct_pil.png"))
print(f"  wrote {RESULTS}/n15_direct_pil.png (PIL, CHW->HWC, no save_grid)")

# and the diffusers reference pipeline, as the ground truth for what this model should produce
from diffusers import DDPMPipeline
pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(dev)
out = pipe(batch_size=1, num_inference_steps=50,
           generator=torch.Generator(device=dev).manual_seed(0)).images[0]
out.save(os.path.join(RESULTS, "n15_diffusers_reference.png"))
ref = np.asarray(out, np.float32) / 255
for c, nm in enumerate("RGB"):
    print(f"  diffusers reference channel {nm}: mean {ref[:, :, c].mean():.3f}")
print(f"  wrote {RESULTS}/n15_diffusers_reference.png")
