"""Build the CelebA attribute probes once, before the backbones need them.

Three n19 runs finish within minutes of each other and every one of them fits these probes at its
evaluation step under the same cache key.  Building them here means all three (and n20/n21/n22)
load rather than write, which removes the race outright rather than relying on the atomic rename to
survive it.

Also prints each probe's validation accuracy, which is the number that decides whether an attribute
is scoreable at all -- an attribute no probe can read measures the probe, not the generator, and
`attr_accuracy` excludes those at 0.65.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE                                                # noqa: E402
from n19_celeba_train import build_probes                               # noqa: E402

from lifelong_ood.data.celeba import IMG_CELEBA, load_celeba_comp            # noqa: E402

if __name__ == "__main__":
    seed = int(os.environ.get("SEED", 0))
    data = load_celeba_comp(img=IMG_CELEBA, min_count=60, n_holdout=24, seed=seed,
                            cache_dir=CACHE)
    print(f"[probes] {len(data['X8'])} real images, seed {seed}", flush=True)
    probes = build_probes(data, seed, 30)
    order = sorted(probes["_attrs"], key=lambda a: -probes[f"{a}_val_acc"])
    print(f"[probes] validation accuracy on real held-out images:")
    for a in order:
        v = probes[f"{a}_val_acc"]
        print(f"    {a:20s} {v:.3f} {'' if v >= 0.65 else '  <- excluded from scoring'}")
    n_ok = sum(1 for a in order if probes[f"{a}_val_acc"] >= 0.65)
    print(f"[probes] {n_ok}/{len(order)} attributes are scoreable")
