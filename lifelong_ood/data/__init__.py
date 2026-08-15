"""Dataset loaders, and the one contract they all satisfy.

Four modules that were scattered at the top level of the package, grouped because they are the same
kind of thing and because a reader looking for "how do I add a dataset" should find them together:

    datasets.py   ColorMNIST -- three named factors, disjoint pixels, the calm explanation
    celeba.py     CelebA -- fourteen binary attributes, factors that overlap spatially
    shapes3d.py   Shapes3D -- six factors as an exact Cartesian product, split into composition
                  and nuisance because a full 6-tuple has exactly ONE image
    factored.py   the CONTRACT: X, factors, names, sizes, slot, seen, img -- what every experiment
                  actually consumes, and what makes a new dataset a loader rather than a refactor

`load_factored(name, ...)` is the entry point; adding a dataset means registering a loader there.

Re-exported at this level so `from lifelong_ood.data import load_colormnist` works, and the old
`from lifelong_ood.datasets import ...` paths are rewritten rather than shimmed -- a compatibility
shim would leave two ways to import the same thing, which is the mess this move exists to remove.
"""
from .datasets import *                                       # noqa: F401,F403
from .factored import (as_factored, decode_slot, encode_slot,  # noqa: F401
                       holdout_mask, load_factored)
