"""lifelong-ood-learner: lifelong learning as repeated composition of hierarchical prototypes
on an analytic diffusion manifold.  See README.md for the idea and experiments/ for the evidence."""
from .tree import GaussianCobweb
from .backend import make_tree, CobwebPrivateTree, merge_snapshots
from .score import ScoreField, cosine_abar
from .compose import poe_teacher, candidates_tree, candidates_ascent, teacher_eps, cfg_weight
from .lifelong import LifelongLearner

__all__ = ["GaussianCobweb", "make_tree", "CobwebPrivateTree", "merge_snapshots", "ScoreField",
           "cosine_abar", "poe_teacher", "candidates_tree", "candidates_ascent", "teacher_eps",
           "cfg_weight", "LifelongLearner"]
