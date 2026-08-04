"""Neural backbone: a trained DDPM standing in for the Cobweb tree, behind the same interface.

Import lazily -- torch is an optional dependency of this repository, and the analytic track never
needs it.
"""
from .unet import UNet
from .ddpm import NeuralScoreField, train_ddpm, load_field, to_model, to_data, DIM
from .adapters import HierarchicalAdapterBank, ConceptTree, fingerprint
from .inversion import learn_embedding, sample_embedding, teacher_init_alpha
from .lifelong_neural import NeuralLifelongLearner

__all__ = ["UNet", "NeuralScoreField", "train_ddpm", "load_field", "to_model", "to_data", "DIM",
           "HierarchicalAdapterBank", "ConceptTree", "fingerprint", "NeuralLifelongLearner",
           "learn_embedding", "sample_embedding", "teacher_init_alpha"]
