"""Compatibility shim.  The Cobweb-form update was split on 2026-09-25 into
backbones/sampler.py (ddim_sample), backbones/reads.py (eps_profile, candidate_losses),
operators/anchor.py (eps_loss, anchor_loss, sq_grad, topk_mask, delta_stats) and
experts/lora.py (GatedLoRA, KeyedLoRA and their helpers).  The legacy fixtures in experiments/legacy
and tests/test_keyed_lora.py import from here; new code imports the modules directly.

The theory this code implements is docs/COBWEB_UPDATE.md: the ideal denoiser's one-instance update is
Cobweb's incremental update with a soft responsibility, and a network realises it as a gated, anchored
weight update.
"""
from ..backbones.layout import to_flat, to_img  # noqa: F401
from ..backbones.reads import candidate_losses, eps_profile  # noqa: F401
from ..backbones.sampler import _cond, _gate, ddim_sample  # noqa: F401
from ..backbones.schedule import abar_of, cosine_abar  # noqa: F401
from ..experts.lora import (GatedLoRA, KeyedLoRA, add_node, drop_node, gate_on,  # noqa: F401
                            inject_gated_lora, inject_keyed_lora, keys_on)
from ..operators.anchor import anchor_loss, delta_stats, eps_loss, sq_grad, topk_mask  # noqa: F401
