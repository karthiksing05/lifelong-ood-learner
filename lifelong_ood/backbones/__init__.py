"""Backbones: the noise schedule, the flat<->image layout, the model loaders and THE one sampler.

Nothing is imported eagerly here on purpose.  The old package roots pulled the whole analytic Cobweb
track into every import (`lifelong_ood/__init__.py` imported tree, backend, score, compose and lifelong;
`neural/__init__.py` imported adapters, inversion and lifelong_neural), so a script that wanted one
adapter class executed six modules it never used.  Import the module you need.
"""
