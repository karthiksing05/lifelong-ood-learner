"""lifelong-ood-learner: lifelong, out-of-distribution learning as the composition of seen prototypes and
the folding of genuinely new ones into a diffusion model's own manifold.

Nothing is imported here on purpose.  This root used to import the analytic Cobweb track (tree, backend,
score, compose, lifelong) eagerly, so every script paid for six modules it never used and nothing could
be archived.  Import the module you need: `lifelong_ood.backbones`, `lifelong_ood.neural.cobweb_update`,
`lifelong_ood.evals`, `lifelong_ood.ledger`.
"""
