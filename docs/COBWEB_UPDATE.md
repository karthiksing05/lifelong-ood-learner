# A diffusion model's update IS a Cobweb update: the correspondence, and the weight update it implies

Written 2026-09-25. The claim this document makes precise: **absorbing one new instance into a
diffusion model's ideal denoiser is exactly Cobweb's incremental update, performed at every noise level
at once, with Cobweb's hard count-share replaced by a soft posterior responsibility.** It then derives
what that implies for a *parametric* network — a sparse, anchored weight update with a principled
parameter selection rule — which is what `lifelong_ood/neural/cobweb_update.py` implements and
`experiments/c1_cobweb_update.py` tests. `tests/test_cobweb_correspondence.py` checks every identity
below numerically.

The exact results (§1–§4) hold for the Bayes-optimal denoiser of an empirical distribution. The
parametric results (§5–§6) are first-order. §7 says what is not covered.

---

## 0. Notation

Forward process at noise level t: `x_t = a x_0 + s ε`, `ε ~ N(0, I)`, with `a = sqrt(ābar(t))`,
`s = sqrt(1 - ābar(t))` (this repository's cosine schedule; nothing below depends on which).

Data: n instances `x_1 … x_n`, the empirical distribution `p_n = (1/n) Σ_i δ(x_i)`. For a
class-conditional model, everything below holds per conditioning value.

Kernel weights: `k_i(x_t) = N(x_t ; a x_i, s² I)`, their sum `K_n = Σ_{i≤n} k_i`.

---

## 1. The ideal denoiser is a responsibility-weighted mean

**Proposition 1.** The MMSE denoiser under `p_n` is

    D_n(x_t) = E[x_0 | x_t] = Σ_i w_i(x_t) x_i,        w_i = k_i / K_n,

and the ideal ε-predictor is `ε*_n(x_t) = (x_t − a D_n(x_t)) / s`. The posterior covariance is
`S_n(x_t) = Σ_i w_i (x_i − D_n)(x_i − D_n)ᵀ`.

*Proof.* The prior on x_0 is `p_n`; the likelihood of x_t given `x_0 = x_i` is `k_i`. Bayes gives
posterior mass `w_i` on `x_i`, whose mean and covariance are the expressions above. The ε relation is
`x_t = a x_0 + s ε` taken in conditional expectation. ∎

So at every noise level the denoiser is a **soft clustering**: `w_i(x_t)` is the posterior probability
that `x_t` was generated from instance i — a responsibility, exactly as in a mixture model.

---

## 2. Adding one instance is Cobweb's incremental update

**Proposition 2.** Adding `x_{n+1}` changes the denoiser, everywhere, by

    D_{n+1}(x_t) = D_n(x_t) + r(x_t) · (x_{n+1} − D_n(x_t)),       r = k_{n+1} / (K_n + k_{n+1}),

    S_{n+1}(x_t) = (1 − r) S_n(x_t) + r (1 − r) (x_{n+1} − D_n)(x_{n+1} − D_n)ᵀ,

    ε*_{n+1}(x_t) = ε*_n(x_t) − (a / s) · r(x_t) · (x_{n+1} − D_n(x_t)).

*Proof.* The new posterior is a two-component mixture: mass `1 − r` on the old posterior (mean
`D_n`, covariance `S_n`) and mass `r` on the point `x_{n+1}`, because
`K_n / (K_n + k_{n+1}) = 1 − r`. The mean of the mixture is the first line. The covariance follows
from the law of total variance: the within-component part is `(1 − r) S_n + r · 0`, the
between-component part is `(1 − r) r (x_{n+1} − D_n)(x_{n+1} − D_n)ᵀ`. The ε line substitutes into
Proposition 1. ∎

**Cobweb's continuous node update** (COBWEB/3, and `GaussianCobweb` in `lifelong_ood/tree.py`), for a
node with count c, mean μ and variance σ² receiving instance x:

    μ' = μ + (x − μ) / (c + 1)
    σ'² = (1 − 1/(c+1)) σ² + (1/(c+1)) (1 − 1/(c+1)) (x − μ)²          (Welford's update)

**These are the same equations with `r = 1/(c+1)`.** The difference is only in what decides the share:

| | Cobweb | diffusion (ideal denoiser) |
|---|---|---|
| share of the new instance | `1/(c+1)` — a HARD count, since x is a member of the node | `r(x_t)` — a SOFT posterior responsibility |
| which "nodes" update | only those on x's root-to-leaf path | every `(x_t, t)`, weighted by `r(x_t, t)` |
| the hierarchy | an explicit tree | the noise levels (§3) |

---

## 3. The noise level is the depth of the tree

Write `r` out:

    r(x_t) = 1 / (1 + Σ_{i≤n} exp(−(‖x_t − a x_i‖² − ‖x_t − a x_{n+1}‖²) / (2 s²))).

**Proposition 3 (limits).**

1. **High noise (the root).** As `a/s → 0` for fixed data, every exponent → 0 on any bounded set of
   x_t, so `r → 1/(n+1)` uniformly: the update is the root node's running-mean step, the same
   minuscule `1/(n+1)` share Cobweb's root receives.
2. **Low noise (the leaves).** As `s → 0`, `r → 1` wherever `x_t / a` is strictly closer to `x_{n+1}`
   than to every other instance, and `r → 0` wherever some other instance is strictly closer: the new
   instance owns exactly its own cell — a leaf containing only itself.
3. **Intermediate noise.** Near `a x_{n+1}`, `r ≈ 1/(1 + n_eff)` with
   `n_eff = Σ_i exp(−a² ‖x_i − x_{n+1}‖² / (2 s²))` (to first order in the cross term) — the
   number of instances within kernel range of the new one at bandwidth `s/a`. That is the count of
   the node the new instance joins at that scale.

*Proof.* Direct from the formula for r. ∎

So the Gaussian scale-space of the data — the sequence of noisy marginals — plays the role of
Cobweb's depth: **coarse levels are nodes with large counts and receive small updates; fine levels
are nodes with small counts and receive large ones.** This is the precise form of the claim in our
group's ACS paper that a diffusion model's noisy marginals are a Cobweb hierarchy, now extended from
the *structure* to the *update*.

---

## 4. Locality: an update does not touch the broader hierarchy

**Proposition 4.** For every `x_t`,

    ‖D_{n+1}(x_t) − D_n(x_t)‖ ≤ r(x_t) · ‖x_{n+1} − D_n(x_t)‖.

*Proof.* Take the norm of Proposition 2's first line. ∎

Consequences, for an old concept whose instances produce inputs x_t:

* where the old concept and the new instance are **separated at that noise level** (the old inputs
  lie many bandwidths from `a x_{n+1}`), `r` is exponentially small and the old concept's denoiser does
  not move;
* where they are **merged** (high noise), the move is `O(1/(n+1))` — the root's minuscule share.

This is the formal content of "a sparse update changes remote, minuscule functionality without
touching the broader hierarchy": in the ideal model, interference with what is already known is
bounded by the new instance's responsibility, which is small exactly where the old knowledge lives.

---

## 5. From function space to weights

A network cannot be edited per input. For a parametric denoiser `D_θ` trained to the minimum `θ_n` of

    L_n(θ) = (1/n) Σ_i ℓ_i(θ),     ℓ_i(θ) = E_{t,ε} λ(t) ‖D_θ(a x_i + s ε, t) − x_i‖²,

adding `x_{n+1}` gives `L_{n+1} = (n/(n+1)) L_n + (1/(n+1)) ℓ_{n+1}`.

**Proposition 5 (first-order).** One Newton step from `θ_n` is

    θ_{n+1} ≈ θ_n − (1/(n+1)) · H^{-1} ∇ℓ_{n+1}(θ_n),       H = ∇² L_n(θ_n) ≈ G = E[Jᵀ J],

with J the Jacobian of `D_θ` in θ (Gauss–Newton, since the loss is squared error). The step is a
natural-gradient step whose size is **`1/(n+1)` — Cobweb's step size**. For a model linear in its
parameters it is exact (it is recursive least squares). The function change it induces at an input z is

    ΔD(z) ≈ (1/(n+1)) · E_{z'}[ K_G(z, z') (x_{n+1} − D_n(z')) ],    K_G(z, z') = J(z) G^{-1} J(z')ᵀ,

the expectation over the new instance's noisy inputs z'. **This is Proposition 2 with the
responsibility `r(z)` replaced by the preconditioned tangent kernel `K_G`**: an old input z moves in
proportion to how much it shares, through the network's own geometry, with the new instance's inputs.

*Proof sketch.* Expand `L_{n+1}` to second order around `θ_n`, where `∇L_n = 0`; minimize the
quadratic. The function change is `J(z) Δθ`. ∎

---

## 6. Which parameters may move: the sparse support

Restrict the update to a parameter subset S. With the old data's curvature approximated diagonally
by the Fisher `F` (so `G ≈ n · diag(F)` for n old instances) and the new loss linearized, the combined
objective for a step Δ supported on S is

    J(Δ) ≈ ℓ_new(θ) + gᵀΔ + ½ Δᵀ (n · diag F) Δ,       g = ∇ℓ_new(θ).

**Proposition 6.** The minimizer on S is `Δ_j = −g_j / (n F_j)` for j in S, and it lowers J by

    ΔJ = Σ_{j ∈ S} g_j² / (2 n F_j).

Hence, among all supports of size k, **the top-k parameters by `g_j² / F_j` give the largest
improvement** in the new concept's fit per unit of damage to the old ones.

*Proof.* The objective separates across coordinates under the diagonal approximation; each coordinate
is a one-dimensional quadratic with minimum value `−g_j² / (2 n F_j)`. Maximizing a sum of
non-negative per-coordinate gains over k coordinates takes the k largest. ∎

This is the mask `cobweb_update.py` uses, `score_j = E[g_new_j²] / (F_old_j + δ)`, with two
practical departures:

* `E[g²]` over single examples instead of `(E g)²`, so a parameter that the new instances pull in
  *different* directions is not mistaken for an unimportant one;
* `δ > 0`, because `F_j = 0` for parameters no old concept uses, where the diagonal model promises a
  free lunch that the linearization cannot deliver. `δ` caps that promise.

And one addition the quadratic model does not contain: **anchoring**. The quadratic model assumes
the old loss stays near its minimum; an explicit anchor loss — the model's current prediction held
to its pre-update prediction on the model's *own* samples of old concepts — enforces the `r ≈ 0`
half of Proposition 2 directly instead of trusting a diagonal curvature to do it.

**The stopping rule.** Proposition 2's update is complete when the denoiser fits the new instance as
well as it fits a trained one. Operationally: stop when the new concept's denoising loss on its own
held-back instances reaches the level a trained class reaches under the same conditioning scheme.
This is a design principle (update only as much as the evidence requires), not a theorem; it is what
makes the update's size track how novel the concept still is.

---

## 7. What this does not cover

1. **The ideal denoiser memorizes.** Propositions 1–4 are exact for `p_n`, whose optimal sampler
   reproduces the training set. Real networks generalize because of their inductive biases. For a
   *local, translation-equivariant* denoiser — the model Kamb & Ganguli (2024) show reproduces
   convolutional diffusion models' creativity as patch mosaics — the same Proposition 2 holds **per
   patch**, with responsibilities over patches of training images instead of whole images. That is a
   Cobweb over **parts**, and it is where compositional generalization comes from in that account.
   It is the natural bridge between this document and the composition half of the thesis, and it is
   not yet tested here.
2. **§5–§6 are first-order**, and §6 uses a diagonal curvature. The experiment is what says whether
   the approximation is good enough.
3. **Nothing here makes preservation exact.** The adapter bank's bit-identity came from isolating
   parameters. This update trades it for *bounded* interference (Proposition 4 in the ideal model,
   anchoring plus the mask in the network), which is why C1 reports drift per noise level rather than a
   bit-identity flag.
4. **The update rule is related to known continual-learning machinery** — online Laplace / K-FAC
   (Ritter et al., 2018), EWC (a diagonal special case), and importance-masked fine-tuning. What this
   document adds is the correspondence: why, for a diffusion model, that machinery implements Cobweb's
   incremental update, and which of its knobs correspond to which of Cobweb's quantities.
