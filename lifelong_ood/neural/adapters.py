"""HierarchicalAdapterBank — how new compositions are added to a NEURAL manifold, forever, safely.

This is the contribution.  The poe-diffusion paper distils each discovered composition into its own
`(LoRA, c_new)` pair on a frozen UNet and stops there: "A separate (theta_LoRA, c_new) pair is
distilled per OOD query."  Nothing accumulates, nothing is shared between concepts, and nothing is
reusable by a later composition.  Scaling that to a lifelong stream gives you either N independent
adapters that never help each other, or one adapter trained sequentially that forgets.

The page we take from Cobweb is its *insertion* discipline, not its statistics: concepts are
organized in an incrementally grown, INSERT-ONLY hierarchy, and a concept's effective adapter is the
SUM OF THE DELTAS ALONG ITS ROOT-TO-LEAF PATH.

    root delta            shared by everything ever consolidated
      +- "warm colours on dark" delta      shared by that neighbourhood
           +- "cyan on deep_red" leaf delta        this concept alone

Two properties follow, and they are the two things a lifelong learner has to have at once:

  NO FORGETTING, structurally.  Inserting a concept creates a NEW leaf and trains ONLY that leaf;
  every ancestor delta is frozen the moment it is created.  An existing concept's path is therefore
  bit-identical before and after any number of later insertions, so its effective adapter -- and
  hence everything it generates -- cannot change.  This is the same guarantee the analytic backbone
  gets from mixture additivity, transplanted to a neural one, and it is exact rather than empirical.

  FORWARD TRANSFER, structurally.  A new leaf inherits its ancestors' accumulated delta as its
  starting point, so a concept inserted near relatives begins where they left off and only has to
  learn its own residual.  That is what makes the tenth consolidation cheaper than the first, and it
  is what the flat per-query adapters of the paper cannot do.

Placement uses a Cobweb-style descent over the concepts' PoE teacher means: descend to the best
child while descending still helps, then attach.  Insert-only, because a merge or split would
restructure existing paths and destroy the guarantee above.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..score import cosine_abar


# ── LoRA modules that sum a PATH of deltas ───────────────────────────────────
class PathLoRA(nn.Module):
    """Wraps a frozen Linear/Conv2d and adds the sum of the low-rank deltas on the ACTIVE path.

    The active path is shared mutable state on the bank (a list of node ids), so switching concepts
    is just switching that list -- no re-injection, no weight copying.
    """

    def __init__(self, base, rank, state):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.state = state                     # {"path": [node_id, ...]}
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.is_conv = isinstance(base, nn.Conv2d)

    def add_node(self, nid, device, generator=None):
        nid = str(nid)
        if nid in self.A:
            return
        b = self.base
        if self.is_conv:
            k = b.kernel_size
            a = torch.empty(self.rank, b.in_channels, *k, device=device)
            bb = torch.zeros(b.out_channels, self.rank, 1, 1, device=device)
        else:
            a = torch.empty(self.rank, b.in_features, device=device)
            bb = torch.zeros(b.out_features, self.rank, device=device)
        nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        self.A[nid] = nn.Parameter(a)
        self.B[nid] = nn.Parameter(bb)         # zero -> a fresh node starts as an exact no-op

    def forward(self, x):
        out = self.base(x)
        for nid in self.state["path"]:
            nid = str(nid)
            if nid not in self.A:
                continue
            if self.is_conv:
                b = self.base
                h = F.conv2d(x, self.A[nid], None, b.stride, b.padding, b.dilation)
                out = out + F.conv2d(h, self.B[nid])
            else:
                out = out + F.linear(F.linear(x, self.A[nid]), self.B[nid])
        return out


    @torch.no_grad()
    def drop_node(self, nid):
        """Delete node `nid`'s delta WITHOUT folding it into the base weight.

        The opposite of `merge_node`: that one absorbs the delta and keeps its effect forever, this
        one discards it and reclaims the parameters.  Because the base weight is never written to,
        dropping a node cannot perturb any other node -- which is what makes concept deletion exact
        rather than approximate.  Caller is responsible for not dropping a SHARED ancestor.
        """
        nid = str(nid)
        if nid not in self.A:
            return 0
        freed = self.A[nid].numel() + self.B[nid].numel()
        del self.A[nid]
        del self.B[nid]
        return int(freed)

    @torch.no_grad()
    def merge_node(self, nid):
        """Fold node `nid`'s delta into the base weight and delete the node.

        This is what lets the adapter be SCAFFOLDING rather than storage: the update is learned in a
        low-rank subspace, then absorbed, so the parameter count returns to the base model's and the
        next concept starts from a clean slate.  Nothing accumulates.

        The fold is exact for both layer types.  For Linear the delta is `B @ A`.  For Conv2d the
        path is `conv(x, A, stride, padding, dilation)` followed by a 1x1 `conv(h, B)`; convolution
        is linear in the kernel and a 1x1 only mixes output channels, so the single equivalent kernel
        is `W[o,i,:,:] = sum_r B[o,r] * A[r,i,:,:]`, applied with the base's own stride/padding.
        """
        nid = str(nid)
        if nid not in self.A:
            return False
        A, B = self.A[nid].detach(), self.B[nid].detach()
        if self.is_conv:
            delta = torch.einsum("or,rikl->oikl", B[:, :, 0, 0], A)
        else:
            delta = B @ A
        self.base.weight.data.add_(delta.to(self.base.weight.dtype))
        del self.A[nid]
        del self.B[nid]
        return True


def inject(model, rank, state, targets=("conv", "linear"), exclude=()):
    """Replace every Conv2d/Linear in `model` with a PathLoRA wrapper.  -> list of wrappers.

    `exclude` skips children whose attribute name contains any of the given strings.  It exists for
    pretrained backbones, where "adapt every Linear" is not the neutral choice it sounds like: in
    `google/ddpm-celebahq-256`, **32 of the 58 Linear modules are `time_emb_proj`** and two more are
    the time embedding's own MLP, so adapting linears means overwhelmingly adapting the path that
    tells the model WHAT NOISE LEVEL IT IS AT.  Perturbing that in 34 places corrupts denoising
    globally -- the same failure as writing to the time embedding directly, arrived at by a different
    route -- and it is what produced melted faces and collapsing diversity under guidance.  Excluding
    it leaves the 24 attention projections, which is where the personalization literature adapts.
    """
    wrapped = []
    ex = tuple(exclude or ())

    def walk(mod):
        for name, child in list(mod.named_children()):
            if any(e in name for e in ex):
                continue
            is_c = isinstance(child, nn.Conv2d) and "conv" in targets
            is_l = isinstance(child, nn.Linear) and "linear" in targets
            if is_c or is_l:
                w = PathLoRA(child, rank, state)
                setattr(mod, name, w)
                wrapped.append(w)
            else:
                walk(child)
    walk(model)
    return wrapped


# ── the insert-only concept hierarchy ────────────────────────────────────────
class ConceptTree:
    """Insert-only hierarchy over concept means.  `insert(mu)` -> (path node ids, new leaf id).

    Descent is Cobweb-flavoured: at each node, go to the best-matching child only while doing so
    beats staying put by `margin` nats, then attach a new leaf.  `branch_max` caps fan-out so the
    tree actually forms levels instead of a star, which is what creates shared ancestors at all.
    """

    def __init__(self, var=0.1, margin=0.0, branch_max=4, max_depth=3, flat=False, rule=None,
                 prior_var=0.05):
        self.var, self.margin = float(var), float(margin)
        self.branch_max, self.max_depth = int(branch_max), int(max_depth)
        self.flat = bool(flat)             # ablation: every concept hangs directly off the root
        # `rule` selects the placement criterion.  **"cu" is the default** -- Cobweb's category
        # utility, insert-only -- because N39 measured what "ll" actually builds: a CATERPILLAR,
        # 31 of 40 concepts strung along one prefix, at leaf purity 0.27 against an oracle's 0.66.
        # Insert-only CU reaches 0.60, i.e. 98% of full Cobweb's 0.61 with none of its restructuring.
        #
        # State the trade, because it is not a free win.  The rule comparison (`rulecmp.sbatch`,
        # 2 experiments x 2 rules x 3 seeds, env-set so the experiment code is byte-identical
        # between arms) found the END-TASK effect small and MIXED:
        #
        #     n9 showdown   ours F1 0.606 -> 0.633,  joint 0.539 -> 0.556      (cu better)
        #     n5 class-inc  A    0.845 -> 0.826,  params +6.3%                 (cu worse)
        #     every non-bank baseline byte-identical under both               (the control held)
        #
        # So cu is adopted for tree SHAPE, which is what the hierarchy claims are about, not for a
        # performance gain there is no evidence for.  Tables collected under "ll" and not yet re-run
        # are marked as such in RESULTS rather than silently reattributed.
        #
        # LIFELONG_TREE_RULE overrides the default process-wide.  It exists so the WHOLE suite --
        # every test and every experiment, including the ones with no --rule flag -- can be re-run
        # under a candidate criterion without editing call sites one at a time, which is how a
        # comparison ends up accidentally mixing rules.  An explicit `rule=` argument still wins.
        # `rule=None` (the default) consults the env; an EXPLICIT rule always wins, including
        # rule="ll" -- a caller that names the shipped rule must get it even under an override,
        # or the override silently rewrites the control arm of its own comparison.
        self.rule = str(rule) if rule is not None else os.environ.get("LIFELONG_TREE_RULE", "cu")
        self.prior_var = float(prior_var)
        self.mean = {0: None}                  # node id -> running mean (None = root, matches all)
        self.count = {0: 0}
        self.sq = {0: None}                    # running sum of squares; only "cu" needs it
        self.children = {0: []}
        self._next = 1

    def _ll(self, nid, mu):
        m = self.mean[nid]
        if m is None:
            return 0.0
        return -0.5 * float(np.mean((mu - m) ** 2)) / self.var

    def _new(self, mu, parent):
        nid = self._next
        self._next += 1
        self.mean[nid] = np.array(mu, np.float64)
        self.count[nid] = 1
        self.sq[nid] = np.asarray(mu, np.float64) ** 2
        self.children[nid] = []
        self.children[parent].append(nid)
        return nid

    _SQ4PI = 1.0 / (2.0 * np.sqrt(np.pi))

    def _stats(self, nid):
        """(n, sum, sumsq) for a node, or None for the empty root."""
        if self.mean.get(nid) is None:
            return None
        n = max(self.count.get(nid, 1), 1)
        return (float(n), self.mean[nid] * n, self.sq[nid])

    def _score(self, st):
        n, s, ss = st
        m = s / n
        var = np.maximum(ss / n - m * m, 0.0) + self.prior_var
        return self._SQ4PI * float(np.mean(1.0 / np.sqrt(var)))

    def _cu(self, stats):
        """Category utility of a partition, given hypothetical child (n, sum, sumsq) triples."""
        stats = [st for st in stats if st is not None]
        if not stats:
            return -np.inf
        n_all = sum(st[0] for st in stats)
        if n_all <= 0:
            return -np.inf
        base = self._score((n_all, sum(st[1] for st in stats), sum(st[2] for st in stats)))
        return sum((st[0] / n_all) * (self._score(st) - base) for st in stats) / len(stats)

    def _descend_cu(self, mu):
        """Cobweb's criterion, insert-only: at each node choose between hosting `mu` in the best
        child and starting a new child here, by CATEGORY UTILITY.

        Why this exists.  The `ll` rule descends to the NEAREST child and, when a node already has
        `branch_max` children, descends *regardless of fit* -- a cap meant to create levels instead
        creates a chain, and N38 measured the result: 31 of 40 concepts on one path prefix, 3 shared
        nodes, branching 10.  CU is a partition-level score rather than a point-to-point distance, so
        it can prefer "this belongs in its own group" over "this is closest to that group", which is
        what produces balance.  N39 measures both against the real `GaussianCobweb` and against an
        offline oracle.

        Cobweb's MERGE and SPLIT are still absent and still cannot be added: they restructure nodes
        that already own sealed adapter parameters (§24f), so this is the host-or-create half only.
        """
        path, cur = [0], 0
        x2 = mu * mu
        for _ in range(self.max_depth):
            kids = list(self.children[cur])
            if not kids:
                break
            base = [self._stats(k) for k in kids]
            if any(b is None for b in base):
                break
            add = lambda st: (st[0] + 1.0, st[1] + mu, st[2] + x2)
            best, best_cu = None, -np.inf
            for i, k in enumerate(kids):
                cu = self._cu([add(b) if j == i else b for j, b in enumerate(base)])
                if cu > best_cu:
                    best, best_cu = k, cu
            cu_new = self._cu(base + [(1.0, mu, x2)])     # start a new child HERE
            if cu_new >= best_cu:
                break
            path.append(best)
            cur = best
        return path

    def _descend(self, mu):
        """Where would `mu` attach?  Returns the path of EXISTING nodes, root first.  No mutation,
        so a whole batch can be placed against one consistent snapshot of the tree."""
        if self.flat:
            return [0]
        if self.rule == "cu":
            return self._descend_cu(mu)
        path, cur = [0], 0
        for _ in range(self.max_depth):
            kids = list(self.children[cur])
            inner = [k for k in kids if self.children[k]] or kids
            if not inner:
                break
            best = max(inner, key=lambda k: self._ll(k, mu))
            if self._ll(best, mu) < self._ll(cur, mu) + self.margin and len(kids) < self.branch_max:
                break
            path.append(best)
            cur = best
        return path

    def _touch(self, path, mu):
        """Ancestors track a running mean for PLACEMENT only; their adapter deltas stay frozen."""
        x2 = mu * mu
        for nid in path:
            if self.mean[nid] is None:
                self.mean[nid] = mu.copy()
                self.count[nid] = 1
                self.sq[nid] = x2.copy()
            else:
                self.count[nid] += 1
                self.mean[nid] += (mu - self.mean[nid]) / self.count[nid]
                self.sq[nid] = x2 if self.sq.get(nid) is None else self.sq[nid] + x2

    def insert(self, mu):
        mu = np.asarray(mu, np.float64).ravel()
        path = self._descend(mu)
        leaf = self._new(mu, path[-1])
        self._touch(path, mu)
        return path + [leaf], leaf

    def insert_batch(self, mus, group=True):
        """Insert K concepts against ONE snapshot of the tree.  -> (paths, leaves, new_node_ids).

        Placement is computed for every member before anything is created, so the batch does not
        depend on its own arrival order -- the same K concepts give the same structure however they
        are shuffled, which sequential insertion does not guarantee.

        `group=True` adds one NEW intermediate node per set of members landing on the same parent.
        That node is trainable for the duration of this batch and frozen afterwards, so it is a real
        shared parameter learned from all its members -- not the warm start a sequential run gets,
        but genuine joint structure -- while every node that existed BEFORE the batch stays frozen
        and therefore bit-identical.  This is what makes batch consolidation different in kind from
        K consolidations in a row, rather than merely faster.
        """
        mus = [np.asarray(m, np.float64).ravel() for m in mus]
        placements = [self._descend(m) for m in mus]
        by_parent = {}
        for i, p in enumerate(placements):
            by_parent.setdefault(p[-1], []).append(i)

        paths, leaves, new_nodes = [None] * len(mus), [None] * len(mus), []
        for parent, members in by_parent.items():
            prefix = placements[members[0]]
            if group and len(members) > 1 and not self.flat:
                gmu = np.mean([mus[i] for i in members], 0)
                gnode = self._new(gmu, parent)
                self.count[gnode] = len(members)
                new_nodes.append(gnode)
                host, hpath = gnode, prefix + [gnode]
            else:
                host, hpath = parent, prefix
            for i in members:
                leaf = self._new(mus[i], host)
                new_nodes.append(leaf)
                leaves[i] = leaf
                paths[i] = hpath + [leaf]
        for i, m in enumerate(mus):
            self._touch(placements[i], m)
        return paths, leaves, new_nodes

    def path_of(self, leaf):
        parent = {c: p for p, ks in self.children.items() for c in ks}
        path, cur = [leaf], leaf
        while cur in parent:
            cur = parent[cur]
            path.append(cur)
        return list(reversed(path))


# ── the bank ─────────────────────────────────────────────────────────────────
class HierarchicalAdapterBank:
    """Frozen base UNet + a growing tree of LoRA deltas + one new class embedding per concept."""

    def __init__(self, model, rank=8, device="cuda", var=0.1, branch_max=4, max_depth=3,
                 freeze_ancestors=True, flat=False, targets=("conv", "linear"),
                 exclude=()):
        self.model = model
        self.device = device
        self.rank = int(rank)
        self.state = {"path": []}
        self.flat = bool(flat)               # ablation: every concept a child of the root
        self.freeze_ancestors = bool(freeze_ancestors)
        for p in model.parameters():
            p.requires_grad_(False)
        # `targets` restricts WHICH layer types get a delta.  On our own small UNet adapting
        # everything is fine; on a 113.7M-parameter pretrained model it perturbs hundreds of
        # layers at once and the samples melt, so ('linear',) -- attention and the
        # projections, which is where the LoRA personalization literature puts it -- is the
        # setting that behaves.
        self.wrapped = inject(model, rank, self.state, targets=targets, exclude=exclude)
        self.tree = ConceptTree(var=var, branch_max=branch_max, max_depth=max_depth, flat=flat)
        self.concepts = {}                   # name -> dict(path, leaf)
        self.frozen = set()                  # created by an earlier batch: immutable
        self._open = set()                   # created by the batch being consolidated right now
        self._emb_extra = {}                 # name -> learned class embedding vector
        self.base_null = model.null_id

    # -- registration ---------------------------------------------------------
    def _register(self, name, path, leaf):
        for nid in path:
            for w in self.wrapped:
                w.add_node(nid, self.device)
        emb = torch.zeros(self.model.emb_dim, device=self.device)
        emb.normal_(0, 0.01)
        self._emb_extra[name] = nn.Parameter(emb)
        self.concepts[name] = dict(path=path, leaf=leaf)
        return self.concepts[name]

    def _seal(self):
        """Freeze whatever the previous batch left open.  Called at the start of every insertion, so
        'open' always means 'created by the batch currently being consolidated'."""
        if self.freeze_ancestors:
            self.frozen.update(self._open)
        self._open = set()

    def add_concept(self, name, mu):
        """Place ONE new concept and create its parameters.  Nothing existing is reset."""
        self._seal()
        path, leaf = self.tree.insert(mu)
        self._open = {leaf}
        return self._register(name, path, leaf)

    def add_concepts(self, names, mus, group=True):
        """Place K concepts as ONE batch.  -> list of concept records.

        Everything created here -- the K leaves plus any shared intermediate nodes -- stays trainable
        until the next insertion seals it, so a group node genuinely learns from all of its members.
        Everything that existed before this call is already frozen and cannot move.
        """
        self._seal()
        paths, leaves, new_nodes = self.tree.insert_batch(mus, group=group)
        self._open = set(new_nodes)
        return [self._register(n, p, l) for n, p, l in zip(names, paths, leaves)]

    def trainable(self, name):
        """Parameters trained when consolidating `name`: the nodes on its path that this batch
        created, plus its own class embedding.  Nothing else in the model is touched."""
        return self.trainable_batch([name])

    def trainable_batch(self, names):
        """Union of the trainable parameters for a set of concepts consolidated together.

        A shared group node appears once even though several members sit under it, so it receives
        gradients from all of them through a single set of parameters -- which is the point.
        """
        ids, ps = set(), []
        for name in names:
            c = self.concepts[name]
            on_path = set(c["path"])
            ids |= (on_path & self._open) if self.freeze_ancestors else on_path
            ps.append(self._emb_extra[name])
        for w in self.wrapped:
            for nid in map(str, sorted(ids)):
                if nid in w.A:
                    ps += [w.A[nid], w.B[nid]]
        return ps

    @torch.no_grad()
    def merge_concept(self, name, keep_embedding=True):
        """Absorb `name`'s adapter into the base weights and drop the adapter.

        The alternative to accumulating a private function per concept.  After this the model IS the
        manifold -- there is no side structure to carry, no path to activate, and the parameter count
        is the base model's regardless of how many concepts have been absorbed.

        `keep_embedding` retains the concept's conditioning vector, which is 256 floats rather than
        45K-550K parameters and is what actually names the concept (RESULTS 5n).  The weights learn
        HOW to render; the embedding says WHAT to render.  Only the first needs merging.

        **Scope of the mutation, stated plainly:** a fold writes to the base weight, so it changes
        EVERY branch -- the unconditional model and every other concept -- by exactly this concept's
        delta.  That is the deliberate trade of merge mode (constant parameters instead of the
        preservation certificate; RESULTS 5r measures the damage and N17 the mitigation).  What is
        NOT part of that trade is leaking a **shared** node: folding an ancestor that other concepts
        still route through would push their joint structure onto every branch while they still list
        the node on their paths.  So only nodes no other concept uses are folded; shared ancestors
        stay on this concept's path, which leaves the function computed on ITS OWN branch exactly
        unchanged either way.

        Returns the number of adapter parameters reclaimed.
        """
        c = self.concepts[name]
        # root exempt for the same reason as in `remove_concept`: it is shared by construction
        shared = {0} | {nid for k, cc in self.concepts.items() if k != name
                        for nid in cc["path"]}
        freed = 0
        for nid in c["path"]:
            if nid in shared:
                continue
            for w in self.wrapped:
                k = str(nid)
                if k in w.A:
                    freed += w.A[k].numel() + w.B[k].numel()
                    w.merge_node(k)
            self.frozen.discard(nid)
            self._open.discard(nid)
        c["path"] = [nid for nid in c["path"] if nid in shared]
        if not keep_embedding:
            self._emb_extra.pop(name, None)
        return freed

    @torch.no_grad()
    def remove_concept(self, name, prune_tree=True):
        """Delete a concept exactly, reclaiming its private parameters.  -> accounting dict.

        The counterpart to `add_concept`, and available for the same structural reason preservation
        is: the frozen base is never written to, and a concept IS a set of nodes on a path.  So
        removing one is dropping the nodes that only IT uses.  Every remaining concept is therefore
        **bit-identical** afterwards, not approximately unchanged -- no gradient step, no re-fitting,
        no distillation.  Methods that protect shared weights cannot offer this at all: to forget one
        task they have to retrain, and the result is a different model for every task that remains.

        **The one thing that would break it** is dropping a node another concept still routes
        through, so the nodes to drop are computed as the path MINUS the union of every other
        registered concept's path, and shared ancestors are kept and reported.

        `prune_tree` also removes the dropped nodes from the placement tree so nothing can later
        descend into a node whose parameters are gone.  Note what this does NOT do: ancestors'
        running means were updated by `_touch` at insertion time, and an incremental mean is not
        exactly reversible in floating point, so *placement statistics* retain a trace of the deleted
        concept even though no parameter does.  That affects where a FUTURE concept lands, not any
        existing one, and the certificate below is unaffected.
        """
        if name not in self.concepts:
            raise KeyError(f"no concept {name!r} to remove")
        rec = self.concepts.pop(name)
        # The ROOT is structural, not owned by anybody: every path begins there, and it is "unused"
        # only in the degenerate moment when the bank holds no other concept.  Pruning it then leaves
        # `tree.children` without node 0 and the next insertion dies in `_descend` -- which is
        # exactly how this was found, deleting the sole concept of a fresh bank (n34).  So the root
        # is permanently exempt from both the drop and the prune.
        still_used = {0} | {nid for n, c in self.concepts.items() for nid in c["path"]}
        to_drop = [nid for nid in rec["path"] if nid not in still_used]
        shared = [nid for nid in rec["path"] if nid in still_used]

        freed = 0
        for nid in to_drop:
            for w in self.wrapped:
                freed += w.drop_node(nid)
            self.frozen.discard(nid)
            self._open.discard(nid)
        emb = self._emb_extra.pop(name, None)
        if emb is not None:
            freed += emb.numel()

        if prune_tree:
            for nid in to_drop:
                for kids in self.tree.children.values():
                    if nid in kids:
                        kids.remove(nid)
                self.tree.children.pop(nid, None)
                self.tree.mean.pop(nid, None)
                self.tree.count.pop(nid, None)
        return dict(name=name, dropped_nodes=to_drop, kept_shared=shared,
                    freed_params=int(freed), n_concepts=len(self.concepts))

    # -- activation -----------------------------------------------------------
    def activate(self, name):
        """Context manager setting the active adapter path.

        Anything that is not a registered concept -- None, or an int class id of the base model's
        own trained classes -- activates the EMPTY path, i.e. the frozen base.  Only consolidated
        concepts have adapters.
        """
        bank = self
        path = list(self.concepts[name]["path"]) if (isinstance(name, str)
                                                     and name in self.concepts) else []

        class _Ctx:
            def __enter__(self):
                self.prev = list(bank.state["path"])
                bank.state["path"] = path
                return self

            def __exit__(self, *a):
                bank.state["path"] = self.prev
                return False
        return _Ctx()

    def embedding(self, name):
        """A concept's learned class embedding, as a live parameter (None -> use the null token).

        Passed to `UNet.forward(emb_override=...)` rather than written into the embedding table, so
        it stays in the autograd graph while a concept is being distilled.
        """
        return None if name is None else self._emb_extra.get(name)

    # -- bookkeeping ----------------------------------------------------------
    def n_params(self):
        return sum(int(p.numel()) for w in self.wrapped for p in list(w.A.values()) + list(w.B.values()))

    def summary(self):
        depth = {n: len(self.tree.path_of(c["leaf"])) - 1 for n, c in self.concepts.items()}
        shared = {}
        for n, c in self.concepts.items():
            for nid in c["path"][:-1]:
                shared[nid] = shared.get(nid, 0) + 1
        return dict(n_concepts=len(self.concepts), n_nodes=self.tree._next,
                    adapter_params=self.n_params(), depths=depth,
                    n_frozen=len(self.frozen), n_open=len(self._open),
                    shared_ancestors={k: v for k, v in shared.items() if v > 1})


@torch.no_grad()
def fingerprint(field, name, seed=0, n=4, u_grid=(0.2, 0.5, 0.8)):
    """A deterministic signature of what concept `name` currently generates.

    Used by the lifelong experiments to test the no-forgetting claim the hard way: re-fingerprint an
    old concept after later consolidations and compare bit-for-bit.  With frozen ancestors this must
    be EXACTLY unchanged, not merely close.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, field.dim))
    return np.concatenate([field.eps(x, u, name).ravel() for u in u_grid])


@torch.no_grad()
def activation_subspace(bank, field, conds, n=4, u_grid=(0.2, 0.5, 0.8), energy=0.97,
                        max_dim=None, seed=0, x0=None):
    """The input-activation subspace each wrapped layer sees on conditionings we already know.

    This is what makes "reconsolidate into the existing weights" survivable.  Merging a delta into
    the base is exact for the concept just learned (`tests/test_merge.py`: relative error 1.1e-06),
    but the NEXT concept's update then lands on shared weights and overwrites it -- measured drift
    1.086 after a single further merge.  Freezing an adapter avoids that by never touching the
    shared weights again, at the cost of carrying a private function per concept forever.

    The alternative is to leave the weights writable but constrain WHERE they may be written: if the
    update's input-side factor is orthogonal to the activations that old conditionings produce, then
    the delta is ~0 on everything already known and free on everything new.  Same guarantee, no
    accumulation.

    We have an unusual advantage in estimating that subspace.  The factor basis is a generator for
    every composition the model knows, so the "old activations" can be SAMPLED rather than stored --
    no exemplars, no replay buffer.

    For Conv2d the input is [B, C, H, W] and is treated as B*H*W observations of a C-vector, so the
    subspace is over channels and ignores spatial structure.  That is an approximation: it constrains
    which channel mixtures the update may read, not which spatial patterns.

    `x0` should be REAL images of known compositions.  The activations are then collected on
    `sqrt(abar) x0 + sqrt(1-abar) eps`, which is the distribution the model actually sees.  Estimating
    the subspace on pure Gaussian input instead measures the wrong thing and roughly halves the
    protection it buys.

    -> {layer_index: P} where P projects OUT the known-activation subspace.
    """
    feats = {}
    handles = []

    def hook(i):
        def fn(mod, inp, out):
            x = inp[0].detach()
            v = (x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) if x.dim() == 4
                 else x.reshape(-1, x.shape[-1]))
            if len(v) > 4096:                      # cap the per-batch contribution
                v = v[torch.randperm(len(v), device=v.device)[:4096]]
            c = (v.T @ v).double()
            feats[i] = c if i not in feats else feats[i] + c
        return fn

    for i, w in enumerate(bank.wrapped):
        handles.append(w.register_forward_hook(hook(i)))
    try:
        rng = np.random.default_rng(seed)
        base = (rng.standard_normal((n, field.dim)) * 0.5 if x0 is None
                else np.asarray(x0, np.float32))
        for ci, c in enumerate(conds):
            emb = (torch.as_tensor(np.asarray(c, np.float32), device=field.device)
                   if not torch.is_tensor(c) else c)
            for u in u_grid:
                if x0 is None:
                    Xu = base
                else:                              # the distribution the model is actually trained on
                    take = base[(ci * n) % len(base):(ci * n) % len(base) + n]
                    if len(take) < n:
                        take = base[:n]
                    ab = float(cosine_abar(np.array([u]))[0])
                    Xu = (ab ** 0.5) * take + ((1 - ab) ** 0.5) * rng.standard_normal(take.shape)
                field.eps(Xu, u, emb=emb)
    finally:
        for h in handles:
            h.remove()

    P = {}
    for i, cov in feats.items():
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals.flip(0), min=0)
        evecs = evecs.flip(1)
        tot = float(evals.sum())
        if tot <= 0:
            continue
        frac = torch.cumsum(evals, 0) / tot
        k = int((frac < energy).sum().item()) + 1
        if max_dim:
            k = min(k, int(max_dim))
        k = max(1, min(k, evecs.shape[1] - 1))     # always leave at least one free direction
        U = evecs[:, :k].float()
        P[i] = (torch.eye(U.shape[0], device=U.device) - U @ U.T)
    return P


@torch.no_grad()
def project_out(bank, node_ids, P):
    """Constrain each node's input-side factor `A` to the complement of the known subspace.

    Applied after every optimizer step, so the update can only read directions that old conditionings
    do not use.  `B` is left alone: it decides how much of an already-orthogonal read to write.
    """
    ids = {str(n) for n in node_ids}
    for i, w in enumerate(bank.wrapped):
        if i not in P:
            continue
        for nid in ids:
            if nid not in w.A:
                continue
            A = w.A[nid]
            if A.dim() == 4:                       # [rank, C, k, k] -> project the channel axis
                r, C, kh, kw = A.shape
                flat = A.permute(0, 2, 3, 1).reshape(-1, C)
                A.data.copy_((flat @ P[i]).reshape(r, kh, kw, C).permute(0, 3, 1, 2))
            else:
                A.data.copy_(A @ P[i])
