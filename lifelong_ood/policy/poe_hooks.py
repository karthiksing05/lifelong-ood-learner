"""PoeHooks: serve a task's PHASE CODES (lifelong_ood/policy/poe.py) through the policy server.

A code names, per phase (the gripper's open / closed runs, in order), up to K + 1 experts by observation-free ids and their
per-dim weights:
    task              the policy under the task's own instruction
    prompt:<noun>     the policy under "pick up the <noun>"
    cell:<i><j>       the policy under the task's instruction, its action expert's attention raised to mass 0.7 on token block
                      (i, j) of the 4 x 4 grid over the agent view
    node:<u>:<id>     a node of the action-marginal tree (mode, extent): a motion prototype
At every query the policy experts are re-run at the CURRENT observation (one batched prefix), the prototypes are analytic, and
the chunk is sampled by poe.sample_poe with the policy family's share `policy_share` at the last steps.  The phase advances on
the policy's own commanded gripper (its previous chunk's 5th action, the last one executed with replan 5).
"""
from __future__ import annotations

import numpy as np

from . import poe


def grid_map(i, j):
    m = np.zeros((16, 16), np.float32)
    m[4 * i:4 * i + 4, 4 * j:4 * j + 4] = 1.0
    return {"agent": m}


class PoeHooks:
    def __init__(self, bk, codes, task_prompt, tree=None, policy_share=0.5, late_t=0.3, exact=False, ablate=""):
        self.bk = bk
        self.codes = codes                          # list of dicts: phase, experts, weights [K, 70], policy_share
        self.task_prompt = task_prompt
        self.tree = tree
        self.lam = float(policy_share)
        self.late_t = float(late_t)
        self.exact = bool(exact)
        self.ablate = ablate                         # "" | "noproto" (options only) | "nopolicy" (the prototype product alone)
        self.k = 0
        self.closed = False
        self.started = False
        self.log = []

    def reset(self, meta):
        self.started, self.k, self.closed = True, 0, False

    def observe(self, actions):
        a = np.asarray(actions)
        closed = bool(a[min(4, len(a) - 1), 6] > 0)
        if closed != self.closed:
            self.k += 1
        self.closed = closed

    def _expert(self, eid):
        if eid == "task":
            return poe.PolicyRow(self.task_prompt, name=eid)
        if eid.startswith("prompt:"):
            return poe.PolicyRow(f"pick up the {eid[7:]}", name=eid)
        if eid.startswith("cell:"):
            return poe.PolicyRow(self.task_prompt, focus=grid_map(int(eid[5]), int(eid[6])), name=eid)
        if eid.startswith("node:"):
            _, u, i = eid.split(":")
            nd = self.tree.nodes[int(i)]
            return poe.ProtoExpert(np.asarray(nd.mode, np.float32), np.maximum(np.asarray(nd.extent_dim, np.float32), 1e-4), eid)
        raise ValueError(eid)

    def compose_poe(self, obs, meta):
        """-> (prefix of the policy rows, rows, prototypes, beta) for this query (the warm-up runs the policy alone)."""
        if not self.started:
            rows = [poe.PolicyRow(obs["prompt"])]
            return poe.policy_prefixes(self.bk, obs, rows), rows, [], np.ones((1, 70), np.float32)
        code = self.codes[min(self.k, len(self.codes) - 1)]
        ids, W = list(code["experts"]), np.asarray(code["weights"], np.float32)
        keep = [i for i, e in enumerate(ids) if not (self.ablate == "noproto" and e.startswith("node:"))]
        ids, W = [ids[i] for i in keep], W[keep]
        ex = [self._expert(e) for e in ids]
        rows = [e for e in ex if isinstance(e, poe.PolicyRow)]
        protos = [e for e in ex if isinstance(e, poe.ProtoExpert)]
        order = [i for i, e in enumerate(ex) if isinstance(e, poe.PolicyRow)] + [i for i, e in enumerate(ex) if isinstance(e, poe.ProtoExpert)]
        W = W[order]
        protos, beta = poe.compose(W, len(rows), protos, policy_share=self.lam, late_t=self.late_t, exact=self.exact)
        if self.ablate == "nopolicy" and protos:     # the no-policy control: the product of the prototypes alone (gripper: policy)
            base = beta

            def beta(t, base=base, n=len(rows)):
                B = base(t).copy()
                arm = ~poe.grip_mask()
                B[:n, arm] = 0.0
                B[n:, arm] = 1.0
                return B
        pre = poe.policy_prefixes(self.bk, obs, rows)
        self.log.append(dict(phase=code["phase"], k=self.k, experts=ids))
        return pre, rows, protos, beta
