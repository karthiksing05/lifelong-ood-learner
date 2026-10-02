"""Serve OUR sampler (guide / gate hooks over the frozen pi0.5) through openpi's websocket protocol, so
openpi's LIBERO client (examples/libero/main.py, python 3.8 venv) drives the simulation unchanged.

    policy = GuidedPolicy(backbone, hooks=my_hooks)      # hooks(obs_dict, prefix) -> (guide, step_gate)
    serve(policy, port)

`infer(obs)` returns {"actions": [H, 7] raw dataset actions, "state": ...} like openpi's Policy.infer.
"""
from __future__ import annotations

import logging
import time

import numpy as np


class GuidedPolicy:
    def __init__(self, backbone, hooks=None, num_steps=10, seed=0, log_every=50):
        self.bk = backbone
        self.hooks = hooks
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.calls = 0
        self.log_every = log_every
        self.secs = 0.0
        self.ep_q = 0                     # query index inside the current episode
        self.branch = None                # a stall-point branch (experiments/r18_branch.py), set per episode by the client

    def _seed(self):
        """The noise seed of this query.  Default: seed + the server's call count (a rerun of the same trials in the same order
        draws the same noise).  A BRANCH (experiments/r18_branch.py; its prefix fast-forwarded on the recorded actions)
        continues from query `at` with the recorded episode's noise, seed_base + query, and one knob changed -- or, for a
        "noise<s>" variant, fresh noise."""
        b = self.branch
        if b is None:
            return self.seed + self.calls
        if self.ep_q >= b["at"] and b["variant"].startswith("noise"):
            return 1_000_003 * int(b["variant"][5:]) + self.ep_q
        return b["seed_base"] + self.ep_q

    def infer(self, obs, *, noise=None):
        t0 = time.monotonic()
        meta = {k[5:]: v for k, v in obs.items() if k.startswith("meta/")}
        obs = {k: v for k, v in obs.items() if not k.startswith("meta/")}
        if meta.get("episode_start"):
            self.ep_q = 0
            self.branch = (dict(seed_base=int(meta["seed_base"]), at=int(meta["branch_at"]), variant=str(meta["variant"]))
                           if "branch_at" in meta else None)
            if hasattr(self.hooks, "reset"):
                self.hooks.reset(meta)
        if hasattr(self.hooks, "rewrite"):
            obs = self.hooks.rewrite(obs, meta)
        if meta.get("ff"):
            # FAST-FORWARD (a branch's prefix): the client executes the recorded actions; the hooks see the observation, so
            # their pointer and stall state evolve exactly as in the recorded episode, and nothing is sampled
            if self.hooks is not None:
                self.hooks(obs, None, meta)
            self.calls += 1
            self.ep_q += 1
            return {"actions": np.zeros((self.bk.H, 7), np.float32), "state": np.asarray(obs["observation/state"], np.float32)}
        if self.branch is not None and self.ep_q == self.branch["at"] and hasattr(self.hooks, "branch"):
            self.hooks.branch(self.branch["variant"])        # the knob changes from this query to the end of the episode
        sd = self._seed()
        if hasattr(self.hooks, "compose_poe"):
            # prototype PoE planning (lifelong_ood/policy/poe.py): a convex mix of live policy experts and motion prototypes per
            # dim, the policy family's share held at the last steps; every policy expert re-runs at this observation
            from . import poe as _poe
            pre, rows, protos, beta = self.hooks.compose_poe(obs, meta)
            nz = None if noise is None else np.asarray(noise).reshape(1, -1)
            rws = _poe.sample_poe(self.bk, pre, rows, protos, beta, seed=sd, num_steps=self.num_steps, noise=nz)
            actions = self.bk.unnormalize(rws, state=pre.state[0].float().cpu().numpy())[0]
            if hasattr(self.hooks, "observe"):
                self.hooks.observe(actions)
            self.calls += 1
            self.ep_q += 1
            self.secs += time.monotonic() - t0
            if self.log_every and self.calls % self.log_every == 0:
                logging.info(f"[serve] {self.calls} calls, {1000 * self.secs / self.calls:.0f} ms each")
            return {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}
        if hasattr(self.hooks, "compose"):
            # factor-composed generation (lifelong_ood/policy/conditionals.py): the policy's own velocities under the code's
            # prompts, combined affinely at every Euler step; nothing external is followed
            from . import conditionals as C
            comp = self.hooks.compose(obs, meta)
            cpre = C.prefixes(self.bk, obs, comp.prompts)
            nz = None if noise is None else np.asarray(noise).reshape(1, -1)
            rows = C.sample_composed(self.bk, cpre, comp.coef, seed=sd, num_steps=self.num_steps, noise=nz)
            actions = self.bk.unnormalize(rows, state=cpre.state[0].float().cpu().numpy())[0]
            if hasattr(self.hooks, "observe"):
                self.hooks.observe(actions)
            self.calls += 1
            self.ep_q += 1
            self.secs += time.monotonic() - t0
            if self.log_every and self.calls % self.log_every == 0:
                logging.info(f"[serve] {self.calls} calls, {1000 * self.secs / self.calls:.0f} ms each")
            return {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}
        observation = self.bk.observation(obs)
        pre = self.bk.prefix(observation)
        if hasattr(self.hooks, "transform_prefix"):
            pre = self.hooks.transform_prefix(pre)            # e.g. a factor token appended to the prompt's cache (token_fold)
        nz = None if noise is None else np.asarray(noise).reshape(1, -1)
        if nz is None and hasattr(self.hooks, "noise_for"):
            nz = self.hooks.noise_for(obs, meta)             # DSRL (lifelong_ood/policy/dsrl.py): a learned initial noise
        if getattr(self.hooks, "needs_prior", False):
            # the PRIOR expert: the frozen policy's own chunk on this observation, before the composed teacher is built
            self.hooks.set_prior(self.bk.sample(1, pre, seed=sd, num_steps=self.num_steps, noise=nz)[0])
        guide, gate = (None, None) if self.hooks is None else self.hooks(obs, pre, meta)
        rows = None
        if guide is not None and getattr(self.hooks, "agree", None) is not None:
            own = self.bk.sample(1, pre, seed=sd, num_steps=self.num_steps, step_gate=gate, noise=nz)
            if self.hooks.agrees(own[0]):
                rows = own                                   # the prior already does what the program asks
        if rows is None and guide is not None and getattr(self.hooks, "no_policy", False):
            rows = np.asarray(self.hooks.last_rows, np.float32)[None]   # the NO-POLICY control: the guide's target, nothing sampled
        if rows is None:
            start = self.hooks.sdedit(obs, meta) if hasattr(self.hooks, "sdedit") else None
            rows = self.bk.sample(1, pre, seed=sd, num_steps=self.num_steps, guide=guide, step_gate=gate, noise=nz,
                                  start=start)
        actions = self.bk.unnormalize(rows, state=observation.state[0].float().cpu().numpy())[0]
        if hasattr(self.hooks, "observe"):
            self.hooks.observe(actions)                      # the policy's own chunk (e.g. its gripper command sets Bind's phase)
        self.calls += 1
        self.ep_q += 1
        self.secs += time.monotonic() - t0
        if self.log_every and self.calls % self.log_every == 0:
            logging.info(f"[serve] {self.calls} calls, {1000 * self.secs / self.calls:.0f} ms each")
        return {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}


class StripMeta:
    """openpi's Policy with the client's "meta/" keys removed (the reference server)."""

    def __init__(self, policy):
        self.policy = policy
        self.metadata = getattr(policy, "metadata", {})

    def infer(self, obs, **kw):
        return self.policy.infer({k: v for k, v in obs.items() if not k.startswith("meta/")}, **kw)


def warm(policy, n=2):
    """Run n inferences on openpi's random LIBERO example so no first-call cost lands inside a client request."""
    try:
        from openpi.policies import libero_policy
        example = libero_policy.make_libero_example
    except ImportError:                                  # a venv without openpi (SmolVLA): the same random example
        def example():
            r = np.random.default_rng()
            return {"observation/state": r.random(8), "observation/image": r.integers(256, size=(224, 224, 3), dtype=np.uint8),
                    "observation/wrist_image": r.integers(256, size=(224, 224, 3), dtype=np.uint8), "prompt": "do something"}
    t0 = time.monotonic()
    for _ in range(int(n)):
        policy.infer(example())
    logging.info(f"[serve] warm-up {n} calls in {time.monotonic() - t0:.1f}s")


def serve(policy, port, host="0.0.0.0", metadata=None):
    from openpi.serving import websocket_policy_server
    logging.info(f"Creating server (host: {host}, port: {port})")
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=host, port=int(port), metadata=metadata or {})
    server.serve_forever()
