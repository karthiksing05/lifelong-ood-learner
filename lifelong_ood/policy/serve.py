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

    def infer(self, obs, *, noise=None):
        t0 = time.monotonic()
        observation = self.bk.observation(obs)
        pre = self.bk.prefix(observation)
        guide, gate = (None, None) if self.hooks is None else self.hooks(obs, pre)
        rows = self.bk.sample(1, pre, seed=self.seed + self.calls, num_steps=self.num_steps, guide=guide, step_gate=gate,
                              noise=None if noise is None else np.asarray(noise).reshape(1, -1))
        actions = self.bk.unnormalize(rows)[0]
        self.calls += 1
        self.secs += time.monotonic() - t0
        if self.log_every and self.calls % self.log_every == 0:
            logging.info(f"[serve] {self.calls} calls, {1000 * self.secs / self.calls:.0f} ms each")
        return {"actions": np.asarray(actions, np.float32), "state": np.asarray(obs["observation/state"], np.float32)}


def serve(policy, port, host="0.0.0.0", metadata=None):
    from openpi.serving import websocket_policy_server
    logging.info(f"Creating server (host: {host}, port: {port})")
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=host, port=int(port), metadata=metadata or {})
    server.serve_forever()
