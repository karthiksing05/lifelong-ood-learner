"""Serve pi05_libero through OUR sampler for openpi's LIBERO client.

    python experiments/serve_guided.py --port 8123 --mode ours            # plain: our Euler sampler, no hooks
    python experiments/serve_guided.py --port 8123 --mode proto:<tree.npz>:<node_id>[:w0]   # forced prototype

Run inside the openpi env (experiments/r1_rollout_pi05.sh MODE=ours does this on a node with the client).
"""
import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", default="ours")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.backbone import PolicyBackbone
    from lifelong_ood.policy.serve import GuidedPolicy, serve
    bk = PolicyBackbone(a.config, device="cuda")
    logging.info(bk.describe())
    hooks = None
    if a.mode.startswith("proto:"):
        _, tree_path, node_id, *rest = a.mode.split(":")
        w0 = float(rest[0]) if rest else 1.2
        from lifelong_ood.operators.decompose import guide as make_guide
        from lifelong_ood.tree.nodes import Tree
        tree = Tree.load(tree_path)
        n = tree.nodes[int(node_id)]
        g = make_guide(n.mode, n.var, w0=w0, path=tree.path)
        logging.info(f"forcing node {node_id} (level {n.level}, u={n.u}) with w0={w0}")
        hooks = lambda obs, pre: (g, None)          # noqa: E731
    elif a.mode != "ours":
        raise SystemExit(f"unknown mode {a.mode}")
    serve(GuidedPolicy(bk, hooks=hooks, num_steps=a.num_steps, seed=a.seed), a.port)


if __name__ == "__main__":
    main()
