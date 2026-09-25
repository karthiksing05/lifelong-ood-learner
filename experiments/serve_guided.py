"""Serve pi05_libero through OUR sampler for openpi's LIBERO client.

    python experiments/serve_guided.py --port 8123 --mode openpi          # reference: openpi's Policy.infer, eager
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
    from lifelong_ood.policy.serve import GuidedPolicy, StripMeta, serve, warm
    if a.mode == "openpi":
        # the REFERENCE: openpi's own Policy (its transforms, its sample_actions) on the PyTorch checkpoint, eager
        # (openpi's max-autotune compile blocks the server's event loop past the client's 20 s keepalive), warmed
        # up before the server listens.
        import dataclasses
        from openpi.policies import policy_config as _pc
        from openpi.training import config as _config
        cfg = _config.get_config(a.config)
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        ck = "/coc/flash10/ksingara3/robot/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch"
        policy = StripMeta(_pc.create_trained_policy(cfg, ck, pytorch_device="cuda"))
        warm(policy)
        serve(policy, a.port, metadata=policy.metadata)
        return
    from lifelong_ood.policy.backbone import PolicyBackbone
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
        hooks = lambda obs, pre, meta=None: (g, None)          # noqa: E731
    elif a.mode.startswith("program:"):
        # program:<program.npz>[:w0[:window[:schedule[:var[:how]]]]] -- COMPOSE: follow one demo's prototype program (or
        # its raw chunks); schedule "variance" (Eq. 16) or "const"; var overrides the teacher's variance (its precision);
        # how = "guide" (default) | "prompt" (swap the instruction only) | "prompt+guide"
        _, prog_path, *rest = a.mode.split(":")
        from lifelong_ood.policy.compose import ComposeHooks, DemoProgram
        prog = DemoProgram.load(prog_path)
        hooks = ComposeHooks(prog, w0=float(rest[0]) if rest else 1.2, window=int(rest[1]) if len(rest) > 1 else 20,
                             schedule=rest[2] if len(rest) > 2 else "variance",
                             var=float(rest[3]) if len(rest) > 3 and rest[3] else None,
                             use_guide=not (len(rest) > 4 and rest[4] == "prompt"),
                             use_prompt=len(rest) > 4 and rest[4] in ("prompt", "prompt+guide"))
        logging.info(f"program {prog_path}: variant {prog.variant}, {len(prog.mu)} frames, {prog.bytes()} B")
    elif a.mode.startswith("hybrid:"):
        # hybrid:<program.npz>:<expert.pt>[:w0[:schedule[:var]]] -- compose where explained, the folded expert where not
        _, prog_path, expert_path, *rest = a.mode.split(":")
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy.compose import DemoProgram, HybridHooks
        state, key, banks = adapt.load(bk, expert_path)
        prog = DemoProgram.load(prog_path)
        hooks = HybridHooks(prog, state, key, w0=float(rest[0]) if rest else 1.2, schedule=rest[1] if len(rest) > 1 else "variance",
                            var=float(rest[2]) if len(rest) > 2 and rest[2] else None)
        logging.info(f"hybrid: program {prog_path} ({int(prog.fold.sum()) if prog.fold is not None else 0} fold frames of {len(prog.mu)}), expert {key}")
    elif a.mode.startswith("lora:"):
        # lora:<expert.pt> -- the few-shot baseline: a flat adapter on the action expert, always on
        from lifelong_ood.policy import adapt
        state, key, banks = adapt.load(bk, a.mode.split(":", 1)[1])
        state["on"] = frozenset({key})
        logging.info(f"expert {key} loaded into {len(banks)} banks and switched on")
    elif a.mode != "ours":
        raise SystemExit(f"unknown mode {a.mode}")
    policy = GuidedPolicy(bk, hooks=hooks, num_steps=a.num_steps, seed=a.seed)
    warm(policy)
    serve(policy, a.port)


if __name__ == "__main__":
    main()
