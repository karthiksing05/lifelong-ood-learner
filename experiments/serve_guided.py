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
    if a.mode.startswith(("playback:", "vinn:")):
        # playback:<task>[:demo] (the demo's actions, open loop) / vinn:<task>[:demo] (nearest demo frame, VINN) -- no policy
        import importlib.util
        from lifelong_ood.policy import baselines as B
        from lifelong_ood.policy import data as D
        kind, task, *rest = a.mode.split(":")
        spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
        r3p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r3p)
        texts, _ = r3p.libero90_texts()
        d90 = D.Demos90()
        ep = d90.load(d90.episodes_of_text(texts[int(task)])[int(rest[0]) if rest else 0])
        if kind == "playback":
            policy = B.PlaybackPolicy(ep["actions"])
        else:
            img, wr = d90.frames(ep["episode_index"])
            policy = B.VinnPolicy(ep["actions"], img, wr)
        logging.info(f"{kind}: task {task} demo episode {ep['episode_index']}, {len(ep['actions'])} frames")
        warm(policy)
        serve(policy, a.port)
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
    elif a.mode.startswith("program:") and "?" in a.mode:
        # program:<program.npz>?w0=1&sched=const&var=0.001&how=guide&stall=3&key=pose&window=20  (named parameters)
        path, q = a.mode[len("program:"):].split("?", 1)
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        from lifelong_ood.policy.compose import ComposeHooks, DemoProgram
        prog = DemoProgram.load(path)
        how = kv.get("how", "guide")
        hooks = ComposeHooks(prog, w0=float(kv.get("w0", 1.2)), window=int(kv.get("window", 20)), schedule=kv.get("sched", "variance"),
                             var=float(kv["var"]) if kv.get("var") else None, use_guide=how != "prompt",
                             use_prompt=how in ("prompt", "prompt+guide"), stall=int(kv.get("stall", 0)),
                             key_mode=kv.get("key", "pos"), agree=float(kv["agree"]) if kv.get("agree") else None,
                             contact=int(kv.get("contact", 0)), arm_only=kv.get("arm", "0") == "1")
        logging.info(f"program {path}: variant {prog.variant}, {len(prog.mu)} frames, {prog.bytes()} B, params {kv}")
    elif a.mode.startswith("composer:"):
        # composer:<r6_comp.npz>?w0=1&sched=const&var=0.001&key=pose&stall=3 -- the learned borrowing PoE over library prototypes
        path, q = (a.mode[len("composer:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        from lifelong_ood.policy.composer import ComposerHooks, ComposerProgram
        from lifelong_ood.tree.nodes import Tree
        cp = ComposerProgram.load(path)
        lib = cp.library if os.path.isabs(cp.library) else os.path.join(ROOT, cp.library)   # the server runs from openpi/
        hooks = ComposerHooks(cp, Tree.load(lib), w0=float(kv.get("w0", 1.0)), window=int(kv.get("window", 4)),
                              schedule=kv.get("sched", "const"), var=float(kv["var"]) if kv.get("var") else 1e-3,
                              stall=int(kv.get("stall", 3)), key_mode=kv.get("key", "pose"))
        logging.info(f"composer {path}: {len(cp.ids)} prototypes, {cp.bytes()} B, params {kv}")
    elif a.mode.startswith("comphybrid:"):
        # comphybrid:<r6_comp.npz>?expert=<r4 expert.pt>&foldprog=<r4 program.npz>&... -- composer + the fold expert in its windows
        path, q = a.mode[len("comphybrid:"):].split("?", 1)
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy.compose import DemoProgram
        from lifelong_ood.policy.composer import ComposerHybridHooks, ComposerProgram
        from lifelong_ood.tree.nodes import Tree
        cp = ComposerProgram.load(path)
        lib = cp.library if os.path.isabs(cp.library) else os.path.join(ROOT, cp.library)
        state, key, banks = adapt.load(bk, kv["expert"])
        fold_mask = DemoProgram.load(kv["foldprog"]).fold
        hooks = ComposerHybridHooks(cp, Tree.load(lib), state, key, fold_mask, w0=float(kv.get("w0", 1.0)), window=int(kv.get("window", 4)),
                                    schedule=kv.get("sched", "const"), var=float(kv["var"]) if kv.get("var") else 1e-3,
                                    stall=int(kv.get("stall", 3)), key_mode=kv.get("key", "pose"))
        logging.info(f"composer+fold {path}: {len(cp.ids)} prototypes, fold frames {int(fold_mask.sum())}/{len(fold_mask)}, expert {key}")
    elif a.mode.startswith("hybrid:") and "?" in a.mode:
        # hybrid:<program.npz>&expert=<expert.pt>?w0=1&sched=const&var=0.001&stall=3&key=pose
        body, q = a.mode[len("hybrid:"):].split("?", 1)
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy.compose import DemoProgram, HybridHooks
        state, key, banks = adapt.load(bk, kv["expert"])
        prog = DemoProgram.load(body)
        if kv.get("foldprog"):                     # the fold windows of the expert's own program (same demo, same frames)
            fm = DemoProgram.load(kv["foldprog"]).fold
            assert len(fm) == len(prog.mu), (len(fm), len(prog.mu))
            prog.fold = fm
        hooks = HybridHooks(prog, state, key, w0=float(kv.get("w0", 1.2)), schedule=kv.get("sched", "variance"),
                            var=float(kv["var"]) if kv.get("var") else None, stall=int(kv.get("stall", 0)),
                            key_mode=kv.get("key", "pos"), window=int(kv.get("window", 20)))
        logging.info(f"hybrid: program {body} ({int(prog.fold.sum()) if prog.fold is not None else 0} fold frames of {len(prog.mu)}), expert {key}, params {kv}")
    elif a.mode.startswith("program:"):
        # program:<program.npz>[:w0[:window[:schedule[:var[:how]]]]] -- COMPOSE: follow one demo's prototype program (or
        # its raw chunks); schedule "variance" (Eq. 16) or "const"; var overrides the teacher's variance (its precision);
        # how = "guide" (default) | "prompt" (swap the instruction only) | "prompt+guide"; stall = release the guide after
        # that many calls without progress (0 = off)
        _, prog_path, *rest = a.mode.split(":")
        from lifelong_ood.policy.compose import ComposeHooks, DemoProgram
        prog = DemoProgram.load(prog_path)
        hooks = ComposeHooks(prog, w0=float(rest[0]) if rest else 1.2, window=int(rest[1]) if len(rest) > 1 else 20,
                             schedule=rest[2] if len(rest) > 2 else "variance",
                             var=float(rest[3]) if len(rest) > 3 and rest[3] else None,
                             use_guide=not (len(rest) > 4 and rest[4] == "prompt"),
                             use_prompt=len(rest) > 4 and rest[4] in ("prompt", "prompt+guide"),
                             stall=int(rest[5]) if len(rest) > 5 and rest[5] else 0)
        logging.info(f"program {prog_path}: variant {prog.variant}, {len(prog.mu)} frames, {prog.bytes()} B")
    elif a.mode.startswith("hybrid:"):
        # hybrid:<program.npz>:<expert.pt>[:w0[:schedule[:var]]] -- compose where explained, the folded expert where not
        _, prog_path, expert_path, *rest = a.mode.split(":")
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy.compose import DemoProgram, HybridHooks
        state, key, banks = adapt.load(bk, expert_path)
        prog = DemoProgram.load(prog_path)
        hooks = HybridHooks(prog, state, key, w0=float(rest[0]) if rest else 1.2, schedule=rest[1] if len(rest) > 1 else "variance",
                            var=float(rest[2]) if len(rest) > 2 and rest[2] else None,
                            stall=int(rest[3]) if len(rest) > 3 and rest[3] else 0)
        logging.info(f"hybrid: program {prog_path} ({int(prog.fold.sum()) if prog.fold is not None else 0} fold frames of {len(prog.mu)}), expert {key}")
    elif a.mode.startswith("lora:"):
        # lora:<expert.pt> -- the few-shot baseline: a flat adapter on the action expert, always on
        from lifelong_ood.policy import adapt
        state, key, banks = adapt.load(bk, a.mode.split(":", 1)[1])
        state["on"] = frozenset({key})
        logging.info(f"expert {key} loaded into {len(banks)} banks and switched on")
    elif a.mode.startswith("prompt:"):
        # prompt:<training task index> -- the prior under a trained task's instruction (instruction retrieval)
        from lifelong_ood.policy import data as D
        from lifelong_ood.policy.baselines import PromptHooks
        text = D.Demos().tasks[int(a.mode.split(":", 1)[1])]
        hooks = PromptHooks(text)
        logging.info(f"prompt replaced by '{text}'")
    elif a.mode != "ours":
        raise SystemExit(f"unknown mode {a.mode}")
    policy = GuidedPolicy(bk, hooks=hooks, num_steps=a.num_steps, seed=a.seed)
    warm(policy)
    serve(policy, a.port)


if __name__ == "__main__":
    main()
