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
    cfg = os.environ.get("BACKBONE_CONFIG") or a.config              # another flow VLA, e.g. pi0_libero (generality checks)
    ck = os.environ.get("BACKBONE_CKPT") or None
    if cfg.startswith("smolvla"):                                      # LeRobot's SmolVLA (its own venv: lifelong_ood/policy/backbone_smolvla.py)
        from lifelong_ood.policy.backbone_smolvla import SmolVLABackbone
        bk = SmolVLABackbone(ck, device="cuda")
    else:
        bk = PolicyBackbone(cfg, checkpoint_dir=ck, device="cuda")
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
        if kv.get("alt"):                          # the next-ranked leaves, executed only by a "leaf2" branch (r18_branch.py)
            hooks.alt = DemoProgram.load(kv["alt"])
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
        if kv.get("alt"):                          # the next-ranked leaves, executed only by a "leaf2" branch (r18_branch.py)
            hooks.alt = DemoProgram.load(kv["alt"])
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
    elif a.mode.startswith("phasebind:"):
        # phasebind:<task>?expert=<fold expert.pt>&foldprog=<fold program.npz>&var=0.001 -- P1 (archive/2026-10-program/PLAN_TRANSFER.md): the task's ONE demo
        # as phases re-anchored on the objects' current poses (the client's --objects), guided through the prior; the fold on the demo's
        # contact frames
        import importlib.util
        import numpy as np
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy import data as D
        from lifelong_ood.policy.phase import PhaseBindHooks
        body, q = (a.mode[len("phasebind:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        t = int(body)
        R = os.path.join(ROOT, "results", "robot")
        spec = importlib.util.spec_from_file_location("p3", os.path.join(ROOT, "experiments", "p3_compose_sim.py"))
        P3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P3)
        dsfx = f"_d{kv['demo']}" if kv.get("demo") else ""        # demo k > 0: IPEC episode k (p0_demo_init --rank k --suffix _dk)
        d = np.load(os.path.join(R, "p0_demo", f"{t}{dsfx}.npz"), allow_pickle=True)
        m = np.load(os.path.join(R, "p0_demo", f"{t}{dsfx}_match.npz"), allow_pickle=True)
        src = dict(names=d["names"], movable=d["movable"], obj_pos=d["demo_obj_pos"], obj_quat=d["demo_obj_quat"],
                   eef_pos=d["demo_eef_pos"], eef_quat=d["demo_eef_quat"], action=np.asarray(m["actions"], np.float32))
        if kv.get("rules") == "2" and kv.get("g"):       # P3c: composed from OTHER tasks' demos (phase.py compose); t's own demo unused
            from lifelong_ood.policy import phase as PH
            tn = [str(x) for x in d["names"]]              # the target scene's objects of interest (the task definition)
            gv = None
            if kv.get("contact") == "1":                   # the grasp source's per-frame prototype verbs (experiments/grammar_readout.py)
                import json as _json
                GL = np.load(os.path.join(R, "grammar_labels.npz"))
                nm = {int(k_): v_["name"].split(" (")[0] for k_, v_ in
                      _json.load(open(os.path.join(R, "grammar_readout.json")))["prototypes"]["0.4"].items()}
                k_ = list(GL["tasks"]).index(int(kv["g"]))
                gv = [nm[int(x)] for x in GL["lab_0.4"][GL["cut"][k_]:GL["cut"][k_ + 1]]]
            parts = PH.compose(tn, PH.at_rest(PH.load_demo(R, int(kv["g"]))), PH.at_rest(PH.load_demo(R, int(kv["p"]))), obj_type=kv.get("obj"),
                               obj_name=kv.get("objn"), rec_name=kv.get("recn"), transport=kv.get("transport") == "1", grasp_verbs=gv)
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (1, len(tn), 1))
            if kv.get("reader"):                           # orientations: the chordal mean over the target's rendered scenes (fixed yaw)
                rp = os.path.join(R, "p6_synth_rj", f"{t}.npz")
                rz = np.load(rp if os.path.exists(rp) else os.path.join(R, "p6_synth", f"{t}_rot.npz"), allow_pickle=True)
                rn = [str(x) for x in rz["names"]]
                quat = np.stack([PH.m2q(PH.mean_rotation(rz["obj_rot"][:, rn.index(n)])) for n in tn])[None]
            src = dict(names=np.array(tn), obj_quat=quat, action=parts[0][0]["action"])
            g = p_ = True
        elif kv.get("rules") == "2":                      # rules v2 (lifelong_ood/policy/phase.py at_rest + cut)
            from lifelong_ood.policy import phase as PH
            src = PH.at_rest(PH.load_demo(R, t, dsfx))
            parts = PH.cut(src)
            g = p_ = True
        else:
            g, p_ = P3.phases(src, "grasp"), P3.phases(src, "place")
            if g and p_:
                parts = g + p_
            else:
                GP, GQ = P3.P0.goals(src)
                parts = [(src, GP, GQ, 0, len(src["action"]), None, None, None)]
        state, key, fold = None, None, None
        if kv.get("expert") and os.path.exists(kv["expert"]) and kv.get("fold", "1") != "0":
            state, key, _ = adapt.load(bk, kv["expert"])
            from lifelong_ood.policy.compose import DemoProgram
            fold = DemoProgram.load(kv["foldprog"]).fold
            assert len(fold) == len(src["action"]), (len(fold), len(src["action"]))
        reader = None
        if kv.get("reader"):                              # P6: a learned pose reader in place of the client's simulator poses
            from lifelong_ood.policy.baselines import VinnPolicy
            sfx = "" if kv["reader"] == "1" else "_" + kv["reader"]          # reader=1: p6_reader_<t>.npz; reader=synth: p6_reader_synth_<t>.npz
            rz = np.load(os.path.join(R, f"p6_reader{sfx}_{t}.npz"), allow_pickle=True)
            enc = VinnPolicy(np.zeros((1, 7), np.float32), np.zeros((1, 224, 224, 3), np.uint8), np.zeros((1, 224, 224, 3), np.uint8), device="cuda")
            reader = dict(W=rz["W"], mx=rz["mx"], my=rz["my"], names=[str(x) for x in rz["names"]], enc=enc,
                          features=str(rz["features"]) if "features" in rz.files else "pooled", pool=int(rz["pool"]) if "pool" in rz.files else 4)
            if kv.get("yaw") == "1":                     # the objects' yaw where it varies (experiments/p6_yaw_reader.py)
                yz = np.load(os.path.join(R, f"p6_yaw_{t}.npz"), allow_pickle=True)
                reader["yaw"] = {k: yz[k] for k in yz.files}
            logging.info(f"phasebind: reader p6_reader{sfx}_{t}.npz ({reader['features']}){' + yaw' if kv.get('yaw') == '1' else ''}")
        hooks = PhaseBindHooks(src, parts, D.Norm.for_backbone(bk), fold=fold, state=state, key=key, var=float(kv.get("var", 1e-3)), H=bk.H,
                               reader=reader, carry="predicted" if kv.get("carry") == "pred" else "live", warp=kv.get("warp") == "1",
                               handoff=kv.get("handoff") == "1", radius=float(kv.get("hr", 0.05)), patience=float(kv.get("hp", 1.5)),
                               events=kv.get("events") == "1", tol=float(kv.get("tol", 0.02)), max_wait=int(kv.get("mw", 8)),
                               cross_pos=kv.get("crosspos") == "1", src_lift=kv.get("lift") == "1")
        hooks.no_policy = kv.get("nopolicy") == "1"       # the control: execute the target, never sample the policy
        logging.info(f"phasebind: task {t}, {len(parts)} phases ({'object phases' if g and p_ else 'world frame'}; anchors "
                     f"{[None if q[6] is None else str(q[0]['names'][q[6]]) for q in parts]}), rules {kv.get('rules', '1')}, "
                     f"sources g={kv.get('g', t)} p={kv.get('p', t)}, fold {key}")
    elif a.mode.startswith("dsrl:"):
        # dsrl:<task>?train=1&save=<actor.pt>&load=<actor.pt>&warmup=10&utd=10&beta=2 -- DSRL (lifelong_ood/policy/dsrl.py): the
        # task's one-demo LoRA switched on, the weights frozen, a SAC agent choosing each chunk's initial noise
        import numpy as np
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy.baselines import VinnPolicy
        from lifelong_ood.policy.dsrl import DSRLHooks
        body, q = (a.mode[len("dsrl:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        t = int(body)
        lp = kv.get("lora", os.path.join(ROOT, "results", "robot", f"r3_lora_{t}.pt"))
        state, key, banks = adapt.load(bk, lp)
        state["on"] = frozenset({key})
        enc = VinnPolicy(np.zeros((1, 7), np.float32), np.zeros((1, 224, 224, 3), np.uint8), np.zeros((1, 224, 224, 3), np.uint8), device="cuda")

        def feat(obs):
            f_ = enc._embed(np.asarray(obs["observation/image"])[None], np.asarray(obs["observation/wrist_image"])[None])
            return np.concatenate([f_.float().cpu().numpy()[0], np.asarray(obs["observation/state"], np.float32)])
        hooks = DSRLHooks(feat, dim=bk.H * bk.A, train=kv.get("train", "1") == "1", save=kv.get("save"), load=kv.get("load"),
                          warmup=int(kv.get("warmup", 10)), utd=int(kv.get("utd", 4)), beta=float(kv.get("beta", 2.0)),
                          seed=int(os.environ.get("SERVE_SEED", "0") or 0), log=logging.info, official=kv.get("official") == "1",
                          H=bk.H, A=bk.A)
        logging.info(f"dsrl: task {t}, LoRA {lp} on, noise dim {bk.H * bk.A}, train {hooks.train}")
    elif a.mode.startswith("lora:"):
        # lora:<expert.pt> -- the few-shot baseline: a flat adapter on the action expert, always on
        from lifelong_ood.policy import adapt
        state, key, banks = adapt.load(bk, a.mode.split(":", 1)[1])
        state["on"] = frozenset({key})
        logging.info(f"expert {key} loaded into {len(banks)} banks and switched on")
    elif a.mode.startswith("learned:"):
        # learned:<task>?w0=1.0&sched=const&var=0.001&key=pose&stall=3&window=20[&mem=0][&fold=orig]
        # compose -> practise -> consolidate: recall a stored composition (demo or practice success) by the first frames,
        # then compose along it with the refit fold (r15_consolidate.py).  mem=0: the demo's composition only; fold=orig:
        # the demo's own fold instead of the refit one
        import numpy as np
        from lifelong_ood.policy import adapt
        from lifelong_ood.policy.compose import DemoProgram, RecallHooks
        body, q = a.mode[len("learned:"):].split("?", 1)
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        R = os.path.join(ROOT, "results", "robot")
        t = int(body)
        nm = kv.get("name", "r15")
        mem = np.load(os.path.join(R, f"{nm}_{t}_memory.npz"))
        n = int(mem["n"]) if kv.get("mem", "1") != "0" else 1
        progs = [DemoProgram.load(os.path.join(R, f"{nm}_{t}_prog{k}.npz")) for k in range(n)]
        fold_path = (os.path.join(R, f"r4_{t}_approach_last2_expert.pt") if kv.get("fold") == "orig"
                     else os.path.join(R, f"{nm}_{t}_expert.pt"))
        state, key = (None, None)
        if bool(mem["grasps"]) and os.path.exists(fold_path):
            state, key, _ = adapt.load(bk, fold_path)
        hooks = RecallHooks(progs, mem["keys"][:n], state, key, w0=float(kv.get("w0", 1.0)), schedule=kv.get("sched", "const"),
                            var=float(kv["var"]) if kv.get("var") else None, stall=int(kv.get("stall", 0)),
                            key_mode=kv.get("key", "pose"), window=int(kv.get("window", 20)))
        logging.info(f"learned: task {t}, {n} stored compositions, fold {fold_path if key else None}")
    elif a.mode.startswith("demodiff:"):
        # demodiff:<task>?s=0.4&index=time|state -- DemoDiffusion: the demo chunk noised to flow time s, denoised by the prior
        from lifelong_ood.policy.baselines import DemoDiffusionHooks
        from lifelong_ood.policy.compose import DemoProgram
        body, q = (a.mode[len("demodiff:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        prog = DemoProgram.load(os.path.join(ROOT, "results", "robot", f"r3_prog_{int(body)}_raw_lib40.npz"))
        hooks = DemoDiffusionHooks(prog, s_star=float(kv.get("s", 0.2)), index=kv.get("index", "time"), key_mode="pose",
                                   window=int(kv.get("window", 20)))
        a.num_steps = int(kv.get("steps", 20))                 # DemoDiffusion's openpi fork integrates with 20 steps
        logging.info(f"demodiff: task {body}, s* {hooks.s}, index {hooks.index}, {a.num_steps} steps")
    elif a.mode.startswith("fullft:"):
        # fullft:<expert_state.pt> -- the action expert fully fine-tuned on the demo (adapt.train_full)
        import torch
        sd = torch.load(a.mode.split(":", 1)[1], map_location="cuda")
        missing = bk.expert.load_state_dict(sd["expert"], strict=True)
        logging.info(f"fully fine-tuned action expert loaded ({missing})")
    elif a.mode.startswith("promptx:"):
        # promptx:<instruction with _ for spaces> -- the prior under an arbitrary instruction (the binding read: which object moves)
        from lifelong_ood.policy.baselines import PromptHooks
        text = a.mode[len("promptx:"):].replace("_", " ")
        hooks = PromptHooks(text)
        logging.info(f"prompt replaced by '{text}'")
    elif a.mode.startswith("ftoken:"):
        # ftoken:<path.pt>[+<path.pt>...] -- the task's own prompt with factor token(s) appended (lifelong_ood/policy/token_fold.py)
        from lifelong_ood.policy.token_fold import TokenHooks
        paths = a.mode[len("ftoken:"):].split("+")
        hooks = TokenHooks(paths)
        logging.info(f"factor tokens {paths}")
    elif a.mode.startswith("factor:"):
        # factor:<libero_90 task>?lam=2&phase=approach|all&ctrl=factor|cfg|placebo&w=2 -- factor-composed generation from the policy's
        # own conditionals (lifelong_ood/policy/factor_hooks.py; the code is built by rule from the task's factors)
        import json as _json
        from lifelong_ood.policy import factor_hooks as FH
        from lifelong_ood.policy import factors as F
        body, q = (a.mode[len("factor:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        tasks = F.load()
        task = next(t for t in tasks if t.suite == "libero_90" and t.index == int(body))
        zs = {int(k): v["zeroshot"] for k, v in _json.load(open(os.path.join(ROOT, "results", "robot", "f0_coverage.json")))["tasks"].items()}
        code = FH.code_for(task, tasks, zs, lam=float(kv.get("lam", 2.0)), phase=kv.get("phase", "approach"))
        hooks = FH.FactorHooks(code, mode=kv.get("ctrl", "factor"), w=kv.get("w"), rng_seed=int(kv.get("seed", 0)),
                               train_prompts=[t.language for t in tasks if t.suite != "libero_90"])
        logging.info(f"factor code {_json.dumps(code)} ({FH.code_bytes(code)} B), control {kv.get('ctrl', 'factor')}")
    elif a.mode.startswith("bind:"):
        # bind:<task>?suite=libero_90&src=oracle&rho=0.5|beta=4&band=all|mid&cams=agent&ctrl=none|placebo|shift&phase=1 -- the
        # policy's own chunk with its action expert's attention biased toward the current phase's referent
        # (lifelong_ood/policy/bind_hooks.py; the Bind plan of 2026-10-01 evening)
        from lifelong_ood.policy import bind_hooks as BH
        from lifelong_ood.policy import factors as F
        body, q = (a.mode[len("bind:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        tasks = F.load()
        task = next(t for t in tasks if t.suite == kv.get("suite", "libero_90") and t.index == int(body))
        hooks = BH.BindHooks(bk, task, src=kv.get("src", "oracle"), rho=float(kv.get("rho", 0.5)),
                             w1_layer=(int(kv["layer"]) if "layer" in kv else None), keep=float(kv.get("keep", 0.5)),
                             phrases=F.scene_phrases(task, tasks),
                             beta=(float(kv["beta"]) if "beta" in kv else None), band=kv.get("band", "all"),
                             cams=kv.get("cams", "agent"), ctrl=kv.get("ctrl", "none"), phase=kv.get("phase", "1") == "1",
                             seed=int(kv.get("seed", 0)))
        logging.info(f"bind {task.suite}:{task.index} referents {hooks.ref} {hooks.cfg} src {hooks.src} ctrl {hooks.ctrl}")
    elif a.mode.startswith("role:"):
        # role:<task>?suite=libero_90&cams=agent[+wrist] -- the DEPLOYABLE role editor (lifelong_ood/policy/role_hooks.py): the base
        # task prototype's instruction, the new task's object and the base's object exchanged in the policy's images by
        # OWLv2 boxes (no simulator state); the code from lifelong_ood/policy/roles.py
        import glob as _glob
        import json as _json
        from lifelong_ood.policy import factors as F
        from lifelong_ood.policy import roles as RO
        from lifelong_ood.policy.role_hooks import RoleHooks
        body, q = (a.mode[len("role:"):].split("?", 1) + [""])[:2]
        kv = dict(x.split("=", 1) for x in q.split("&") if x)
        tasks = F.load()
        task = next(t for t in tasks if t.suite == kv.get("suite", "libero_90") and t.index == int(body))
        zs = {}
        for f in _glob.glob(os.path.join(ROOT, "ledger", "zeroshot90", "*.json")):
            zs.update({int(k): v["success_rate"] for k, v in _json.load(open(f))["metrics"]["tasks"].items()})
        code = RO.role_code(task, tasks, zs, _json.load(open(os.path.join(ROOT, "results", "robot", "f1_bind_libero_90_binding.json"))))
        if kv.get("ctrl") == "placebo":
            code = RO.placebo_code(task, code)
        ph = F.scene_phrases(task, tasks)
        phrases = {n: ph.get(task.objects.get(n), str(task.objects.get(n, n)).replace("_", " ")) for p_ in code["pairs"] for n in p_}
        hooks = RoleHooks(code, phrases, cams=kv.get("cams", "agent"))
        logging.info(f"role {task.suite}:{task.index} code {code} phrases {phrases} cams {hooks.cams}")
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
