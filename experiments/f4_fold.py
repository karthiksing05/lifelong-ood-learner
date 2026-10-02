"""The overhaul's FOLD (step 4): a factor token (lifelong_ood/policy/token_fold.py) fitted with every policy weight frozen.

    source=demo     the arrival's one demonstration (THE demo of experiments/r3_fewshot.py, the same frames and chunks the
                    one-demo LoRA trains on): for a primitive no composition reaches; the demo is training data, never executed
    source=<json>   a composition's own successful rollouts (recorded with --record; self-distillation, no demo): later

The token is appended to the task's own prompt (its cache entries start as the mean of the prompt's language entries) and
fitted by flow matching; only its 18.4 KB move.  Saves results/robot/ftoken_<task><tag>.pt and a ledger row f4_fold.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/f4_fold.py --task 48" slurm/jobs/robot_gpu.sbatch
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.policy import adapt                                  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402
from lifelong_ood.policy import token_fold as TF                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="a task id, or several joined by + (one token fitted on all of their demos: "
                    "a value shared by arrivals that differ elsewhere is identifiable as a factor, docs/COBWEB_UPDATE.md P7)")
    ap.add_argument("--source", default="demo")
    ap.add_argument("--n-tokens", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rec", default="", help="source=rollouts: '+'-joined globs of recording dirs (<out>_rec); the SUCCESSFUL "
                    "episodes' real frames and executed actions are the training data, under the task's own instruction")
    ap.add_argument("--stride", type=int, default=5, help="source=rollouts: one frame in `stride` (5 = the policy's query frames)")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--grasp-only", action="store_true", help="source=rollouts: keep only frames up to the first gripper close + 5 "
                    "steps (the approach and the grasp: the object, without the source's placement)")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    tids = [int(x) for x in str(a.task).split("+")]
    text = " | ".join(texts[t] for t in tids)
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    for q in bk.model.parameters():
        q.requires_grad_(False)
    t0 = time.time()
    obs, fulls, eps_ = [], [], []
    if a.source in ("rollouts", "demo+rollouts"):
        # SELF-DISTILLATION (no demonstration): the policy's own successful episodes, recorded with the REAL scene, relabelled
        # with the task's own instruction (e.g. role-rebound episodes: base prompt + exchanged appearances at execution)
        import glob as _glob
        files = sorted(f for g in a.rec.split("+") for f in _glob.glob(os.path.join(g, "*.npz")))
        norm = D.Norm.load()
        for f in files:
            z = np.load(f, allow_pickle=True)
            tt = int(z["task"])
            if not bool(z["success"]) or tt not in tids or "image" not in z.files:
                continue
            acts, sts = np.asarray(z["action"], np.float32), np.asarray(z["state"], np.float32)
            T_ = len(acts)
            idx, _ = D.chunk_indices(T_, bk.H)
            rows = D.pad_rows(D.to_rows(acts[idx], norm), bk.H)
            stop = T_
            if a.grasp_only:
                cl = np.where(acts[:, 6] > 0)[0]
                stop = min(T_, int(cl[0]) + 5) if len(cl) else T_
            for i in range(0, stop, a.stride):
                obs.append({"observation/image": z["image"][i], "observation/wrist_image": z["wrist"][i],
                            "observation/state": sts[i], "prompt": texts[tt]})
                fulls.append(rows[i:i + 1])
            eps_.append(os.path.relpath(f, ROOT))
        if len(obs) > a.max_frames:
            keep = np.linspace(0, len(obs) - 1, a.max_frames).astype(int)
            obs, fulls = [obs[i] for i in keep], [fulls[i] for i in keep]
        if not obs and a.source == "rollouts":
            raise SystemExit("no successful recorded episodes with images for these tasks")
    if a.source in ("demo", "demo+rollouts"):
        d90 = D.Demos90()
        for t in tids:
            ep_idx = d90.episodes_of_text(texts[t])[a.demo]
            ep = d90.load(ep_idx)
            T_ = len(ep["actions"])
            obs += [d90.observation(ep, i) for i in range(T_)]
            idx, _ = D.chunk_indices(T_, bk.H)
            fulls.append(D.pad_rows(D.to_rows(ep["actions"][idx], D.Norm.load()), bk.H))
            eps_.append(int(ep_idx))
    if a.source not in ("demo", "rollouts", "demo+rollouts"):
        raise SystemExit(f"unknown source {a.source}")
    T = len(obs)
    bank = adapt.PrefixBank(bk, obs, store="cpu")
    full = np.concatenate(fulls)
    ep_idx = eps_[0] if len(eps_) == 1 else eps_
    print(f"[f4] tasks {tids} '{text}': source {a.source}, episodes {len(eps_)}, {T} frames cached in {time.time() - t0:.0f}s", flush=True)
    pre0 = bank.take([0])
    pad = pre0.pad_masks[0].cpu().numpy().astype(bool)
    L = len(pad)
    n_lang = 200 if L > 200 else L // 2                                   # pi0.5: the prompt's 200 language slots end the prefix
    lang = [i for i in range(L - n_lang, L) if pad[i]]
    init = TF.KVToken.from_prefix(pre0, lang)
    token = TF.KVToken(init.k.mean(2, keepdim=True).repeat(1, 1, a.n_tokens, 1), init.v.mean(2, keepdim=True).repeat(1, 1, a.n_tokens, 1))
    token.to(bk.device)
    print(f"[f4] token: {token.n} entries per layer over {len(token.k)} layers, {token.nbytes() / 1e3:.1f} KB fp16; "
          f"initialised from {len(lang)} language entries", flush=True)
    t1 = time.time()
    hist = TF.fit(bk, token, bank, full, steps=a.steps, lr=a.lr, batch=a.batch, seed=a.seed, log=lambda x: print(x, flush=True))
    path = os.path.join(ROOT, "results", "robot", f"ftoken_{str(a.task).replace('+', '_')}{a.tag}.pt")
    torch.save(dict(token=token.state_dict_fp16(), task=a.task, text=text, source=a.source, demo_episode=ep_idx), path)
    m = dict(task=a.task, text=text, source=a.source, demo_episode=ep_idx, n_frames=int(T), n_tokens=token.n,
             first_loss=hist["loss"][0], final_loss=hist["loss"][-1], curve=hist, train_secs=time.time() - t1)
    print(ledger.record("f4_fold", bench="libero_90", tag=f"token_{a.source}{a.tag}", seed=a.seed, keys=dict(task=str(a.task)),
                        metrics=m, bytes=dict(experts=token.nbytes(), tree=0, codes=0, samples=0, teacher_copies=0,
                                              total=token.nbytes()),
                        config=vars(a), script="experiments/f4_fold.py", command=" ".join(sys.argv), artifacts=[path]))
    print(f"[f4] saved {path}: flow loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
