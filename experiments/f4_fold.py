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
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--source", default="demo")
    ap.add_argument("--n-tokens", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    text = texts[a.task]
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    for q in bk.model.parameters():
        q.requires_grad_(False)
    t0 = time.time()
    if a.source != "demo":
        raise SystemExit("only source=demo is implemented so far")
    d90 = D.Demos90()
    ep_idx = d90.episodes_of_text(text)[a.demo]
    ep = d90.load(ep_idx)
    T = len(ep["actions"])
    obs = [d90.observation(ep, i) for i in range(T)]
    bank = adapt.PrefixBank(bk, obs, store="cpu")
    idx, _ = D.chunk_indices(T, bk.H)
    full = D.pad_rows(D.to_rows(ep["actions"][idx], D.Norm.load()), bk.H)
    print(f"[f4] task {a.task} '{text}': demo episode {ep_idx}, {T} frames cached in {time.time() - t0:.0f}s", flush=True)
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
    path = os.path.join(ROOT, "results", "robot", f"ftoken_{a.task}{a.tag}.pt")
    torch.save(dict(token=token.state_dict_fp16(), task=a.task, text=text, source=a.source, demo_episode=int(ep_idx)), path)
    m = dict(task=a.task, text=text, source=a.source, demo_episode=int(ep_idx), n_frames=int(T), n_tokens=token.n,
             first_loss=hist["loss"][0], final_loss=hist["loss"][-1], curve=hist, train_secs=time.time() - t1)
    print(ledger.record("f4_fold", bench="libero_90", tag=f"token_{a.source}{a.tag}", seed=a.seed, keys=dict(task=a.task),
                        metrics=m, bytes=dict(experts=token.nbytes(), tree=0, codes=0, samples=0, teacher_copies=0,
                                              total=token.nbytes()),
                        config=vars(a), script="experiments/f4_fold.py", command=" ".join(sys.argv), artifacts=[path]))
    print(f"[f4] saved {path}: flow loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
