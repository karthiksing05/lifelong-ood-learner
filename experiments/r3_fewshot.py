"""M3 baseline: few-shot (ONE demo) flat LoRA on pi0.5's action expert for a held-out LIBERO-90 task.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/r3_fewshot.py --task 35" slurm/jobs/robot_gpu.sbatch

THE demo = the first episode of the task's OWN scene block (data.Demos90.episodes_of_task; before 2026-10-04 the first
episode of its text, another scene's for tasks that share an instruction: experiments/l90_scenes.py).  Every frame is a
training pair (observation, 10-step chunk); the VLM prefix is cached once; KeyedLoRA (rank r) on every Linear of the
expert (lm_head / embeddings excluded) is trained with openpi's flow-matching loss.  Saves
results/robot/r3_lora_<task>{tag}.pt; ledger row r3_fewshot with the loss curve, parameters and bytes (fp16).
"""
import argparse
import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/coc/flash10/ksingara3/robot/openpi/src")
from lifelong_ood import ledger                                        # noqa: E402
from lifelong_ood.experts.lora import add_node                         # noqa: E402
from lifelong_ood.policy import adapt                                  # noqa: E402
from lifelong_ood.policy import data as D                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help="LIBERO-90 benchmark id")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--demo", type=int, default=0, help="which IPEC episode of the text is THE demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scope", default="expert", help="where the adapter goes (adapt.SCOPES): expert | last2 | out")
    ap.add_argument("--full", action="store_true", help="FULL fine-tuning of the action expert instead of a LoRA (lr 2.5e-5)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sys.argv_saved = list(sys.argv)
    import importlib.util
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec); spec.loader.exec_module(r3p)
    texts, names = r3p.libero90_texts()
    text = texts[a.task]
    from lifelong_ood.policy.backbone import PolicyBackbone
    bk = PolicyBackbone("pi05_libero", device="cuda")
    print(f"[r3f] {bk.describe()}", flush=True)
    d90 = D.Demos90()
    ep_idx = d90.episodes_of_task(a.task)[a.demo]
    ep = d90.load(ep_idx)
    T = len(ep["actions"])
    t0 = time.time()
    obs = [d90.observation(ep, i) for i in range(T)]
    bank = adapt.PrefixBank(bk, obs)
    idx, _ = D.chunk_indices(T, bk.H)
    rows = D.to_rows(ep["actions"][idx], D.Norm.load())
    full = D.pad_rows(rows, bk.H)
    print(f"[r3f] task {a.task} '{text}': demo episode {ep_idx}, {T} frames; prefixes cached in {time.time() - t0:.0f}s ({bank.bytes() / 1e9:.2f} GB)", flush=True)
    if a.full:
        import torch
        t1 = time.time()
        hist = adapt.train_full(bk, bank, full, steps=a.steps, lr=2.5e-5 if a.lr == 1e-4 else a.lr, batch=a.batch, seed=a.seed,
                                log=lambda x: print(x, flush=True))
        path = f"results/robot/r3_full_{a.task}{a.tag}.pt"
        torch.save(dict(expert={k: v.detach().cpu() for k, v in bk.expert.state_dict().items()}, task=a.task), path)
        nbytes = int(sum(v.numel() * v.element_size() for v in bk.expert.state_dict().values()))
        m = dict(task=a.task, text=text, demo_episode=int(ep_idx), n_frames=int(T), final_loss=hist["loss"][-1],
                 first_loss=hist["loss"][0], curve=hist, train_secs=time.time() - t1, full=True)
        ledger.record("r3_fewshot", bench="libero_90", tag=a.tag or "_full", seed=a.seed, keys=dict(task=a.task), metrics=m,
                      bytes=dict(experts=nbytes, tree=0, codes=0, samples=0, teacher_copies=0, total=nbytes),
                      config=vars(a), script="experiments/r3_fewshot.py", command=" ".join(sys.argv_saved), artifacts=[path])
        print(f"[r3f] saved {path} ({nbytes / 1e6:.0f} MB); flow loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f}", flush=True)
        return
    state, banks = adapt.inject(bk, scope=a.scope)
    key = f"l90_{a.task}"
    params = add_node(banks, key, a.rank)
    npar = adapt.n_params(params)
    print(f"[r3f] {len(banks)} banks, rank {a.rank}: {npar:,} parameters ({2 * npar / 1e6:.2f} MB fp16)", flush=True)
    t1 = time.time()
    hist = adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=a.lr, batch=a.batch, seed=a.seed)
    secs = time.time() - t1
    os.makedirs("results/robot", exist_ok=True)
    path = f"results/robot/r3_lora_{a.task}{a.tag}.pt"
    nbytes = adapt.save(banks, key, path, meta=dict(task=a.task, text=text, demo_episode=int(ep_idx), rank=a.rank, scope=a.scope))
    m = dict(task=a.task, text=text, demo_episode=int(ep_idx), n_frames=int(T), n_params=npar, n_banks=len(banks),
             final_loss=hist["loss"][-1], first_loss=hist["loss"][0], curve=hist, train_secs=secs)
    ledger.record("r3_fewshot", bench="libero_90", tag=a.tag, seed=a.seed, keys=dict(task=a.task), metrics=m,
                  bytes=dict(experts=nbytes, tree=0, codes=0, samples=0, teacher_copies=0, total=nbytes),
                  config=vars(a), script="experiments/r3_fewshot.py", command=" ".join(sys.argv_saved), artifacts=[path])
    print(f"[r3f] saved {path} ({nbytes / 1e6:.2f} MB); flow loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f} in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
