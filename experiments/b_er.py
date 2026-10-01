"""Baseline (2026 continual-learning literature for VLAs: experience replay barely forgets, arXiv 2603.03818): ONE rank-16 LoRA on
pi0.5's action expert trained jointly on every held-out task's one demo, each frame with its own task's instruction, every batch
balanced across tasks -- the adapter an experience-replay learner holds at the end of the 21-task stream (all demos replayed).
Storage: one 18.7 MB adapter plus every raw demo kept for replay.

    sbatch --export=ALL,ENV=openpi,SCRIPT="experiments/b_er.py" slurm/jobs/robot_gpu.sbatch
    -> results/robot/er_lora.pt, linked as er_<task>_lora.pt (arm lora@er); ledger b_baselines tag er
"""
import argparse
import importlib.util
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

HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-task", type=int, default=2, help="rows per task in every batch")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    r3p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r3p)
    texts, _ = r3p.libero90_texts()
    from lifelong_ood.policy.backbone import PolicyBackbone
    import numpy as np
    bk = PolicyBackbone("pi05_libero", device="cuda")
    d90 = D.Demos90()
    norm = D.Norm.load()
    obs, rows, groups, per = [], [], [], {}
    for t in HELD:
        ep = d90.load(d90.episodes_of_text(texts[t])[0])
        T = len(ep["actions"])
        idx, _ = D.chunk_indices(T, bk.H)
        start = len(obs)
        obs += [d90.observation(ep, i) for i in range(T)]
        rows.append(D.pad_rows(D.to_rows(ep["actions"][idx], norm), bk.H))
        groups.append((np.arange(start, start + T), a.per_task))
        per[str(t)] = T
    full = np.concatenate(rows)
    bank = adapt.PrefixBank(bk, obs, store="cpu")             # ~26 MB per frame in host memory (request ~128 GB)
    print(f"[er] {len(HELD)} demos, {bank.n} frames; batch {a.per_task * len(HELD)}", flush=True)
    state, banks = adapt.inject(bk, scope="expert")
    key = "er_all"
    params = add_node(banks, key, a.rank)
    t1 = time.time()
    hist = adapt.train(bk, state, key, params, bank, full, steps=a.steps, lr=a.lr, batch=a.per_task * len(HELD), seed=a.seed, groups=groups)
    secs = time.time() - t1
    path = os.path.join(ROOT, "results", "robot", "er_lora.pt")
    nbytes = adapt.save(banks, key, path, meta=dict(tasks=HELD, rank=a.rank, scope="expert"))
    for t in HELD:
        link = os.path.join(ROOT, "results", "robot", f"er_{t}_lora.pt")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink("er_lora.pt", link)
    m = dict(tasks=HELD, frames=per, n_frames=int(bank.n), final_loss=hist["loss"][-1], first_loss=hist["loss"][0], curve=hist, train_secs=secs)
    print(ledger.record("b_baselines", bench="libero_90", tag="er", seed=a.seed, metrics=m,
                        bytes=dict(experts=nbytes, tree=0, codes=0, samples=0, teacher_copies=0, total=nbytes), config=vars(a),
                        script="experiments/b_er.py", command=" ".join(sys.argv), artifacts=[path]))
    print(f"[er] saved {path} ({nbytes / 1e6:.2f} MB); loss {hist['loss'][0]:.4f} -> {hist['loss'][-1]:.4f} in {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
