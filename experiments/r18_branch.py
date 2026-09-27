"""Experiment 2 of reports/RL from practice for composition.md: branch at the stall to locate what rescues the persistent failures.

Instances: the 123 (task, init state) pairs of init states 0-19 on which the first composition (m5_practice, recorded)
AND both fresh-noise retries (m5_redo _seed1, _seed2) failed.  Per instance, the stall = the first policy query at which the
composition's demo-alignment pointer (recomputed from the recorded states, the same locator as Experiment 1) reaches its final
maximum.  A BRANCH re-enters the recorded episode at query `at`: the client rebuilds the recorded session's simulator (re-seed,
then as many resets as preceded the episode: MuJoCo's solver warm-start is not in LIBERO's init state and depends on them) and
executes the RECORDED actions up to there, while the same server -- program, fold, hooks -- follows every prefix observation
without sampling (its pointer and stall state evolve exactly as recorded); from `at` on the policy acts with ONE change:

    late  at = max(0, stall - 2): replay (the recorded noise, nothing changed: the knobs' matched control), noise1..noise4
          (fresh noise), wx0.5 / wx2 (guide weight), leaf2 (every action dimension's next-ranked leaf, from the aligned frame
          on), release (the frozen prior alone: no guide, no fold); where the task has a fold, also nofold (the program guides
          the fold windows too) and foldall (the fold expert everywhere, no guide) -- the guide knobs do nothing inside fold
          windows, where 54 of the 123 stalls sit
    early at = max(0, stall - 8): noise1, noise2
    full replay: the recorded actions for the whole episode (the first all-fail instance of each task): the simulator check

Smoke tests 2026-09-27: without the reset count only the episode in the recorded session position replayed (the others
diverged at the first contact, step 65 of task 58 state 3; one such branch succeeded before its knob switched on); with it,
all replay bit-identically (experiments/r18_sim_check.py).  A branch RESCUES its instance if the episode then succeeds.  Every branch records its states; the prefix
must match the recording (the prefix gate of the pre-registration, ledger m3_prereg tags branch and branch_amend1).

    python experiments/r18_branch.py prep                # stalls, seeds, next-ranked-leaf programs, branch shards (CPU)
    python experiments/r18_branch.py lists               # the branch shards again from r18_branch_specs.json
    python experiments/r18_branch.py launch [--dry]      # packs of 3 instances per GPU (roll_pack.sbatch)
    python experiments/r18_branch.py stats [--record]    # the prefix gate and the rescue reads
    python experiments/r18_branch.py recover [--dry]     # part b (if >= 10 concentrated instances rescued): r18rec training
                                                         # (r17_nft.py --recovery) + its scoring on init states 30-49
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                        # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SPEC = os.path.join(R, "r18_branch_specs.json")
LATE = ["replay", "noise1", "noise2", "noise3", "noise4", "wx0.5", "wx2", "leaf2", "release"]
EARLY = ["noise1", "noise2"]
FOLDKNOBS = ["nofold", "foldall"]
CONCENTRATED = [42, 32, 1, 35, 40, 36, 52, 62, 41]
SHARD = 30
WARM = 2                                   # serve.warm(): two calls before the first episode


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def practice_files():
    """task -> (tag, practice json) of the first composition (m5_practice)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(R, "m5_practice_*.json"))):
        b = os.path.basename(f)[:-5].split("_")
        out[int(b[2])] = ("_".join(b[3:]), f)
    return out


def episodes(path, t):
    return json.load(open(path))["tasks"][str(t)]["episodes"]


def all_fail():
    """task -> sorted init states that failed the practice and both fresh-noise retries."""
    res = {}
    for t, (tag, f) in practice_files().items():
        r1 = {e["trial"]: e for e in episodes(os.path.join(R, f"m5_redo_{t}_{tag}_seed1.json"), t)}
        r2 = {e["trial"]: e for e in episodes(os.path.join(R, f"m5_redo_{t}_{tag}_seed2.json"), t)}
        ks = [e["trial"] for e in episodes(f, t) if not e["success"] and not r1[e["trial"]]["success"] and not r2[e["trial"]]["success"]]
        if ks:
            res[t] = sorted(ks)
    return res


def seed_bases(path, t):
    """trial -> the practice server's call count at the episode's first query (serve.GuidedPolicy: seed = 0 + calls)."""
    out, c = {}, WARM
    for e in episodes(path, t):
        out[e["trial"]] = c
        c += int(e["calls"])
    return out


def leaf2_program(t, lib, d90, norm, texts):
    """The task's borrowing program with every dimension's NEXT-ranked leaf among the frame's K prototypes."""
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import dimcode
    from lifelong_ood.policy.compose import DemoProgram
    from lifelong_ood.operators.decompose import loglik_perdim
    prog = DemoProgram.load(os.path.join(R, f"r7_dim_{t}_s1.npz"))
    ep = d90.load(d90.episodes_of_text(texts[t])[0])
    T = len(ep["actions"])
    rows = D.to_rows(ep["actions"][D.chunk_indices(T, 10)[0]], norm)
    assert len(rows) == len(prog.mu)
    mus, vs, ch2, gap = [], [], [], []
    for f, r in enumerate(np.asarray(rows, np.float32)):
        ids = prog.dim_ids[f]
        L = loglik_perdim(r, lib.M, lib.V)
        Lg = np.stack([L[:, g].sum(1) for g in dimcode.GROUPS], 1)[[lib.index[int(i)] for i in ids]]   # [K, 7] (dimcode.encode's)
        assert np.array_equal(np.argmax(Lg, 0), prog.dim_choice[f]), f"task {t} frame {f}: the stored choice is not the best leaf"
        order = np.argsort(-Lg, 0)
        c2 = order[1].astype(np.int64)
        m, v = dimcode.decode(lib, ids, c2)
        mus.append(m); vs.append(v); ch2.append(c2)
        gap.append(float(np.sqrt(((m - prog.mu[f]) ** 2).mean())))
    alt = DemoProgram(mu=np.stack(mus), var=np.stack(vs), key=prog.key, codes=[], variant="dimcode", states=prog.states,
                      dim_ids=prog.dim_ids, dim_choice=np.stack(ch2), fold=prog.fold)
    path = os.path.join(R, f"r18_leaf2_{t}.npz")
    alt.save(path)
    return path, float(np.median(gap))


def prep(a):
    r17 = load_module("r17", os.path.join(ROOT, "experiments", "r17_nft.py"))
    r3p = load_module("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    from lifelong_ood.policy import data as D
    from lifelong_ood.policy import dimcode
    from lifelong_ood.policy.compose import DemoProgram
    from lifelong_ood.tree.nodes import Tree
    texts, _ = r3p.libero90_texts()
    d90, norm = D.Demos90(), D.Norm.load()
    lib = dimcode.Library(Tree.load(os.path.join(R, "r2b_libero_40_tree.npz")))
    modes = {}
    for r in ledger.load("m5_practice"):
        if r.get("retired"):
            continue
        modes[int(r["keys"]["tasks"])] = r["keys"]["mode"]
    fails, pf = all_fail(), practice_files()
    print(f"[r18] {sum(map(len, fails.values()))} all-fail instances over {len(fails)} tasks")
    spec = dict(tasks={}, instances=[])
    import time
    for t, ks in sorted(fails.items()):
        t0 = time.time()
        tag, f = pf[t]
        sb = seed_bases(f, t)
        demo = DemoProgram.load(os.path.join(R, f"r3_prog_{t}_raw_lib40.npz"))
        alt, gap = leaf2_program(t, lib, d90, norm, texts)
        spec["tasks"][str(t)] = dict(tag=tag, mode=modes[t] + f"&alt={alt}", alt=alt, leaf2_rms_median=gap, text=texts[t])
        for k in ks:
            rec = np.load(os.path.join(R, f"m5_practice_{t}_{tag}_rec", f"task{t}_trial{k}.npz"))
            steps, trace = r17.stall_queries(demo, rec["state"], 1)
            q = int(steps[-1] // 5)
            spec["instances"].append(dict(task=t, trial=k, seed_base=sb[k], stall_query=q, stall_frame=int(trace.max()),
                                          demo_frames=len(demo.mu), queries=len(trace), late=max(0, q - 2), early=max(0, q - 8)))
        print(f"[r18] task {t}: {len(ks)} instances, stall queries {[i['stall_query'] for i in spec['instances'] if i['task'] == t]}, "
              f"next-ranked leaves {gap:.3f} rms from the executed chunks (median) [{time.time() - t0:.0f}s]", flush=True)
    json.dump(spec, open(SPEC, "w"), indent=1)
    lists(a)


def lists(a):
    """The branch lists, one per (task, shard)."""
    spec = json.load(open(SPEC))
    shards = []
    for t in sorted({i["task"] for i in spec["instances"]}):
        items = []
        inst = [i for i in spec["instances"] if i["task"] == t]
        for n, i in enumerate(inst):
            tag = spec["tasks"][str(t)]["tag"]
            base = dict(trial=i["trial"], seed_base=i["seed_base"], resets=i["trial"],       # practice sessions ran trials 0-19 in order
                        rec=os.path.join(R, f"m5_practice_{t}_{tag}_rec", f"task{t}_trial{i['trial']}.npz"))
            if n == 0:
                items.append(dict(base, id=f"k{i['trial']}_full_replay", at=10 ** 6, variant="replay"))
            items += [dict(base, id=f"k{i['trial']}_late_{v}", at=i["late"], variant=v) for v in LATE]
            if tag.startswith("dimfold"):
                items += [dict(base, id=f"k{i['trial']}_late_{v}", at=i["late"], variant=v) for v in FOLDKNOBS]
            items += [dict(base, id=f"k{i['trial']}_early_{v}", at=i["early"], variant=v) for v in EARLY]
        for s in range(0, len(items), SHARD):
            p = os.path.join(R, f"r18_branch_{t}_{s // SHARD}.list.json")
            json.dump(items[s:s + SHARD], open(p, "w"), indent=0)
            shards.append(dict(task=t, shard=s // SHARD, path=p, n=len(items[s:s + SHARD])))
    json.dump(shards, open(os.path.join(R, "r18_branch_shards.json"), "w"), indent=1)
    print(f"[r18] {sum(s['n'] for s in shards)} branch episodes in {len(shards)} shards -> {SPEC}")


def launch(a):
    spec = json.load(open(SPEC))
    shards = json.load(open(os.path.join(R, "r18_branch_shards.json")))
    os.makedirs(os.path.join(R, ".packs"), exist_ok=True)
    todo = [s for s in shards if not os.path.exists(os.path.join(R, f"r18_branch_{s['task']}_{s['shard']}.json"))]
    todo.sort(key=lambda s: -s["n"])
    n_pack = len(os.listdir(os.path.join(R, ".packs")))
    for g in range(0, len(todo), a.per):
        grp = todo[g:g + a.per]
        n_pack += 1
        path = os.path.join(R, ".packs", f"p_{n_pack:04d}.tsv")
        with open(path, "w") as f:
            for s in grp:
                t = s["task"]
                kv = dict(MODE=spec["tasks"][str(t)]["mode"], SUITE="libero_90", TASKS=t, TRIALS=0, FIRST=0, EXPERIMENT="r18_branch",
                          TAG=f"{t}_{s['shard']}", PROGRAM_BYTES="0", OUT=os.path.join(R, f"r18_branch_{t}_{s['shard']}.json"),
                          RECORD="1", BRANCHES=s["path"])
                f.write("\t".join(f"{k}={v}" for k, v in kv.items()) + "\n")
        args = ["sbatch", "--parsable", f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        if a.dry:
            print("  [dry]", " ".join(args), [f"{s['task']}_{s['shard']}" for s in grp])
            continue
        j = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
        print(f"pack {os.path.basename(path)} [{', '.join(str(s['task']) + '_' + str(s['shard']) for s in grp)}] -> job {j}")


def recover(a):
    """Part b of the pre-registration: the recovery LoRA r18rec on the 9 concentrated tasks and every task with a rescue."""
    st = json.load(open(os.path.join(R, "r18_branch_stats.json")))
    if st["rescued_concentrated"] < 10:
        print(f"[r18] {st['rescued_concentrated']} concentrated instances rescued (< 10): part b does not run (pre-registered)")
        return
    tasks = sorted(set(CONCENTRATED) | {int(t) for t, r in st["per_task"].items() if r["rescued"] > 0})
    jobs = {}
    for t in tasks:
        if os.path.exists(os.path.join(R, f"r18rec_{t}_lora.pt")):
            continue
        # the recovery data outgrow the GPU (about 26 MB of cached prefix per frame): the bank lives in host memory
        args = ["sbatch", "--parsable", "--partition=tail-lab", "--account=tail-lab", "--mem=128G",
                f"--export=ALL,ENV=openpi,SCRIPT=experiments/r17_nft.py --task {t} --name r18rec --recovery --bank-device cpu",
                "slurm/jobs/robot_gpu.sbatch"]
        if a.dry:
            print("  [dry]", " ".join(args))
            jobs[t] = "DRY"
            continue
        jobs[t] = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
        print(f"[r18] r18rec task {t} -> job {jobs[t]}")
    PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
    cmd = [PY, "experiments/launch_pack.py", "--per", "3", "--tasks", ",".join(map(str, tasks)), "--arms", "lora@r18rec"]
    if jobs:
        cmd += ["--task-after", ",".join(f"{t}:{j}" for t, j in jobs.items())]
    print(subprocess.run(cmd + (["--dry"] if a.dry else []), capture_output=True, text=True).stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "lists", "launch", "recover", "stats"])
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    os.chdir(ROOT)
    {"prep": prep, "lists": lists, "launch": launch, "recover": recover, "stats": lambda a: load_module("r18s", os.path.join(ROOT, "experiments", "r18_branch_stats.py")).main(a)}[a.cmd](a)


if __name__ == "__main__":
    main()
