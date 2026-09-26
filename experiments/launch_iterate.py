"""Chain the iterated-practice loop (ledger m3_prereg tag iterate) with SLURM dependencies.

round 1: practise L1 (r15_<t>_lora.pt) on instances 0-29, seed 1, recorded (m6_round1)
learners after round k: r16L<k+1> = LoRA rank 16 on the demo + rounds 0..k; r16eq<k+1> = rank 2 on the last two layers (258 KB)
round 2: practise r16L2 on instances 0-29, seed 2, recorded (m6_round2)
every learner is scored on init states 30-49 (m3_final, tag lora_<name>).

    python experiments/launch_iterate.py [--dry]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")


def sh(args, dry):
    if dry:
        print("  [dry]", " ".join(args))
        return "DRY"
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def pack(args, dry):
    """launch_pack.py -> {task: [pack job ids]}"""
    out = sh([PY, "experiments/launch_pack.py", "--per", "3"] + args + (["--dry"] if dry else []), False)
    jobs = {}
    for line in out.splitlines():
        m = re.match(r"pack \S+ \[(.*)\].* -> job (\S+)", line)
        if m:
            for x in m.group(1).split(","):
                jobs.setdefault(int(x.strip().split(":")[0]), []).append(m.group(2))
    return jobs


def train(t, practice, name, after, extra, dry):
    dep = [f"--dependency=afterok:{':'.join(after)}"] if after else []
    out = sh(["sbatch", "--parsable", "--partition=tail-lab", "--account=tail-lab"] + dep +
             [f"--export=ALL,ENV=openpi,SCRIPT=experiments/r15_consolidate.py --task {t} --lora --practice {practice} --name {name}{extra}",
              "slurm/jobs/robot_gpu.sbatch"], dry)
    return out.strip().split(";")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    T = ",".join(map(str, TASKS))
    SMALL = " --lora-rank 2 --lora-scope last2"
    # round 1 practice with L1, and the small learner on round-0 data
    p1 = pack(["--experiment", "m6_round1", "--first", "0", "--trials", "30", "--record", "--seed", "1", "--tasks", T, "--arms", "lora_learn"], a.dry)
    eq1 = {t: train(t, "m5_practice", "r16eq1", [], SMALL, a.dry) for t in TASKS}
    pack(["--tasks", T, "--arms", "lora@r16eq1", "--task-after", ",".join(f"{t}:{j}" for t, j in eq1.items())], a.dry)
    # learners after round 1
    L2 = {t: train(t, "m5_practice+m6_round1", "r16L2", p1.get(t, []), "", a.dry) for t in TASKS}
    eq2 = {t: train(t, "m5_practice+m6_round1", "r16eq2", p1.get(t, []), SMALL, a.dry) for t in TASKS}
    pack(["--tasks", T, "--arms", "lora@r16L2", "--task-after", ",".join(f"{t}:{j}" for t, j in L2.items())], a.dry)
    pack(["--tasks", T, "--arms", "lora@r16eq2", "--task-after", ",".join(f"{t}:{j}" for t, j in eq2.items())], a.dry)
    # round 2 practice with L2, then the learners after round 2
    p2 = pack(["--experiment", "m6_round2", "--first", "0", "--trials", "30", "--record", "--seed", "2", "--tasks", T,
               "--arms", "lora@r16L2", "--task-after", ",".join(f"{t}:{j}" for t, j in L2.items())], a.dry)
    L3 = {t: train(t, "m5_practice+m6_round1+m6_round2", "r16L3", p2.get(t, []), "", a.dry) for t in TASKS}
    eq3 = {t: train(t, "m5_practice+m6_round1+m6_round2", "r16eq3", p2.get(t, []), SMALL, a.dry) for t in TASKS}
    pack(["--tasks", T, "--arms", "lora@r16L3", "--task-after", ",".join(f"{t}:{j}" for t, j in L3.items())], a.dry)
    pack(["--tasks", T, "--arms", "lora@r16eq3", "--task-after", ",".join(f"{t}:{j}" for t, j in eq3.items())], a.dry)
    print(f"[iterate] round-1 practice packs {sorted({j for v in p1.values() for j in v})}")
    print(f"[iterate] learners: eq1 {len(eq1)}, L2 {len(L2)}, eq2 {len(eq2)}, L3 {len(L3)}, eq3 {len(eq3)} jobs; round-2 practice "
          f"packs {len({j for v in p2.values() for j in v})}")


if __name__ == "__main__":
    sys.exit(main())
