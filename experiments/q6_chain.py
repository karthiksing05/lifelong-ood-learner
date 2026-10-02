"""Submit the deployable scheme's fold rounds and scoring as a SLURM dependency chain (ledger m3_prereg tag q6_main).

Per task with a form: round 1 = the form (attention binding with the demo's object prototype, approach only) on practice states
10-29, recorded with images -> fit a token on its successes (_w1) -> round 2 = the token alone on states 10-29 -> refit on both
rounds (_w2) -> score the token alone, the form, and zero-shot on states 30-49.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/q6_chain.py --tasks 52+53+59 [--dry]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/flash/miniconda3/envs/chunks/bin/python")
FORM = "bindw4@rho0.7_approach"
FORM_TAG = "bindw4_rho0.7_approach"
PRE = "q6"
if "--form" in sys.argv:                       # e.g. --form bind@rho0.7_all_approach --pre q8 (oracle where during practice)
    FORM = sys.argv[sys.argv.index("--form") + 1]
    FORM_TAG = FORM.replace("@", "_")
if "--pre" in sys.argv:
    PRE = sys.argv[sys.argv.index("--pre") + 1]


def run(cmd, dry):
    print("$", " ".join(cmd), flush=True)
    if dry:
        return "DRY"
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    print(out.strip()[-300:], flush=True)
    m = re.findall(r"job (\d+)", out) or re.findall(r"^(\d+)", out, flags=re.M)
    return m[-1] if m else None


def pack(task, arm, exp, first, trials, after=None, record=False, dry=False):
    cmd = [PY, "experiments/launch_pack.py", "--tasks", str(task), "--arms", arm, "--experiment", exp, "--first", str(first),
           "--trials", str(trials), "--bind", "track", "--partition", "overcap", "--per", "1"]
    if record:
        cmd.append("--record")
    if after:
        cmd += ["--after", after]
    return run(cmd, dry)


def fit(task, recs, tag, after, dry):
    script = f"experiments/f4_fold.py --task {task} --source rollouts --rec {'+'.join(recs)} --tag {tag}"
    cmd = ["sbatch", "--parsable", f"--dependency=afterok:{after}", f"--export=ALL,ENV=openpi,SCRIPT={script}",
           "slurm/jobs/robot_gpu.sbatch"]
    return run(cmd, dry)


def main():
    tasks = [int(x) for x in sys.argv[sys.argv.index("--tasks") + 1].split("+")]
    dry = "--dry" in sys.argv
    for t in tasks:
        w1, w2 = f"_{PRE}1", f"_{PRE}2"
        p1 = pack(t, FORM, f"{PRE}_p1", 10, 20, record=True, dry=dry)
        f1 = fit(t, [f"results/robot/{PRE}_p1_{t}_{FORM_TAG}_rec"], w1, p1, dry)
        p2 = pack(t, f"ftoken@self{w1}", f"{PRE}_p2", 10, 20, after=f1, record=True, dry=dry)
        f2 = fit(t, [f"results/robot/{PRE}_p1_{t}_{FORM_TAG}_rec", f"results/robot/{PRE}_p2_{t}_ftoken_self{w1}_rec"], w2, p2, dry)
        s_tok = pack(t, f"ftoken@self{w2}", f"{PRE}_main", 30, 20, after=f2, dry=dry)
        s_form = pack(t, FORM, f"{PRE}_main", 30, 20, dry=dry)
        s_zs = pack(t, "zs", f"{PRE}_main", 30, 20, dry=dry)
        print(f"[q6] task {t}: practice {p1} fit {f1} token practice {p2} refit {f2} | score token {s_tok} form {s_form} zs {s_zs}",
              flush=True)


if __name__ == "__main__":
    main()
