"""Launch the P5 stream's remaining runs three per GPU (experiments/roll_pack.sbatch): own-demo runs (phasebind2_rjpred) and every
composed (task, g, p) of the gated and ungated plans whose m3_final result does not exist yet.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/p5_launch.py [--dry] [--experiment dev_p5 --first 0 --trials 10 --kinds streamc,streamw]
"""
import importlib.util
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "robot")
spec = importlib.util.spec_from_file_location("lp", os.path.join(ROOT, "experiments", "launch_pack.py"))
LP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LP)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--experiment", default="m3_final")
    ap.add_argument("--first", type=int, default=30)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--kinds", default="own,streamc", help="own, streamc (composed), streamw (composed, warped carry)")
    ap.add_argument("--plans", default="p5_stream_plan_gated.json,p5_stream_plan.json")
    a = ap.parse_args()
    dry = a.dry
    kinds = a.kinds.split(",")
    pool = json.load(open(os.path.join(R, "p5_pool.json")))
    want = [(int(t), "phasebind2_rjpred") for t in pool] if "own" in kinds else []
    for f in a.plans.split(","):
        for o in json.load(open(os.path.join(R, f))).values():
            for k in ("streamc", "streamw"):
                if k in kinds:
                    want += [(s["task"], f"{k}@{s['g']}-{s['p']}") for s in o["steps"] if s["action"] == "compose"]
    running = set(subprocess.run(["bash", "-c", "cat results/robot/.packs/p_*.tsv"], capture_output=True, text=True, cwd=ROOT).stdout.split())
    specs = []
    for t, arm in dict.fromkeys(want):
        tag = arm.replace("streamc@", "streamc_").replace("streamw@", "streamw_").replace("-", "_")
        out = os.path.join(R, f"{a.experiment}_{t}_{tag}.json")
        if os.path.exists(out) or f"OUT={out}" in running:
            continue
        mode, pb, _ = LP.arm_spec(t, arm, False)
        specs.append(dict(MODE=mode, SUITE="libero_90", TASKS=t, TRIALS=a.trials, FIRST=a.first, EXPERIMENT=a.experiment, TAG=tag,
                          PROGRAM_BYTES=pb, OUT=out))
    n = len(os.listdir(os.path.join(R, ".packs")))
    for i in range(0, len(specs), 3):
        n += 1
        path = os.path.join(R, ".packs", f"p_{n:04d}.tsv")
        grp = specs[i:i + 3]
        if not dry:
            with open(path, "w") as f:
                for sp in grp:
                    f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
            j = subprocess.run(["sbatch", "--parsable", f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"],
                               capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
        else:
            j = "dry"
        print(f"{os.path.basename(path)} [{', '.join(str(sp['TASKS']) + ':' + sp['TAG'] for sp in grp)}] -> {j}")
    print(f"{len(specs)} runs")


if __name__ == "__main__":
    main()
