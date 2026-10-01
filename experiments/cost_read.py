"""COST and LATENCY of every method on the 21 held-out tasks: success, bytes stored, demos, practice episodes, compute to adapt a new
task, and inference latency per policy query (from the scoring runs' server logs: '[serve] N calls, X ms each', three servers
sharing one A40, as every method was scored).  -> results/robot/cost_read.json; ledger cost tag read.

    python experiments/cost_read.py [--record]
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
HELD = [35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63]
ARMS = [("zero-shot pi0.5", "zeroshot"), ("LoRA, one demo", "lora"), ("LoRA + failure negatives", "lora_r17bal"),
        ("STRAP", "lora_strap"), ("STRAP, authors' retrieval", "lora_strap2"), ("DSRL (our variant)", "dsrl"),
        ("DSRL, authors' configuration", "dsrlo"), ("experience replay", "lora_er"),
        ("bindings, simulator poses", "phasebind2_nofold"), ("bindings, full reader", "phasebind2_reader_spatial4_synth"),
        ("bindings, 7 KB reader head", "phasebind2_rjpred_pca512_rj")]


def pack_index():
    """OUT path -> (job id, instance) from the pack records and the jobs' logs."""
    spec_job = {}
    for f in glob.glob(os.path.join(ROOT, "slurm", "*.out")):
        try:
            with open(f, errors="ignore") as fh:
                head = fh.read(2000)
        except OSError:
            continue
        m = re.search(r"specs (\S+\.tsv)", head)
        if m:
            spec_job[m.group(1)] = os.path.basename(f)[:-4]
    out = {}
    for tsv, job in spec_job.items():
        if not os.path.exists(tsv):
            continue
        for i, line in enumerate(open(tsv)):
            m = re.search(r"OUT=(\S+)", line)
            if m:
                out[m.group(1)] = (job, i)
    return out


def latency(index, tag):
    ms, calls = [], []
    for t in HELD:
        o = os.path.join(R, f"m3_final_{t}_{tag}.json")
        if o not in index or not os.path.exists(o):
            continue
        job, i = index[o]
        log = os.path.join(R, f"server_{job}_{i}.log")
        if not os.path.exists(log):
            continue
        hits = re.findall(r"\[serve\] (\d+) calls, (\d+) ms each", open(log, errors="ignore").read())
        if hits:
            n, m = map(int, hits[-1])
            ms.append(m); calls.append(n)
    if not ms:
        return None
    return dict(ms_per_query=float(np.average(ms, weights=calls)), runs=len(ms), calls=int(sum(calls)))


def successes(tag):
    k = n = 0
    for t in HELD:
        f = os.path.join(R, f"m3_final_{t}_{tag}.json")
        if os.path.exists(f):
            e = json.load(open(f))["tasks"][str(t)]["episodes"]
            k += sum(x["success"] for x in e); n += len(e)
    return k, n


def main():
    from lifelong_ood import ledger
    idx = pack_index()
    sto = [r for r in ledger.load("storage", tag="read") if not r.get("retired")][-1]["metrics"]
    lora_secs = np.mean([r["metrics"]["train_secs"] for r in ledger.load("r3_fewshot") if r.get("tag", "") == "" and "train_secs" in r["metrics"]])
    strap_secs = np.mean([r["metrics"]["train_secs"] for r in ledger.load("b_baselines", tag="strap")])
    s2 = [r["metrics"]["train_secs"] for r in ledger.load("b_baselines", tag="strap2") if "train_secs" in r["metrics"]]
    strap2_secs = float(np.mean(s2)) if s2 else None
    er_secs = [r["metrics"]["train_secs"] for r in ledger.load("b_baselines", tag="er")][-1]
    dsrl_secs = {}
    for k in ("dsrl_train", "dsrlo_train"):
        v = [json.load(open(f))["secs"] for f in glob.glob(os.path.join(R, f"dsrl_practice_*_{k}.json"))]
        dsrl_secs[k] = float(np.mean(v)) if v else None
    timing = os.path.join(R, "p6_synth_timing.json")
    render = json.load(open(timing))["secs"] if os.path.exists(timing) else None
    adapt = {"zeroshot": 0.0, "lora": lora_secs, "lora_strap": strap_secs, "lora_strap2": strap2_secs, "lora_er": er_secs / 21,
             "dsrl": dsrl_secs["dsrl_train"], "dsrlo": dsrl_secs["dsrlo_train"],
             "phasebind2_rjpred_pca512_rj": render, "phasebind2_reader_spatial4_synth": render, "phasebind2_nofold": 0.0}
    practice = {"lora_r17bal": 30, "dsrl": 100, "dsrlo": 100}
    res = dict(arms={}, render_secs=render, notes={
        "latency": "mean ms per policy query (one query = one 10-step chunk, 5 executed), three servers per A40, weighted by calls",
        "adapt": "seconds of compute to adapt one new task: training (LoRA, STRAP, experience replay per task), the practice episodes' "
                 "wall time (DSRL), rendering the task's 400 placements for the reader (bindings with a reader; experiments/p6_synth.py on task 42 with 8 CPU workers and EGL on one A40; the ridge fit adds seconds); STRAP's one-time "
                 "embedding of the prior data is not included; the phases themselves are cut from the demo in milliseconds"})
    M = [r for r in ledger.load("m3_stats", tag="final_23tasks") if not r.get("retired")][-1]["metrics"]["methods"]
    for lab, tag in ARMS:
        k, n = successes(tag)
        if tag == "lora":                                  # four development tasks were scored in an earlier experiment (m3_stats)
            k, n = M["practice_lora.prior_fails"]["k"], 420
        if n == 0:
            continue
        res["arms"][tag] = dict(label=lab, successes=k, episodes=n, latency=latency(idx, tag), adapt_secs=adapt.get(tag),
                                practice_episodes=practice.get(tag, 0))
        L = res["arms"][tag]["latency"]
        ad = adapt.get(tag)
        print(f"{lab:36s} {k:4d}/{n}  latency {L['ms_per_query'] if L else float('nan'):6.0f} ms  adapt {float('nan') if ad is None else ad:7.0f} s  practice {practice.get(tag, 0)}")
    json.dump(res, open(os.path.join(R, "cost_read.json"), "w"), indent=1)
    if "--record" in sys.argv:
        print(ledger.record("cost", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/cost_read.py", command=" ".join(sys.argv), artifacts=[os.path.join(R, "cost_read.json")]))


if __name__ == "__main__":
    main()
