"""STORAGE: what a task costs to keep, per method, and the cumulative bytes over the 50-task stream (the lifelong claim: cost
tracks the binding vocabulary, not the number of tasks; nothing is overwritten, so nothing is forgotten).

Per task, the bytes each method must keep for the task to be performed later:
- the rank-16 LoRA (with failure negatives, the best learned arm): its fp16 adapter (ledger r17_nft r17bal);
- the raw demonstration a replay-based learner keeps: LIBERO-90's raw hdf5 episode (two 128 x 128 cameras, states, actions),
  measured from the files (results/robot/.. robot/data/libero_raw);
- the phase program (rules v2): experiments/m34_report.phasebind_bytes (fp16 controller goals + 28 B per phase);
- the pose reader: the deployed ridge (fp32, experiments/m34_report.reader_bytes) or the compressed head (fp16 over a shared
  512-direction basis stored once; experiments/p6_compress.py --export pca512);
- a COMPOSED task keeps two source ids (8 B) and its reader; its phases are other tasks'.
The stream: the practice-gated plans (experiments/p5_validated.py), in arrival order.

    python experiments/storage_read.py [--record]
"""
import glob
import importlib.util
import json
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402

R = os.path.join(ROOT, "results", "robot")
RAW = "/coc/flash10/ksingara3/robot/data/libero_raw/libero_90"


def m34():
    spec = importlib.util.spec_from_file_location("m34", os.path.join(ROOT, "experiments", "m34_report.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def raw_demo_bytes():
    """Mean stored bytes of one raw LIBERO-90 episode (datasets' storage sizes, over every downloaded task file)."""
    per = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.hdf5"))):
        with h5py.File(f, "r") as h:
            eps = list(h["data"].keys())
            tot = 0
            for e in eps:
                h["data"][e].visititems(lambda n, o: None)
                def acc(n, o):
                    nonlocal tot
                    if isinstance(o, h5py.Dataset):
                        tot += o.id.get_storage_size()
                h["data"][e].visititems(acc)
            per.append(tot / len(eps))
    return float(np.mean(per)), len(per)


def main():
    M = m34()
    pool = sorted(int(t) for t in json.load(open(os.path.join(R, "p5_pool.json"))))
    lora = [r for r in ledger.load("r17_nft") if r.get("tag") == "r17bal"]
    lora_b = int(np.median([r["bytes"]["experts"] for r in lora]))
    raw_b, raw_files = raw_demo_bytes()
    exp = json.load(open(os.path.join(R, "p6_compress_export_pca512.json")))
    per = {}
    for t in pool:
        per[t] = dict(program=int(M.phasebind_bytes(t, fold=False, rules="v2")),
                      reader_full=int(M.reader_bytes(t, "spatial4_synthonly_rj")),
                      reader_pca512=int(exp["head_bytes"][str(t)]))
    basis = int(exp["basis_bytes"])
    res = dict(tasks=pool, lora_bytes=lora_b, raw_demo_bytes=raw_b, raw_files=raw_files, basis_pca512=basis, per_task=per)
    med = lambda k: float(np.median([v[k] for v in per.values()]))
    res["median"] = {k: med(k) for k in ("program", "reader_full", "reader_pca512")}
    res["own_demo_task_full"] = res["median"]["program"] + res["median"]["reader_full"]
    res["own_demo_task_pca512"] = res["median"]["program"] + res["median"]["reader_pca512"]
    # the 50-task stream: every task's own demo vs the practice-gated composition stream (orders A, B; C, D)
    streams = {}
    for f, names in (("p5_stream_plan_practice.json", "AB"), ("p5_stream_plan_practice_orders1.json", "CD")):
        P = json.load(open(os.path.join(R, f)))
        for o, pl in P.items():
            cum = dict(lora=[], raw=[], own_full=[], own_pca=[], stream_pca=[])
            tot = dict(lora=0, raw=0, own_full=0, own_pca=basis, stream_pca=basis)
            for s in pl["steps"]:
                t = s["task"]
                tot["lora"] += lora_b
                tot["raw"] += raw_b
                tot["own_full"] += per[t]["program"] + per[t]["reader_full"]
                tot["own_pca"] += per[t]["program"] + per[t]["reader_pca512"]
                tot["stream_pca"] += (8 if s["action"] == "compose" else per[t]["program"]) + per[t]["reader_pca512"]
                for k in cum:
                    cum[k].append(int(tot[k]))
            streams[o] = dict(totals={k: int(v) for k, v in tot.items()}, cumulative=cum, demos=pl["demos"])
    res["streams"] = streams
    res["ratios_50"] = {o: dict(lora_over_own_pca=v["totals"]["lora"] / v["totals"]["own_pca"],
                                lora_over_own_full=v["totals"]["lora"] / v["totals"]["own_full"],
                                raw_over_own_pca=v["totals"]["raw"] / v["totals"]["own_pca"]) for o, v in streams.items()}
    out = os.path.join(R, "storage_read.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"per task: LoRA {lora_b / 1e6:.1f} MB; raw demo {raw_b / 1e6:.1f} MB ({raw_files} files); program {res['median']['program'] / 1e3:.1f} KB; "
          f"reader {res['median']['reader_full'] / 1e3:.0f} KB -> head {res['median']['reader_pca512'] / 1e3:.1f} KB (+ basis {basis / 1e6:.1f} MB once)")
    for o, v in streams.items():
        tt = v["totals"]
        print(f"order {o} (50 tasks): LoRA {tt['lora'] / 1e6:.0f} MB, raw demos {tt['raw'] / 1e6:.0f} MB, own-demo bindings + reader "
              f"{tt['own_full'] / 1e6:.1f} MB, + compressed reader {tt['own_pca'] / 1e6:.1f} MB, composition stream {tt['stream_pca'] / 1e6:.2f} MB "
              f"({v['demos']} demos)")
    if "--record" in sys.argv:
        print(ledger.record("storage", bench="libero_90", tag="read", seed=0, metrics=res,
                            bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
                            script="experiments/storage_read.py", command=" ".join(sys.argv), artifacts=[out]))


if __name__ == "__main__":
    main()
