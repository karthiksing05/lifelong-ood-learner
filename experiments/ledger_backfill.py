"""Backfill ledger rows for the runs docs/REVIEW_2026-09.md quotes, from the result files on disk.

The review's numbers were produced at commit af8f07a by the commands in its appendix; the result files
are gitignored.  This writes one small tracked row per (run, seed[, arm]) with `backfilled: true`, the
source commit, and an explicit note of what the ORIGINAL accounting did not charge (the c-series kept 8
self-generated samples and a `copy.deepcopy(model)` teacher per stored concept and charged neither).
The aggregation mirrors experiments/c_report.py so the rows reproduce the review's tables.

    python experiments/ledger_backfill.py --dry-run
    python experiments/ledger_backfill.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                       # noqa: E402

SOURCE = "af8f07a"
NOTE_C = ("legacy accounting: the c-series charged adapters and codes only; 8 self-generated samples "
          "and a deepcopy of the backbone kept per stored concept were not charged "
          "(docs/REVIEW_2026-09.md, audit 2026-09-25)")

C1 = {"colormnist": ["F3A", "F3B", "F3C", "F3D"], "shapes3d": ["S3A", "S3Ae", "S3Av", "S3Ac", "S3B"]}
C1_CMD = ("experiments/c1_cobweb_update.py --bench {bench} --seed {seed} --criterion all --u-high "
          "--u-min-crit 0.3 --tol 0.05 --arms {arms} --tag _{tag}")
C5 = {"colormnist": ["FINAL2", "FINALFLAT2"], "shapes3d": ["FINAL", "FINAL2", "FINALFLAT"]}
C5_CMD = ("experiments/c5_router_cobweb.py --bench {bench} --seed {seed} --address-rule retrieve "
          "--q-row 1.0 --read-band {mode} --tag _{tag}")
C3_BYTES = {"address (oracle labels)": 72, "address (read, 1 image)": 72, "address (read, 16 images)": 72,
            "inversion (1 KB, 1 image)": 1024, "nearest trained class": 0, "unconditional": 0}


def _bytes(experts=0, codes=0, samples=0, teacher=0, tree=0):
    return dict(experts=int(experts), tree=int(tree), codes=int(codes), samples=int(samples),
                teacher_copies=int(teacher), total=int(experts + codes + samples + teacher + tree))


def c1_rows(path, bench, tag):
    r = json.load(open(path))
    seed = r["config"]["seed"]
    arms = []
    for row in r["rows"]:
        if row["arm"] not in arms:
            arms.append(row["arm"])
    out = []
    for a in arms:
        rs = [x for x in r["rows"] if x["arm"] == a]
        last = max(rs, key=lambda x: x["task"])
        added = 0 if a.startswith("lora-consol") else 4 * int(last["lora_params"])
        m = dict(taught_joint=last["taught_joint"],
                 taught_new=last.get("taught_new", last.get("taught_bg")),
                 untaught_joint=last["untaught_joint"],
                 untaught_new=last.get("untaught_new", last.get("untaught_bg")),
                 untaught_mean=float(np.mean([x["untaught_joint"] for x in rs])),
                 old_drop=last["old_drop"], drift_pct=100.0 * last["drift_mean"],
                 changed_pct=100.0 * last["changed_frac"], steps=float(np.mean([x["steps_weight"] for x in rs])),
                 taught_cov=last.get("taught_cov"), taught_vendi=last.get("taught_vendi"),
                 taught_mem=last.get("taught_mem"), last_task=int(last["task"]), n_tasks=len(rs),
                 absorbed=last.get("absorbed"))
        out.append(dict(experiment="c1_cobweb_update", bench=bench, tag=tag, seed=seed, keys={"arm": a},
                        metrics=m, bytes=_bytes(experts=added), config=r["config"],
                        script="experiments/c1_cobweb_update.py",
                        command=C1_CMD.format(bench=bench, seed=seed, arms=r["config"].get("arms", "?"), tag=tag),
                        backbone=dict(bench=bench, new_factor=r.get("new_factor")),
                        artifacts=dict(results_json=os.path.relpath(path, ROOT), sha256=ledger.sha256_file(path))))
    return out


def c5_rows(path, bench, tag):
    r = json.load(open(path))
    cfg, s = r["config"], r["summary"]
    seed = cfg["seed"]
    m = {}
    for pop in ("in-dist", "new-comb", "new-value"):
        rs = [x for x in r["records"] if x["pop"] == pop]
        if not rs:
            continue
        tiers = {}
        for x in rs:
            tiers[x["tier"]] = tiers.get(x["tier"], 0) + 1
        m[pop] = dict(joint=float(np.mean([x["joint"] for x in rs])),
                      joint_end=float(np.mean([x["joint_end"] for x in rs])),
                      stored_ok=float(np.mean([x.get("stored_ok", True) for x in rs])),
                      bytes_per_concept=float(np.sum([x["cost_bytes"] for x in rs]) / len(rs)),
                      tiers=tiers, n=len(rs))
    if "untaught" in s:
        m["untaught"] = dict(addressable=s["untaught"].get("addressable"), joint=s["untaught"].get("joint"))
    m.update(old_joint_start=s.get("old_joint_start"), old_joint_end=s.get("old_joint_end"),
             n_leaves=s.get("n_leaves"), n_merged=s.get("n_merged"), n_blocks=s.get("n_blocks"),
             merges_right_factor=sum(int(bool(x.get("right_factor"))) for x in s.get("merges", [])),
             n_merges=len(s.get("merges", [])), minutes=s.get("minutes"))
    adapter = int(s.get("adapter_bytes", 0))
    total = int(s.get("bytes_total", 0))
    n_adapters = int(s.get("n_merged", 0) or 0) + int(s.get("n_blocks", 0) or 0)
    experts = min(total, adapter * max(n_adapters, 1)) if adapter else 0
    mode = "--flat" if cfg.get("flat") else "--incorporate"
    return [dict(experiment="c5_router_cobweb", bench=bench, tag=tag, seed=seed, keys={"layout": "flat" if cfg.get("flat") else "tree"},
                 metrics=m, bytes=_bytes(experts=experts, codes=max(total - experts, 0)), config=cfg,
                 script="experiments/c5_router_cobweb.py",
                 command=C5_CMD.format(bench=bench, seed=seed, mode=mode, tag=tag),
                 backbone=dict(bench=bench),
                 artifacts=dict(results_json=os.path.relpath(path, ROOT), sha256=ledger.sha256_file(path)))]


def c3_rows(path):
    r = json.load(open(path))
    out = []
    for arm, d in r["summary"].items():
        if not isinstance(d, dict):
            continue
        ci = d.get("ci") or [None, None]
        m = dict(joint=d.get("joint"), ci_lo=ci[0], ci_hi=ci[1], cells_ge_half=d.get("cells_ge_half"),
                 coverage=d.get("coverage"), vendi=d.get("vendi"), memorized=d.get("memorized"),
                 read_acc_1=r["summary"].get("read_acc_1"), read_acc_16=r["summary"].get("read_acc_16"),
                 n_cells=len(r.get("cells", [])), basis_r2=r.get("basis_r2"))
        out.append(dict(experiment="c3_flagship", bench="shapes3d", tag="ALL", seed=0, keys={"arm": arm},
                        metrics=m, bytes=_bytes(codes=C3_BYTES.get(arm, 0)), config=r["config"],
                        script="experiments/c3_flagship.py",
                        command="experiments/c3_flagship.py --seeds 3 --cells a:b --tag _partK; --merge --tag _ALL",
                        backbone=dict(bench="shapes3d", note="eurecom pretrained UNet, legacy rows, pretraining contains the held-out cells"),
                        artifacts=dict(results_json=os.path.relpath(path, ROOT), sha256=ledger.sha256_file(path))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger-dir", default=None)
    args = ap.parse_args()
    rows = []
    for bench, tags in C1.items():
        for tag in tags:
            for f in sorted(glob.glob(os.path.join(ROOT, "results", f"c1_cobweb_update_{bench}_{tag}__seed*.json"))):
                rows += c1_rows(f, bench, tag)
    for bench, tags in C5.items():
        for tag in tags:
            for f in sorted(glob.glob(os.path.join(ROOT, "results", f"c5_router_cobweb_{bench}_{tag}__seed*.json"))):
                rows += c5_rows(f, bench, tag)
    f3 = os.path.join(ROOT, "results", "c3_flagship_ALL__seed0.json")
    if os.path.exists(f3):
        rows += c3_rows(f3)
    print(f"[backfill] {len(rows)} rows from {SOURCE}")
    for row in rows:
        rid = ledger.run_id(row["experiment"], row["bench"], row["tag"], row["seed"], SOURCE, row["keys"])
        if args.dry_run:
            print(f"  {rid}  bytes={row['bytes']['total']}")
            continue
        ledger.record(backfilled=True, source_commit=SOURCE, notes=NOTE_C, ledger_dir=args.ledger_dir, **row)
    if not args.dry_run:
        print(f"[backfill] wrote {len(rows)} rows under {args.ledger_dir or ledger.LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
