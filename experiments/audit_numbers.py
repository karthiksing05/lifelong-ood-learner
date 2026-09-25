"""Re-derive every number quoted in the repository's documents from the ledger.

    python experiments/audit_numbers.py            # prints each check, exits 1 on any discrepancy

A check is (label, a function returning ledger.mean(...), the string the document quotes).  The
value is rounded to the quoted precision and must appear in at least one audited document on a
numeric boundary (so "0.71" cannot be satisfied by "0.716").  `retired_checks.json` lists labels
whose rows were retired on purpose.  tests/test_audit.py asserts this passes, that a perturbed
number is caught, and that audit_coverage reports no unaudited numeral.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger as L                                   # noqa: E402

DOCS = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "docs", "THE_TASK.md")]
RETIRED = os.path.join(ROOT, "experiments", "retired_checks.json")


def c1(metric, bench="colormnist", tag="F3A", arm="value-lora-r1"):
    return lambda: L.mean("c1_cobweb_update", metric, bench=bench, tag=tag, keys__arm=arm)


def c1_s3d(metric):
    return lambda: L.mean("c1_cobweb_update", metric, bench="shapes3d", tag=["S3A", "S3Av"], keys__arm="value-lora-r1")


def c5(metric, tag):
    return lambda: L.mean("c5_router_cobweb", metric, bench="colormnist", tag=tag)


def c3(arm):
    return lambda: L.mean("c3_flagship", "metrics.joint", keys__arm=arm)


def p0(bench):
    return lambda: L.mean("p0_sampler_parity", "metrics.max_abs_diff", bench=bench, tag="")


def p1(metric):
    return lambda: L.mean("p1_tree_build", metric, bench="colormnist", tag="", seed=0)


def p3s(metric, arm, task):
    return lambda: L.mean("p3_fold_frontier", metric, bench="colormnist", tag="_smoke", keys__arm=arm, keys__task=task)


CHECKS = [
    ("p1 tree (phase 0): best background purity, level 1", p1("metrics.levels.1.purity_bg"), "0.84"),
    ("p1 tree (phase 0): best digit purity, level 0", p1("metrics.levels.0.purity_digit"), "0.25"),
    ("p3 smoke: label-gated reference taught joint, task 0", p3s("metrics.taught_joint", "lora-all-c1@gate=label", 0), "0.778"),
    ("p3 smoke: label-gated reference taught joint, task 1", p3s("metrics.taught_joint", "lora-all-c1@gate=label", 1), "0.736"),
    ("p3 smoke: label-gated reference never-taught joint, task 0", p3s("metrics.untaught_joint", "lora-all-c1@gate=label", 0), "0.938"),
    ("p3 smoke: label-gated reference never-taught joint, task 1", p3s("metrics.untaught_joint", "lora-all-c1@gate=label", 1), "0.708"),
    ("p3 smoke: analytic-gate arm taught joint, task 0", p3s("metrics.taught_joint", "lora-all", 0), "0.083"),
    ("C1 ColorMNIST label-gated rank-1 LoRA: taught joint", c1("metrics.taught_joint"), "0.716"),
    ("C1 ColorMNIST label-gated rank-1 LoRA: never-taught joint", c1("metrics.untaught_joint"), "0.715"),
    ("C1 ColorMNIST label-gated rank-1 LoRA: old-class drop", c1("metrics.old_drop"), "0.000"),
    ("C1 ColorMNIST label-gated rank-1 LoRA: bytes", c1("bytes.total"), "275,576"),
    ("C1 Shapes3D label-gated rank-1 LoRA: taught joint", c1_s3d("metrics.taught_joint"), "0.972"),
    ("C1 Shapes3D label-gated rank-1 LoRA: never-taught joint", c1_s3d("metrics.untaught_joint"), "0.972"),
    ("C5 tree router: new-value joint", c5("metrics.new-value.joint_end", "FINAL2"), "0.771"),
    ("C5 tree router: never-taught joint", c5("metrics.untaught.joint", "FINAL2"), "0.795"),
    ("C5 tree router: bytes", c5("bytes.total", "FINAL2"), "277,601"),
    ("C5 flat store: new-value joint", c5("metrics.new-value.joint_end", "FINALFLAT2"), "0.646"),
    ("C5 flat store: never-taught joint", c5("metrics.untaught.joint", "FINALFLAT2"), "0.550"),
    ("C5 flat store: bytes", c5("bytes.total", "FINALFLAT2"), "1,844,635"),
    ("C3 flagship: address read from one image", c3("address (read, 1 image)"), "0.980"),
    ("C3 flagship: 1 KB inversion", c3("inversion (1 KB, 1 image)"), "0.913"),
    ("C3 flagship: nearest trained class", c3("nearest trained class"), "0.001"),
    ("p0 sampler parity: ColorMNIST max abs diff", p0("colormnist_nobg3"), "0.0"),
    ("p0 sampler parity: Shapes3D max abs diff", p0("s3d_full"), "0.0"),
]


def _dp(quoted):
    return len(quoted.split(".")[1]) if "." in quoted else 0


def fmt(value, quoted):
    dp = _dp(quoted)
    return format(round(float(value), dp), f",.{dp}f")


def present(quoted, texts):
    pat = re.compile(r"(?<![\d.,])" + re.escape(quoted) + r"(?![\d])")
    return any(pat.search(t) for t in texts)


def docs_text(paths=None):
    return [open(p, encoding="utf-8").read() for p in (paths or DOCS) if os.path.exists(p)]


def run(paths=None, verbose=True):
    texts = docs_text(paths)
    retired = set(json.load(open(RETIRED))) if os.path.exists(RETIRED) else set()
    bad = 0
    for label, fn, quoted in CHECKS:
        if label in retired:
            continue
        m, se, n = fn()
        if n == 0:
            bad += 1
            if verbose:
                print(f"MISSING   {label}: no ledger rows (quoted {quoted})")
            continue
        got = fmt(m, quoted)
        ok_val = got == quoted
        ok_doc = present(quoted, texts)
        if not (ok_val and ok_doc):
            bad += 1
        if verbose:
            tag = "ok       " if ok_val and ok_doc else ("VALUE    " if not ok_val else "UNQUOTED ")
            print(f"{tag} {label}: ledger {got} (n={n}, se={se:.4f}) quoted {quoted}")
    if verbose:
        print(f"{bad} discrepancies over {len(CHECKS) - len(retired)} checks")
    return bad


def main():
    return 0 if run() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
