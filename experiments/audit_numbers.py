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


def emb(experiment, metric, **f):
    return lambda: L.mean(experiment, metric, **f)


EMBODIED = [
    ("M0: LIBERO-Goal ms per simulation step", emb("r0_env_smoke", "metrics.ms_per_step_median", bench="libero_goal"), "16.9"),
    ("M1: parity with openpi's sampler", emb("r1_policy_smoke", "metrics.parity_max_abs_diff", bench="libero_goal"), "0.0"),
    ("M1: LIBERO-Goal success, openpi's policy", emb("r1_rollout", "metrics.success_rate", bench="libero_goal", tag="openpi"), "0.95"),
    ("M1: LIBERO-Goal success, our sampler", emb("r1_rollout", "metrics.success_rate", bench="libero_goal", tag="ours"), "0.98"),
    ("M2 conditional: mode distance to its own chunk, t=0.1", emb("r2_prototypes", "metrics.levels.0.1.mode_rms_to_own_chunk", bench="libero_goal"), "0.043"),
    ("M2 conditional: distance between chunks", emb("r2_prototypes", "metrics.between_chunk_rms", bench="libero_goal"), "0.788"),
    ("M2 conditional: distinct modes at t=0.1", emb("r2_prototypes", "metrics.levels.0.1.n_distinct", bench="libero_goal"), "942"),
    ("M2 conditional: chunks", emb("r2_prototypes", "metrics.n_chunks", bench="libero_goal"), "960"),
    ("M2 marginal: prototypes at t=0.4", emb("r2b_marginal", "metrics.policy.0.4.incidence.n_prototypes", bench="libero_goal"), "130"),
    ("M2 marginal: tasks per prototype at t=0.4", emb("r2b_marginal", "metrics.policy.0.4.incidence.tasks_per_prototype", bench="libero_goal"), "1.97"),
    ("M2 marginal: frames in shared prototypes at t=0.4", emb("r2b_marginal", "metrics.policy.0.4.incidence.shared_frames", bench="libero_goal"), "0.78"),
    ("M2 marginal: prototypes at t=0.7", emb("r2b_marginal", "metrics.policy.0.7.incidence.n_prototypes", bench="libero_goal"), "2"),
    ("M2 marginal: within-observation std", emb("r2b_marginal", "metrics.unimodality.within_obs_std", bench="libero_goal"), "0.024"),
    ("M2 marginal: between-frame rms", emb("r2b_marginal", "metrics.unimodality.between_frame_rms", bench="libero_goal"), "0.741"),
    ("ADDRESS vs Goal library: turn off the stove", emb("r3_prepare", "metrics.unexplained_frac", bench="libero_90", tag="_goallib", keys__task=39), "0.38"),
    ("ADDRESS vs Goal library: open the microwave", emb("r3_prepare", "metrics.unexplained_frac", bench="libero_90", tag="_goallib", keys__task=35), "0.28"),
    ("ADDRESS vs Goal library: white mug on the left plate", emb("r3_prepare", "metrics.unexplained_frac", bench="libero_90", tag="_goallib", keys__task=67), "0.06"),
    ("growth: frying pan on the shelf, base", emb("r5_growth", "metrics.steps.1.unexplained_base", bench="libero_90", tag="_goallib_probe"), "0.31"),
    ("growth: frying pan on the shelf, after the first fold", emb("r5_growth", "metrics.steps.1.unexplained_stream", bench="libero_90", tag="_goallib_probe"), "0.07"),
    ("growth: open the microwave, base", emb("r5_growth", "metrics.steps.4.unexplained_base", bench="libero_90", tag="_goallib_probe"), "0.17"),
    ("growth: open the microwave, after earlier folds", emb("r5_growth", "metrics.steps.4.unexplained_stream", bench="libero_90", tag="_goallib_probe"), "0.17"),
    ("40-task library: prototypes at t=0.4", emb("r2b_marginal", "metrics.policy.0.4.incidence.n_prototypes", bench="libero_40"), "242"),
    ("40-task library: tasks per prototype at t=0.4", emb("r2b_marginal", "metrics.policy.0.4.incidence.tasks_per_prototype", bench="libero_40"), "2.94"),
    ("40-task library: frames in shared prototypes at t=0.4", emb("r2b_marginal", "metrics.policy.0.4.incidence.shared_frames", bench="libero_40"), "0.94"),
    ("40-task library: gripper events at a boundary", emb("r2b_marginal", "metrics.policy.0.4.semantics.hit_rate", bench="libero_40"), "0.99"),
    ("40-task library: random boundaries", emb("r2b_marginal", "metrics.policy.0.4.semantics.random_hit_rate", bench="libero_40"), "0.51"),
    ("M3 pilot: fading guide", emb("r3_compose", "metrics.success_rate", tag="raw_pilot_w1.2_variance"), "0.0"),
    ("M3 pilot: constant guide", emb("r3_compose", "metrics.success_rate", tag="raw_pilot_w1.0_const_v0.001"), "0.4"),
    ("M3 held-out: zero-shot successes where the prior fails", emb("m3_summary", "metrics.arms.zeroshot.prior_fails.k", tag="heldout"), "1"),
    ("M3 held-out: episodes where the prior fails", emb("m3_summary", "metrics.arms.zeroshot.prior_fails.n", tag="heldout"), "80"),
    ("M3 held-out: library codes where the prior fails", emb("m3_summary", "metrics.arms.code_replay.prior_fails.rate", tag="heldout"), "0.45"),
    ("M3 held-out: library codes successes", emb("m3_summary", "metrics.arms.code_replay.prior_fails.k", tag="heldout"), "36"),
    ("M3 held-out: LoRA where the prior fails", emb("m3_summary", "metrics.arms.lora.prior_fails.rate", tag="heldout"), "0.40"),
    ("M3 held-out: LoRA successes", emb("m3_summary", "metrics.arms.lora.prior_fails.k", tag="heldout"), "32"),
    ("M3 held-out: LoRA size MB", emb("m3_summary", "metrics.lora_mb", tag="heldout"), "18.7"),
    ("M3 held-out: turn off the stove, zero-shot", emb("m3_heldout", "metrics.success_rate", tag="zeroshot", keys__tasks="39"), "0.0"),
    ("M3 held-out: turn off the stove, library codes", emb("m3_heldout", "metrics.success_rate", tag="code_replay", keys__tasks="39"), "0.9"),
    ("M3 held-out: practice then compose, all six", emb("m3_summary", "metrics.arms.practice.all.rate", tag="heldout"), "0.625"),
    ("M3 held-out: practice successes", emb("m3_summary", "metrics.arms.practice.all.k", tag="heldout"), "75"),
    ("M3 held-out: episodes, all six", emb("m3_summary", "metrics.arms.practice.all.n", tag="heldout"), "120"),
    ("M3 held-out: zero-shot, all six", emb("m3_summary", "metrics.arms.zeroshot.all.rate", tag="heldout"), "0.333"),
    ("M3 held-out: zero-shot successes, all six", emb("m3_summary", "metrics.arms.zeroshot.all.k", tag="heldout"), "40"),
    ("M3 held-out: always compose, all six", emb("m3_summary", "metrics.arms.code_replay.all.rate", tag="heldout"), "0.408"),
    ("M3 held-out: always compose successes", emb("m3_summary", "metrics.arms.code_replay.all.k", tag="heldout"), "49"),
    ("M4: hybrid, open the microwave", emb("m4_fold", "metrics.success_rate", tag="hybrid_lib40", keys__tasks="35"), "0.6"),
    ("M4: codes, open the microwave", emb("m3_compose", "metrics.success_rate", tag="code_replay", keys__tasks="35"), "0.6"),
    ("M4: hybrid, turn off the stove", emb("m4_fold", "metrics.success_rate", tag="hybrid_lib40", keys__tasks="39"), "0.5"),
    ("M4: codes, turn off the stove", emb("m3_compose", "metrics.success_rate", tag="code_replay", keys__tasks="39"), "0.6"),
    ("M4: expert on everywhere, stove", emb("m4_fold", "metrics.success_rate", tag="expert_on_lib40", keys__tasks="39"), "0.0"),
    ("gate: white mug restored", emb("m3_compose", "metrics.success_rate", tag="code_agree20", keys__tasks="67"), "0.8"),
    ("gate: moka pot", emb("m3_compose", "metrics.success_rate", tag="code_agree20", keys__tasks="19"), "0.0"),
    ("gate: stove", emb("m3_compose", "metrics.success_rate", tag="code_agree20", keys__tasks="39"), "0.0"),
]

CHECKS = EMBODIED + [
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
