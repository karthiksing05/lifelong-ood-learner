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


def neg(fn):
    """A check quoted without its minus sign ("-0.094" in the text matches the quoted "0.094")."""
    return lambda: (-fn()[0],) + tuple(fn()[1:])


def kb(metric):
    return lambda: tuple([L.mean("m3_summary", metric, tag="final")[0] / 1e3] + list(L.mean("m3_summary", metric, tag="final")[1:]))


def mb(metric):
    return lambda: tuple([L.mean("m3_summary", metric, tag="final")[0] / 1e6] + list(L.mean("m3_summary", metric, tag="final")[1:]))


def st(metric, scale=1.0, sign=1.0, tag="final_23tasks"):
    """A metric of the pre-registered statistics row (experiments/m3_stats.py), optionally rescaled or negated."""
    def f():
        m, se, n = L.mean("m3_stats", metric, tag=tag)
        return (sign * m / scale, se / scale, n)
    return f


def st_len(key, tag="final_23tasks"):
    return lambda: (float(len(L.load("m3_stats", tag=tag)[0]["metrics"][key])), 0.0, 1)


def st_ratio(num, den, tag="final_23tasks"):
    return lambda: (L.mean("m3_stats", num, tag=tag)[0] / L.mean("m3_stats", den, tag=tag)[0], 0.0, 1)


def fam_tasks(arm, tasks, tag="read"):
    """Successes of a family-transfer arm summed over some target tasks (ledger r20_family tag read / read_r21)."""
    def f():
        rows = L.load("r20_family", tag=tag)
        if not rows:
            return (float("nan"), float("nan"), 0)
        pt = rows[-1]["metrics"]["all_targets"]["per_task"]
        return (float(sum(pt[str(t)][arm] for t in tasks)), 0.0, 1)
    return f


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
    ("pre-registered: held-out tasks", st_len("tasks"), "23"),
    ("pre-registered: tasks where the practice fails", st_len("prior_fails"), "21"),
    ("pre-registered: paired episodes per method", st("metrics.methods.ours_dim1_last2.prior_fails.n"), "420"),
    ("pre-registered: borrowing + fold successes", st("metrics.methods.ours_dim1_last2.prior_fails.k"), "253"),
    ("pre-registered: borrowing + fold MB", st("metrics.methods.ours_dim1_last2.prior_fails.bytes", 1e6), "5.3"),
    ("pre-registered: LoRA successes", st("metrics.methods.practice_lora.prior_fails.k"), "252"),
    ("pre-registered: LoRA MB", st("metrics.methods.practice_lora.prior_fails.bytes", 1e6), "392.9"),
    ("pre-registered: borrowing + fold - LoRA", st("metrics.compare.ours_dim1_last2-practice_lora.prior_fails.diff"), "0.002"),
    ("pre-registered: one-sided lower bound (negated)", st("metrics.compare.ours_dim1_last2-practice_lora.prior_fails.lo90", sign=-1.0), "0.091"),
    ("pre-registered: LoRA bytes / borrowing + fold bytes", st_ratio("metrics.methods.practice_lora.prior_fails.bytes", "metrics.methods.ours_dim1_last2.prior_fails.bytes"), "74"),
    ("pre-registered: replay successes", st("metrics.methods.practice_replay.prior_fails.k"), "171"),
    ("pre-registered: replay KB", st("metrics.methods.practice_replay.prior_fails.bytes", 1e3), "106.2"),
    ("pre-registered: borrowing + fold - replay", st("metrics.compare.ours_dim1_last2-practice_replay.prior_fails.diff"), "0.195"),
    ("borrowing every 5th frame successes", st("metrics.methods.ours_dim.prior_fails.k"), "176"),
    ("borrowing every 5th frame KB", st("metrics.methods.ours_dim.prior_fails.bytes", 1e3), "23.6"),
    ("borrowing every 5th frame - replay", st("metrics.compare.ours_dim-practice_replay.prior_fails.diff"), "0.012"),
    ("borrowing every 5th frame below the LoRA", st("metrics.compare.ours_dim-practice_lora.prior_fails.diff", sign=-1.0), "0.181"),
    ("what the fold adds", st("metrics.compare.ours_dim1_last2-ours_dim1.prior_fails.diff"), "0.202"),
    ("practice allocation successes", st("metrics.methods.ours_alloc.prior_fails.k"), "270"),
    ("practice allocation MB", st("metrics.methods.ours_alloc.prior_fails.bytes", 1e6), "97.4"),
    ("practice allocation - LoRA", st("metrics.compare.ours_alloc-practice_lora.prior_fails.diff"), "0.043"),
    ("practice allocation vs LoRA, sign test p", st("metrics.compare.ours_alloc-practice_lora.prior_fails.sign_p"), "0.11"),
    ("test-retest Pearson r", st("metrics.retest.pearson"), "0.958"),
    ("test-retest cells", st("metrics.retest.cells"), "22"),
    ("forced execution: median realised fraction", emb("r12_forced", "metrics.moving_frac_median"), "1.05"),
    ("forced execution: median cosine", emb("r12_forced", "metrics.moving_cos_median"), "0.998"),
    ("lenders: share from the finest level", emb("r8_inspect", "metrics.lenders.level_0.2", tag="provenance_23tasks"), "0.89"),
    ("library: leaves' distance to the demo", emb("r9_library_ablation", "metrics.summary.bank_policy_K8.rms", tag="offline_23tasks"), "0.025"),
    ("library: tree prototypes' distance", emb("r9_library_ablation", "metrics.summary.band_K8.rms", tag="offline_23tasks"), "0.033"),
    ("library: shared level alone", emb("r9_library_ablation", "metrics.summary.t0.4_K8.rms", tag="offline_23tasks"), "0.080"),
    ("shared-prototype borrowing successes", st("metrics.methods.ours_shared1.prior_fails.k"), "65"),
    ("shared-prototype borrowing episodes", st("metrics.methods.ours_shared1.prior_fails.n"), "240"),
    ("shared-prototype borrowing below the band", st("metrics.compare.ours_shared1-ours_dim1.prior_fails.diff", sign=-1.0), "0.133"),
    ("provenance: word overlap of the sources", emb("r8_inspect", "metrics.overlap_mean", tag="provenance_23tasks"), "0.122"),
    ("provenance: shuffled pairings", emb("r8_inspect", "metrics.null_mean", tag="provenance_23tasks"), "0.106"),
    ("second demo: LoRA successes", st("metrics.second_demo.rates.lora_d1.k"), "298"),
    ("second demo: borrowing + fold successes", st("metrics.second_demo.rates.ours_d1.k"), "261"),
    ("second demo: pooled LoRA lead", st("metrics.second_demo.compare.ours_both-lora_both.diff", sign=-1.0), "0.043"),
    ("second demo: pooled sign test p", st("metrics.second_demo.compare.ours_both-lora_both.sign_p"), "0.037"),
    ("second demo: pooled lead over replay", st("metrics.second_demo.compare.ours_both-replay_both.diff"), "0.110"),
    ("baseline: open-loop playback successes", st("metrics.methods.base_playback.prior_fails.k"), "187"),
    ("baseline: DemoDiffusion successes", st("metrics.methods.base_demodiff.prior_fails.k"), "36"),
    ("baseline: VINN successes", st("metrics.methods.base_vinn.prior_fails.k"), "167"),
    ("baseline: equal-bytes adapter successes", st("metrics.methods.base_lora_eq.prior_fails.k"), "184"),
    ("baseline: borrowing + fold over the equal-bytes adapter", st("metrics.compare.ours_dim1_last2-base_lora_eq.prior_fails.diff"), "0.164"),
    ("baseline: LoRA rank 64 successes", st("metrics.methods.base_lora_r64.prior_fails.k"), "267"),
    ("baseline: full fine-tuning successes", st("metrics.methods.base_fullft.prior_fails.k"), "267"),
    ("practice: LoRA round 1 successes", st("metrics.methods.base_lora_learn.prior_fails.k"), "274"),
    ("practice: LoRA round 2 successes", st("metrics.methods.loop_L2.prior_fails.k"), "281"),
    ("practice: LoRA round 3 successes", st("metrics.methods.loop_L3.prior_fails.k"), "294"),
    ("practice: round 3 over the one-demo LoRA", st("metrics.compare.loop_L3-practice_lora.prior_fails.diff"), "0.100"),
    ("practice: failure negatives successes", st("metrics.methods.nft_bal.prior_fails.k"), "302"),
    ("practice: failure negatives over success-only", st("metrics.compare.nft_bal-loop_L2.prior_fails.diff"), "0.050"),
    ("practice: failure negatives sign test p", st("metrics.compare.nft_bal-loop_L2.prior_fails.sign_p"), "0.035"),
    ("practice: 258 KB learner round 3", st("metrics.methods.loop_eq3.prior_fails.k"), "198"),
    ("practice: 258 KB learner with negatives", st("metrics.methods.nft_fold.prior_fails.k"), "175"),
    ("redo: consolidated on failed instances", emb("m5_learn_stats", "metrics.L2.learned.k", tag="redo_21tasks"), "52"),
    ("redo: failed instances", emb("m5_learn_stats", "metrics.L2.learned.n", tag="redo_21tasks"), "164"),
    ("redo: fresh-noise retry on failed instances", emb("m5_learn_stats", "metrics.L2.retry1.k", tag="redo_21tasks"), "36"),
    ("branching: persistent instances", emb("r18_branch", "metrics.D1.late_replay.n", tag="diagnosis"), "123"),
    ("branching: fresh-noise rescues", emb("r18_branch", "metrics.D1.any_late_noise.k", tag="diagnosis"), "7"),
    ("branching: knob rescues", emb("r18_branch", "metrics.D1.any_late_knob.k", tag="diagnosis"), "7"),
    ("branching: rescues by any branch", emb("r18_branch", "metrics.D1.any.k", tag="diagnosis"), "16"),
    ("branching: stalls at the demo's last frames", emb("r18_branch", "metrics.exploratory.at_end", tag="diagnosis"), "79"),
    ("branching: exact prefixes", emb("r18_branch", "metrics.G1.exact", tag="diagnosis"), "1,569"),
    ("recovery: r18rec on the concentrated tasks", emb("r18_branch", "metrics.R1_concentrated.r18rec.k", tag="recovery"), "105"),
    ("recovery: negatives learner on the concentrated tasks", emb("r18_branch", "metrics.R1_concentrated.r17bal.k", tag="recovery"), "102"),
    ("recovery: concentrated-task episodes", emb("r18_branch", "metrics.R1_concentrated.r18rec.n", tag="recovery"), "180"),
    ("recovery: R1 sign test p", emb("r18_branch", "metrics.R1_concentrated.r18rec-r17bal.sign_p", tag="recovery"), "0.76"),
    ("levels: task by nearest human demo, t=0.2", emb("r19_invariance", "metrics.levels.0.2.acc_human"), "0.72"),
    ("levels: robot rollout by nearest human demo, t=0.1", emb("r19_invariance", "metrics.levels.0.1.acc_robot_by_human"), "0.97"),
    ("levels: robot rollout by nearest human demo, raw", emb("r19_invariance", "metrics.raw.acc_robot_by_human"), "0.76"),
    ("levels: within-task edit distance, t=0.55", emb("r19_invariance", "metrics.levels.0.55.within"), "0.105"),
    ("levels: between-task edit distance, t=0.55", emb("r19_invariance", "metrics.levels.0.55.between"), "0.433"),
    ("family: family pool successes", emb("r20_family", "metrics.all_targets.fam.k", tag="read"), "104"),
    ("family: random pool successes", emb("r20_family", "metrics.all_targets.rnd.k", tag="read"), "137"),
    ("family: episodes", emb("r20_family", "metrics.all_targets.fam.n", tag="read"), "240"),
    ("family: sign test p", emb("r20_family", "metrics.all_targets.fam-rnd.sign_p", tag="read"), "0.002"),
    ("family: one-demo LoRA successes", emb("r20_family", "metrics.all_targets.onedemo.k", tag="read"), "153"),
    ("family: drawer tasks, family pool", fam_tasks("fam", (3, 4, 5)), "8"),
    ("family: drawer tasks, episodes", lambda: (float(sum(L.load("m3_final", tag="lora_r20fam", keys__tasks=str(t))[-1]["metrics"]["n_episodes"]
                                                           for t in (3, 4, 5))), 0.0, 1), "60"),
    ("family2: family pretraining successes", emb("r20_family", "metrics.all_targets.fam.k", tag="read_r21"), "148"),
    ("family2: random pretraining successes", emb("r20_family", "metrics.all_targets.rnd.k", tag="read_r21"), "144"),
    ("family2: one-demo LoRA successes", emb("r20_family", "metrics.all_targets.onedemo.k", tag="read_r21"), "153"),
    ("family2: drawer tasks, family pretraining", fam_tasks("fam", (3, 4, 5), tag="read_r21"), "48"),
    ("shared fold successes", st("metrics.methods.ours_shared_all.prior_fails.k"), "146"),
    ("other half's fold successes", st("metrics.methods.ours_shared_xfit.prior_fails.k"), "107"),
    ("borrowing without a fold successes", st("metrics.methods.ours_dim1.prior_fails.k"), "168"),
    ("composer: stove, development", emb("m3_dev", "metrics.success_rate", tag="composer_v1", keys__tasks="39"), "0.3"),
    ("dim-borrow every frame: stove, development", emb("m3_dev", "metrics.success_rate", tag="dimcode_s1", keys__tasks="39"), "1.0"),
    ("composer: program KB", lambda: tuple([L.mean("r6_composer", "metrics.program_bytes", tag="", keys__task=39)[0] / 1e3] + list(L.mean("r6_composer", "metrics.program_bytes", tag="", keys__task=39)[1:])), "4.3"),
    ("prior composer: weight on the prior, white mug", emb("r6_composer", "metrics.w_prior", tag="_prior", keys__task=67), "0.47"),
    ("prior composer: weight on the prior, stove", emb("r6_composer", "metrics.w_prior", tag="_prior", keys__task=39), "0.10"),
    ("atlas: gripper purity, t=0.4", emb("r2c_atlas", "metrics.levels.0.4.gripper_purity", bench="libero_40"), "0.99"),
    ("atlas: words at t=0.2", emb("r2c_atlas", "metrics.levels.0.2.n_words", bench="libero_40"), "111"),
    ("atlas: word purity at t=0.2", emb("r2c_atlas", "metrics.levels.0.2.word_purity", bench="libero_40"), "0.90"),
    ("fold size, last two layers, KB", lambda: tuple([L.mean("r4_fold", "bytes.experts", tag="_approach_last2")[0] / 1e3] + list(L.mean("r4_fold", "bytes.experts", tag="_approach_last2")[1:])), "258"),
]

TRANSFER = [   # docs/PLAN_TRANSFER.md probes (pre-registered: ledger m3_prereg tags p0, p1, p1_amend1, p2, p3a, p3b, p4, g38, p6)
    ("P0b: persistent movable-object failures", emb("p0_audit", "metrics.primary.n", tag="geometry"), "106"),
    ("P0b: aim-failure share", emb("p0_audit", "metrics.primary.aim_share", tag="geometry"), "0.83"),
    ("P0c: retargeted demo successes with no policy", emb("p0_audit", "metrics.totals.retarget", tag="retarget"), "58"),
    ("P1: bindings alone, 9 tasks", emb("p1_bindings", "metrics.B.x", tag="read"), "143"),
    ("P1: bindings + fold, 9 tasks", emb("p1_bindings", "metrics.A.x", tag="read"), "97"),
    ("P1: control, 9 tasks", emb("p1_bindings", "metrics.B.control", tag="read"), "66"),
    ("P1: bindings alone, difference", emb("p1_bindings", "metrics.B.diff.diff", tag="read"), "0.428"),
    ("P1: bindings + fold, difference", emb("p1_bindings", "metrics.A.diff.diff", tag="read"), "0.172"),
    ("P1 secondary: bindings alone, 21 tasks", emb("p1_bindings", "metrics.B.x", tag="secondary"), "314"),
    ("P1 secondary: bindings + fold, 21 tasks", emb("p1_bindings", "metrics.A.x", tag="secondary"), "264"),
    ("P1 secondary: control, 21 tasks", emb("p1_bindings", "metrics.B.control", tag="secondary"), "253"),
    ("P1 secondary: bindings alone, difference", emb("p1_bindings", "metrics.B.diff.diff", tag="secondary"), "0.145"),
    ("3.8: generated successes", emb("g38_generate", "metrics.generated_successes", tag="demo_retarget"), "331"),
    ("3.8: LoRA on the generated demonstrations", emb("g38_generate", "metrics.gen-onedemo.x", tag="read"), "293"),
    ("3.8: sign p against the one-demo LoRA", emb("g38_generate", "metrics.gen-onedemo.sign_p", tag="read"), "0.0014"),
    ("P3a: bindings looked up from other tasks", emb("p3_compose", "metrics.cross", tag="p3a"), "39"),
    ("P3a: own bindings", emb("p3_compose", "metrics.own", tag="p3a"), "91"),
    ("P3a: episodes per arm", emb("p3_compose", "metrics.n", tag="p3a"), "100"),
    ("P3a corrected executor (exploratory): looked up", emb("p3_compose", "metrics.totals.cross", tag="p3a_carry"), "95"),
    ("P3a corrected executor (exploratory): own", emb("p3_compose", "metrics.totals.own", tag="p3a_carry"), "92"),
    ("P3b: looked-up bindings on new targets", emb("p3_compose", "metrics.cross", tag="p3b"), "11"),
    ("P2: phase-localized negatives", emb("p2_negatives", "metrics.all_21.phase", tag="read"), "284"),
    ("P4: grown library", emb("p4_growth", "metrics.grown", tag="rollouts"), "70"),
    ("P4: base library", emb("p4_growth", "metrics.base", tag="rollouts"), "74"),
    ("P6: reader offline median error, cm", emb("p6_reader", "metrics.median_of_medians_cm", tag="offline"), "0.54"),
    ("P6: mean-placement error, cm", emb("p6_reader", "metrics.mean_placement_median_of_medians_cm", tag="offline"), "1.81"),
    ("P6: learned reader, 9 tasks", emb("p6_reader", "metrics.reader_vs_oracle.x", tag="read"), "126"),
    ("P6: reader - oracle (quoted negated)", neg(emb("p6_reader", "metrics.reader_vs_oracle.diff.diff", tag="read")), "0.094"),
    ("P6: reader - oracle lower bound (quoted negated)", neg(emb("p6_reader", "metrics.reader_vs_oracle.diff.lo90", tag="read")), "0.256"),
    ("P6: reader - control", emb("p6_reader", "metrics.reader_vs_control.diff.diff", tag="read"), "0.333"),
    ("P6: milk task, oracle", emb("p6_reader", "metrics.per_task.52.B", tag="read"), "17"),
    ("P6: salad dressing task, oracle", emb("p6_reader", "metrics.per_task.62.B", tag="read"), "20"),
    ("transfer: bindings alone, bytes for 21 tasks, KB", st("metrics.methods.ours_phasebind_nofold.prior_fails.bytes", 1e3), "54.9"),
    ("transfer: LoRA + failure negatives, MB", st("metrics.methods.nft_bal.prior_fails.bytes", 1e6), "392.9"),
    ("v2: largest rest shift of a demo object, cm", emb("p0_audit", "metrics.max_cm", tag="demo_at_rest"), "12.2"),
    ("v2: simulator poses, 21 tasks", emb("p1_bindings", "metrics.V1.x", tag="v2"), "385"),
    ("v2: v1 on the same pairs", emb("p1_bindings", "metrics.V1.control", tag="v2"), "314"),
    ("v2: LoRA + failure negatives on the same pairs", emb("p1_bindings", "metrics.V2.control", tag="v2"), "302"),
    ("v2: 30-placement reader", emb("p1_bindings", "metrics.V3.x", tag="v2"), "362"),
    ("v2: 30-placement reader lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.V3.diff.lo90", tag="v2")), "0.121"),
    ("v2: rendered-scene reader offline median, cm", emb("p6_reader", "metrics.median_of_medians_cm", tag="offline_spatial4_synth"), "0.18"),
    ("v2: rendered-scene reader, 21 tasks", emb("p1_bindings", "metrics.V5.x", tag="v2_synth"), "376"),
    ("v2: rendered-scene reader - simulator (quoted negated)", neg(emb("p1_bindings", "metrics.V5.diff.diff", tag="v2_synth")), "0.021"),
    ("v2: rendered-scene reader lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.V5.diff.lo90", tag="v2_synth")), "0.050"),
    ("v2: rendered-scene reader, MB for 21 tasks", st("metrics.methods.ours_phasebind2_synth.prior_fails.bytes", 1e6), "9.3"),
    ("stove knob: lowest initial angle, rad", emb("p0_audit", "metrics.init_min_rad", tag="stove_knob"), "0.53"),
    ("stove knob: highest initial angle, rad", emb("p0_audit", "metrics.init_max_rad", tag="stove_knob"), "2.09"),
    ("v2 generator: successes on the practice placements", emb("g38_generate", "metrics.generated_successes", tag="demo_retarget_v2"), "386"),
    ("v2 generator: LoRA", emb("p1_bindings", "metrics.V4.x", tag="v2"), "327"),
    ("v2 generator: LoRA sign p", emb("p1_bindings", "metrics.V4.sign_p", tag="v2"), "0.003"),
    ("P3c: composed from other tasks", emb("p3_compose", "metrics.C1.x", tag="p3c"), "114"),
    ("P3c: own demo", emb("p3_compose", "metrics.C1.control", tag="p3c"), "117"),
    ("P3c: lower bound (quoted negated)", neg(emb("p3_compose", "metrics.C1.diff.lo90", tag="p3c")), "0.067"),
    ("P3c: sibling's whole demo", emb("p3_compose", "metrics.C2.control", tag="p3c"), "42"),
    ("P3c: composed with the rendered-scene reader", emb("p3_compose", "metrics.R1.x", tag="p3c_reader"), "92"),
    ("P3c: composed, reader, predicted carry", emb("p3_compose", "metrics.R2.x", tag="p3c_reader2"), "112"),
    ("P3c: predicted carry lower bound (quoted negated)", neg(emb("p3_compose", "metrics.R2.diff.lo90", tag="p3c_reader2")), "0.058"),
    ("C1: change points per demo at t=0.55", emb("c1_grammar", "metrics.levels.0.55.change_points_per_demo", tag="read"), "2.9"),
    ("C1: recall at t=0.55", emb("c1_grammar", "metrics.levels.0.55.recall", tag="read"), "0.99"),
    ("C1: random recall at t=0.55", emb("c1_grammar", "metrics.levels.0.55.random_recall", tag="read"), "0.20"),
    ("C1: precision at t=0.55", emb("c1_grammar", "metrics.levels.0.55.precision", tag="read"), "0.68"),
    ("C1: random precision at t=0.55", emb("c1_grammar", "metrics.levels.0.55.random_precision", tag="read"), "0.12"),
    ("type reader: fixtures, median cm", emb("p6_reader", "metrics.fixture_median_cm", tag="type_transfer"), "0.20"),
    ("type reader: movable objects, median cm", emb("p6_reader", "metrics.movable_median_cm", tag="type_transfer"), "3.12"),
    ("P5: order A demos", emb("p5_stream", "metrics.orders.A.demos", tag="read"), "35"),
    ("P5: order B demos", emb("p5_stream", "metrics.orders.B.demos", tag="read"), "33"),
    ("P5: composed only when late", emb("p5_stream", "metrics.S1.only_late", tag="read"), "20"),
    ("P5: composed only when early", emb("p5_stream", "metrics.S1.only_early", tag="read"), "4"),
    ("P5: composed arrivals", emb("p5_stream", "metrics.S2.x", tag="read"), "252"),
    ("P5: composed arrivals, episodes", emb("p5_stream", "metrics.S2.n", tag="read"), "640"),
    ("P5: the same tasks' own demos", emb("p5_stream", "metrics.S2.control", tag="read"), "561"),
    ("P5 dev: carry warp", emb("p5_stream", "metrics.totals.streamw", tag="dev_fixes"), "82"),
    ("P5 dev: generic transport", emb("p5_stream", "metrics.totals.streamt", tag="dev_fixes"), "6"),
    ("P5 dev: as registered", emb("p5_stream", "metrics.totals.streamc", tag="dev_fixes"), "123"),
    ("P5 dev: episodes", emb("p5_stream", "metrics.n", tag="dev_fixes"), "320"),
    ("yaw reader: median deg", emb("p6_reader", "metrics.median_deg", tag="yaw"), "0.8"),
    ("yaw reader: fixed yaw median deg", emb("p6_reader", "metrics.fixed_median_deg", tag="yaw"), "10.2"),
    ("yaw read: books with yaw", emb("p6_reader", "metrics.Y1.x", tag="yaw_read"), "173"),
    ("yaw read: books with a fixed yaw", emb("p6_reader", "metrics.Y1.control", tag="yaw_read"), "175"),
    ("yaw read: episodes", emb("p6_reader", "metrics.Y1.n", tag="yaw_read"), "220"),
    ("P5 validated: order A stream", emb("p5_stream", "metrics.orders.A.stream", tag="validated"), "855"),
    ("P5 validated: order B stream", emb("p5_stream", "metrics.orders.B.stream", tag="validated"), "832"),
    ("P5 validated: order A demos", emb("p5_stream", "metrics.orders.A.demos", tag="validated"), "43"),
    ("P5 validated: order B demos", emb("p5_stream", "metrics.orders.B.demos", tag="validated"), "42"),
    ("P5 validated multi: order A stream", emb("p5_stream", "metrics.orders.A.stream", tag="validated_multi"), "817"),
    ("P5 validated multi: order B stream", emb("p5_stream", "metrics.orders.B.stream", tag="validated_multi"), "820"),
    ("P5 validated multi: order A demos", emb("p5_stream", "metrics.orders.A.demos", tag="validated_multi"), "40"),
    ("P5 validated multi: order B demos", emb("p5_stream", "metrics.orders.B.demos", tag="validated_multi"), "37"),
    ("P5: every task its own demo", emb("p5_stream", "metrics.orders.A.all_demo", tag="validated"), "874"),
    ("P5 validated: composed arrivals", emb("p5_stream", "metrics.composed_vs_own.x", tag="validated"), "201"),
    ("P5 validated: their own demos", emb("p5_stream", "metrics.composed_vs_own.control", tag="validated"), "262"),
    ("P5 validated: composed episodes", emb("p5_stream", "metrics.composed_vs_own.n", tag="validated"), "300"),
    ("P5: episodes per order", emb("p5_stream", "metrics.orders.A.n", tag="validated"), "1,000"),
    ("allocator: successes", emb("p1_bindings", "metrics.AL1.x", tag="alloc_v2"), "391"),
    ("allocator: phases alone", emb("p1_bindings", "metrics.AL1.control", tag="alloc_v2"), "376"),
    ("allocator: sign p", emb("p1_bindings", "metrics.AL1.sign_p", tag="alloc_v2"), "0.0015"),
    ("allocator: MB for 21 tasks", lambda: tuple([L.mean("p1_bindings", "metrics.bytes_total", tag="alloc_v2")[0] / 1e6] + list(L.mean("p1_bindings", "metrics.bytes_total", tag="alloc_v2")[1:])), "45.6"),
    ("allocator: the stove, borrowing", emb("p1_bindings", "metrics.per_task.39.alloc", tag="alloc_v2"), "17"),
    ("allocator: the stove, phases", emb("p1_bindings", "metrics.per_task.39.A", tag="alloc_v2"), "6"),
]

CHECKS = EMBODIED + TRANSFER + [
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
