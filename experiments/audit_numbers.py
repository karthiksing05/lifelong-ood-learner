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

DOCS = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "docs", "THE_TASK.md"),
        os.path.join(ROOT, "archive", "2026-10-program", "README_demo_rows.md")]
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


def gr(*path, fn=None):
    """A value of the grammar readout row (ledger grammar, tag readout) by its key path (keys may contain dots)."""
    def f():
        v = L.load("grammar", tag="readout")[-1]["metrics"]
        for k in path:
            v = v[k]
        return (float(fn(v) if fn else v), 0.0, 1)
    return f


def sto(*path, scale=1.0):
    """A value of the storage row (ledger storage, tag read) by its key path, divided by `scale`."""
    def f():
        v = L.load("storage", tag="read")[-1]["metrics"]
        for k in path:
            v = v[k]
        return (float(v) / scale, 0.0, 1)
    return f


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

TRANSFER = [   # archive/2026-10-program/PLAN_TRANSFER.md probes (pre-registered: ledger m3_prereg tags p0, p1, p1_amend1, p2, p3a, p3b, p4, g38, p6)
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
    ("P5 practice: order A stream", emb("p5_stream", "metrics.orders.A.stream", tag="practice"), "875"),
    ("P5 practice: order B stream", emb("p5_stream", "metrics.orders.B.stream", tag="practice"), "870"),
    ("P5 practice: order A demos", emb("p5_stream", "metrics.orders.A.demos", tag="practice"), "46"),
    ("P5 practice: order B demos", emb("p5_stream", "metrics.orders.B.demos", tag="practice"), "44"),
    ("P5 practice: composed arrivals", emb("p5_stream", "metrics.composed_vs_own.x", tag="practice"), "186"),
    ("P5 practice: their own demos", emb("p5_stream", "metrics.composed_vs_own.control", tag="practice"), "189"),
    ("P5 practice: composed episodes", emb("p5_stream", "metrics.composed_vs_own.n", tag="practice"), "200"),
    ("P5 practice: lower bound (quoted negated)", neg(emb("p5_stream", "metrics.composed_vs_own.diff.lo90", tag="practice")), "0.085"),
    ("seed 1: v2 bindings", emb("p1_bindings", "metrics.R1.x", tag="seed1"), "386"),
    ("seed 1: reader arm", emb("p1_bindings", "metrics.R2.x", tag="seed1"), "381"),
    ("seed 1: reader lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.R2.diff.lo90", tag="seed1")), "0.048"),
    ("seed 1: allocator", emb("p1_bindings", "metrics.R3.x", tag="seed1"), "388"),
    ("seed 1: LoRA + failure negatives", emb("p1_bindings", "metrics.R1.control", tag="seed1"), "306"),
    ("seed 1: lowest per-task correlation", lambda: (min(v["pearson_per_task"] for v in L.load("p1_bindings", tag="seed1")[-1]["metrics"]["retest"].values()), 0.0, 1), "0.91"),
    ("demo 1: v2 bindings", emb("p1_bindings", "metrics.D1.x", tag="demo1_v2"), "381"),
    ("demo 1: episodes", emb("p1_bindings", "metrics.D1.n", tag="demo1_v2"), "400"),
    ("demo 1: the demo-1 LoRA", emb("p1_bindings", "metrics.D1.control", tag="demo1_v2"), "278"),
    ("demo 1: v2 with the first demo, same tasks", emb("p1_bindings", "metrics.demo0_vs_demo1.v2.demo0", tag="demo1_v2"), "365"),
    ("demo 1: reader arm", emb("p1_bindings", "metrics.D2.x", tag="demo1_v2"), "370"),
    ("demo 1: reader lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.D2.diff.lo90", tag="demo1_v2")), "0.083"),
    ("P5 confirm: composed arrivals", emb("p5_stream", "metrics.composed_vs_own.x", tag="confirm"), "157"),
    ("P5 confirm: their own demos", emb("p5_stream", "metrics.composed_vs_own.control", tag="confirm"), "162"),
    ("P5 confirm: composed episodes", emb("p5_stream", "metrics.composed_vs_own.n", tag="confirm"), "180"),
    ("P5 confirm: lower bound (quoted negated)", neg(emb("p5_stream", "metrics.composed_vs_own.diff.lo90", tag="confirm")), "0.167"),
    ("P5 confirm: order C stream", emb("p5_stream", "metrics.orders.C.stream", tag="confirm"), "861"),
    ("P5 confirm: order D stream", emb("p5_stream", "metrics.orders.D.stream", tag="confirm"), "883"),
    ("P5 confirm: order C demos", emb("p5_stream", "metrics.orders.C.demos", tag="confirm"), "46"),
    ("P5 confirm: order D demos", emb("p5_stream", "metrics.orders.D.demos", tag="confirm"), "45"),
    ("allocator: successes", emb("p1_bindings", "metrics.AL1.x", tag="alloc_v2"), "391"),
    ("allocator: phases alone", emb("p1_bindings", "metrics.AL1.control", tag="alloc_v2"), "376"),
    ("allocator: sign p", emb("p1_bindings", "metrics.AL1.sign_p", tag="alloc_v2"), "0.0015"),
    ("allocator: MB for 21 tasks", lambda: tuple([L.mean("p1_bindings", "metrics.bytes_total", tag="alloc_v2")[0] / 1e6] + list(L.mean("p1_bindings", "metrics.bytes_total", tag="alloc_v2")[1:])), "45.6"),
    ("allocator: the stove, borrowing", emb("p1_bindings", "metrics.per_task.39.alloc", tag="alloc_v2"), "17"),
    ("allocator: the stove, phases", emb("p1_bindings", "metrics.per_task.39.A", tag="alloc_v2"), "6"),
    ("sota: ours", emb("p1_bindings", "metrics.B1.x", tag="sota"), "372"),
    ("sota: episodes", emb("p1_bindings", "metrics.B1.n", tag="sota"), "420"),
    ("sota: strap", emb("p1_bindings", "metrics.B1.control", tag="sota"), "168"),
    ("sota: dsrl", emb("p1_bindings", "metrics.B2.control", tag="sota"), "234"),
    ("sota: er", emb("p1_bindings", "metrics.B3.control", tag="sota"), "138"),
    ("sota: LoRA + negatives", emb("p1_bindings", "metrics.vs_nft.strap.control", tag="sota"), "302"),
    ("sota: retrieved sub-trajectories", lambda: (float(L.load("b_baselines", tag="strap")[-1]["metrics"]["retrieved"]), 0.0, 1), "100"),
    ("sota: DSRL warm-up successes", lambda: (float(sum(sum(x["success"] for x in __import__("json").load(open(f"results/robot/dsrl_practice_{t}_dsrl_train.json"))["tasks"][str(t)]["episodes"][:10]) for t in (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63))), 0.0, 1), "126"),
    ("sota: DSRL warm-up episodes", lambda: (210.0, 0.0, 1), "210"),
    ("sota: DSRL last-20 successes", lambda: (float(sum(sum(x["success"] for x in __import__("json").load(open(f"results/robot/dsrl_practice_{t}_dsrl_train.json"))["tasks"][str(t)]["episodes"][-20:]) for t in (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63))), 0.0, 1), "234"),
    ("sota amend1: ours", emb("p1_bindings", "metrics.B1.x", tag="sota_amend1"), "372"),
    ("sota amend1: STRAP, authors' retrieval", emb("p1_bindings", "metrics.B1.control", tag="sota_amend1"), "138"),
    ("sota amend1: DSRL, authors' configuration", emb("p1_bindings", "metrics.B2.control", tag="sota_amend1"), "282"),
    ("sota amend1: LoRA + negatives", emb("p1_bindings", "metrics.vs_nft.dsrl.control", tag="sota_amend1"), "302"),
    ("sota amend1: DSRL vs LoRA + negatives, sign test p", emb("p1_bindings", "metrics.vs_nft.dsrl.sign_p", tag="sota_amend1"), "0.098"),
    ("sota amend1: DSRL warm-up successes", lambda: (float(sum(sum(x["success"] for x in __import__("json").load(open(f"results/robot/dsrl_practice_{t}_dsrlo_train.json"))["tasks"][str(t)]["episodes"][:10]) for t in (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63))), 0.0, 1), "116"),
    ("sota amend1: DSRL last-20 successes", lambda: (float(sum(sum(x["success"] for x in __import__("json").load(open(f"results/robot/dsrl_practice_{t}_dsrlo_train.json"))["tasks"][str(t)]["episodes"][-20:]) for t in (35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62, 3, 4, 5, 32, 41, 43, 48, 58, 63))), 0.0, 1), "302"),
    ("carries: compositions", emb("p5_stream", "metrics.compositions", tag="dev_carries"), "54"),
    ("carries: route", emb("p5_stream", "metrics.totals.streamc", tag="dev_carries"), "197"),
    ("carries: episodes", emb("p5_stream", "metrics.n", tag="dev_carries"), "540"),
    ("carries: prior", emb("p5_stream", "metrics.totals.streamh", tag="dev_carries"), "188"),
    ("carries: prior + events", emb("p5_stream", "metrics.totals.streamhe", tag="dev_carries"), "200"),
    ("carries: transport on arrival", emb("p5_stream", "metrics.totals.streamte", tag="dev_carries"), "307"),
    ("carries: lifted transport", emb("p5_stream", "metrics.totals.streamtle", tag="dev_carries"), "316"),
    ("carries: lifted vs route", emb("p5_stream", "metrics.vs_route.streamtle.diff.diff", tag="dev_carries"), "0.220"),
    ("carries: lifted vs route lo", emb("p5_stream", "metrics.vs_route.streamtle.diff.lo", tag="dev_carries"), "0.111"),
    ("carries: lifted vs route hi", emb("p5_stream", "metrics.vs_route.streamtle.diff.hi", tag="dev_carries"), "0.337"),
    ("carries: the mismatch band's threshold (cm)", lambda: (float([k for k in L.load("p5_stream", tag="dev_carries")[-1]["metrics"]["groups"] if ">" in k][0].split(">")[1]), 0.0, 1), "30"),
    ("carries: prior beyond 30 cm", emb("p5_stream", "metrics.groups.mismatch >30.streamh", tag="dev_carries"), "52"),
    ("carries: route beyond 30 cm", emb("p5_stream", "metrics.groups.mismatch >30.streamc", tag="dev_carries"), "11"),
    ("carries: episodes beyond 30 cm", lambda: (10.0 * L.load("p5_stream", tag="dev_carries")[-1]["metrics"]["groups"]["mismatch >30"]["compositions"], 0.0, 1), "140"),
    ("carries: set passes", emb("p5_stream", "metrics.passes_union_set", tag="dev_carries"), "34"),
    ("carries: route passes", emb("p5_stream", "metrics.passes.streamc", tag="dev_carries"), "16"),
    ("override: pi0.5 guided chunk from the plan (rms)", lambda: (__import__("json").load(open("results/robot/guide_override_pi05_libero.json"))["guided_minus_target_rms"], 0.0, 1), "0.008"),
    ("override: SmolVLA guided chunk from the plan (rms)", lambda: (__import__("json").load(open("results/robot/guide_override_smolvla_libero.json"))["guided_minus_target_rms"], 0.0, 1), "0.008"),
    ("override: two observations' guided chunks (rms)", lambda: (__import__("json").load(open("results/robot/guide_override_pi05_libero.json"))["guided_obs0_minus_obs1_rms"], 0.0, 1), "0.0003"),
    ("override: two observations' own chunks (rms)", lambda: (__import__("json").load(open("results/robot/guide_override_pi05_libero.json"))["own_obs0_minus_obs1_rms"], 0.0, 1), "0.39"),
    ("no policy: successes", emb("p1_bindings", "metrics.N.x", tag="nopolicy"), "372"),
    ("no policy: bindings through pi0.5", emb("p1_bindings", "metrics.N.control", tag="nopolicy"), "372"),
    ("no policy: discordant (no policy wins)", emb("p1_bindings", "metrics.N.discordant.0", tag="nopolicy"), "5"),
    ("no policy: discordant (pi0.5 wins)", emb("p1_bindings", "metrics.N.discordant.1", tag="nopolicy"), "5"),
    ("no policy: lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.N.diff.lo90", tag="nopolicy")), "0.017"),
    ("no policy: margin", lambda: (0.05, 0.0, 1), "0.05"),
    ("pi0: bindings", emb("p1_bindings", "metrics.Z.x", tag="pi0"), "375"),
    ("pi0: zero-shot", emb("p1_bindings", "metrics.Z.control", tag="pi0"), "18"),
    ("pi0: vs pi0.5 bindings", emb("p1_bindings", "metrics.vs_pi05.control", tag="pi0"), "372"),
    ("pi0: vs pi0.5 diff", emb("p1_bindings", "metrics.vs_pi05.diff.diff", tag="pi0"), "0.007"),
    ("pi0: vs pi0.5 sign p", emb("p1_bindings", "metrics.vs_pi05.sign_p", tag="pi0"), "0.51"),
    ("pi0: task 43 zero-shot", emb("p1_bindings", "metrics.per_task.43.z", tag="pi0"), "16"),
    ("smolvla: decoding check successes", lambda: (float(sum(sum(e["success"] for v in __import__("json").load(open(f))["tasks"].values() for e in v["episodes"]) for f in __import__("glob").glob("results/robot/m3_smoke_libero_spatial_*_zeroshot_smolvla.json"))), 0.0, 1), "6"),
    ("overhaul: LIBERO-90 tasks", emb("f0_coverage", "metrics.n_tasks", tag="read"), "90"),
    ("overhaul: training instructions", lambda: (40.0, 0.0, 1), "40"),
    ("overhaul: rule success on", emb("f0_coverage", "metrics.features.rule_family_named.success_on", tag="read"), "0.54"),
    ("overhaul: rule tasks", emb("f0_coverage", "metrics.features.rule_family_named.n_on", tag="read"), "29"),
    ("overhaul: rule success off", emb("f0_coverage", "metrics.features.rule_family_named.success_off", tag="read"), "0.14"),
    ("overhaul: rule AUC", emb("f0_coverage", "metrics.features.rule_family_named.auc", tag="read"), "0.72"),
    ("overhaul: contrast AUC", emb("f0_coverage", "metrics.features.contrast_obj.auc", tag="read"), "0.74"),
    ("overhaul: living room 1 habit (alphabet soup first, of 10)", lambda: (float(min(v["bound_counts"].get("alphabet_soup", 0) for k, v in L.load("f1_binding", tag="f1_bind_libero_90")[-1]["metrics"]["rows"].items() if v["task"] in (46, 47, 48, 49))), 0.0, 1), "10"),
    ("overhaul: LIBERO-Object swaps followed", lambda: (float(__import__("numpy").mean([v["BA"] for v in L.load("f1_binding", tag="f1_bind_libero_object")[-1]["metrics"]["rows"].values() if v["named"] not in {"libero_object:0": "alphabet_soup", "libero_object:1": "cream_cheese", "libero_object:2": "salad_dressing", "libero_object:3": "bbq_sauce", "libero_object:4": "ketchup"}.get(f"libero_object:{v['task']}", "")])), 0.0, 1), "0.43"),
    ("overhaul: first-chunk validity PA", emb("f1_factor_probe", "metrics.groups.validity.PA", tag="read"), "0.43"),
    ("overhaul: first-chunk validity gate", lambda: (0.8, 0.0, 1), "0.8"),
    ("overhaul: first-chunk contrast cosine", emb("f1_factor_probe", "metrics.groups.validity.CC", tag="read"), "0.72"),
    ("overhaul: pilot composition", emb("zc_pilot", "metrics.totals.fcomp_8", tag="read"), "14"),
    ("overhaul: pilot episodes", emb("zc_pilot", "metrics.episodes_per_arm", tag="read"), "70"),
    ("overhaul: pilot guidance", emb("zc_pilot", "metrics.totals.cfg_4", tag="read"), "4"),
    ("overhaul: pilot placebo", emb("zc_pilot", "metrics.totals.placebo_4", tag="read"), "8"),
    ("overhaul: main tasks", lambda: (float(len(L.load("p1_bindings", tag="zc_main")[-1]["metrics"]["per_task"])), 0.0, 1), "12"),
    ("overhaul: main composition", emb("p1_bindings", "metrics.H1.x", tag="zc_main"), "22"),
    ("overhaul: main zero-shot", emb("p1_bindings", "metrics.H1.control", tag="zc_main"), "9"),
    ("overhaul: main episodes", emb("p1_bindings", "metrics.H1.n", tag="zc_main"), "240"),
    ("overhaul: main lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.H1.diff.lo90", tag="zc_main")), "0.004"),
    ("overhaul: main margin", lambda: (0.05, 0.0, 1), "0.05"),
    ("overhaul: main guidance", emb("p1_bindings", "metrics.H1b.control", tag="zc_main"), "19"),
    ("overhaul: main ketchup task", emb("p1_bindings", "metrics.per_task.48.f", tag="zc_main"), "9"),
    ("overhaul: token bytes (KB)", lambda: (L.load("f4_fold", tag="token_demo")[-1]["bytes"]["total"] / 1e3, 0.0, 1), "18.4"),
    ("overhaul: token", emb("p1_bindings", "metrics.H2a.x", tag="zc_fold"), "140"),
    ("overhaul: token episodes", emb("p1_bindings", "metrics.H2a.n", tag="zc_fold"), "420"),
    ("overhaul: token zero-shot", emb("p1_bindings", "metrics.Z.control", tag="zc_fold"), "6"),
    ("overhaul: token LoRA", emb("p1_bindings", "metrics.H2a.control", tag="zc_fold"), "252"),
    ("overhaul: token lower bound (quoted negated)", neg(emb("p1_bindings", "metrics.H2a.diff.lo90", tag="zc_fold")), "0.362"),
    ("overhaul: token demo 1", emb("p1_bindings", "metrics.H2b.x", tag="zc_fold"), "113"),
    ("overhaul: LoRA demo 1", emb("p1_bindings", "metrics.H2b.control", tag="zc_fold"), "298"),
    ("overhaul: ketchup task token", emb("p1_bindings", "metrics.per_task.48.tok", tag="zc_fold"), "13"),
    ("overhaul: ketchup task LoRA", emb("p1_bindings", "metrics.per_task.48.lora", tag="zc_fold"), "12"),
    ("overhaul: butter drawer token", emb("p1_bindings", "metrics.per_task.3.tok", tag="zc_fold"), "5"),
    ("overhaul: butter drawer LoRA", emb("p1_bindings", "metrics.per_task.3.lora", tag="zc_fold"), "19"),
    ("overhaul: ketchup token on the tray task (binding)", lambda: (L.load("f1_binding", tag="f4_pilot_libero_90")[-1]["metrics"]["rows"]["58:ftoken_48"]["BA"], 0.0, 1), "0.6"),
    ("overhaul: joint ketchup token on the tray task (binding)", lambda: (L.load("f1_binding", tag="f4_pilot_libero_90")[-1]["metrics"]["rows"]["58:ftoken_48_32"]["BA"], 0.0, 1), "0.9"),
    ("overhaul: task prototypes", emb("f0_task_library", "metrics.n_prototypes", tag="read"), "49"),
    ("scheme: units", emb("p5_stream", "metrics.H1.units", tag="scheme"), "29"),
    ("scheme: composed", emb("p5_stream", "metrics.H1.x", tag="scheme"), "520"),
    ("scheme: episodes", emb("p5_stream", "metrics.H1.n", tag="scheme"), "580"),
    ("scheme: own demos", emb("p5_stream", "metrics.H1.control", tag="scheme"), "493"),
    ("scheme: diff", emb("p5_stream", "metrics.H1.diff.diff", tag="scheme"), "0.047"),
    ("scheme: lower bound (quoted negated)", neg(emb("p5_stream", "metrics.H1.diff.lo90", tag="scheme")), "0.014"),
    ("scheme: sign p", emb("p5_stream", "metrics.H1.sign_p", tag="scheme"), "0.0067"),
    ("scheme: order E demos", emb("p5_stream", "metrics.orders.E.demos", tag="scheme"), "41"),
    ("scheme: order F demos", emb("p5_stream", "metrics.orders.F.demos", tag="scheme"), "39"),
    ("scheme: order G demos", emb("p5_stream", "metrics.orders.G.demos", tag="scheme"), "43"),
    ("scheme: order H demos", emb("p5_stream", "metrics.orders.H.demos", tag="scheme"), "38"),
    ("scheme: 83 composed", emb("p5_stream", "metrics.per_unit.83_74_75_streamtle.composed", tag="scheme"), "15"),
    ("scheme: 83 own", emb("p5_stream", "metrics.per_unit.83_74_75_streamtle.own", tag="scheme"), "4"),
    ("scheme: 16 composed", emb("p5_stream", "metrics.per_unit.16_17_17_streamtle.composed", tag="scheme"), "11"),
    ("grammar: demos", gr("tasks", fn=len), "50"),
    ("grammar: t=0.55 open motion tasks", gr("prototypes", "0.55", "0", "tasks"), "50"),
    ("grammar: t=0.55 closed transport tasks", gr("prototypes", "0.55", "1", "tasks"), "46"),
    ("grammar: t=0.55 closed lift tasks", gr("prototypes", "0.55", "2", "tasks"), "23"),
    ("grammar: t=0.4 prototypes used", gr("prototypes", "0.4", fn=len), "58"),
    ("grammar: reach down tasks", gr("prototypes", "0.4", "2", "tasks"), "46"),
    ("grammar: lower while holding tasks", gr("prototypes", "0.4", "6", "tasks"), "34"),
    ("grammar: grip change tasks", gr("prototypes", "0.4", "9", "tasks"), "46"),
    ("grammar: carry strings within relation", gr("string_by_key", "carry", "same_key_median"), "0.27"),
    ("grammar: carry strings across relations", gr("string_by_key", "carry", "other_key_median"), "0.56"),
    ("grammar: carry permutation p", gr("string_by_key", "carry", "perm_p"), "0.001"),
    ("grammar: grasp strings within type", gr("string_by_key", "grasp", "same_key_median"), "0.22"),
    ("grammar: grasp strings across types", gr("string_by_key", "grasp", "other_key_median"), "0.25"),
    ("grammar: grasp strings permutation p", gr("string_by_key", "grasp", "perm_p"), "0.13"),
    ("grammar: grasp geometry within type (cm)", gr("grasp_offset_cm", "within_type_median"), "2.0"),
    ("grammar: grasp geometry across types (cm)", gr("grasp_offset_cm", "across_type_median"), "9.8"),
    ("grammar: grasp geometry permutation p", gr("grasp_geometry_perm_p"), "0.0005"),
    ("reader: full reader KB (median)", sto("median", "reader_full", scale=1e3), "459"),
    ("reader: phases KB (median)", sto("median", "program", scale=1e3), "2.4"),
    ("reader: feature dimension", lambda: (16384.0, 0.0, 1), "16,384"),
    ("reader: head KB (median)", sto("median", "reader_pca512", scale=1e3), "7.2"),
    ("reader: basis MB", sto("basis_pca512", scale=1e6), "16.8"),
    ("reader: pca512 offline median cm", emb("p6_reader", "metrics.summary.pca512.median_err_cm", tag="compress_offline"), "0.214"),
    ("reader: full offline median cm", emb("p6_reader", "metrics.summary.full.median_err_cm", tag="compress_offline"), "0.152"),
    ("reader: pca512 online", emb("p6_reader", "metrics.RC1.x", tag="compress_read"), "861"),
    ("reader: full online", emb("p6_reader", "metrics.RC1.control", tag="compress_read"), "874"),
    ("reader: online episodes", emb("p6_reader", "metrics.RC1.n", tag="compress_read"), "1,000"),
    ("reader: lower bound (quoted negated)", neg(emb("p6_reader", "metrics.RC1.diff.lo90", tag="compress_read")), "0.034"),
    ("reader: random projection dimension", lambda: (float(max(L.load("p6_reader", tag="compress_offline_large")[-1]["metrics"]["ks"])), 0.0, 1), "2,048"),
    ("reader: rand2048 offline cm", emb("p6_reader", "metrics.summary.rand2048.median_err_cm", tag="compress_offline_large"), "0.183"),
    ("reader: rand2048 KB", lambda: tuple([L.mean("p6_reader", "metrics.summary.rand2048.mean_bytes", tag="compress_offline_large")[0] / 1e3] + [0.0, 1]), "27.7"),
    ("storage: LoRA MB", sto("lora_bytes", scale=1e6), "18.7"),
    ("storage: raw demo MB", sto("raw_demo_bytes", scale=1e6), "15.5"),
    ("storage: phases + head KB", lambda: ((sto("median", "program")()[0] + sto("median", "reader_pca512")()[0]) / 1e3, 0.0, 1), "9.6"),
    ("storage: 50 LoRAs MB", sto("streams", "A", "totals", "lora", scale=1e6), "936"),
    ("storage: 50 raw demos MB", sto("streams", "A", "totals", "raw", scale=1e6), "773"),
    ("storage: 50 tasks phases + heads MB", sto("streams", "A", "totals", "own_pca", scale=1e6), "17.3"),
]

def _row(exp, tag):
    return L.load(exp, tag=tag)[-1]["metrics"]


def q2(tag, key, field, scale=1.0):
    """b2_q2 reads (experiments/b2_q2_read.py): rows keyed '<task>:<arm>' (arm names contain dots, so no dotted paths)."""
    return lambda: (scale * float(_row("b2_q2", tag)["rows"][key][field]), 0.0, 1)


BIND = [   # the Bind plan's reads (no demonstration; ledger m3_prereg tags b_bind_q1, b_bind_q2, b_bind_q2_amend1)
    ("bind Q1a: W1 LIBERO-Goal layer 10", lambda: (_row("b1_where", "read_all")["groups"]["validity"]["w1_by_layer"]["10"], 0.0, 1), "0.74"),
    ("bind Q1a: W1 class a layer 10", lambda: (_row("b1_where", "read_all")["groups"]["class_a"]["w1_by_layer"]["10"], 0.0, 1), "0.51"),
    ("bind Q1a: OWLv2 LIBERO-Goal", lambda: (_row("b1_where", "read_w3center")["groups"]["validity"]["w3"], 0.0, 1), "0.65"),
    ("bind Q1a: OWLv2 class a", lambda: (_row("b1_where", "read_w3center")["groups"]["class_a"]["w3"], 0.0, 1), "0.71"),
    ("bind Q1b: CC lowest config", lambda: (min(v["CC"] for v in _row("b2_steer", "offline_class_a")["groups"]["class_a"]["configs"].values()), 0.0, 1), "0.45"),
    ("bind Q1b: CC highest config", lambda: (max(v["CC"] for v in _row("b2_steer", "offline_class_a")["groups"]["class_a"]["configs"].values()), 0.0, 1), "0.62"),
    ("bind Q1b: PA mass 0.7", lambda: (_row("b2_steer", "offline_class_a")["groups"]["class_a"]["configs"]["mass0.7_all"]["PA"], 0.0, 1), "0.43"),
    ("bind Q1b: PA unbiased", lambda: (_row("b2_steer", "offline_class_a")["groups"]["class_a"]["PA_base"], 0.0, 1), "0.30"),
    ("bind Q2: ketchup attention (of 10)", q2("b_q2mini", "48:bind_rho0.7_all", "BA", 10), "7"),
    ("bind Q2: milk attention (of 10)", q2("b_q2mini", "52:bind_rho0.7_all", "BA", 10), "7"),
    ("bind Q2: milk zero-shot (of 10)", q2("b_q2mini", "52:zs", "BA", 10), "1"),
    ("bind Q2: tomato sauce attention none (of 10)", q2("b_q2mini", "49:bind_rho0.7_all", "NONE", 10), "10"),
    ("bind Q2: attention whole-episode success mean", lambda: (_row("b2_q2", "b_q2full")["arms"]["bind_rho0.7_all"]["success_mean"], 0.0, 1), "0.01"),
    ("bind Q2: tomato sauce approach-only (of 10)", q2("b_q2full", "59:bind_rho0.7_all_approach", "success", 10), "10"),
    ("bind Q2: tomato sauce zero-shot success (of 10)", q2("b_q2mini", "59:zs", "success", 10), "1"),
    ("role: milk BA", q2("b_q2mini", "52:role", "BA"), "0.9"),
    ("role: OJ BA", q2("b_q2mini", "53:role", "BA"), "1.0"),
    ("role: milk zero-shot BA", q2("b_q2mini", "52:zs", "BA"), "0.1"),
    ("role: milk success", q2("b_q2mini", "52:role", "success"), "0.4"),
    ("role: OJ success", q2("b_q2mini", "53:role", "success"), "0.3"),
    ("role: milk approach-only success", q2("b_q2mini", "52:role_open", "success"), "0.7"),
    ("role: OJ approach-only success", q2("b_q2mini", "53:role_open", "success"), "0.7"),
    ("role: ketchup BA", q2("b_q2mini", "48:role", "BA"), "0.0"),
    ("role: ketchup none (of 10)", q2("b_q2mini", "48:role", "NONE", 10), "7"),
    ("role: same-shape placebo pooled", lambda: (_row("b2_q2", "b_q2mini")["arms"]["roleplacebo"]["BA_placebo_pooled"], 0.0, 1), "0.0"),
    ("role: carton placebo task 52", q2("b_q2mini", "52:roleshape", "BA_placebo"), "1.0"),
    ("role: carton placebo task 53", q2("b_q2mini", "53:roleshape", "BA_placebo"), "0.8"),
    ("fold: token bytes (KB)", lambda: (L.load("f4_fold", tag="token_rollouts_role")[-1]["bytes"]["total"] / 1e3, 0.0, 1), "18.4"),
    ("fold: OJ token BA (of 10)", q2("b_q2mini", "53:ftoken_self_role", "BA", 10), "10"),
    ("fold: OJ token success", q2("b_q2mini", "53:ftoken_self_role", "success"), "0.6"),
    ("fold: milk token BA", q2("b_q2mini", "52:ftoken_self_role", "BA"), "0.5"),
    ("fold: milk token success", q2("b_q2mini", "52:ftoken_self_role", "success"), "0.2"),
    ("scheme: tasks with a role code", lambda: (float(len(_row("b5_scheme", "main")["per_task"])), 0.0, 1), "14"),
    ("scheme: tasks with a token", lambda: (float(len(_row("b5_scheme", "main")["tokens"])), 0.0, 1), "4"),
    ("scheme: tasks without a form", lambda: (float(len(_row("b5_scheme", "main")["per_task"]) - len(_row("b5_scheme", "main")["tokens"])), 0.0, 1), "10"),
    ("scheme H2: form", lambda: (float(_row("b5_scheme", "main")["hypotheses"]["H2"]["x"]), 0.0, 1), "65"),
    ("scheme H2: episodes", lambda: (float(_row("b5_scheme", "main")["hypotheses"]["H2"]["n"]), 0.0, 1), "80"),
    ("scheme H2: zero-shot", lambda: (float(_row("b5_scheme", "main")["hypotheses"]["H2"]["control"]), 0.0, 1), "21"),
    ("scheme H1: token", lambda: (float(_row("b5_scheme", "main")["hypotheses"]["H1"]["x"]), 0.0, 1), "51"),
    ("scheme H1 passes (1 = yes)", lambda: (float(_row("b5_scheme", "main")["hypotheses"]["H1"]["passes"]), 0.0, 1), "1"),
    ("scheme: milk token", lambda: (float(_row("b5_scheme", "main")["per_task"]["52"]["token"]), 0.0, 1), "3"),
    ("scheme: OJ token", lambda: (float(_row("b5_scheme", "main")["per_task"]["53"]["token"]), 0.0, 1), "13"),
    ("scheme: OJ zero-shot", lambda: (float(_row("b5_scheme", "main")["per_task"]["53"]["zs"]), 0.0, 1), "2"),
    ("scheme: tomato sauce token", lambda: (float(_row("b5_scheme", "main")["per_task"]["59"]["token"]), 0.0, 1), "15"),
    ("scheme: tomato sauce zero-shot", lambda: (float(_row("b5_scheme", "main")["per_task"]["59"]["zs"]), 0.0, 1), "7"),
    ("scheme: bowl token", lambda: (float(_row("b5_scheme", "main")["per_task"]["60"]["token"]), 0.0, 1), "20"),
    ("scheme: bowl zero-shot", lambda: (float(_row("b5_scheme", "main")["per_task"]["60"]["zs"]), 0.0, 1), "12"),
    ("scheme: token binding", lambda: (_row("b5_scheme", "main")["binding"]["token"], 0.0, 1), "0.75"),
    ("scheme: zero-shot binding", lambda: (_row("b5_scheme", "main")["binding"]["zs"], 0.0, 1), "0.21"),
    ("scheme H4: transfer", lambda: (float(_row("b5_scheme", "main")["hypotheses"]["H4"]["x"]), 0.0, 1), "0"),
    ("scheme: population episodes", lambda: (float(_row("b5_scheme", "main")["population"]["episodes"]), 0.0, 1), "280"),
    ("split binding: ketchup (of 10)", q2("b_q2full", "48:bindrole", "success", 10), "0"),
    ("split binding: tomato sauce (of 10)", q2("b_q2full", "49:bindrole", "success", 10), "1"),
    ("PoE P0: early effect", lambda: (_row("q0_share", "read")["early_effect_max"], 0.0, 1), "0.38"),
    ("PoE P0: sensitivity at 0.5", lambda: (_row("q0_share", "read")["summary"]["0.5_0.5"]["sens"], 0.0, 1), "0.33"),
    ("PoE P2: tasks", lambda: (float(_row("q2_explain", "read")["tasks_with_demo1"]), 0.0, 1), "21"),
    ("PoE P2: beats policy on demo 1", lambda: (float(_row("q2_explain", "read")["recomputed_beats_policy"]), 0.0, 1), "21"),
    ("PoE P2: options NLL cut", lambda: (_row("q2_explain", "read")["options_nll_cut"], 0.0, 1), "0.57"),
    ("PoE P2: recomputed beats frozen", lambda: (float(_row("q2_explain", "read")["recomputed_vs_frozen"]), 0.0, 1), "10"),
    ("PoE P2: code KB", lambda: (_row("q2_explain", "read")["code_bytes_mean"] / 1e3, 0.0, 1), "1.6"),
    ("PoE pilot: lambda 0.5", lambda: (float(_row("q3_pilot", "read")["totals"]["poe_0.5"]), 0.0, 1), "3"),
    ("PoE pilot: zero-shot", lambda: (float(_row("q3_pilot", "read")["totals"]["zs"]), 0.0, 1), "1"),
    ("PoE pilot: lambda 0.3", lambda: (float(_row("q3_pilot", "read")["totals"]["poe_0.3"]), 0.0, 1), "0"),
]

CHECKS = EMBODIED + TRANSFER + BIND + [
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
