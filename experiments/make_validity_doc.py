"""Write docs/VALIDITY.md from the recorded results (no hand-copied numbers).

Sources: results/robot/m3_stats.json (experiments/m3_stats.py), results/robot/r8_provenance{,_leaf,_seg}.json
(experiments/r8_inspect.py), the latest r9_library_ablation ledger row, results/robot/r12_forced.json, the m3_prereg rows.

    python experiments/make_validity_doc.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                          # noqa: E402

R = os.path.join(ROOT, "results", "robot")
def _names():
    """The arm names of experiments/m3_stats.py (one source)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m3s", os.path.join(ROOT, "experiments", "m3_stats.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.NAMES


NAMES = _names()


def kb(b):
    return f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.1f} KB"


def _row(exp, tag):
    rows = [r for r in ledger.load(exp, tag=tag) if not r.get("retired")]
    return rows[-1]["metrics"] if rows else None


def _cmp(v):
    d = v["diff"]
    return (f"{v['x']}/{v['n']} vs {v['control']}/{v['n']}, {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], one-sided 95% bound "
            f"{d['lo90']:+.3f}, discordant {v['discordant'][0]}/{v['discordant'][1]}, sign p {v['sign_p']:.2g}")


def transfer_probes(w):
    """The probe ladder of docs/PLAN_TRANSFER.md, each read from its ledger row (pre-registered: ledger m3_prereg tags p0, p1,
    p1_amend1, p2, p3a, p3b, p4, g38, p6)."""
    geo, ret, p1, p1s = _row("p0_audit", "geometry"), _row("p0_audit", "retarget"), _row("p1_bindings", "read"), _row("p1_bindings", "secondary")
    if not (geo and p1):
        return
    w("## Transfer plan probes (docs/PLAN_TRANSFER.md; pre-registered: ledger m3_prereg tags p0, p1, p1_amend1, p2, p3a, p3b, p4, g38, p6)\n")
    w("Object poses come from the simulator (an ORACLE the client sends) unless a row says learned reader.  Init states 30-49, "
      "paired by episode; bounds from the two-level bootstrap; sign tests exact and two-sided.\n")
    pr = geo["primary"]
    w(f"- P0b (`ledger/p0_audit` tag geometry): of the {pr['n']} persistent movable-object failures replayed bit for bit "
      f"({geo['gate']['exact']}/{geo['gate']['replayed']} replays exact), {pr['aim_share']:.2f} are aim failures (the hand closes "
      f"off the object): C2 {geo['decision_C2']}.")
    if ret:
        w(f"- P0c (`ledger/p0_audit` tag retarget): the demo retargeted to new placements with no policy succeeds "
          f"{ret['totals']['retarget']}/{ret['n']} (open-loop replay {ret['totals']['replay']}/{ret['n']}): {ret['decision']}.")
    for lab, key in (("A (bindings + the 258 KB fold)", "A"), ("B (bindings alone)", "B")):
        v = p1[key]
        w(f"- P1 arm {lab}, 9 concentrated-failure tasks (`ledger/p1_bindings` tag read): {_cmp(v)}, Holm p {v['holm_p']:.2g}: "
          f"{'PASSES' if v['passes'] else 'fails'}{'; killed' if v['killed'] else ''}.")
    if p1s:
        for lab, key in (("A", "A"), ("B", "B")):
            v = p1s[key]
            w(f"- P1 secondary, arm {lab}, all 21 tasks (`ledger/p1_bindings` tag secondary): {_cmp(v)}, Holm p {v['holm_p']:.2g}: "
              f"{'passes' if v['passes'] else 'fails'}{'; the kill rule is met' if v['killed'] else ''}.")
        w("")
        w("| task | A: bindings + fold | B: bindings | control (borrowing + fold) |")
        w("|---|---|---|---|")
        for t, v in sorted(p1s["per_task"].items(), key=lambda x: int(x[0])):
            w(f"| {t} | {v['A']} | {v['B']} | {v['ctrl']} |")
        w("")
    p6o, p6r = _row("p6_reader", "offline"), _row("p6_reader", "read")
    if p6o:
        w(f"- P6 reader offline (`ledger/p6_reader` tag offline): median position error {p6o['median_of_medians_cm']:.2f} cm on the first frame "
          f"of init states 20-29 (the mean placement: {p6o['mean_placement_median_of_medians_cm']:.2f} cm).")
    if p6r:
        a, b = p6r["reader_vs_oracle"], p6r["reader_vs_control"]
        w(f"- P6 read, 9 tasks (`ledger/p6_reader` tag read): learned reader vs oracle {_cmp(a)}: "
          f"{'NON-INFERIOR' if a['non_inferior'] else 'not non-inferior'} at -0.10; vs the control {_cmp(b)}: "
          f"{'passes' if b['passes'] else 'fails'}.  Per task (reader / oracle / control): " +
          "; ".join(f"{t} {v['R']}/{v['B']}/{v['ctrl']}" for t, v in p6r["per_task"].items()) + ".")
    p6n = _row("p6_reader", "online")
    if p6n:
        sb = p6n["success_by_worst_error"]
        w(f"- P6 reader on the scored episodes (`ledger/p6_reader` tag online, experiments/p6_online_error.py): median position error "
          f"{p6n['median_cm']:.2f} cm, 90th percentile {p6n['p90_cm']:.2f} cm; episodes whose worst object is read within 1 cm succeed "
          f"{sb['<1 cm']['k']}/{sb['<1 cm']['n']}, within 2 cm {sb['<2 cm']['k']}/{sb['<2 cm']['n']}, beyond 2 cm {sb['>=2 cm']['k']}/{sb['>=2 cm']['n']}.  "
          "Median error per object (cm): " + "; ".join(
              f"{t} " + ", ".join(f"{k.rsplit('_', 1)[0]} {e:.2f}" for k, e in v["err_cm_median"].items())
              for t, v in p6n["per_task"].items()) + ".")
    gv2, v2 = _row("g38_generate", "demo_retarget_v2"), _row("p1_bindings", "v2")
    if gv2 and v2:
        w("")
        w("**Rules v2 (pre-registered: ledger m3_prereg tag v2).** A defect found after the reads above: the living-room demos' first "
          "kept steps catch the object still falling after the scene spawned (ketchup 12 cm above rest), and v1 anchored the approach "
          "on that pose, so its grasp aimed under the object. v2 (lifelong_ood/policy/phase.py `at_rest`, `cut`) reads the demo's "
          "objects at rest, finds the manipulated object by displacement (lifted or pushed), and anchors carries and fixture tasks on "
          f"fixtures too. Development check on the practice states 0-19 with no policy (`ledger/g38_generate` tag demo_retarget_v2): "
          f"{gv2['generated_successes']}/{gv2['attempts']} generated successes against {gv2['v1_generated_successes']} for v1, gains only "
          "on 42, 48, 52, 58.")
        for k, lab in (("V1", "V1, v2 vs v1 (simulator poses)"), ("V2", "V2, v2 vs LoRA + failure negatives"),
                       ("V4", "V4, LoRA on the v2 generator's demonstrations vs 3.8's")):
            v = v2[k]
            w(f"- {lab} (`ledger/p1_bindings` tag v2): {_cmp(v)}, Holm p {v['holm_p']:.2g}: {'PASSES' if v['passes'] else 'fails'}.")
        v = v2["V3"]
        w(f"- V3, the learned first-frame reader vs the simulator's poses under v2: {_cmp(v)}: "
          f"{'non-inferior' if v['non_inferior'] else 'NOT non-inferior'} at -0.10.")
        for k, lab in (("reader_vs_nft", "the learned reader vs LoRA + failure negatives"), ("g27_vs_nft", "the v2 generator's LoRA vs LoRA + failure negatives")):
            w(f"- descriptive, {lab}: {_cmp(v2[k])}.")
        w("")
        w("| task | v2 | v2, learned reader | v1 | LoRA + failure negatives | LoRA on v2 demonstrations | control |")
        w("|---|---|---|---|---|---|---|")
        for t, r in sorted(v2["per_task"].items(), key=lambda x: int(x[0])):
            w(f"| {t} | {r['v2']} | {r['v2r']} | {r['v1']} | {r['nft']} | {r['g27']} | {r['ctrl']} |")
        w("")
    pc, pcr = _row("p3_compose", "p3c"), _row("p3_compose", "p3c_reader")
    if pc:
        w("**P3c: tasks composed from other tasks' phases (pre-registered: ledger m3_prereg tags p3c, p3c_amend1, p3c_reader).** "
          "The approach of a task that manipulates the target's object type + the carry and the rest of a task with the target's goal "
          "relation (experiments/p3c_table.py), relabelled onto the target scene and followed by the frozen prior; the target's own "
          "demo is not used. Amendment 1: the sibling control crashed at server start before any episode (type-based relabelling); "
          "fixed by role-based relabelling.")
        w(f"- C1, composed vs the task's own demo (v2, simulator poses): {_cmp(pc['C1'])}: {'NON-INFERIOR' if pc['C1']['non_inferior'] else 'not non-inferior'} at -0.10.")
        w(f"- C2, composed vs the sibling's whole demo (another object type's grasp): {_cmp(pc['C2'])}: {'PASSES' if pc['C2']['passes'] else 'fails'}.")
        if pcr:
            w(f"- R1, composed with a pose reader trained only on rendered scenes vs composed with the simulator's poses: {_cmp(pcr['R1'])}: "
              f"{'non-inferior' if pcr['R1']['non_inferior'] else 'NOT non-inferior'} at -0.10.")
        w("")
        w("| task | composed | own demo | sibling | composed, rendered-scene reader | LoRA + failure negatives |")
        w("|---|---|---|---|---|---|")
        for t, r in sorted(pc["per_task"].items(), key=lambda x: int(x[0])):
            w(f"| {t} | {r['c']} | {r['own']} | {r['sib']} | {pcr['per_task'][t]['cr'] if pcr else '-'} | {r['nft']} |")
        w("")
    rd = [(tag, _row("p6_reader", tag)) for tag in ("near_offline", "offline_spatial4", "offline_synth", "offline_spatial4_synth",
                                                     "offline_spatial4_synthonly_p3c", "offline_spatial4_synthonly_rj_p3c")]
    lab = {"near_offline": "re-reading the grasped object when the hand is within 10 cm of its grasp point",
           "offline_spatial4": "the last feature map pooled to 4 x 4 instead of the global pool",
           "offline_synth": "400 simulator-rendered placements per task added to the training frames (labels from simulator state, training only)",
           "offline_spatial4_synth": "both",
           "offline_spatial4_synthonly_p3c": "the P3c targets, rendered scenes ONLY (the arm at reset()'s fixed pose): a domain gap",
           "offline_spatial4_synthonly_rj_p3c": "the P3c targets, rendered scenes only with the arm's joints from a practice init state"}
    got = [(t, r) for t, r in rd if r]
    if got:
        w("Pose-reader variants, offline on the first frames of init states 20-29 (`ledger/p6_reader`):")
        for t, r in got:
            if t == "near_offline":
                w(f"- {lab[t]}: median {r['median_near_cm']:.2f} cm against {r['median_first_cm']:.2f} cm for the first frame (the same episodes).")
            else:
                w(f"- {lab[t]}: median {r['median_of_medians_cm']:.2f} cm; per object: " + "; ".join(
                    f"{tk} " + ", ".join(f"{k.rsplit('_', 1)[0]} {e:.1f}" for k, e in v["err_cm_median"].items()) for tk, v in r["per_task"].items()) + ".")
        w("")
    p2 = _row("p2_negatives", "read")
    if p2:
        v = p2["all_21"]
        w(f"- P2 (`ledger/p2_negatives` tag read): phase-localized negatives {v['phase']}/{v['n']} vs pointer-localized {v['stall']}/{v['n']}, "
          f"discordant {v['discordant'][0]}/{v['discordant'][1]}, sign p {v['sign_p']:.2g}: {p2['P2']}.")
    p3a, p3b = _row("p3_compose", "p3a"), _row("p3_compose", "p3b")
    if p3a and p3b:
        w(f"- P3a (`ledger/p3_compose` tag p3a): bindings looked up from other tasks {p3a['cross']}/{p3a['n']} vs own {p3a['own']}/{p3a['n']}: "
          f"X1 {p3a['X1'].split(' (')[0].lower()}.  P3b, confirmatory on new targets (tag p3b): {p3b['cross']}/{p3b['n']}; "
          f"{p3b['reading'].split(':')[0]}.  The minimal stream (P5) is off by the plan's rule.")
    p4 = _row("p4_growth", "rollouts")
    if p4:
        w(f"- P4 (`ledger/p4_growth` tag rollouts): the success-grown library {p4['grown']}/{p4['diff']['n_pairs']} vs the base "
          f"{p4['base']}/{p4['diff']['n_pairs']}: killed (growth stays bookkeeping).")
    g = _row("g38_generate", "read")
    if g:
        v1, v2 = g["gen-onedemo"], g["gen-learn"]
        w(f"- 3.8 (`ledger/g38_generate` tag read): a LoRA on the demo + demonstrations generated by retargeting it (no practice "
          f"rollouts) {v1['x']}/{v1['n']} vs the one-demo LoRA {v1['y']}/{v1['n']} (sign p {v1['sign_p']:.2g}; G1 {g['G1']}) and vs the "
          f"practice LoRA {v2['y']}/{v2['n']} (G2 {g['G2']}).")
    w("")


def main():
    S = json.load(open(os.path.join(R, "m3_stats.json")))
    L = []
    w = L.append
    w("# Validity of the composition results\n")
    w("Written for: collaborators checking whether the embodied claims hold.  Generated by `experiments/make_validity_doc.py` from "
      "`results/robot/m3_stats.json` and the ledger rows named below; rerun it after new rows land.\n")
    w("## Protocol\n")
    w(f"- Held-out tasks: {len(S['tasks'])} LIBERO-90 tasks pi05_libero never saw; the first six chosen during development, the rest drawn "
      "by `random.Random(0)` from the unique-language tasks the prior fails zero-shot (ledger `m3_prereg`, tag wave_a).")
    w("- ONE demonstration per task (IPEC episode 0 of the task's text).  Every method first practises the prior for 10 episodes "
      "(init states 0-9) and keeps it where at least half succeed; methods are scored on init states 30-49, 20 episodes each, "
      "which no development run touched.")
    w("- Bytes are recomputed from the artifacts for every arm (`experiments/m34_report.arm_bytes`).  A replayed demo is "
      "charged its T x 7 action sequence (corrected 2026-09-26: earlier rows charged a full chunk per frame, about five times too much).")
    w(f"- Statistics (`lifelong_ood/stats.py`): Wilson intervals per arm; pooled rates and paired differences with a two-level "
      f"bootstrap (tasks, then init states, the same states for both arms); the exact sign test on discordant pairs.  "
      f"Pre-registered before the wave results: non-inferiority to the LoRA at a margin of {S['margin']:.2f} (one-sided 95% lower "
      "bound of the pooled paired difference), superiority to zero-shot, non-inferiority to replay; the practice-allocation rule "
      "(ledger `m3_prereg`, tag alloc).\n")
    w(f"Tasks where the prior's practice fails: {len(S['prior_fails'])} of {len(S['tasks'])}; kept prior on {', '.join(S['kept_prior']) or 'none'}.\n")
    for scope, title in (("prior_fails", "tasks where pi0.5 fails its practice"), ("all", "all tasks")):
        w(f"## Pooled success, {title}\n")
        w("| method | successes | rate (Wilson 95%) | bootstrap 95% | tasks | bytes stored |")
        w("|---|---|---|---|---|---|")
        for m, name in NAMES.items():
            r = S["methods"].get(f"{m}.{scope}")
            if not r:
                continue
            flag = "" if r["complete"] else " (not every task)"
            w(f"| {name}{flag} | {r['k']}/{r['n']} | {r['rate']:.3f} ({r['wilson'][0]:.2f}-{r['wilson'][1]:.2f}) | {r['lo']:.2f}-{r['hi']:.2f} | "
              f"{r['tasks']} | {kb(r['bytes'])} |")
        w("")
        w("| comparison | pooled difference | 95% interval | one-sided lower | non-inferior at the margin | discordant | sign test p | pairs, tasks |")
        w("|---|---|---|---|---|---|---|---|")
        for key, d in S["compare"].items():
            if not key.endswith("." + scope):
                continue
            x, y = key[: -len(scope) - 1].split("-")
            w(f"| {NAMES[x]} vs {NAMES[y]} | {d['diff']:+.3f} | {d['lo']:+.3f} to {d['hi']:+.3f} | {d['lo90']:+.3f} | "
              f"{'yes' if d['noninferior'] else 'not shown'} | {d['discordant'][0]} vs {d['discordant'][1]} | {d['sign_p']:.3f} | "
              f"{d['n_pairs']}, {d['n_tasks']} |")
        w("")
    if "retest" in S:
        r = S["retest"]
        w("## Reliability\n")
        w(f"- Test-retest: the same (task, arm) on two disjoint init-state sets (10-29 and 30-49) over {r['cells']} cells: Pearson r = "
          f"{r['pearson']:.3f}, mean absolute difference {r['mean_abs_diff']:.3f}.")
    if "demo_robustness" in S:
        D = S["demo_robustness"]
        w(f"- A second demonstration (IPEC episode 1) on tasks {', '.join(D['tasks'])}, same init states: "
          + "; ".join(f"{k} {v['demo0']['k']}/{v['demo0']['n']} with demo 0, {v['demo1']['k']}/{v['demo1']['n']} with demo 1"
                      for k, v in D["arms"].items()) + ".")
    f12 = os.path.join(R, "r12_forced.json")
    if os.path.exists(f12):
        F = json.load(open(f12))
        w(f"- Forced execution (M2 validation iv, run retroactively; `ledger/r12_forced`): the {len(F['prototypes'])} most-shared t = 0.4 "
          f"prototypes executed open-loop in 5 scenes x {F['states']} init states; the {F['moving']} that command at least 1.5 cm realise a median "
          f"fraction {F['moving_frac_median']:.2f} of the commanded displacement, cosine {F['moving_cos_median']:.3f}.  Kill (< 0.5): "
          f"{'triggered' if F['killed'] else 'not triggered'}.")
    if "exploratory_novelty" in S:
        E = S["exploratory_novelty"]
        w(f"- EXPLORATORY (not pre-registered): the library's novelty read (fraction of the demo's frames it leaves unexplained) against "
          f"success over {len(E['tasks'])} tasks: borrowing + fold Spearman {E['ours']['rho']:+.2f} (permutation p {E['ours']['perm_p']:.3f}), "
          f"LoRA {E['lora']['rho']:+.2f} (p {E['lora']['perm_p']:.3f}), their difference {E['ours_minus_lora']['rho']:+.2f} (p {E['ours_minus_lora']['perm_p']:.3f}).")
    w("")
    if "second_demo" in S:
        D2 = S["second_demo"]
        w("## A second demonstration (pre-registered: ledger m3_prereg tag demo1)\n")
        w(f"Every arm rebuilt from IPEC episode 1 of each task's text, same init states 30-49, {len(D2['tasks'])} tasks. Borrowing + "
          "fold uses the fold of that demo (borrowing alone where it does not grasp).\n")
        names = {"ours_d1": "borrowing + fold, demo 1", "lora_d1": "LoRA, demo 1", "replay_d1": "replay, demo 1",
                 "ours_both": "borrowing + fold, both demos", "lora_both": "LoRA, both demos", "replay_both": "replay, both demos"}
        w("| arm | successes |")
        w("|---|---|")
        for k, v in D2["rates"].items():
            w(f"| {names.get(k, k)} | {v['k']}/{v['n']} ({v['k'] / v['n']:.3f}) |")
        w("")
        w("| comparison | difference | 95% interval | one-sided lower | non-inferior at -0.10 | sign test p |")
        w("|---|---|---|---|---|---|")
        for k, d in D2["compare"].items():
            w(f"| {k} | {d['diff']:+.3f} | {d['lo']:+.3f} to {d['hi']:+.3f} | {d['lo90']:+.3f} | {'yes' if d['noninferior'] else 'no'} | {d['sign_p']:.3f} |")
        w("")
        w("RETRACTION: non-inferiority of borrowing + fold to the per-task LoRA holds with the first demo only; with the second demo "
          "the LoRA is better, and over both demos pooled it is not shown.\n")
    f5 = os.path.join(R, "m5_learn_stats.json")
    if os.path.exists(f5):
        LS = json.load(open(f5))
        w("## Redo on the practice instances (pre-registered: ledger m3_prereg tag learn_from_practice; ledger m5_learn_stats)\n")
        w("Reruns with the practice's own noise seed replay the practice episode when they run the same init states in the same "
          "order (LIBERO's simulator state depends on the episodes before it in a session, not only on the init state: see the "
          "stall-point branching section); the controls are fresh-noise retries (policy seeds 1 and 2).\n")
        for scope, title in (("L1", "instances the first composition solved"), ("L2", "instances the first composition failed")):
            w(f"**{title}**\n")
            w("| arm | successes |")
            w("|---|---|")
            for k, v in LS[scope].items():
                if "-" not in k:
                    w(f"| {k} | {v['k']}/{v['n']} ({v['rate']:.3f}) |")
            w("")
            w("| comparison | difference | 95% interval | sign test p |")
            w("|---|---|---|---|")
            for k, v in LS[scope].items():
                if "-" in k:
                    w(f"| {k} | {v['diff']:+.3f} | {v['lo']:+.3f} to {v['hi']:+.3f} | {v['sign_p']:.4f} |")
            w("")
    f18 = os.path.join(R, "r18_branch_stats.json")
    if os.path.exists(f18):
        B = json.load(open(f18))
        n_inst = max(v["n"] for k, v in B["D1"].items())
        w("## Stall-point branching (pre-registered: ledger m3_prereg tags branch, branch_amend1-4; ledger r18_branch tag diagnosis)\n")
        w(f"The {n_inst} practice instances (init states 0-19) that the first composition and both fresh-noise retries failed.  Each "
          "branch re-enters the recorded episode two policy queries before its stall (the first query at which the demo-alignment "
          "pointer reaches its final maximum), or eight for the early arms, and changes one thing to the end of the episode.  The "
          "prefix executes the recorded actions in a fresh environment built as at the start of the recorded session and reset as "
          "often as the session had been; the server follows every prefix observation without sampling.\n")
        w(f"- Gates: prefixes exact in {B['G1']['exact']} of {B['G1']['branches']} branches; full replays reproduce the recorded states "
          f"in {B['G2']['states_reproduced']} of {B['G2']['replays']}; the recorded-noise continuation reproduces the recorded actions "
          f"in {B['G3']['reproduced']} of {B['G3']['late_replays']} instances.")
        w("- Run 1 of this experiment was discarded before any read (amendment 4): one environment per client process let earlier "
          "branches leave simulator state that re-seeding did not clear on seven tasks, and knob-free branches then 'succeeded'.\n")
        lab = {"late_replay": "recorded noise, nothing changed", "late_noise1": "fresh noise, seed 1", "late_noise2": "fresh noise, seed 2",
               "late_noise3": "fresh noise, seed 3", "late_noise4": "fresh noise, seed 4", "late_wx0.5": "guide weight x0.5",
               "late_wx2": "guide weight x2", "late_leaf2": "next-ranked leaves", "late_release": "the prior alone",
               "late_nofold": "the program through the fold windows", "late_foldall": "the fold everywhere",
               "early_noise1": "8 queries earlier, fresh noise 1", "early_noise2": "8 queries earlier, fresh noise 2",
               "any_late_noise": "any of the four fresh-noise branches", "any_late_knob": "any knob", "any_late": "any late branch",
               "any_early": "any early branch", "any": "any branch"}
        w("| branch | rescued | rate (Wilson 95%) | pointer past the stall | actions changed |")
        w("|---|---|---|---|---|")
        for k, name in lab.items():
            v = B["D1"].get(k)
            if v:
                w(f"| {name} | {v['k']}/{v['n']} | {v['rate']:.3f} ({v['wilson'][0]:.2f}-{v['wilson'][1]:.2f}) | "
                  f"{v.get('passes_stall', '')} | {v.get('active', '')} |")
        w("")
        w("| knob vs control | knob | control | discordant | sign test p | Holm p |")
        w("|---|---|---|---|---|---|")
        for k, d in B["D2"].items():
            w(f"| {k} | {d['knob']}/{d['n']} | {d['control']}/{d['n']} | {d['discordant'][0]}/{d['discordant'][1]} | {d['sign_p']:.3f} | {d['holm_p']:.3f} |")
        w("")
        D3 = B["D3"]
        w(f"Early vs late fresh noise (seeds 1-2, paired): {D3['early']} vs {D3['late']} of {D3['n']}, discordant {D3['discordant'][0]}/{D3['discordant'][1]}, "
          f"sign test p {D3['sign_p']:.3f}.  Instances of the nine concentrated tasks rescued by some branch: {B['rescued_concentrated']}.\n")
        w("Pre-registered reading: every late arm rescues at most 0.10 and no knob beats its control, so the persistent failures are "
          "not a bad draw near the stall: the budget goes to geometry or an earlier cause (Experiment 5 of the report).\n")
        ex = B.get("exploratory")
        if ex:
            w(f"Exploratory: {ex['at_end']} of {n_inst} stalls sit at the demo's last frames (the alignment pointer follows the hand, so "
              f"there the hand finished the demo's path and the task still failed; {ex['at_end_rescued']} rescued), {ex['mid']} elsewhere "
              f"({ex['mid_rescued']} rescued).  Where the demo holds something at the stall frame ({ex['demo_holds']} instances), the "
              f"robot's gripper is more than 5 mm tighter than the demo's in {ex['tighter']}.\n")
        f18b = os.path.join(R, "r18_recovery_stats.json")
        if os.path.exists(f18b):
            RB = json.load(open(f18b))
            w("**Recovery data (part b, pre-registered R1 / R2).**  r18rec = Experiment 1's learner (LoRA + failure negatives) plus the "
              "rescued post-branch segments as positives and the failed branches' own stalls as negatives, scored on init states 30-49.\n")
            w("| scope | r18rec | LoRA + failure negatives | LoRA round 3 (success-only) | r18rec - negatives (sign p) | r18rec - round 3 (sign p) |")
            w("|---|---|---|---|---|---|")
            for sc, name in (("R1_concentrated", "R1: the 9 concentrated tasks"), ("R2_rescued", "R2: tasks with a rescue"),
                             ("all_trained", "all 14 trained tasks")):
                o_ = RB[sc]
                d1, d2 = o_["r18rec-r17bal"], o_["r18rec-L3"]
                w(f"| {name} | {o_['r18rec']['k']}/{o_['r18rec']['n']} | {o_['r17bal']['k']}/{o_['r17bal']['n']} | {o_['L3']['k']}/{o_['L3']['n']} | "
                  f"{d1['diff']:+.3f} ({d1['sign_p']:.3f}) | {d2['diff']:+.3f} ({d2['sign_p']:.3f}) |")
            w("")
            w("R1 fails: the recovery data add nothing detectable to the learner with failure negatives.  Its lead over the round-3 "
              "learner (which used 630 more practice episodes) is the negatives' lead: the learner with negatives alone shows it too.\n")
    f19, f20 = os.path.join(R, "r19_invariance.json"), os.path.join(R, "r20_family_stats.json")
    if os.path.exists(f19):
        V = json.load(open(f19))
        w("## What the levels of the hierarchy carry (ledger r19_invariance)\n")
        w(f"{V['n_human']} human demos and {V['n_robot']} successful closed-loop practice rollouts of {len(V['tasks'])} held-out tasks; every "
          "chunk labelled at every level by the tree's basin rule, an episode = its label sequence with repeats collapsed.  Task "
          f"identification by the nearest other episode (normalised edit distance), chance {V['chance']:.3f}.\n")
        w("| level t | prototypes used | labels per episode | edit distance within / between tasks | human demos | robot rollouts | robot by human demos |")
        w("|---|---|---|---|---|---|---|")
        for u, r in V["levels"].items():
            w(f"| {u} | {r['used']} | {r['seq_len']:.1f} | {r['within']:.3f} / {r['between']:.3f} | {r['acc_human']:.3f} | {r['acc_robot']:.3f} | "
              f"{r['acc_robot_by_human']:.3f} |")
        w(f"| raw trajectories | - | - | - | {V['raw']['acc_human']:.3f} | {V['raw']['acc_robot']:.3f} | {V['raw']['acc_robot_by_human']:.3f} |")
        w("")
    if os.path.exists(f20):
        F = json.load(open(f20))
        w("## Family transfer (pre-registered: ledger m3_prereg tag family; ledger r20_family tag read)\n")
        w(f"Families at t = 0.4: {F['families']}.  Each target: its one demo + 1,000 frames of practice data from the other family members "
          "(fam) or from as many random tasks outside the family (rnd, size-matched); LoRA rank 16, half of every batch from the own demo; "
          "the target's own practice is never used; init states 30-49.\n")
        w("| scope | fam | rnd | one-demo LoRA | fam - rnd (sign p) | fam - one demo (sign p) |")
        w("|---|---|---|---|---|---|")
        for sc, name in (("all_targets", "all 12 targets (F1)"), ("pick_place_family", "pick-and-place family (F2)")):
            o_ = F[sc]
            d1, d2 = o_["fam-rnd"], o_["fam-onedemo"]
            w(f"| {name} | {o_['fam']['k']}/{o_['fam']['n']} | {o_['rnd']['k']}/{o_['rnd']['n']} | {o_['onedemo']['k']}/{o_['onedemo']['n']} | "
              f"{d1['diff']:+.3f} ({d1['sign_p']:.3f}) | {d2['diff']:+.3f} ({d2['sign_p']:.3f}) |")
        w("")
        w("F1 fails in the reverse direction: family data hurt.  Per task (fam / rnd / one demo): " +
          "; ".join(f"{t} {v['fam']}/{v['rnd']}/{v['onedemo']}" for t, v in F["all_targets"]["per_task"].items()) +
          ".  On the drawer tasks the family-trained adapter carries out a sibling's instruction (figures/robot/r20_task3_objects.png).\n")
    f21 = os.path.join(R, "r21_family_stats.json")
    if os.path.exists(f21):
        F2 = json.load(open(f21))
        w("## Two-stage family transfer (pre-registered: ledger m3_prereg tag family2; ledger r20_family tag read_r21)\n")
        w("Stage 1: LoRA on the family's (fam) or the random tasks' (rnd) pool alone; stage 2: the one-demo LoRA's exact recipe from the "
          "stage-1 adapter.  T1: fam vs the one-demo LoRA; T2: fam vs rnd; Holm over the two.\n")
        w("| scope | fam | rnd | one-demo LoRA | T1 fam - one demo (sign p, Holm) | T2 fam - rnd (sign p, Holm) |")
        w("|---|---|---|---|---|---|")
        for sc, name in (("all_targets", "all 12 targets"), ("pick_place_family", "pick-and-place family")):
            o_ = F2[sc]
            d1, d2 = o_["fam-onedemo"], o_["fam-rnd"]
            w(f"| {name} | {o_['fam']['k']}/{o_['fam']['n']} | {o_['rnd']['k']}/{o_['rnd']['n']} | {o_['onedemo']['k']}/{o_['onedemo']['n']} | "
              f"{d1['diff']:+.3f} ({d1['sign_p']:.3f}, {d1['holm_p']:.3f}) | {d2['diff']:+.3f} ({d2['sign_p']:.3f}, {d2['holm_p']:.3f}) |")
        w("")
        w("Both fail: the second stage restores the target's binding, and the family's pretraining adds nothing to it.\n")
    transfer_probes(w)
    w("## What composition borrows\n")
    for src, name in (("", "tree prototypes (band t = 0.2-0.4)"), ("_leaf", "leaves (the policy's chunk means)"), ("_seg", "segments of leaves")):
        p = os.path.join(R, f"r8_provenance{src}.json")
        if os.path.exists(p):
            P = json.load(open(p))
            eff = [v["effective_sources"] for v in P["tasks"].values()]
            w(f"- {name}, {len(P['held'])} programs: {min(eff):.0f}-{max(eff):.0f} effective source tasks each; word overlap of the sources "
              f"with the task's instruction {P['overlap_mean']:.3f} against {P['null_mean']:.3f} +- {P['null_std']:.3f} for shuffled pairings "
              f"(permutation p {'< 0.0001' if P['perm_p'] < 0.0001 else '= ' + format(P['perm_p'], '.4f')}, {P['n_perms']:,} permutations).")
            if src == "" and "lenders" in P:
                g = P["lenders"]
                w(f"  Lenders: {g['level_0.2']:.2f} / {g['level_0.3']:.2f} / {g['level_0.4']:.2f} of the borrowed pairs from t = 0.2 / 0.3 / 0.4; "
                  f"{g['le3_members']:.2f} from prototypes with at most 3 member frames, {g['single_task']:.2f} from single-task prototypes, "
                  f"{g['ge3_tasks']:.2f} from prototypes that at least 3 tasks share.")
    r9 = sorted(ledger.load("r9_library_ablation"), key=lambda r: len(r["metrics"]["held"]))       # the row over the most demos
    if r9:
        Sm = r9[-1]["metrics"]["summary"]
        w(f"- Library controls (`ledger/r9_library_ablation`, {len(r9[-1]['metrics']['held'])} held-out demos, K = 8): distance of the rebuilt chunks "
          "to the demo's and jerk (the demo's own jerk is higher):")
        w("")
        w("| library | entries | distance to the demo | jerk |")
        w("|---|---|---|---|")
        lab = {"band_K8": "tree prototypes, t = 0.2-0.4 (the method)", "t0.2_K8": "t = 0.2 alone", "t0.3_K8": "t = 0.3 alone",
               "t0.4_K8": "t = 0.4 alone (the shared prototypes)", "t0.3+0.4_K8": "t = 0.3 + 0.4", "bank_demo_K8": "raw demo chunks (no discovery)",
               "bank_policy_K8": "the policy's chunk means (leaves)", "bank_rand_K8": "a random subset of raw chunks, the band's size",
               "band_K1": "tree prototypes, K = 1", "band_K2": "tree prototypes, K = 2", "band_K4": "tree prototypes, K = 4"}
        for k, name in lab.items():
            if k in Sm:
                w(f"| {name} | {Sm[k]['library']:,} | {Sm[k]['rms']:.3f} | {Sm[k]['jerk']:.4f} |")
        w("")
    out = os.path.join(ROOT, "docs", "VALIDITY.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[doc] {out}: {len(L)} lines")


if __name__ == "__main__":
    main()
