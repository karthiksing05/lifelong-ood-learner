"""Launch rollouts two per GPU (experiments/roll_pack.sbatch), after each task's preparation (experiments/prep.sbatch).

Per task: the practice (zero-shot, states 0-9, m3_compose) and the final arms (states 30-49, m3_final).  A (task, arm) is
skipped when its result file exists or is in --running; preparation is submitted when the task's LoRA adapter is missing
(the prep's last step), or --prep-job TASK:JOB names a prep already queued.  Arms of one task are packed in pairs so a
pack waits on one prep.

    python experiments/launch_pack.py --tasks 1,6 --arms zeroshot,raw_replay,... [--running OUT1,OUT2] [--prep-job 34:3958606] [--dry]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = os.path.join(ROOT, "results", "robot")
C = "w0=1.0&sched=const&var=0.001&key=pose&stall=3"


def grasps(tid, demo=0):
    from lifelong_ood.policy import data as D
    spec = importlib.util.spec_from_file_location("r3p", os.path.join(ROOT, "experiments", "r3_prepare.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    texts, _ = m.libero90_texts()
    d = D.Demos90()
    ep = d.load(d.episodes_of_text(texts[tid])[demo])
    return len(D.gripper_events(ep["actions"])) > 0


GROUP_A = {35, 36, 39, 42, 1, 6, 34, 37, 40, 52, 53, 62}             # the cross-fitting split of the shared-fold test


def arm_spec(tid, arm, fold, demo=0):
    """-> (MODE, PROGRAM_BYTES, needs_prep) or None when the arm does not apply.  demo k > 0: the programs, fold and LoRA
    built from IPEC episode k of the task's text (files with the suffix _d<k>)."""
    if demo:
        d = f"_d{demo}"
        P = lambda name: os.path.join(R, name)                           # noqa: E731
        F0, EX = P(f"r4_{tid}_approach_last2{d}_program.npz"), P(f"r4_{tid}_approach_last2{d}_expert.pt")
        table = {"raw_replay": (f"program:{P(f'r3_prog_{tid}_raw_lib40{d}.npz')}?{C}", f"prog:{P(f'r3_prog_{tid}_raw_lib40{d}.npz')}", False),
                 "dimcode_s1": (f"program:{P(f'r7_dim_{tid}_s1{d}.npz')}?{C}&window=20", f"prog:{P(f'r7_dim_{tid}_s1{d}.npz')}", False),
                 "dimfold_s1_last2": (f"hybrid:{P(f'r7_dim_{tid}_s1{d}.npz')}?expert={EX}&foldprog={F0}&{C}&window=20",
                                      f"prog:{P(f'r7_dim_{tid}_s1{d}.npz')}@{F0}", True) if fold else None,
                 "lora": (f"lora:{P(f'r3_lora_{tid}{d}.pt')}", "0", True),
                 "seg_s5": (f"program:{P(f'r10_seg_{tid}_s5{d}.npz')}?{C}&window=4", f"prog:{P(f'r10_seg_{tid}_s5{d}.npz')}", False)}
        table["phasebind2_nofold"] = (f"phasebind:{tid}?fold=0&rules=2&demo={demo}&var=0.001", "0", False)
        table["phasebind2_reader_spatial4_synth"] = (f"phasebind:{tid}?fold=0&rules=2&demo={demo}&reader=spatial4_synth&var=0.001", "0", False)
        if arm == "dimbest":
            return table["dimfold_s1_last2"] if fold else table["dimcode_s1"]
        return table[arm]
    P = lambda name: os.path.join(R, name)                               # noqa: E731
    F0 = P(f"r4_{tid}_approach_last2_program.npz")
    EX = P(f"r4_{tid}_approach_last2_expert.pt")
    table = {
        "zeroshot": ("ours", "0", False),
        "raw_replay": (f"program:{P(f'r3_prog_{tid}_raw_lib40.npz')}?{C}", f"prog:{P(f'r3_prog_{tid}_raw_lib40.npz')}", True),
        "dimcode_s5": (f"program:{P(f'r7_dim_{tid}_s5.npz')}?{C}&window=4", f"prog:{P(f'r7_dim_{tid}_s5.npz')}", True),
        "dimcode_s1": (f"program:{P(f'r7_dim_{tid}_s1.npz')}?{C}&window=20", f"prog:{P(f'r7_dim_{tid}_s1.npz')}", True),
        "dimfold_s1_last2": (f"hybrid:{P(f'r7_dim_{tid}_s1.npz')}?expert={EX}&foldprog={F0}&{C}&window=20",
                             f"prog:{P(f'r7_dim_{tid}_s1.npz')}@{F0}", True) if fold else None,
        "lora": (f"lora:{P(f'r3_lora_{tid}.pt')}", "0", True),
        "leaf_s5": (f"program:{P(f'r9_leaf_{tid}_s5.npz')}?{C}&window=4", f"prog:{P(f'r9_leaf_{tid}_s5.npz')}", False),
        "leaf_s1": (f"program:{P(f'r9_leaf_{tid}_s1.npz')}?{C}&window=20", f"prog:{P(f'r9_leaf_{tid}_s1.npz')}", False),
        "leaffold_s1_last2": (f"hybrid:{P(f'r9_leaf_{tid}_s1.npz')}?expert={EX}&foldprog={F0}&{C}&window=20",
                              f"prog:{P(f'r9_leaf_{tid}_s1.npz')}@{F0}", True) if fold else None,
        "shared_s1": (f"program:{P(f'r9_shared_{tid}_s1.npz')}?{C}&window=20", f"prog:{P(f'r9_shared_{tid}_s1.npz')}", False),
        "seg_s5": (f"program:{P(f'r10_seg_{tid}_s5.npz')}?{C}&window=4", f"prog:{P(f'r10_seg_{tid}_s5.npz')}", False),
        "dimshared_all": (f"hybrid:{P(f'r7_dim_{tid}_s1.npz')}?expert={P('r13_shared_all_expert.pt')}&foldprog={F0}&{C}&window=20",
                          f"prog:{P(f'r7_dim_{tid}_s1.npz')}@{F0}", False) if fold else None,
        "dimshared_xfit": (f"hybrid:{P(f'r7_dim_{tid}_s1.npz')}?expert={P('r13_xfit_' + ('B' if tid in GROUP_A else 'A') + '_expert.pt')}"
                           f"&foldprog={F0}&{C}&window=20", f"prog:{P(f'r7_dim_{tid}_s1.npz')}@{F0}", False) if fold else None,
    }
    if arm in ("learned", "learned_mem0", "learned_origfold"):           # compose -> practise -> consolidate (r15_consolidate.py)
        extra = {"learned": "", "learned_mem0": "&mem=0", "learned_origfold": "&fold=orig"}[arm]
        return (f"learned:{tid}?{C}&window=20{extra}", "0", False)
    if arm.startswith("lora@"):                                          # lora@<name>: results/robot/<name>_<task>_lora.pt (rounds)
        return (f"lora:{os.path.join(R, arm[5:] + '_' + str(tid) + '_lora.pt')}", "0", False)
    if arm == "lora_learn":                                              # LoRA on the demo + the same successful rollouts
        return (f"lora:{os.path.join(R, 'r15_' + str(tid) + '_lora.pt')}", "0", False)
    if arm in ("demodiff", "demodiff_state"):                            # DemoDiffusion (SDEdit of the demo chunk), s* = 0.2, 20 steps
        return (f"demodiff:{tid}?s=0.2&steps=20&index={'time' if arm == 'demodiff' else 'state'}&window=20", "0", False)
    if arm in ("lora_eq", "lora_r64"):                                   # equal-bytes adapter / stronger LoRA (r3_fewshot.py)
        return (f"lora:{os.path.join(R, 'r3_lora_' + str(tid) + ('_eq' if arm == 'lora_eq' else '_r64') + '.pt')}", "0", False)
    if arm == "fullft":                                                  # full fine-tuning of the action expert
        return (f"fullft:{os.path.join(R, 'r3_full_' + str(tid) + '.pt')}", "0", False)
    if arm in ("playback", "vinn"):                                      # no policy: the demo's actions (open loop / nearest frame)
        return (f"{arm}:{tid}", "0", False)
    if arm in ("promptret", "loraret"):                                  # retrieval baselines (experiments/r14_retrieval.py)
        ch = json.load(open(os.path.join(R, "r14_retrieval.json")))[str(tid)]
        return ((f"prompt:{ch['prompt_task']}", "0", False) if arm == "promptret"
                else (f"lora:{os.path.join(R, 'r3_lora_' + str(ch['adapter_task']) + '.pt')}", "0", False))
    if arm == "phasebind":                                               # P1: the demo's phases re-anchored on the objects + the fold
        return (f"phasebind:{tid}?expert={EX if fold else ''}&foldprog={F0 if fold else ''}&var=0.001", "0", False)
    if arm == "phasebind_nofold":                                        # P1 arm B: the bindings alone through the prior
        return (f"phasebind:{tid}?fold=0&var=0.001", "0", False)
    if arm == "phasebind_reader":                                        # P6: arm B with the learned pose reader (no simulator poses)
        return (f"phasebind:{tid}?fold=0&reader=1&var=0.001", "0", False)
    if arm == "phasebind2_nofold":                                       # rules v2 (phase.py at_rest + cut), simulator poses
        return (f"phasebind:{tid}?fold=0&rules=2&var=0.001", "0", False)
    if arm == "phasebind2_reader":                                       # rules v2 with the learned pose reader
        return (f"phasebind:{tid}?fold=0&rules=2&reader=1&var=0.001", "0", False)
    if arm in ("compose2", "sibling2", "compose2_reader", "compose2_reader_pred"):   # P3c: composed from other tasks' phases (p3c_table.py)
        tab = json.load(open(os.path.join(R, "p3c_table.json")))["table"]
        if str(tid) not in tab:
            return None
        g = tab[str(tid)]["g" if arm != "sibling2" else "sibling"]
        rd = "&reader=spatial4_synthonly_rj" if arm.startswith("compose2_reader") else ""   # the reader trained on rendered scenes only
        rd += "&carry=pred" if arm == "compose2_reader_pred" else ""                        # the in-hand pose predicted from the grasp binding
        return (f"phasebind:{tid}?fold=0&rules=2&g={g}&p={tab[str(tid)]['p']}&obj={tab[str(tid)]['object']}{rd}&var=0.001", "0", False)
    if arm == "phasebind2_rjpred":                                       # P5: own demo, rendered-only reader, predicted carry
        return (f"phasebind:{tid}?fold=0&rules=2&reader=spatial4_synthonly_rj&carry=pred&var=0.001", "0", False)
    if arm == "phasebind2_rjpred_yaw":                                   # the same + the objects' yaw where it varies (p6_yaw_reader.py)
        return (f"phasebind:{tid}?fold=0&rules=2&reader=spatial4_synthonly_rj&carry=pred&yaw=1&var=0.001", "0", False)
    if arm.startswith("phasebind2_rjpred@"):                             # the same with reader variant <v> (p6_reader_<v>_<t>.npz), e.g. pca512_rj
        return (f"phasebind:{tid}?fold=0&rules=2&reader={arm.split('@', 1)[1]}&carry=pred&var=0.001", "0", False)
    if arm == "dsrl_train":                                              # DSRL practice: 100 episodes on init states 0-29 (STATE_MOD)
        return (f"dsrl:{tid}?train=1&save={os.path.join(R, f'dsrl_{tid}_actor.pt')}&warmup=10&utd=4&beta=2", "0", False)
    if arm == "dsrl":                                                    # DSRL scored: the trained actor, deterministic noise
        return (f"dsrl:{tid}?train=0&load={os.path.join(R, f'dsrl_{tid}_actor.pt')}&beta=2", "0", False)
    if arm == "phasebind2_rjpred_e":                                     # own demo, events fire on arrival (PhaseBindHooks events=1)
        return (f"phasebind:{tid}?fold=0&rules=2&reader=spatial4_synthonly_rj&carry=pred&events=1&var=0.001", "0", False)
    if arm == "phasebind2_rjpred_h":                                     # own demo, the prior carries between grasp and descent (handoff)
        return (f"phasebind:{tid}?fold=0&rules=2&reader=spatial4_synthonly_rj&carry=pred&handoff=1&var=0.001", "0", False)
    if arm.startswith("stream") and "@" in arm:                         # P5: composed arrival, streamc@<g>-<p> (p5_pool.json names)
        kind, gp = arm.split("@")                                        # streamw: carry warped; streamt: generic transport + place binding;
        g, p = gp.split("-")                                             # streamh: the prior carries; suffix e: events fire on arrival
        wp = {"w": "&warp=1", "t": "&transport=1", "h": "&handoff=1",      # x: contact-local grasp (prototype verbs) + transport + events
              "x": "&transport=1&contact=1&events=1"}.get(kind[6], "") + ("&events=1" if kind.endswith("e") else "") \
            + ("&crosspos=1" if "p" in kind[7:] else "") \
            + ("&lift=1" if "l" in kind[7:] else "")                     # suffix p: cross-type places transfer position only;
        #                                                                  suffix l: the transport lifts by the place source's own lift
        pool = json.load(open(os.path.join(R, "p5_pool.json")))[str(tid)]
        return (f"phasebind:{tid}?fold=0&rules=2&g={g}&p={p}&objn={pool['object']}&recn={pool['receptacle']}"
                f"&reader=spatial4_synthonly_rj&carry=pred{wp}&var=0.001", "0", False)
    if arm.startswith("phasebind2_reader_"):                             # rules v2 with reader variant <v> (p6_reader_<v>_<t>.npz)
        return (f"phasebind:{tid}?fold=0&rules=2&reader={arm[len('phasebind2_reader_'):]}&var=0.001", "0", False)
    if arm == "growleaf":                                                # P4: leaf borrowing over the success-grown library + the fold
        prog = P(f"r26_grow_{tid}_s1.npz")
        return ((f"hybrid:{prog}?expert={EX}&foldprog={F0}&{C}&window=20", f"prog:{prog}@{F0}", False) if fold
                else (f"program:{prog}?{C}&window=20", f"prog:{prog}", False))
    if arm == "leafbest":                                                # leaf borrowing + the fold where the demo grasps
        return table["leaffold_s1_last2"] if fold else table["leaf_s1"]
    if arm == "dimbest":                                                 # borrowing + the fold where the demo grasps
        return table["dimfold_s1_last2"] if fold else table["dimcode_s1"]
    return table[arm]


def sbatch(args, dry):
    if dry:
        print("  [dry] sbatch " + " ".join(args))
        return "DRY"
    return subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--practice", action="store_true", help="also the practice run (zero-shot, states 0-9)")
    ap.add_argument("--running", default="", help="result files being produced by running jobs (skip them)")
    ap.add_argument("--prep-job", default="", help="TASK:JOB,... preps already queued or running")
    ap.add_argument("--per", type=int, default=2, help="instances per GPU")
    ap.add_argument("--demo", type=int, default=0, help="k > 0: the arms built from IPEC episode k (tags and files _d<k>)")
    ap.add_argument("--experiment", default="m3_final", help="m3_final (states 30-49) or m3_practice (practice states 0-9)")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="the policy's noise seed (0 = every run so far); tags gain _seed<k>")
    ap.add_argument("--record", action="store_true", help="the client saves every episode (frames, states, actions, outcome)")
    ap.add_argument("--task-after", default="", help="TASK:JOB[:JOB...],... jobs a task's packs wait for (e.g. its consolidation)")
    ap.add_argument("--after", default="", help="job ids every pack of this run waits for (e.g. the shared-fold trainings)")
    ap.add_argument("--partition", default="", help="overcap: the packs run on the preemptible overcap partition (requeued)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join(R, ".packs"), exist_ok=True)
    running = {os.path.basename(x) for x in a.running.split(",") if x}
    prep_job = dict(x.split(":") for x in a.prep_job.split(",") if x)
    n_pack = len(os.listdir(os.path.join(R, ".packs")))
    queue = []
    for tid in [int(x) for x in a.tasks.replace("+", ",").split(",")]:
        fold = grasps(tid, a.demo)
        dsfx = f"_d{a.demo}" if a.demo else ""
        specs_free, specs_dep = [], []
        if a.practice:
            out = os.path.join(R, f"m3_compose_{tid}_zeroshot.json")
            if not os.path.exists(out) and os.path.basename(out) not in running:
                specs_free.append(dict(MODE="ours", SUITE="libero_90", TASKS=tid, TRIALS=10, FIRST=0, EXPERIMENT="m3_compose",
                                       TAG="zeroshot", PROGRAM_BYTES="0", OUT=out))
        for arm in a.arms.split(","):
            s = arm_spec(tid, arm, fold, a.demo)
            if s is None:
                continue
            tag = ("leaffold_s1_last2" if fold else "leaf_s1") if arm == "leafbest" else arm.replace("lora@", "lora_")
            tag = tag.replace("@", "_").replace("-", "_")
            tag = ("dimfold_s1_last2" if fold else "dimcode_s1") if arm == "dimbest" else tag
            tag += dsfx + (f"_seed{a.seed}" if a.seed else "")
            first = a.first if a.first is not None else (30 if a.experiment == "m3_final" else 0)
            trials = a.trials if a.trials is not None else (20 if a.experiment == "m3_final" else 10)
            out = os.path.join(R, f"{a.experiment}_{tid}_{tag}.json")
            if os.path.exists(out) or os.path.basename(out) in running:
                continue
            spec = dict(MODE=s[0], SUITE="libero_90", TASKS=tid, TRIALS=trials, FIRST=first, EXPERIMENT=a.experiment, TAG=tag,
                        PROGRAM_BYTES=s[1], OUT=out, **({"RECORD": "1"} if a.record else {}), **({"SERVE_SEED": str(a.seed)} if a.seed else {}),
                        **({"OBJECTS": "1"} if arm in ("phasebind", "phasebind_nofold", "phasebind2_nofold", "compose2", "sibling2") else {}),
                        **({"STATE_MOD": "30"} if arm == "dsrl_train" else {}))
            (specs_dep if s[2] else specs_free).append(spec)
        dep = None
        if specs_dep and not os.path.exists(os.path.join(R, f"r3_lora_{tid}{dsfx}.pt")):
            if str(tid) in prep_job:
                dep = prep_job[str(tid)]
            else:
                dep = sbatch([f"--export=ALL,TID={tid},FOLD={int(fold)},DEMO={a.demo}", "experiments/prep.sbatch"], a.dry)
                print(f"task {tid} prep -> job {dep} (fold {int(fold)})")
        tdep = ":".join(x.split(":", 1)[1] for x in a.task_after.split(",") if x and x.split(":")[0] == str(tid)) or None
        both = ":".join(d for d in (dep, tdep) if d) or None
        queue += [(sp, tdep) for sp in specs_free] + [(sp, both) for sp in specs_dep]
    # pack across tasks: specs without a dependency first, then those with one; a pack waits on the union of its deps
    queue.sort(key=lambda x: x[1] is not None)
    for i in range(0, len(queue), a.per):
        grp = queue[i:i + a.per]
        n_pack += 1
        path = os.path.join(R, ".packs", f"p_{n_pack:04d}.tsv")
        with open(path, "w") as f:
            for sp, _ in grp:
                f.write("\t".join(f"{k}={v}" for k, v in sp.items()) + "\n")
        deps = sorted({x for _, d in grp if d for x in d.split(":")} | {x for x in a.after.split(",") if x})
        args = ([f"--dependency=afterok:{':'.join(deps)}"] if deps else []) + ([f"--partition={a.partition}", f"--account={a.partition}"] if a.partition else []) \
            + [f"--export=ALL,SPECS={path}", "experiments/roll_pack.sbatch"]
        j = sbatch(args, a.dry)
        print(f"pack {os.path.basename(path)} [{', '.join(str(sp['TASKS']) + ':' + sp['TAG'] + ('/practice' if sp['EXPERIMENT'] == 'm3_compose' else '') for sp, _ in grp)}]"
              f"{' after ' + ','.join(deps) if deps else ''} -> job {j}")


if __name__ == "__main__":
    main()
