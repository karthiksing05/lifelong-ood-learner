"""Pre-registration of the no-demonstration scheme's main read (ledger m3_prereg tag b_scheme_main), written before any
practice episode of this protocol and before any episode on the scoring states 30-49 of these tasks.

Scheme: compose a new task from a task prototype the policy performs (role code, lifelong_ood/policy/roles.py), bind its
object slot by the form practice selects, practise, fold the successes into an 18.4 KB factor token, practise with the token
alone, refit; at test the policy gets ONLY the task's own instruction and the token."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger
from lifelong_ood.policy import factors as F

TASKS = [18, 21, 26, 48, 49, 52, 53, 58, 59, 60, 62, 65, 66, 71]
print(ledger.record(
    "m3_prereg", bench="libero_90", tag="b_scheme_main", seed=0,
    metrics={
        "plan": "~/.claude/plans/i-m-interested-in-re-workshopping-quizzical-crane.md (Bind and Chain) as revised by the Q2 reads; "
                "no demonstration of any held-out task is used anywhere",
        "parser_hash": F.parser_hash(),
        "population": {"tasks": TASKS, "rule": "every LIBERO-90 task pi0.5 fails zero-shot (< 0.8 on states 0-4) that has a role "
                       "code (a performed task of the same scene whose goal equals the task's with the moved object replaced)"},
        "forms": {"role_open": "the base task prototype's instruction; the new object and the base's object exchange appearance "
                               "in the simulator's renderer while the policy's commanded gripper is open (ORACLE; b_q2mini role@open)",
                  "attn": "the task's own instruction; the action expert's attention raised to mass 0.7 on the moved object's "
                          "agent-view tokens (simulator segmentation, ORACLE), all layers, while the gripper is open "
                          "(b_q2full bind@rho0.7_all_approach)"},
        "form_rule": "per task, the form with the highest success on practice states 0-9 (ledger b2_q2 tags b_q2mini and "
                     "b_q2full, already run); ties -> role_open; if the best practice success is below 0.2 the task gets no "
                     "form and no token (the scheme leaves it to the frozen policy)",
        "form_choice_from_the_rule": {"52": "role_open (0.7)", "53": "role_open (0.7)", "59": "attn (1.0)", "60": "role_open (1.0)",
                                      "others": "none (best practice success <= 0.1)"},
        "fold": {"round1": "the chosen form on practice states 10-29 (20 episodes, real scene recorded); its successes, "
                           "relabelled with the task's own instruction, fit a 1-entry KV token (experiments/f4_fold.py "
                           "--source rollouts, 600 steps, lr 1e-2; tag _s1)",
                 "round2": "the token alone (own instruction, no form, no simulator state) on states 10-29; its successes plus "
                           "round 1's refit the token (tag _s2); a task with no success in round 1 gets no token"},
        "scoring": "init states 30-49, 20 episodes per arm and task, client bind tracking; arms: zs (all 14 tasks), form (tasks "
                   "with a form), token = own instruction + the round-2 token (tasks with a token), transfer = task 49 with "
                   "task 59's token (tomato sauce learned in living room 3, used in living room 1, no practice on 49)",
        "hypotheses": {"H1": "token > zs, pooled over the tasks with a token (one-sided exact sign test on discordant paired "
                             "episodes)",
                       "H2": "form > zs, pooled over the tasks with a form",
                       "H4": "transfer > zs on task 49",
                       "correction": "Holm over H1, H2, H4",
                       "population": "descriptive: successes of the scheme (token where it exists, else zero-shot) against "
                                     "zero-shot over all 14 tasks (280 paired episodes); secondary: two-level bootstrap"},
        "deployability": "the forms use the simulator (renderer or segmentation) and are an oracle teacher during practice; the "
                         "token arm uses nothing but the instruction and the frozen policy's inputs",
        "written_before": "any practice episode of this protocol (tags _s1/_s2) and any episode on states 30-49 of these tasks",
    },
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/b_scheme_main.py", command="pre-registration of the scheme's main read"))
