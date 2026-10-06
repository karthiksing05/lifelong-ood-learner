"""Plan W, read W2 (docs/PLAN_WANG.md): W1's stream with RE-VALIDATION -- nothing damaged.  E4 showed the failure: an arrival
kept on a composition whose concepts later arrivals refit can fall from 13 of 20 to 0 (ledger e4_stream, order b).  Here, after
every refit, each earlier arrival served by the library whose composition CHANGED re-runs its practice (stored states 0-9, 10
episodes, noise seed 13) with the new composition, and must pass the rule it was kept under (beat the policy; and, without a
demonstration of its own, reach 8 of 10).  An arrival that fails and never gave its demonstration requests it now (its concepts are
refit and re-validated in turn); one that already gave it falls back to the policy.  So every served composition has passed its
practice with the library as it is at the end.  Same orders, prior library and rules as W1 otherwise (W1's and E4's acceptance runs
of an identical composition are reused).  Resumable: results/robot/w2_state.json.

    nohup ~/flash/miniconda3/envs/chunks/bin/python experiments/w2_stream.py --run > results/robot/w2_driver.log 2>&1 &
    ~/flash/miniconda3/envs/chunks/bin/python experiments/w2_stream.py --read [--record]
"""
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec_ = importlib.util.spec_from_file_location("w1", os.path.join(ROOT, "experiments", "w1_stream.py"))
W1 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(W1)
E4 = W1.E4

R = E4.R
STATE = os.path.join(R, "w2_state.json")
THETA = W1.THETA
E4.name = lambda o, k, e: f"W2{o}{k[0]}{e['idx']}v{e['v']}"


def acc_of(t, arm):
    for exp in ("w2_accept", "w1_accept", "e4_accept"):
        x = E4.succ(exp, t, arm)
        if x is not None:
            return x
    return None


def passes(a, acc):
    """The rule the arrival was kept under: beat the policy, and reach THETA when it gave no demonstration."""
    return acc is not None and acc > a["zs"] and (bool(a.get("demo")) or acc >= THETA)


def changed(s, meta, skip=None):
    """Earlier arrivals served by the library whose composition differs from the one they last passed with."""
    out = []
    for a in s["arrivals"]:
        if "kept" not in a or a["kept"] == "zs" or a["task"] == skip:
            continue
        c = E4.compose(s, meta["keys"][str(a["task"])])
        if c != a["valid"]:
            out.append((a, c))
    return out


def keep(s, a, comp, acc):
    a["kept"] = comp if passes(a, acc) else "zs"
    a.setdefault("arrival_kept", a["kept"])              # what the arrival first kept: the damage read's reference
    a["valid"] = a["kept"] if a["kept"] != "zs" else None


def served(st):
    """-> {order: {task: (served now, kept at arrival)}}: the final library's composition for an arrival still served by the
    library, the policy for one kept on the policy or fallen back to it."""
    out = {}
    for o in E4.ORDERS:
        s = st[o]
        out[o] = {str(a["task"]): (E4.compose(s, st["_meta"]["keys"][str(a["task"])]) if a["kept"] != "zs" else "zs",
                                   a.get("arrival_kept", a["kept"])) for a in s["arrivals"]}
    return out


def step(o, s, meta, has):
    if s["i"] >= len(s["order"]) and s["phase"] == "start":
        return True
    if E4.running(s["jobs"]):
        return False
    lib = s["library"]
    if s["phase"] == "revalidate":                       # read the re-checks; failures request their demonstration or fall back
        need = None
        for t in list(s["recheck"]):
            a = next(x for x in s["arrivals"] if x["task"] == t)
            c = E4.compose(s, meta["keys"][str(t)])
            acc = acc_of(t, c)
            if acc is None:
                s["jobs"] = E4.pack(t, [c], "w2_accept", 10)
                return False
            a.setdefault("rechecks", []).append(dict(comp=c, acc=acc))
            if passes(a, acc):
                a["valid"] = c
            elif not a.get("demo") and t in has and need is None:
                need = a                                 # one demonstration at a time; the others are re-checked after it
                continue
            elif not a.get("demo") and t in has:
                continue
            else:
                a["kept"], a["valid"], a["fallback"] = "zs", None, True
            s["recheck"].remove(t)
        if need is not None:
            s["recheck"].remove(need["task"])
            s["current"] = need["task"]
            s["jobs"], s["phase"] = E4.request(o, s, need, meta["keys"][str(need["task"])], True), "fit"
            need["late_demo"] = True
            E4.log(o, "re-check", need["task"], "fails; demonstration requested now", s["jobs"])
            return False
        if s["recheck"]:
            return False
        s["phase"] = "settle"
        return False
    if s["phase"] == "settle":                           # anything still changed since its last pass? re-check it
        ch = changed(s, meta)
        if ch:
            s["recheck"] = [a["task"] for a, _ in ch]
            js = []
            for a, c in ch:
                if acc_of(a["task"], c) is None:
                    js += E4.pack(a["task"], [c], "w2_accept", 10)
            s["jobs"], s["phase"] = js, "revalidate"
            E4.log(o, "re-check", s["recheck"], js)
            return False
        s["phase"], s["current"] = "start", None
        return False
    if s["phase"] == "start":
        if s["i"] >= len(s["order"]):
            return True
        t = s["order"][s["i"]]
        ks = meta["keys"][str(t)]
        comp = E4.compose(s, ks)
        a = dict(task=t, zs=E4.succ("e3_accept", t, "zs"), comp1=comp, library_before={k: e["name"] for k, e in lib.items()})
        s["arrivals"].append(a)
        s["current"] = t
        s["i"] += 1
        if comp and acc_of(t, comp) is None:
            s["jobs"] = E4.pack(t, [comp], "w2_accept", 10)
            E4.log(o, "arrival", t, "try", comp, s["jobs"])
        s["phase"] = "accept1"
        return False
    t = s["current"]
    a = next(x for x in s["arrivals"] if x["task"] == t)
    ks = meta["keys"][str(t)]
    if s["phase"] == "accept1":
        a["acc1"] = acc_of(t, a["comp1"]) if a["comp1"] else None
        if a["comp1"] and a["acc1"] is None:
            s["jobs"] = E4.pack(t, [a["comp1"]], "w2_accept", 10)
            return False
        best, best_acc = (a["comp1"], a["acc1"]) if a["acc1"] is not None and a["acc1"] > a["zs"] else ("zs", a["zs"])
        if best_acc >= THETA:
            a["demo"] = False
            keep(s, a, a["comp1"], a["acc1"]) if best != "zs" else a.update(kept="zs", valid=None, arrival_kept="zs")
            s["phase"] = "settle"
            E4.log(o, "arrival", t, "holds:", best, best_acc, "of 10 (policy", a["zs"], ") -- no demonstration")
            return False
        if t not in has:
            a.update(demo=True, demo_unavailable=True, kept="zs", valid=None, arrival_kept="zs")
            s["phase"] = "settle"
            E4.log(o, "arrival", t, "no demonstration exists; kept zs")
            return False
        s["jobs"], s["phase"] = E4.request(o, s, a, ks, True), "fit"
        E4.log(o, "arrival", t, "best", best, best_acc, "<", THETA, "-> demonstration", s["jobs"])
        return False
    if s["phase"] == "fit":
        js = []
        for k in [ks["I"], ks["O"], ks["G"]]:
            if not k or not lib[k].get("pending"):
                continue
            nm = lib[k]["pending"]
            if os.path.exists(os.path.join(R, f"prim_{nm}_0.pt")):
                lib[k]["name"], lib[k]["pending"] = nm, None
            else:
                lib[k]["v"] -= 1
                js += E4.fit(o, k, lib[k])
        if js:
            s["jobs"] = js
            return False
        a["comp2"] = E4.compose(s, ks)
        if acc_of(t, a["comp2"]) is None:
            s["jobs"] = E4.pack(t, [a["comp2"]], "w2_accept", 10)
        s["phase"] = "accept2"
        E4.log(o, "arrival", t, "accept refit", a["comp2"], s["jobs"])
        return False
    if s["phase"] == "accept2":
        a["acc2"] = acc_of(t, a["comp2"])
        if a["acc2"] is None:
            s["jobs"] = E4.pack(t, [a["comp2"]], "w2_accept", 10)
            return False
        keep(s, a, a["comp2"], a["acc2"])
        s["phase"] = "settle"
        E4.log(o, "arrival", t, "kept", a["kept"], a["acc2"], "vs", a["zs"], "(demonstration)")
        return False
    return False


def init():
    st = W1.init()
    for o in E4.ORDERS:
        st[o].update(recheck=[], current=None)
    return st


def run():
    from lifelong_ood.policy import data as D
    st = json.load(open(STATE)) if os.path.exists(STATE) else init()
    json.dump(st, open(STATE, "w"), indent=1)
    d90 = D.Demos90()
    has = {t for t in st["_meta"]["stream"] if d90.episodes_of_task(t)}
    while True:
        done = [step(o, st[o], st["_meta"], has) for o in E4.ORDERS]
        json.dump(st, open(STATE, "w"), indent=1)
        if all(done):
            break
        time.sleep(90)
    test = st.setdefault("_test", {})
    if not test.get("jobs"):
        js = []
        for o, sv in served(st).items():
            for t, (fin, arr) in sv.items():
                arms = [x for x in sorted({fin, arr} - {"zs"})
                        if all(E4.load(m, t, x) is None for m in ("w2_main", "w1_main", "e4_main"))]
                if arms:
                    js += E4.pack(t, arms, "w2_main", 20, fresh=True)
        test["jobs"] = js
        json.dump(st, open(STATE, "w"), indent=1)
        E4.log("test packs", len(js))
    while E4.running(test["jobs"]):
        time.sleep(120)
    E4.log("stream done")


if __name__ == "__main__":
    if "--init" in sys.argv:
        json.dump(init(), open(STATE, "w"), indent=1)
    if "--run" in sys.argv:
        run()
    if "--read" in sys.argv:
        E4.served = served                              # served now / kept at arrival, falls back included
        E4.read("--record" in sys.argv, state=STATE, mains=("w2_main", "w1_main", "e4_main"), ledger_name="w2_stream", out="w2_read.json")
