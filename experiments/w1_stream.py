"""Plan W, read W1 (docs/PLAN_WANG.md): E4's stream with the REQUEST RULE the thesis states -- a demonstration is requested only
when what the learner already holds fails its own practice.  What it holds is the policy AND the library's composition: the
arrival keeps the better of the two (ties: the policy) without a demonstration when that one succeeds in at least THETA of its 10
practice episodes; otherwise it requests its one demonstration, which updates every concept on its path (instance child, type
concept, goal concept: refit on all their demonstrations), and the refit composition is kept only if it beats the policy.

E4's rule requested a demonstration whenever the composition did not beat the policy, so tasks the policy already solves (10 of 10)
requested one, and a composition that beat a failing policy by one episode (1 vs 0) was kept without one.  Same orders and prior
library as E4 (its prior fits are reused; so are its acceptance runs of an identical composition); noise seed 13; served arms
tested on E3's fresh start states (set b).  Resumable: results/robot/w1_state.json.

    nohup ~/flash/miniconda3/envs/chunks/bin/python experiments/w1_stream.py --run > results/robot/w1_driver.log 2>&1 &
    ~/flash/miniconda3/envs/chunks/bin/python experiments/w1_stream.py --read [--record]
"""
import copy
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec_ = importlib.util.spec_from_file_location("e4", os.path.join(ROOT, "experiments", "e4_stream.py"))
E4 = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(E4)

R = E4.R
STATE = os.path.join(R, "w1_state.json")
THETA = 8
E4.name = lambda o, k, e: f"W1{o}{k[0]}{e['idx']}v{e['v']}"           # W1's own refits; the prior keeps E4's names


def init():
    e4 = json.load(open(E4.STATE))
    st = {"_meta": e4["_meta"]}
    for o in E4.ORDERS:
        s = copy.deepcopy(e4[o])
        for k, e in s["library"].items():                                # the prior library as E4 fitted it (v1), nothing else
            e["tasks"] = [t for t in e["tasks"] if t not in st["_meta"]["stream"]]
            if e["tasks"]:
                e["name"], e["v"] = f"E4{o}{k[0]}{e['idx']}v1", 1
                assert os.path.exists(os.path.join(R, f"prim_{e['name']}_0.pt")), e["name"]
            else:
                e["name"], e["v"] = None, 0
            e["jobs"], e["pending"] = [], None
        st[o] = dict(order=s["order"], i=0, phase="start", jobs=[], library=s["library"], arrivals=[], demos=0)
    return st


def acc_of(t, arm):
    for exp in ("w1_accept", "e4_accept"):
        x = E4.succ(exp, t, arm)
        if x is not None:
            return x
    return None


def step(o, s, meta, has):
    if s["i"] >= len(s["order"]):
        return True
    if E4.running(s["jobs"]):
        return False
    lib = s["library"]
    t = s["order"][s["i"]]
    ks = meta["keys"][str(t)]
    zs = E4.succ("e3_accept", t, "zs")
    if s["phase"] == "start":
        comp = E4.compose(s, ks)
        a = dict(task=t, zs=zs, comp1=comp, library_before={k: e["name"] for k, e in lib.items()})
        s["arrivals"].append(a)
        if comp and acc_of(t, comp) is None:
            s["jobs"], s["phase"] = E4.pack(t, [comp], "w1_accept", 10), "accept1"
            E4.log(o, "arrival", t, "try", comp, s["jobs"])
        else:
            s["phase"] = "accept1"
        return False
    a = s["arrivals"][-1]
    if s["phase"] == "accept1":
        a["acc1"] = acc_of(t, a["comp1"]) if a["comp1"] else None
        if a["comp1"] and a["acc1"] is None:
            s["jobs"] = E4.pack(t, [a["comp1"]], "w1_accept", 10)
            return False
        best, best_acc = (a["comp1"], a["acc1"]) if a["acc1"] is not None and a["acc1"] > zs else ("zs", zs)
        if best_acc >= THETA:
            a.update(demo=False, kept=best)
            s["i"], s["phase"] = s["i"] + 1, "start"
            E4.log(o, "arrival", t, "holds:", best, best_acc, "of 10 (policy", zs, ") -- no demonstration")
            return False
        s["jobs"], s["phase"] = E4.request(o, s, a, ks, t in has), "fit"
        E4.log(o, "arrival", t, "best", best, best_acc, "<", THETA, "-> demonstration", s["jobs"])
        return False
    if s["phase"] == "fit":
        if a.get("demo_unavailable"):
            a["kept"] = a["comp1"] if a["acc1"] is not None and a["acc1"] > zs else "zs"
            s["i"], s["phase"] = s["i"] + 1, "start"
            E4.log(o, "arrival", t, "no demonstration exists; kept", a["kept"])
            return False
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
        s["jobs"], s["phase"] = E4.pack(t, [a["comp2"]], "w1_accept", 10), "accept2"
        E4.log(o, "arrival", t, "accept refit", a["comp2"], s["jobs"])
        return False
    if s["phase"] == "accept2":
        a["acc2"] = acc_of(t, a["comp2"])
        if a["acc2"] is None:
            s["jobs"] = E4.pack(t, [a["comp2"]], "w1_accept", 10)
            return False
        a["kept"] = a["comp2"] if a["acc2"] > zs else "zs"
        s["i"], s["phase"] = s["i"] + 1, "start"
        E4.log(o, "arrival", t, "kept", a["kept"], a["acc2"], "vs", zs, "(demonstration)")
        return False
    return False


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
        for o, sv in E4.served(st).items():
            for t, (fin, arr) in sv.items():
                arms = sorted({fin, arr} - {"zs"})
                arms = [x for x in arms if E4.load("e4_main", t, x) is None and E4.load("w1_main", t, x) is None]
                if arms:
                    js += E4.pack(t, arms, "w1_main", 20, fresh=True)
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
        E4.read(False, state=STATE, mains=("w1_main", "e4_main"), ledger_name="w1_stream", out="w1_read.json")
        res = json.load(open(os.path.join(R, "w1_read.json")))
        w1, e4 = json.load(open(STATE)), json.load(open(E4.STATE))
        sw, se = E4.served(w1), E4.served(e4)
        for o in E4.ORDERS:                                              # H2: W1's served arms against E4's, same tasks and states
            def arm(t, a):
                return E4.load("e3_main", t, "zs") if a == "zs" else next(
                    (z for z in (E4.load(m, t, a) for m in ("w1_main", "e4_main")) if z is not None), None)
            tab = {t: {"x": arm(t, sw[o][t][0]), "y": arm(t, se[o][t][0])} for t in sw[o] if t in se[o]}
            if any(v["x"] is None or v["y"] is None for v in tab.values()):
                res["missing"].append(f"{o}:vs_e4")
                continue
            h2 = E4.E1.compare(tab)
            h2["noninferior"] = bool(h2["diff"]["lo"] > -0.05)
            res[o]["vs_e4"] = h2
            res[o]["e4_demos"] = sum(bool(a.get("demo")) and not a.get("demo_unavailable") for a in e4[o]["arrivals"])
            print(f"[w1] order {o}: served {h2['x']} vs E4's {h2['y']} of {h2['n']}, diff {h2['diff']['diff']:+.3f} "
                  f"[{h2['diff']['lo']:+.3f}, {h2['diff']['hi']:+.3f}]; demos {res[o]['demos']} vs E4's {res[o]['e4_demos']}", flush=True)
        json.dump(res, open(os.path.join(R, "w1_read.json"), "w"), indent=1)
        if "--record" in sys.argv and not res["missing"]:
            from lifelong_ood import ledger
            print(ledger.record("w1_stream", bench="libero_90", tag="read", seed=E4.SEED, metrics=res,
                                bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config=dict(theta=THETA),
                                script="experiments/w1_stream.py", command=" ".join(sys.argv)))
