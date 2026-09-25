"""Tables for the c-series, straight from the result files: mean +- SE over seeds.

    python experiments/c_report.py c1 results/c1_cobweb_update_colormnist_FINAL__seed*.json
    python experiments/c_report.py c2 results/c2_router_loop_colormnist_FINAL__seed*.json

C1: per arm, the LAST task's taught / untaught / old-class numbers, and the mean over tasks of the
untaught joint (how early the value became addressable).  C2: per population, the tiers the router
chose, what was stored, its accuracy at storage and at the end of the stream, and the cost.
No number here is typed by hand; the documents quote this output.
"""
import json
import sys

import numpy as np


def mse(x):
    x = np.asarray([v for v in x if v == v], float)
    if not len(x):
        return "   -   "
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{x.mean():.3f}±{se:.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def c1(files):
    runs = [json.load(open(f)) for f in files]
    arms = []
    for r in runs:
        for row in r["rows"]:
            if row["arm"] not in arms:
                arms.append(row["arm"])
    nf = runs[0].get("new_factor", "new")
    seeds = sorted({r["config"]["seed"] for r in runs})
    print(f"C1 {runs[0].get('bench', '?')} -- seeds {seeds} ({len(runs)} files); new value on "
          f"'{nf}'; last task unless noted\n")
    hdr = (f"{'arm':30s} {'taught':>12s} {'t-new':>12s} {'untaught':>12s} {'u-new':>12s} "
           f"{'untaught(mean)':>14s} {'old drop':>12s} {'drift %':>10s} {'changed %':>10s} "
           f"{'added B':>9s} {'steps':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for a in arms:
        last = [max((x for x in r["rows"] if x["arm"] == a), key=lambda x: x["task"])
                for r in runs if any(x["arm"] == a for x in r["rows"])]
        allt = [[x for x in r["rows"] if x["arm"] == a] for r in runs]
        um = [np.mean([x["untaught_joint"] for x in rs]) for rs in allt if rs]
        steps = [np.mean([x["steps_weight"] for x in rs]) for rs in allt if rs]
        tn = [x.get("taught_new", x.get("taught_bg")) for x in last]
        un = [x.get("untaught_new", x.get("untaught_bg")) for x in last]
        # a consolidating arm folds its adapter into the backbone and zeroes it: nothing is kept
        added = [0 if a.startswith("lora-consol") else 4 * x["lora_params"] for x in last]
        print(f"{a[:30]:30s} {mse([x['taught_joint'] for x in last]):>12s} {mse(tn):>12s} "
              f"{mse([x['untaught_joint'] for x in last]):>12s} {mse(un):>12s} {mse(um):>14s} "
              f"{mse([x['old_drop'] for x in last]):>12s} "
              f"{mse([100 * x['drift_mean'] for x in last]):>10s} "
              f"{mse([100 * x['changed_frac'] for x in last]):>10s} {int(np.mean(added)):>9d} "
              f"{np.mean(steps):>6.0f}")


def c2(files):
    runs = [json.load(open(f)) for f in files]
    print(f"C2 -- {len(runs)} seed(s)\n")
    for pop in ("in-dist", "new-comb", "new-value"):
        tiers, joint, jend, cost, ok = {}, [], [], [], []
        for r in runs:
            rs = [x for x in r["records"] if x["pop"] == pop]
            for x in rs:
                tiers[x["tier"]] = tiers.get(x["tier"], 0) + 1
            if rs:
                joint.append(np.mean([x["joint"] for x in rs]))
                jend.append(np.mean([x["joint_end"] for x in rs]))
                cost.append(np.sum([x["cost_bytes"] for x in rs]) / len(rs))
                ok.append(np.mean([x.get("stored_ok", True) for x in rs]))
        print(f"{pop:10s} tiers {tiers} | joint at storage {mse(joint)} end {mse(jend)} | "
              f"stored as the right tuple {mse(ok)} | bytes/concept {np.mean(cost):.0f}")
    print()
    for key in ("old_joint_start", "old_joint_end", "changed_frac", "bytes_total"):
        if all(key in r["summary"] for r in runs):
            print(f"{key:18s} {mse([r['summary'][key] for r in runs])}")
    # tier of each new-value concept by its position among new-value arrivals: does it get cheaper?
    pos = {}
    for r in runs:
        k = 0
        for x in r["records"]:
            if x["pop"] == "new-value":
                pos.setdefault(k, []).append(x["tier"])
                k += 1
    print("\nnew-value concepts by arrival order (tiers over seeds):")
    for k in sorted(pos):
        c = {}
        for t in pos[k]:
            c[t] = c.get(t, 0) + 1
        print(f"  #{k + 1}: {c}")


def c4(files):
    """Each router decision alone: D1 novel?, D2 address or node?, D3 which factor."""
    for f in files:
        r = json.load(open(f))
        s, cfg = r["summary"], r["config"]
        print(f"C4 {cfg['bench']} (leaf mode {cfg.get('leaf_mode', 'free')}) -- {f}")
        print(f"  D1 in-dist vs novel, row-ratio AUROC {s['D1_auroc_row']:.3f}; IN-DIST rate "
              + " ".join(f"{k} {v:.2f}" for k, v in s["D1_indist_rate"].items()))
        print(f"  D2 new-comb vs new-value AUROC: best-address ratio {s['D2_auroc_addr']:.3f}, "
              f"informativeness {s['D2_auroc_inform']:.3f}")
        print(f"     ADDRESS rate, gamma rule ({s['gamma']:.3f}): "
              + " ".join(f"{k} {v:.2f}" for k, v in s["D2_gamma_rule"].items()))
        for kap, d in s["D2_kappa_rule"].items():
            print(f"     ADDRESS rate, ratio <= {kap}: " + " ".join(f"{k} {v:.2f}" for k, v in d.items()))
        print("     basis read exactly right: " + " ".join(f"{k} {v:.2f}" for k, v in s["D2_read_ok"].items()))
        # compose vs retrieve: the best address's worst level over the best trained row's
        cr = {pop: np.array([x["addr_ratio"] / x["row_ratio"] for x in r["records"] if x["pop"] == pop])
              for pop in ("in-dist", "new-comb", "new-value")}
        au = float((cr["new-value"][:, None] > cr["new-comb"][None, :]).mean()
                   + 0.5 * (cr["new-value"][:, None] == cr["new-comb"][None, :]).mean())
        print(f"     compose-vs-retrieve (address / row): AUROC {au:.3f}; " + " ".join(
            f"{pop} [{v.min():.2f}, {v.max():.2f}]" for pop, v in cr.items() if len(v)))
        for rho in (0.8, 0.9, 0.95):
            print(f"     ADDRESS rate, address <= {rho} x row: " + " ".join(
                f"{pop} {np.mean(v <= rho):.2f}" for pop, v in cr.items() if len(v)))
        recs = r["records"]
        for pop in ("in-dist", "new-comb", "new-value"):
            ar = np.array([x["addr_ratio"] for x in recs if x["pop"] == pop])
            im = np.array([x["inf_min"] for x in recs if x["pop"] == pop])
            if len(ar):
                print(f"     {pop:9s} n={len(ar):2d} best-address ratio median {np.median(ar):6.2f} "
                      f"[{ar.min():.2f}, {ar.max():.2f}]  inform min median {np.median(im):.2f}")
        nv = [x for x in recs if x["pop"] == "new-value"]
        nfac = len(nv[0]["true"]) if nv else 0
        if nv:
            new_f = next(k for k in range(nfac) if len({x["true"][k] for x in nv}) == 1)
            oth = np.mean([all(x["read"][h] == x["true"][h] for h in range(nfac) if h != new_f)
                           for x in nv])
            print(f"  D3a informativeness names the new factor: {s['D3a_inform_acc']:.2f}; the basis "
                  f"read gets the OTHER factors of a new-value concept right: {oth:.2f}")
            if "loo_reads" in nv[0]:
                # JSON keys are strings: loo_reads[f] maps each OTHER factor to its read value
                loo = np.mean([all(int(x["loo_reads"][new_f][str(h)]) == x["true"][h]
                                   for h in range(nfac) if h != new_f) for x in nv])
                print(f"      ... and with the new factor LEFT OUT of the read: {loo:.2f}")
        if "D3b" in s:
            d = s["D3b"]
            print(f"  D3b leaf offered on every factor ({len(s['leaves'])} leaves): new-value targets "
                  f"merge {d['merge_rate']:.2f}; merged on the right factor {d['factor_acc_given_merge']:.2f}"
                  f" (best factor right, merge or not: {d['factor_acc_any']:.2f})")
            print(f"      joint of the merged composition {d['joint_best']:.3f}; on the right factor "
                  f"{d['joint_true_f']:.3f}; other factors read right there {d['other_read_ok']:.2f}")
            print("      false merges: " + " ".join(f"{k} {v:.2f}" for k, v in d["false_merge"].items()))
            for lf in s["leaves"]:
                parts = lf.get("parts") or [lf]
                print(f"      leaf from {tuple(lf['true'])}: " + "; ".join(
                    f"{p_.get('f') or 'all'} {p_['steps']} steps absorbed {p_['absorbed']} own joint "
                    f"{p_['own_joint']:.2f}" for p_ in parts))
        print()


def c5(files):
    """C2's table for the Cobweb loop, plus what the leaves did."""
    c2(files)
    runs = [json.load(open(f)) for f in files]
    ut = [r["summary"]["untaught"] for r in runs if "untaught" in r["summary"]]
    if ut:
        print(f"\nnever-taught new-value combinations: addressable {mse([u['addressable'] for u in ut])}"
              f", joint {mse([u['joint'] for u in ut])}")
    print("\nleaves and merges:")
    for r in runs:
        s = r["summary"]
        print(f"  seed {r['config']['seed']}: {s['n_leaves']} leaves, {s['n_merged']} merged, "
              f"{s['n_blocks']} left as blocks; bytes {s['bytes_total']}; merges " + "; ".join(
                  f"{m['src_pop']}#{m['src']} + {m['by_pop']}#{m['by']} on {m['factor']}"
                  f"{' (right)' if m['right_factor'] else ''}" for m in s["merges"]))


if __name__ == "__main__":
    kind, files = sys.argv[1], sys.argv[2:]
    {"c1": c1, "c2": c2, "c4": c4, "c5": c5}[kind](files)
