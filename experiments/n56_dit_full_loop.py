"""N56 -- the DiT with the FULL loop: infer the composition from one query, then compose it.

N55/D3/D4 report ADDRESS on the DiT at joint 0.807, and that number is narrower than it sounded.
It calls `compose(fac, (d, f, b))` with the factor values decoded from the held-out SLOT ID -- the
ground truth.  It answers "can the DiT's basis express this composition", not "can the system work
out which composition is needed".  Discovery was skipped.

This runs the loop the UNet actually runs (`learn_concept_factored` -> `compose_factored` ->
`learn_factored`): from ONE held-out query and NO labels, optimise one softmax per factor so that the
composed conditioning best explains that query, read the argmax as the inferred composition, compose
it, and generate.  The latent analogue of `idealwords.learn_factored`, against the DiT's own score.

Three numbers, and the middle one is the one that was missing:

    READ accuracy      did the inversion recover the right factor values from one query, no labels
    FULL-LOOP joint    accuracy of samples generated from the INFERRED composition
    ORACLE joint       the same but composed from ground-truth values -- N55's 0.807, the ceiling

The gap between the last two is exactly what discovery costs, and reporting only the oracle (as N55
did) overstates the pipeline.

    python experiments/n56_dit_full_loop.py --n-cases 12
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from common import CACHE, RESULTS                                      # noqa: E402

from lifelong_ood.data.datasets import load_colormnist, slot_name           # noqa: E402
from lifelong_ood.evals import fit_probes, save_json                   # noqa: E402
from lifelong_ood.neural.dit import DiT                                # noqa: E402
from lifelong_ood.neural.idealwords import compose, factorize, slot_factors  # noqa: E402
from lifelong_ood.neural.latent import ConvAE, decode, encode          # noqa: E402
from lifelong_ood.score import cosine_abar                             # noqa: E402
from n55_dit_train import ddim_latent                                  # noqa: E402


def learn_factored_latent(model, z_q, fac, n_steps=300, lr=0.05, seed=0,
                          u_range=(0.05, 0.95), gains=None, device="cuda"):
    """`idealwords.learn_factored` against a DiT in latent space.

    One softmax per factor (10 + 4 + 4 = 18 parameters here), optimised so the composed conditioning
    minimises the model's epsilon error on the query.  Composition is structural rather than
    something the optimiser has to discover, which is why one query is enough to determine 18
    parameters where it could never determine 120.

    -> (c_new [d], per-factor weight vectors, argmax read).
    """
    F = len(fac["V"])
    Vt = [torch.as_tensor(np.asarray(v, np.float32), device=device) for v in fac["V"]]
    mu = torch.as_tensor(np.asarray(fac["mu"], np.float32), device=device)
    gain = [1.0] * F if gains is None else [float(x) for x in gains]
    logits = [torch.zeros(fac["sizes"][f], device=device, requires_grad=True) for f in range(F)]
    opt = torch.optim.Adam(logits, lr=lr)
    z0 = torch.as_tensor(np.asarray(z_q, np.float32), device=device).reshape(1, *z_q.shape)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(n_steps):
        c = mu.clone()
        for f in range(F):
            c = c + gain[f] * (torch.softmax(logits[f], 0) @ Vt[f])
        u = float(np.random.default_rng(int(torch.randint(0, 10 ** 6, (1,), generator=g,
                                                          device=device).item())).uniform(*u_range))
        a = float(cosine_abar(np.array([u]))[0])
        e = torch.randn(z0.shape, generator=g, device=device)
        zt = (a ** 0.5) * z0 + ((1 - a) ** 0.5) * e
        pred = model(zt, torch.full((1,), u, device=device), emb_override=c.reshape(1, -1))
        loss = ((pred - e) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ws = [torch.softmax(l, 0).cpu().numpy() for l in logits]
        c = mu.clone()
        for f in range(F):
            c = c + gain[f] * (torch.as_tensor(ws[f], device=device, dtype=torch.float32) @ Vt[f])
    return c.detach().cpu().numpy(), ws, [int(np.argmax(w)) for w in ws]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit", default=os.path.join(CACHE, "dit_colormnist.pt"))
    p.add_argument("--ae", default=os.path.join(CACHE, "ae_colormnist.pt"))
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--inv-steps", type=int, default=300)
    p.add_argument("--inv-lr", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-per-slot", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = ConvAE(zdim=ack["zdim"], base=ack["base"]).to(dev)
    ae.load_state_dict(ack["model"]); ae.eval()
    scale = float(ack["scale"])
    dck = torch.load(args.dit, map_location="cpu", weights_only=False)
    dit = DiT(n_classes=dck["n_classes"], in_ch=dck["zdim"], size=8, patch=dck["patch"],
              d=dck["dim"], depth=dck["depth"]).to(dev)
    dit.load_state_dict(dck["model"]); dit.eval()
    for q in dit.parameters():
        q.requires_grad_(False)
    slots_seen = np.array(sorted(int(k) for k in dck["remap"]))
    fac = factorize(dit.class_table().cpu().numpy(), slot_factors(slots_seen))
    print(f"[n56] DiT basis R^2 {fac['r2']:.3f}, factor sizes {fac['sizes']}", flush=True)

    data = load_colormnist(n_per_slot=args.n_per_slot, seed=args.seed)
    probes = fit_probes(data, seed=args.seed, epochs=30)
    X = np.asarray(data["X"], np.float32)
    Z = encode(ae, X, scale=scale, device=dev)
    held = np.array(sorted(set(int(s) for s in np.unique(data["slot"])) - set(slots_seen.tolist())))
    rng = np.random.default_rng(args.seed)
    cases = rng.permutation(held)[:args.n_cases]
    zsh = (ack["zdim"], 8, 8)

    def score(S, d_, f_, b_):
        S = np.asarray(S, np.float32)
        return dict(joint=float(((probes["digit"].predict(S) == d_)
                                 & (probes["fg"].predict(S) == f_)
                                 & (probes["bg"].predict(S) == b_)).mean()),
                    digit=float((probes["digit"].predict(S) == d_).mean()),
                    fg=float((probes["fg"].predict(S) == f_).mean()),
                    bg=float((probes["bg"].predict(S) == b_).mean()))

    rows = []
    for ci, s in enumerate(cases):
        d_, f_, b_ = int(s) // 16, (int(s) % 16) // 4, int(s) % 4
        i0 = int(np.where(data["slot"] == int(s))[0][0])
        zq = Z[i0].numpy()

        # FULL LOOP: read the composition off ONE query, no labels anywhere
        c_read, ws, read = learn_factored_latent(dit, zq, fac, n_steps=args.inv_steps,
                                                 lr=args.inv_lr, seed=args.seed + ci, device=dev)
        ok = [int(read[0] == d_), int(read[1] == f_), int(read[2] == b_)]
        S_full = decode(ae, ddim_latent(dit, c_read, args.n_samples, zsh, steps=args.steps,
                                        seed=args.seed, device=dev).cpu(), scale=scale, device=dev)
        # ORACLE: the same composition built from ground-truth values -- N55's number
        c_true = compose(fac, (d_, f_, b_))
        S_or = decode(ae, ddim_latent(dit, c_true, args.n_samples, zsh, steps=args.steps,
                                      seed=args.seed, device=dev).cpu(), scale=scale, device=dev)

        a_full, a_or = score(S_full, d_, f_, b_), score(S_or, d_, f_, b_)
        rows.append(dict(slot=int(s), read=read, truth=[d_, f_, b_], read_ok=ok,
                         full=a_full, oracle=a_or))
        print(f"[n56] {ci + 1:2d}/{len(cases)} slot {int(s):3d} {slot_name(d_, f_, b_):32s} "
              f"read ({read[0]},{read[1]},{read[2]}) vs ({d_},{f_},{b_}) "
              f"{'.'.join('Y' if o else 'n' for o in ok)} | "
              f"full {a_full['joint']:.3f}  oracle {a_or['joint']:.3f}", flush=True)

    R = np.array([r["read_ok"] for r in rows])
    mf = {k: float(np.mean([r["full"][k] for r in rows])) for k in ("joint", "digit", "fg", "bg")}
    mo = {k: float(np.mean([r["oracle"][k] for r in rows])) for k in ("joint", "digit", "fg", "bg")}
    print("\n[n56] " + "=" * 70)
    print(f"[n56] READ accuracy from ONE query, no labels: "
          f"digit {R[:, 0].mean():.3f}  fg {R[:, 1].mean():.3f}  bg {R[:, 2].mean():.3f}  "
          f"all-three {float(np.all(R, 1).mean()):.3f}")
    print(f"[n56] FULL LOOP joint {mf['joint']:.3f}  (digit {mf['digit']:.3f} fg {mf['fg']:.3f} "
          f"bg {mf['bg']:.3f})")
    print(f"[n56] ORACLE    joint {mo['joint']:.3f}  (digit {mo['digit']:.3f} fg {mo['fg']:.3f} "
          f"bg {mo['bg']:.3f})   <- what N55 reported")
    print(f"[n56] discovery costs {mo['joint'] - mf['joint']:+.3f} joint")
    save_json(dict(rows=rows, full=mf, oracle=mo, basis_r2=float(fac["r2"]),
                   read_acc=dict(digit=float(R[:, 0].mean()), fg=float(R[:, 1].mean()),
                                 bg=float(R[:, 2].mean()), all=float(np.all(R, 1).mean())),
                   config=vars(args)),
              os.path.join(RESULTS, "n56_dit_full_loop.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
