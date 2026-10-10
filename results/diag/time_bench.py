import os, sys, time
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
t0 = time.time()
from lifelong_ood.data.datasets import load_colormnist
d = load_colormnist(n_per_slot=200, seed=0); print(f"load_colormnist {time.time()-t0:.1f}s X {d['X'].shape}", flush=True)
t0 = time.time()
from lifelong_ood.neural.ddpm import load_field
f = load_field("/coc/flash10/ksingara3/lifelong-ood-learner/results/cache/ddpm_colormnist_nobg3.pt", device="cpu"); print(f"load_field {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
from lifelong_ood.evals import fit_probes
pr = fit_probes(d, seed=0, epochs=2); print(f"fit_probes epochs=2 {time.time()-t0:.1f}s -> {[(k, round(v,3)) for k, v in pr.items() if k.endswith('val_acc')]}", flush=True)
t0 = time.time()
from lifelong_ood.streams.bench import load_bench
B = load_bench("colormnist", seed=0, device="cpu", probe_epochs=2); print(f"load_bench(probe_epochs=2) {time.time()-t0:.1f}s; seen {len(B.seen)}", flush=True)
